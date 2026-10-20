"""Phase 4 runner: Module III (proposal Fig. 2) on every rep of the in-scope exercises.

  1. Reference template per exercise from the primary PT demo (``references.primary``; 30 fps grid, 3D angles,
     channels for the moving arm so left arms are mirrored): template.build_reference (M1, M4).
  2. Every rep is DTW-aligned to the template once (the alignment does not depend on any fitted value).
  3. Per GroupKFold-5 fold (the Phase 3 folds, E1): the library (view bias, bands, ROM and speed tolerances) is fitted
     on the training subjects' expert reps of videos scoring >= templates.min_score, and the test subjects' reps are
     compared, scored and checked for errors with it (M2, M8-M12). The deployable library is fitted on all subjects
     and saved to templates.library_dir (demo).
  4. Per-video JSON reports (M13), with the embedding-to-reference distance of ``eval.proposed_default`` if
     outputs/phase4/embedding_distance_reps.csv exists (scripts/phase4/embedding_distance.py, run first).

Full scope (M17): with ``configs/per_exercise.yaml`` (T17) the folds are each exercise's own GroupKFold-5 (the folds
of the per-exercise models), every discovered exercise is assessed (``--exercises`` for a subset), E03's template
averages both demo arms over the 90/90 working phase (M3), and E06's wrist channel is the Hands angle of the hand the
side camera sees (P1c; front views: no wrist channel), with that arm's other channels.

usage: python -m cvdc.assess.run [--config C] [--exercises E03 ...] [--workers 8] [--posture-centre pt_demo|experts]
       [--out DIR] [--no-json]
Writes <out> (default: the folder above ``assess.json_dir``, outputs/phase4/ for the pilot): module3_reps.csv,
module3_errors.csv, module3_videos.csv, reference_curves.csv, library_fold_counts.csv, json/<video_id>.json;
<templates.library_dir>/<EX>.{npz,json}. ``--posture-centre`` overrides
``assess.posture_centre`` (M14); the before / after comparison runs the old setting with
``--posture-centre pt_demo --out outputs/phase4/posture_pt --no-json``.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ..config import data_path, exercise_cfg, load_config, resolve
from ..eval.baselines import hand_angle_2d
from ..data.labels import movement_quality
from ..eval.splits import check_no_leakage
from ..train.trainer import fold_plan
from .module3 import assess_config, evaluate, rep_entry, set_errors, validate_report, video_report
from .angles import ASSESS_ANGLES, other
from .template import aligned_rep, build_reference, fit_library, rep_signals, save_library

ITEMS = ["PF1", "PF2", "PF3", "PF4", "CF1", "CF2", "CF3"]
OUT = "outputs/phase4"


def hands_extra(ac, view, kp_path, hand_path, n, side):
    """E06 (P1c): {'wrist_extension': (n,)} = the Hands wrist angle of the hand the side camera sees, and the side to
    assess (that arm); front views (or no Hands data): all-NaN and the rep's own side. Other exercises: (None, side)."""
    if ac.get("angle_source") != "hands":
        return None, side
    if view in (ac.get("angle_views") or []) and isinstance(hand_path, str) and hand_path:
        h = ac["near_hand"][view]
        return {ac["angle"]: hand_angle_2d(kp_path, hand_path, h, n)}, h
    return {ac["angle"]: np.full(n, np.nan)}, side


def working_phase(world, start, end, side, wp):
    """M3: the longest run of frames of start..end where channel ``wp['channel']`` >= ``wp['min_deg']`` (E03's 90/90
    phase of the demo), at least 10 frames. Returns (start, end) or None."""
    a = ASSESS_ANGLES[wp["channel"]](np.asarray(world[start:end + 1], float), side)
    ok = np.r_[False, a >= wp["min_deg"], False]
    d = np.diff(ok.astype(int))
    runs = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))
    if not runs:
        return None
    a0, b0 = max(runs, key=lambda r: r[1] - r[0])
    return (start + int(a0), start + int(b0)) if b0 - a0 >= 9 else None


def reference_for(cfg, ex, ac):
    """(template, PT file name) from the exercise's primary PT demo; with ``template_both_sides`` both arms of every
    demo rep (the other one mirrored), with ``template_working_phase`` cut to that phase (M3)."""
    ref_idx = pd.read_csv(resolve(cfg["data"]["ref_index_csv"]))
    rr = pd.read_csv(data_path(cfg, "reps_ref_csv"))
    row = ref_idx[(ref_idx.exercise == ex) & (ref_idx.role == "primary")].iloc[0]
    world = np.load(resolve(row.pre_path))["world"].astype(float)
    reps = rr[(rr.ref_id == row.ref_id) & (rr.exercise == ex)]
    sigs = []
    for r in reps.itertuples():
        for side in ([r.side, other(r.side)] if ac.get("template_both_sides") else [r.side]):
            se = (r.start, r.end)
            if ac.get("template_working_phase"):
                se = working_phase(world, r.start, r.end, side, ac["template_working_phase"])
                if se is None:
                    continue
            extra = None
            if ac.get("angle_source") == "hands":  # the demo's own hand of the moving side (front-oblique view)
                hp = row.hand_path if isinstance(row.hand_path, str) else ""
                extra = ({ac["angle"]: hand_angle_2d(row.kp_path, hp, side, len(world))} if hp else
                         {ac["angle"]: np.full(len(world), np.nan)})
            sigs.append(rep_signals(world, se[0], se[1], side, ac, ac["fps"], extra))
    meta = {"file": row.file, "ref_id": row.ref_id, "side": reps.side.iloc[0], "pt_reps_used": len(sigs)}
    return build_reference(sigs, ac, ac["fps"], meta), row.file


def video_job(args):
    """Signals and alignment of every rep of one video (worker)."""
    vid, pre_path, rows, ac, ref, view, kp_path, hand_path = args
    world = np.load(pre_path)["world"].astype(float)
    out = []
    for r in rows:
        extra, side = hands_extra(ac, view, kp_path, hand_path, len(world), r["side"])
        sig = rep_signals(world, r["start"], r["end"], side, ac, ac["fps"], extra)
        out.append((r["rep"], sig, aligned_rep(sig, ref)))
    return vid, out


def flat_row(r, sig, s, X, meas, errs):
    row = dict(video_id=r.video_id, rep=int(r.rep), exercise=r.exercise, subject=r.subject, group=r.group,
               view=r.view, variation=r.variation, side=sig["side"], start=sig["start"], end=sig["end"],
               duration_s=sig["duration_s"], rom=sig["rom"], excursion=sig["excursion"], ref_score=s,
               **{f"X_{g}": x for g, x in X.items()}, **meas, n_errors=len(errs))
    for e in errs:
        row[f"err_{e['type']}"] = 1
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--exercises", nargs="+", help="default: data.include_exercises, or every exercise if empty")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--posture-centre", choices=["pt_demo", "experts"], help="override assess.posture_centre (M14)")
    ap.add_argument("--out", help=f"output directory (default: the folder above assess.json_dir; pilot {OUT})")
    ap.add_argument("--no-json", action="store_true", help="skip the per-video JSON files")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    if a.posture_centre:
        cfg["assess"]["posture_centre"] = a.posture_centre
    out = resolve(a.out) if a.out else resolve(cfg["assess"]["json_dir"]).parent
    (out / "json").mkdir(parents=True, exist_ok=True)
    print(f"posture checks centred on: {cfg['assess'].get('posture_centre', 'pt_demo')}; output {out}")
    tc = cfg["templates"]
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    idx["MQ"] = movement_quality(idx, cfg["labels"]["movement_quality_items"])
    reps = pd.read_csv(data_path(cfg, "reps_csv"))
    rpv = pd.read_csv(resolve(cfg["outputs"]["phase1"]) / "reps_per_video.csv").set_index("video_id")
    per_ex = bool(cfg["eval"].get("per_exercise_methods"))  # T17: each exercise's own folds

    def plans_for(ex):
        r = reps[reps.exercise == ex] if per_ex else reps
        pl = fold_plan(cfg, r)
        check_no_leakage([(p["train"], p["test"]) for p in pl], r.subject.unique())
        return pl

    pre = idx.set_index("video_id").pre_path
    vrow = idx.set_index("video_id")
    vscore = idx.set_index("video_id").score
    emb_path = out / "embedding_distance_reps.csv"
    emb = None
    if emb_path.exists():
        e = pd.read_csv(emb_path)
        e = e[(e.method == cfg["eval"]["proposed_default"]) & e.is_test]  # the fold where the subject is held out
        emb = {(r.video_id, int(r.rep)): {"model": r.method, "fold": int(r.fold), "cosine_centred": round(r.d_cos, 4),
                                           "l2": round(r.d_l2, 3)} for r in e.itertuples()}
    print(f"embedding distances for the JSON: {'yes (' + cfg['eval']['proposed_default'] + ')' if emb else 'no'}")

    rep_rows, err_rows, vid_rows, curve_rows, count_rows = [], [], [], [], []
    exercises = a.exercises or cfg["data"]["include_exercises"] or sorted(reps.exercise.unique())
    for ex in exercises:
        plans = plans_for(ex)
        fold_of = {s: p["fold"] for p in plans for s in p["test"]}
        ex_cfg = exercise_cfg(cfg, ex)
        ac = assess_config(cfg, ex)
        ref, ref_file = reference_for(cfg, ex, ac)
        for c in ref["channels"]:
            curve_rows += [dict(exercise=ex, channel=c, frame=t, pt=v) for t, v in enumerate(ref["curves"][c])]
        print(f"{ex}: template from {ref_file!r}: {ref['n_pt_reps']} PT reps {ref['pt_durations_s']} s -> "
              f"{ref['n']} frames, end-range ROM {ref['rom']:.1f} deg, excursion {ref['excursion']:.1f} deg ({ref['main']})")
        R = reps[reps.exercise == ex].reset_index(drop=True)
        jobs = [(vid, str(resolve(pre[vid])), d[["rep", "start", "end", "side"]].to_dict("records"), ac, ref,
                 vrow.loc[vid, "view"], vrow.loc[vid, "kp_path"],
                 vrow.loc[vid, "hand_path"] if "hand_path" in vrow.columns else "")
                for vid, d in R.groupby("video_id", sort=False)]
        aligned = {}
        with ProcessPoolExecutor(a.workers) as pool:
            for vid, res in pool.map(video_job, jobs, chunksize=4):
                for rep, sig, al in res:
                    aligned[(vid, rep)] = (sig, al)
        print(f"  {len(aligned)} reps aligned ({time.time() - t0:.0f} s)")
        R["vscore"] = R.video_id.map(vscore)
        excl = set((cfg["data"].get("template_exclude_subjects") or {}).get(ex, []))  # D8 (E03 P02)
        good = R[(R.group == "expert") & (R.vscore >= tc["min_score"]) & ~R.subject.isin(excl)]

        def library(subjects):
            g = good[good.subject.isin(set(subjects))]
            curves = {v: [aligned[(x.video_id, x.rep)][1]["curves"] for x in d.itertuples()] for v, d in g.groupby("view")}
            scal = {v: {"rom": [aligned[(x.video_id, x.rep)][0]["rom"] for x in d.itertuples()],
                        "log2_speed": [aligned[(x.video_id, x.rep)][1]["log2_speed"] for x in d.itertuples()]}
                    for v, d in g.groupby("view")}
            return fit_library(ref, curves, scal, ac["sigma_floor_deg"], tc["min_reps"], cfg["data"]["views"]), g

        lib_all, g_all = library(R.subject.unique())
        save_library(resolve(tc["library_dir"]) / ex, ref, lib_all,
                     {"exercise": ex, "file": ref_file, "fitted_on": "all subjects", "min_score": tc["min_score"],
                      "n_videos": int(g_all.video_id.nunique()), "n_subjects": int(g_all.subject.nunique())})
        libs = {}
        for p in plans:
            libs[p["fold"]], g = library(p["train"])
            for v in cfg["data"]["views"]:
                gv = g[g.view == v]
                count_rows.append(dict(exercise=ex, fold=p["fold"], view=v, reps=len(gv), videos=gv.video_id.nunique(),
                                       subjects=gv.subject.nunique()))
        for vid, d in R.groupby("video_id", sort=False):
            f = fold_of[d.subject.iloc[0]]
            entries = []
            for r in d.itertuples():
                sig, al = aligned[(vid, r.rep)]
                s, X, meas, errs, _ = evaluate(sig, al, ref, libs[f][r.view], ac)
                rep_rows.append(dict(flat_row(r, sig, s, X, meas, errs), fold=f))
                err_rows += [dict(video_id=vid, rep=int(r.rep), exercise=ex, subject=r.subject, group=r.group,
                                  view=r.view, fold=f, **{k: (json.dumps(v) if isinstance(v, list) else v)
                                                          for k, v in e.items()}) for e in errs]
                entries.append(rep_entry(r.rep, sig, s, meas, X, errs, ac, emb.get((vid, int(r.rep))) if emb else None))
            if emb:
                assert all(emb[(vid, int(r.rep))]["fold"] == f for r in d.itertuples()), vid
            n_set = int(rpv.loc[vid, "n_reps_set"])
            serr = set_errors(n_set, (ex_cfg.get("eaaq") or {}).get("expected_reps"))
            meta = {"exercise": f"{ex}_{ex_cfg['name']}", "video_id": vid, "view": d.view.iloc[0],
                    "angle_source": (ac["angle_source"] if ac["angle_source"] != "hands" else
                                     "2d_mediapipe_hands_side_views + 3d_mediapipe_world"),
                    "template": {"source": tc["source"], "file": ref_file, "library": f"fold{f}",
                                 "min_score": tc["min_score"], "posture_centre": ac.get("posture_centre", "pt_demo")},
                    "n_reps_set": n_set, "expected_reps": (ex_cfg.get("eaaq") or {}).get("expected_reps")}
            rep_ = video_report(meta, entries, serr, ac)
            validate_report(rep_)
            if not a.no_json:
                (out / "json" / f"{vid}.json").write_text(json.dumps(rep_, indent=1), encoding="utf-8")
            vid_rows.append(dict(video_id=vid, exercise=ex, fold=f, ref_set_score=rep_["set_score"],
                                 n_reps=len(entries), n_reps_set=n_set, too_few_reps=int(bool(serr)),
                                 n_rep_errors=sum(len(x["errors"]) for x in entries)))
        print(f"  {ex}: {R.video_id.nunique()} videos assessed ({time.time() - t0:.0f} s)")

    RR = pd.DataFrame(rep_rows)
    err_cols = sorted(c for c in RR.columns if c.startswith("err_"))
    RR[err_cols] = RR[err_cols].fillna(0).astype(int)
    RR.to_csv(out / "module3_reps.csv", index=False)
    pd.DataFrame(err_rows).to_csv(out / "module3_errors.csv", index=False)
    num = [c for c in RR.columns if c.startswith(("X_", "dev_", "D_", "rom", "z_", "omega", "log2"))
           or c in ("duration_s", "excursion")]
    agg = RR.groupby("video_id")[num + err_cols].mean().add_prefix("mean_").reset_index()
    V = pd.DataFrame(vid_rows).merge(agg, on="video_id").merge(
        idx[["video_id", "subject", "group", "view", "variation", "group_id", "score", "MQ", *ITEMS]], on="video_id")
    V.to_csv(out / "module3_videos.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out / "reference_curves.csv", index=False)
    pd.DataFrame(count_rows).to_csv(out / "library_fold_counts.csv", index=False)
    print(f"{len(RR)} reps, {len(V)} videos, {len(err_rows)} rep errors; JSON in {out / 'json'} "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
