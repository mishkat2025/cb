"""Proposal completion (2026-10-22), quick measure 4: is the CVDC embedding depth-aware?

The proposal calls the CVDC representation "implicit depth-aware" (Fig. 3); MPJPE cannot be computed (no 3D ground
truth, E8). This check uses the synchronised cameras instead: for the same rep filmed at the same time, the side
camera sees the hand's forward-backward position (depth for the front camera) directly in its 2D image. Target, from
the side view that sees the moving arm best (the moving wrist's median visibility over the rep, the higher of L / R):
  forward       the moving wrist's position relative to the moving shoulder along the image x axis, oriented so that
                forward (the direction of the nose from the mid-shoulder) is positive, in 2D torso lengths: its mean
                over the rep ('mean forward reach') and its range max - min ('depth excursion')
Predictors, from the FRONT view of the same rep only (what a single front camera gives):
  embeddings    the pooled 256-d embedding of each GroupKFold-5 fold model (random-init encoder, B1, B2-KTPFormer,
                the frozen CVDC encoder, Proposed lambda 1.0), centred with the fold's training mean
  2D skeleton   the front rep's 2D pack itself (17 joints x 64 frames, flattened): what a linear model gets from the
                2D shape alone
  explicit 3D   MediaPipe's world landmarks and KTPFormer's lifted 3D of the front view: the same forward measure
                (towards the camera = forward) computed from their depth axis (one feature each)
Ridge regression fitted on the fold's training subjects (alpha by subject-grouped inner CV), predicted for its test
subjects; pooled over the five folds per exercise (E01-E04, experts, synced performances). R2 (vs the mean) and
Spearman, 95% CIs from the subject bootstrap, paired differences vs B1 on the same draws.

usage: python scripts/full/completion/depth_check.py [--boot 2000]
Writes <results>/completion/depth/{reps.csv, oof_predictions.csv, metrics.csv, summary.md}. Needs encode.py --z.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "full" / "phase3"))
sys.path.insert(0, str(REPO / "scripts" / "pilot" / "phase4"))
from cvdc.config import data_path, load_config, resolve  # noqa: E402
from cvdc.eval.metrics import interval, item_weights, regression  # noqa: E402
from cvdc.train.trainer import fold_plan  # noqa: E402
from report_full import draws_for  # noqa: E402

EXS = ["E01", "E02", "E03", "E04"]
EMB = ["init_xy", "b1", "b2_ktpformer", "proposed_lp", "proposed_j1"]
NAMES = {"init_xy": "random-init encoder", "b1": "B1 (2D, no CVDC)", "b2_ktpformer": "B2-KTPFormer (3D input)",
         "proposed_lp": "CVDC encoder (frozen)", "proposed_j1": "Proposed λ1 (CVDC + fine-tuning)",
         "pack2d": "2D skeleton, linear", "mp3d": "MediaPipe 3D (explicit depth)", "ktp3d": "KTPFormer 3D (explicit depth)"}
MP = {"nose": 0, "left_shoulder": 11, "right_shoulder": 12, "left_wrist": 15, "right_wrist": 16}
CO = {"nose": 0, "left_shoulder": 5, "right_shoulder": 6, "left_wrist": 9, "right_wrist": 10}
ALPHAS = np.logspace(-1, 5, 13)
TARGETS = {"mean": "mean forward reach", "range": "depth excursion"}


def forward_series(P, side, idx, axis, sign):
    """Moving wrist - moving shoulder along ``axis`` times ``sign`` (n,)."""
    return sign * (P[:, idx[f"{side}_wrist"], axis] - P[:, idx[f"{side}_shoulder"], axis])


def summ(x):
    x = x[np.isfinite(x)]
    return (float(np.mean(x)), float(np.max(x) - np.min(x))) if len(x) >= 5 else (np.nan, np.nan)


def build_table(cfg, reps):
    sync = pd.read_csv(resolve(cfg["sync"]["report_csv"]))
    synced = set(sync.group_id[sync.cls == "synced"])
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"])).set_index("video_id")
    lift_dir = resolve(cfg["lifter"]["out_dir"])
    r = reps[(reps.exercise.isin(EXS)) & (reps.group == "expert") & reps.group_id.isin(synced)]
    cache = {}

    def load(vid):
        if vid not in cache:
            p = resolve(idx.loc[vid, "pre_path"])
            if not p.exists():
                cache[vid] = None
            else:
                with np.load(p) as z:
                    d = {"xy": z["xy"].astype(float), "vis": z["vis"].astype(float), "world": z["world"].astype(float)}
                lp = lift_dir / f"{vid}.npz"
                if lp.exists():
                    with np.load(lp) as z:
                        d["lift"] = z["world"].astype(float)
                cache[vid] = d
        return cache[vid]

    rows = []
    for uid, d in r.groupby("rep_uid"):
        views = set(d.view)
        if "F" not in views or not (views & {"L", "R"}):
            continue
        f = d[d.view == "F"].iloc[0]
        side = f.side
        best = None
        for v in ("L", "R"):
            if v not in views:
                continue
            s = d[d.view == v].iloc[0]
            k = load(s.video_id)
            if k is None:
                continue
            vis = np.nanmedian(k["vis"][s.start:s.end + 1, MP[f"{side}_wrist"]])
            if best is None or vis > best[0]:
                best = (vis, s, k)
        kf = load(f.video_id)
        if best is None or kf is None or best[0] < 0.5:
            continue
        _, s, k = best
        seg = k["xy"][s.start:s.end + 1]
        nose = seg[:, MP["nose"], 0] - 0.5 * (seg[:, MP["left_shoulder"], 0] + seg[:, MP["right_shoulder"], 0])
        sign = np.sign(np.nanmedian(nose)) or 1.0
        tm, tr = summ(forward_series(seg, side, MP, 0, sign))
        fw = kf["world"][f.start:f.end + 1]
        mm, mr = summ(forward_series(fw, side, MP, 2, -1.0))  # MediaPipe z: negative = closer to the camera
        km = kr = np.nan
        if "lift" in kf:
            km, kr = summ(forward_series(kf["lift"][f.start:f.end + 1], side, CO, 2, -1.0))
        rows.append(dict(rep_uid=uid, item=int(f["index"]), exercise=f.exercise, subject=f.subject, group=f.group,
                         side=side, side_view=s.view, side_vis=float(best[0]), target_mean=tm, target_range=tr,
                         mp3d_mean=mm, mp3d_range=mr, ktp3d_mean=km, ktp3d_range=kr))
    return pd.DataFrame(rows).dropna(subset=["target_mean", "target_range"])


def fit_predict(Xtr, ytr, gtr, Xte):
    n_splits = min(5, len(np.unique(gtr)))
    cv = list(GroupKFold(n_splits).split(Xtr, ytr, gtr))
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=cv))
    model.fit(Xtr, ytr)
    return model.predict(Xte)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise.yaml")
    ap.add_argument("--boot", type=int)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    cfg = load_config(REPO / a.config)
    ec = cfg["eval"]
    n_boot, ci, seed = a.boot or ec["bootstrap"]["n"], ec["bootstrap"]["ci"], ec["bootstrap"]["seed"]
    res = resolve(ec["results_dir"])
    out = res / "completion" / "depth"
    out.mkdir(parents=True, exist_ok=True)
    reps = pd.read_csv(data_path(cfg, "reps_csv")).reset_index()
    T = build_table(cfg, reps)
    T.to_csv(out / "reps.csv", index=False)
    print(f"{len(T)} synced front + side rep sets ({time.time() - t0:.0f} s)", flush=True)
    pc = cfg["preprocess"]
    with np.load(resolve(pc["pack_dir"]) / f"reps_{pc['joint_set']}_T{pc['T_rep']}.npz") as z:
        pack = z["xy"]
    assert len(pack) == len(reps), "the rep pack must have one entry per reps.csv row"
    preds = []
    for ex in EXS:
        te_all = T[T.exercise == ex]
        plans = fold_plan(cfg, reps[reps.exercise == ex].drop(columns="index"))
        for p in plans:
            tr = te_all[te_all.subject.isin(set(p["train"]))]
            te = te_all[te_all.subject.isin(set(p["test"]))]
            if te.empty:
                continue
            with np.load(res / "completion" / "encode" / "perex_gkf5" / ex / f"fold{p['fold']}.npz") as z:
                pos = {int(x): i for i, x in enumerate(z["item"])}
                feats = {m: z[f"z_{m}"][[pos[i] for i in np.r_[tr.item, te.item]]].astype(float) - z[f"centre_{m}"]
                         for m in EMB}
            feats["pack2d"] = pack[np.r_[tr.item, te.item]].reshape(len(tr) + len(te), -1).astype(float)
            ntr = len(tr)
            for t in TARGETS:
                y = tr[f"target_{t}"].to_numpy()
                row = te[["rep_uid", "exercise", "subject", "group", f"target_{t}"]].rename(columns={f"target_{t}": "y"})
                row = row.assign(target=t, fold=p["fold"])
                for m, X in feats.items():
                    preds.append(row.assign(method=m, pred=fit_predict(X[:ntr], y, tr.subject.to_numpy(), X[ntr:])))
                for m in ("mp3d", "ktp3d"):
                    ok_tr, ok_te = tr[f"{m}_{t}"].notna().to_numpy(), te[f"{m}_{t}"].notna().to_numpy()
                    pr = np.full(len(te), np.nan)
                    pr[ok_te] = fit_predict(tr[f"{m}_{t}"].to_numpy()[ok_tr, None], y[ok_tr],
                                            tr.subject.to_numpy()[ok_tr], te[f"{m}_{t}"].to_numpy()[ok_te, None])
                    preds.append(row.assign(method=m, pred=pr))
        print(f"{ex} done ({time.time() - t0:.0f} s)", flush=True)
    P = pd.concat(preds, ignore_index=True)
    P.to_csv(out / "oof_predictions.csv", index=False)

    rows = []
    for ex in EXS + [None]:
        for t in TARGETS:
            d = P[(P.target == t) & ((P.exercise == ex) if ex else True)]
            ref = d[d.method == "b1"].sort_values("rep_uid")
            subs, M = draws_for(ref, n_boot, seed)
            W = np.vstack([np.ones(len(ref)), item_weights(ref.subject.to_numpy(), subs, M)])
            y = ref.y.to_numpy()
            if ex is None:  # pooled: centre within exercise, so exercise differences do not count
                cen = ref.groupby("exercise").y.transform("mean").to_numpy()
            else:
                cen = np.zeros(len(ref))
            yc = y - cen
            var = (W * (yc - ((W * yc).sum(1) / W.sum(1))[:, None]) ** 2).sum(1) / W.sum(1)
            res_m = {}
            for m in EMB + ["pack2d", "mp3d", "ktp3d"]:
                pm = d[d.method == m].set_index("rep_uid").pred.reindex(ref.rep_uid).to_numpy()
                ok = np.isfinite(pm)
                pmc = np.where(ok, pm - cen, 0.0)
                Wk = W * ok
                mse = (Wk * (pmc - yc) ** 2).sum(1) / Wk.sum(1)
                r2 = 1 - mse / var
                rho = regression(yc[ok], pmc[ok], W[:, ok])["spearman"]
                res_m[m] = (r2, rho)
                lo, hi = interval(r2[1:], ci)
                rlo, rhi = interval(rho[1:], ci)
                row = dict(exercise=ex or "E01–E04", target=t, method=m, reps=int(ok.sum()), subjects=len(subs),
                           r2=float(r2[0]), r2_lo=lo, r2_hi=hi, spearman=float(rho[0]), spearman_lo=rlo, spearman_hi=rhi)
                rows.append(row)
            for m in EMB + ["pack2d", "mp3d", "ktp3d"]:
                if m == "b1":
                    continue
                dr2 = res_m[m][0] - res_m["b1"][0]
                lo, hi = interval(dr2[1:], ci)
                for row in rows[-len(res_m):]:
                    if row["method"] == m:
                        row.update(d_r2_vs_b1=float(dr2[0]), d_r2_vs_b1_lo=lo, d_r2_vs_b1_hi=hi)
    Mt = pd.DataFrame(rows)
    Mt.to_csv(out / "metrics.csv", index=False)
    write_summary(out, Mt, T, n_boot)
    print(f"written to {out} ({time.time() - t0:.0f} s)")


def write_summary(out, Mt, T, n_boot):
    f = lambda x: "–" if not np.isfinite(x) else f"{x:.2f}".replace("-", "−")  # noqa: E731
    L = ["# Proposal completion: is the CVDC embedding depth-aware? (E01–E04 experts)\n",
         "Generated by `scripts/full/completion/depth_check.py`. The side camera measures how far forward the moving "
         "hand is (2D, torso lengths) for the same rep that the front camera filmed at the same time; each predictor "
         "sees the front view only. Out-of-fold R² (GroupKFold-5 by subject) and Spearman ρ [95% CI, subject bootstrap, "
         f"{n_boot} draws]; ΔR² vs B1 on the same draws (bold: CI excludes 0). Pooled rows: centred within exercise.\n",
         f"Rep sets: " + ", ".join(f"{e} {int((T.exercise == e).sum())}" for e in EXS) + ".\n"]
    for t, tname in TARGETS.items():
        L.append(f"\n## Target: {tname}\n")
        exs = list(dict.fromkeys(Mt.exercise))
        L.append("| Predictor (front view only) | " + " | ".join(exs) + " |")
        L.append("|" + "---|" * (len(exs) + 1))
        for m in EMB + ["pack2d", "mp3d", "ktp3d"]:
            cells = []
            for e in exs:
                r = Mt[(Mt.exercise == e) & (Mt.target == t) & (Mt.method == m)].iloc[0]
                s = f"R² {f(r.r2)} [{f(r.r2_lo)}, {f(r.r2_hi)}], ρ {f(r.spearman)}"
                if m != "b1" and np.isfinite(r.get("d_r2_vs_b1", np.nan)):
                    dd = f"Δ {f(r.d_r2_vs_b1)} [{f(r.d_r2_vs_b1_lo)}, {f(r.d_r2_vs_b1_hi)}]"
                    s += f"; {'**' + dd + '**' if (r.d_r2_vs_b1_lo > 0 or r.d_r2_vs_b1_hi < 0) else dd}"
                cells.append(s)
            L.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
