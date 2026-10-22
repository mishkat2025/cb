"""Proposal completion (2026-10-22): B2-DepthAnything (A11) next to the other networks, E01-E04.

The proposal's baseline 2 is "ST-GCN using estimated 3D/depth information without CVDC" and names DepthAnything and
KTPFormer as the methods to investigate. This adds the DepthAnything variant (2D x, y + relative depth at the joints,
depth_extract.py / depth_pack.py, trained by depth_chain.py) to the published score tables of both schemes: its
video-head predictions are appended to <results>/<scheme>/video_predictions.csv (read only; the B1, B2-KTPFormer,
Proposed and baseline rows are the published ones) and every metric is recomputed with the same code as the final
report (report_full.regression_tables: video level, within groups first, subject bootstrap, paired differences),
plus the subject-level Wilcoxon tests of the LOSO report for B2-DepthAnything vs B1, B2-KTPFormer and Proposed.

usage: python scripts/full/completion/depth_baseline_report.py [--boot 2000]
Writes <results>/completion/depth_baseline/{score_metrics.csv, paired_deltas.csv, subject_tests.csv, summary.md}.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "full" / "phase3"))
sys.path.insert(0, str(REPO / "scripts" / "full" / "loso"))
sys.path.insert(0, str(REPO / "scripts" / "pilot" / "phase4"))
from cvdc.config import data_path, load_config, resolve  # noqa: E402
from cvdc.data.labels import movement_quality  # noqa: E402
from cvdc.eval.baselines import ALL_ITEMS, item_activity, video_table  # noqa: E402
from cvdc.train.trainer import fold_plan, loso_plans  # noqa: E402
import report_perex as RP  # noqa: E402
from report_full import VIDEO_TARGETS, noise_floors, regression_tables  # noqa: E402
from report_loso import subject_tests  # noqa: E402

EXS = ["E01", "E02", "E03", "E04"]
NEW = "b2_depthanything"
METHODS = ["b1", "b2_ktpformer", NEW, "proposed_j1", "proposed_lp"]
SHORT = {**RP.SHORT, NEW: "B2-DepthAny", "base_group": "group-only", "base_setfeat": "set-features"}
SCHEMES = {"LOSO": ("perex_loso", "perex_loso", True), "GroupKFold-5": ("perex_gkf5", "final_E01_E04/step1", False)}
POOLED = [("experts, E01–E04", None, "expert"), ("non-experts, E01–E04", None, "non-expert")]


def new_predictions(cfg, root, reps, v, active, loso):
    pv, _, done = RP.load_runs(root, EXS, [NEW])
    plans = {}
    for ex in EXS:
        pl = (loso_plans if loso else fold_plan)(cfg, reps[reps.exercise == ex])
        for p in pl:
            plans[RP.fkey(ex, p["fold"])] = {**p, "exercise": ex}
    missing = [(ex, p["fold"]) for k, p in plans.items() for ex in [p["exercise"]] if (ex, NEW, p["fold"]) not in done]
    if missing:
        raise SystemExit(f"{root}: {len(missing)} {NEW} runs are not finished yet (e.g. {missing[:3]})")
    pv = RP.add_mq(pv, v, plans, active)
    vp = pv[["method", "fold", "video_id", *[f"{t}_pred" for t in VIDEO_TARGETS]]].copy()
    vp["source"] = "video head"
    return vp.merge(v[["video_id", "subject", "group", "exercise", "view", "variation", *VIDEO_TARGETS]], on="video_id",
                    how="left", validate="many_to_one")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise.yaml")
    ap.add_argument("--boot", type=int)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    cfg = load_config(REPO / a.config)
    ec, ia = cfg["eval"], cfg["heads"]["item_activity"]
    n_boot, ci, seed = a.boot or ec["bootstrap"]["n"], ec["bootstrap"]["ci"], ec["bootstrap"]["seed"]
    res = resolve(ec["results_dir"])
    out = res / "completion" / "depth_baseline"
    out.mkdir(parents=True, exist_ok=True)
    v = video_table(cfg)
    v["MQ"] = movement_quality(v)
    index = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    index["MQ"] = movement_quality(index)
    reps = pd.read_csv(data_path(cfg, "reps_csv"))
    act = item_activity(v, ALL_ITEMS, ia["min_subjects_off_mode"], ia["min_sd"])
    active = {(e, g): set(d[d.varies].item) for (e, g), d in act.groupby(["exercise", "group"])}
    floors = noise_floors(index, ["score", "MQ"])
    S_all, D_all, T_all = [], [], []
    for name, (run_tag, res_tag, loso) in SCHEMES.items():
        P = pd.read_csv(res / res_tag / "video_predictions.csv")
        P = P[P.exercise.isin(EXS)]
        Pn = new_predictions(cfg, resolve(cfg["train"]["runs_dir"]) / run_tag, reps, v, active, loso)
        n_ref = P[(P.method == "b1") & (P.source == "video head")].video_id.nunique()
        assert Pn.video_id.nunique() == n_ref, f"{name}: {NEW} predicts {Pn.video_id.nunique()} videos, B1 {n_ref}"
        P = pd.concat([P, Pn[P.columns]], ignore_index=True)
        within = [(f"{e} {g}", e, g) for e, g in sorted({(e, g) for e, g in zip(P.exercise, P.group)})]
        strata = within + POOLED
        entries = [(m, "video head") for m in METHODS] + [("base_group", "baseline"), ("base_setfeat", "baseline")]
        vs = [("base_group", "baseline"), ("b1", None), ("b2_ktpformer", None), ("proposed_j1", None)]
        S, D = regression_tables(P, strata, entries, active, floors, n_boot, seed, ci, vs)
        T = subject_tests(P, strata, [(NEW, "b1"), (NEW, "b2_ktpformer"), ("proposed_j1", NEW), (NEW, "base_group")])
        for X in (S, D, T):
            X.insert(0, "scheme", name)
        S_all.append(S)
        D_all.append(D)
        T_all.append(T)
        print(f"{name} done ({time.time() - t0:.0f} s)", flush=True)
    S, D, T = (pd.concat(x, ignore_index=True) for x in (S_all, D_all, T_all))
    S.to_csv(out / "score_metrics.csv", index=False)
    D.to_csv(out / "paired_deltas.csv", index=False)
    T.to_csv(out / "subject_tests.csv", index=False)
    write_summary(out, S, D, T, n_boot)
    print(f"written to {out} ({time.time() - t0:.0f} s)")


def f(x, d=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}".replace("-", "−")


def write_summary(out, S, D, T, n_boot):
    L = ["# B2-DepthAnything next to the other networks (E01–E04)\n",
         "Generated by `scripts/full/completion/depth_baseline_report.py`. Video head, total score; RMSE / Spearman ρ "
         f"(pooled rows: ρ centred within exercise); 95% CIs from the subject bootstrap ({n_boot} draws). The rows of "
         "the other methods are the published ones, recomputed with the same code.\n"]
    for scheme in ("LOSO", "GroupKFold-5"):
        s = S[(S.scheme == scheme) & (S.target == "score")]
        strata = list(dict.fromkeys(s.stratum))
        L.append(f"\n## {scheme}: RMSE / ρ\n")
        L.append("| Stratum | " + " | ".join(SHORT.get(m, m) for m in METHODS + ["base_group", "base_setfeat"]) + " |")
        L.append("|" + "---|" * (len(METHODS) + 3))
        for st in strata:
            g = s[s.stratum == st]
            cells = []
            for m in METHODS + ["base_group", "base_setfeat"]:
                r = g[(g.method == m) & (g.source == ("baseline" if m.startswith("base_") else "video head"))]
                if r.empty:
                    cells.append("–")
                    continue
                r = r.iloc[0]
                rho = r.get("spearman_centred", np.nan) if r.get("spearman_centred") is not None else np.nan
                rho = r.spearman if not np.isfinite(rho) else rho
                cells.append(f(r.rmse) + ("" if m == "base_group" else f" / {f(rho)}"))
            L.append(f"| {st} | " + " | ".join(cells) + " |")
        d = D[(D.scheme == scheme) & (D.target == "score") & (D.method == NEW)]
        L.append(f"\n**B2-DepthAnything, paired RMSE difference [95% CI]** (negative = lower error than the "
                 f"comparison; bold = CI excludes 0):\n")
        vss = list(dict.fromkeys(d.vs))
        L.append("| Stratum | " + " | ".join(f"vs {SHORT.get(x, x)}" for x in vss) + " |")
        L.append("|" + "---|" * (len(vss) + 1))
        for st in dict.fromkeys(d.stratum):
            cells = []
            for x in vss:
                r = d[(d.stratum == st) & (d.vs == x)]
                if r.empty:
                    cells.append("–")
                    continue
                r = r.iloc[0]
                c = f"{f(r.delta)} [{f(r.lo)}, {f(r.hi)}]"
                cells.append(f"**{c}**" if (r.lo > 0 or r.hi < 0) else c)
            L.append(f"| {st} | " + " | ".join(cells) + " |")
        t = T[(T.scheme == scheme) & (T.target == "score") & T.stratum.isin([s_ for s_, _, _ in POOLED])]
        if len(t):
            L.append("\nSubject-level tests (per-subject RMSE, Wilcoxon signed-rank, pooled strata unadjusted):\n")
            L.append("| Stratum | comparison | subjects lower / higher | p | rank-biserial r |")
            L.append("|---|---|---|---|---|")
            for r in t.itertuples():
                L.append(f"| {r.stratum} | {SHORT.get(r.method, r.method)} vs {SHORT.get(r.vs, r.vs)} | "
                         f"{r.lower} / {r.higher} | {f(r.wilcoxon_p, 3)} | {f(r.rank_biserial)} |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
