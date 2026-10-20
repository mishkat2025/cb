"""Final report, final scope E01-E04 (D18), with E05-E09 as a labelled supplementary appendix.

Collects the tables of the finished analyses; nothing is re-estimated here:
  step 1     one model per exercise, GroupKFold-5 (T17): <results>/final_E01_E04/step1/ (report_perex.py and
             group_probe.py with --exercises E01 E02 E03 E04, so the pooled strata are E01-E04)
  cross-view per exercise, train on two views / test on the third (E24): <results>/perex_crossview/
             (crossview_distance_perex.py, report_crossview_perex.py)
  Module III <outputs>/phase4/agreement.csv and calibration/ (M17, M18)
  appendix   E05-E09 from the nine-exercise step-1 report (<results>/perex_gkf5/) and Module III
Every score result is within groups first (E9), video level, with 95% subject-bootstrap CIs (paired differences; bold
= a paired CI that excludes 0).

usage: python scripts/final/report_final.py [--config configs/per_exercise.yaml] [--crossview-tag perex_crossview]
Writes <results>/final_E01_E04/tables.md and its committed copy reports/final_E01_E04_tables.md.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve  # noqa: E402

FINAL = ["E01", "E02", "E03", "E04"]
APPX = ["E05", "E06", "E07", "E08", "E09"]
SHORT = {"b1": "B1", "b2_ktpformer": "B2-KTP", "proposed_j1": "Prop λ1", "proposed_lp": "Prop frozen",
         "init_xy": "random init", "base_group": "group-only", "base_setfeat": "set-features"}
MODELS = ["b1", "b2_ktpformer", "proposed_j1", "proposed_lp"]


def f2(x, d=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}".replace("-", "−")


def ci(row, k, lo=None, hi=None, d=2):
    """value [lo, hi], bold when the CI excludes 0."""
    v, a, b = row[k], row[lo or f"{k}_lo"], row[hi or f"{k}_hi"]
    s = f"{f2(v, d)} [{f2(a, d)}, {f2(b, d)}]"
    return f"**{s}**" if np.isfinite(a) and np.isfinite(b) and (a > 0 or b < 0) else s


def strata_of(exs, pooled):
    out = []
    for e in exs:
        out += [f"{e} expert", f"{e} non-expert"]
    return out + pooled


def low_conf(stratum, n_subjects, low=15):
    return f"{stratum} ⚠" if n_subjects < low else stratum


class Report:
    def __init__(self):
        self.L = []

    def w(self, s=""):
        self.L.append(s)

    def table(self, head, rows):
        self.w("| " + " | ".join(head) + " |")
        self.w("|" + "---|" * len(head))
        for r in rows:
            self.w("| " + " | ".join(str(x) for x in r) + " |")
        self.w()


def score_tables(R, S, D, NF, exs, pooled, target):
    """RMSE / rho per stratum for the four networks (video head) and the baselines, then the paired differences."""
    rows, drows = [], []
    for st in strata_of(exs, pooled):
        s = S[(S.stratum == st) & (S.target == target)]
        if s.empty:
            continue
        cells = []
        rho = "spearman_centred" if st in pooled else "spearman"  # pooled strata: within-exercise (centred) rho
        for m in MODELS:
            r = s[(s.method == m) & (s.source == "video head")]
            cells.append(f"{f2(r.rmse.iloc[0])} / {f2(r[rho].iloc[0])}" if len(r) else "–")
        for m in ("base_group", "base_setfeat"):
            r = s[(s.method == m)]
            cells.append(f"{f2(r.rmse.iloc[0])}" + (f" / {f2(r[rho].iloc[0])}" if m != "base_group" and len(r)
                                                     else "") if len(r) else "–")
        ex = st.split()[0]
        nf = NF.loc[ex, "score" if target == "score" else "MQ"] if ex in NF.index and "non-expert" not in st else np.nan
        n_sub = int(s.subjects.max())
        rows.append([low_conf(st, n_sub) + f" ({n_sub})", *cells, f2(nf)])
        d = D[(D.stratum == st) & (D.target == target) & (D.source == "video head") & (D.metric == "rmse")]
        for m in ("proposed_j1", "proposed_lp", "b1"):
            dd = d[d.method == m]
            g = dd[dd.vs == "base_group"]
            b = dd[dd.vs == "b1"]
            drows.append([st, SHORT[m], ci(g.iloc[0], "delta", "lo", "hi") if len(g) else "–",
                          ci(b.iloc[0], "delta", "lo", "hi") if len(b) and m != "b1" else "–"])
    R.table(["Stratum (subjects)", "B1", "B2-KTP", "Prop λ1", "Prop frozen", "group-only", "set-features",
             "noise floor"], rows)
    R.w(f"Paired RMSE differences, {target} (negative = lower error than the comparison):")
    R.w()
    R.table(["Stratum", "method", "vs group-only", "vs B1"], drows)


def vci_table(R, V, exs):
    rows = []
    for st in strata_of(exs, []):
        v = V[(V.stratum == st) & (V.variant == "centred")]
        if v.empty:
            continue

        def g(m, stage):
            r = v[(v.method == m) & (v.stage == stage)]
            return ci(r.iloc[0], "vci") if len(r) else "–"
        pairs = v[(v.method == "proposed_j1") & (v.stage == "final")].same_pairs
        rows.append([st, g("proposed_j1", "init"), g("proposed_j1", "pretrained"), g("b1", "final"),
                     g("b2_ktpformer", "final"), g("proposed_j1", "final"), int(pairs.iloc[0]) if len(pairs) else "–"])
    R.table(["Stratum", "random init", "after CVDC", "B1", "B2-KTP", "Prop λ1", "same-rep pairs"], rows)


def distance_table(R, DV, exs):
    rows = []
    for ex in exs:
        d = DV[DV.exercise == ex]
        if d.empty:
            continue

        def g(m, delta=False):
            r = d[d.method == m]
            if not len(r):
                return "–"
            r = r.iloc[0]
            s = f2(r.icc)
            if delta and np.isfinite(r.get("d_icc_vs_b1", np.nan)):
                s += f" ({ci(r, 'd_icc_vs_b1')})"
            return s
        rows.append([ex, g("init_xy"), g("b1"), g("b2_ktpformer", True), g("proposed_j1", True),
                     g("proposed_lp", True), int(d.perfs.iloc[0])])
    R.table(["Exercise", "random init", "B1", "B2-KTP (Δ vs B1)", "Prop λ1 (Δ vs B1)", "Prop frozen (Δ vs B1)",
             "synced performances"], rows)


def group_probe_table(R, G, exs, pooled_label):
    rows = []
    for ex in [*exs, "pooled"]:
        g = G[(G.exercise == ex) & (G.kind == "all")]
        if g.empty:
            continue
        cells = []
        for enc in ["init_xy", *MODELS]:
            r = g[g.encoder == enc]
            cells.append(f"{f2(r.auc.iloc[0])} [{f2(r.auc_lo.iloc[0])}, {f2(r.auc_hi.iloc[0])}]" if len(r) else "–")
        r0 = g.iloc[0]
        rows.append([pooled_label if ex == "pooled" else ex, f"{int(r0.experts)} / {int(r0.non_experts)}", *cells])
    R.table(["Exercise", "experts / non-experts", "random init", "B1", "B2-KTP", "Prop λ1", "Prop frozen"], rows)


def module3_table(R, A, exs):
    rows = []
    for ex in exs:
        for grp in ("expert", "non-expert"):
            a = A[(A.exercise == ex) & (A.group == grp)]
            if a.empty:
                continue
            s, q = a[a.target == "score"], a[a.target == "MQ"]
            n = int(a.subjects.iloc[0])
            rows.append([low_conf(f"{ex} {grp}", n) + f" ({n})", ci(s.iloc[0], "spearman") if len(s) else "–",
                         ci(q.iloc[0], "spearman") if len(q) else "–"])
    R.table(["Stratum (subjects)", "total score ρ", "MQ ρ"], rows)


def crossview_tables(R, X):
    M, MD, V, VD, DS, DSD = (X.get(k) for k in ("mq", "mqd", "vci", "vcid", "dvs", "dvsd"))
    if M is None:
        R.w("_Not available yet (the cross-view runs for E01–E04 are still running)._")
        R.w()
        return
    order = list(dict.fromkeys(M.stratum))
    for target, name in (("MQ", "Movement-quality score (MQ)"), ("score", "Total score")):
        R.w(f"**{name} on the held-out view** (RMSE / ρ: held-out · seen · all-view; group-only RMSE):")
        R.w()
        rows = []
        for st in order:
            m = M[(M.stratum == st) & (M.target == target)]
            cells = []
            for meth in ("b1", "b2_ktpformer", "proposed_j1"):
                parts = []
                for role in ("held-out", "seen", "all-view"):
                    r = m[(m.method == meth) & (m.role == role)]
                    parts.append(f"{f2(r.rmse.iloc[0])}/{f2(r.spearman.iloc[0])}" if len(r) else "–")
                cells.append(" · ".join(parts))
            g = m[m.method == "base_group"]
            rows.append([st, *cells, f2(g.rmse.iloc[0]) if len(g) else "–"])
        R.table(["Stratum", "B1", "B2-KTP", "Prop λ1", "group-only"], rows)
        R.w(f"Paired RMSE differences, {target}:")
        R.w()
        rows = []
        for st in order:
            d = MD[(MD.stratum == st) & (MD.target == target) & (MD.metric == "rmse")]
            for meth in ("b1", "proposed_j1"):
                dd = d[d.method == meth]

                def c(comp):
                    r = dd[dd.comparison == comp]
                    return ci(r.iloc[0], "delta") if len(r) and np.isfinite(r.delta.iloc[0]) else "–"
                rows.append([st, SHORT[meth], c("held-out − seen (gap)"), c("held-out − all-view"), c("vs group-only"),
                             c("vs B1 (held-out)") if meth != "b1" else "–", c("gap − B1's gap") if meth != "b1" else "–"])
        R.table(["Stratum", "method", "held-out − seen", "held-out − all-view", "vs group-only", "vs B1",
                 "gap − B1's gap"], rows)
    R.w("**Within-group VCI on the held-out view** (centred 256-d; pairs with one member in the held-out view and one "
        "in a seen view, next to seen↔seen pairs of two different seen cameras):")
    R.w()
    rows = []
    for st in list(dict.fromkeys(V.stratum)):
        v = V[(V.stratum == st) & (V.variant == "centred")]

        def g(meth, stage, pairing):
            r = v[(v.method == meth) & (v.stage == stage) & (v.pairing == pairing)]
            return ci(r.iloc[0], "vci") if len(r) else "–"
        rows.append([st, g("proposed_j1", "init", "held-out↔seen"), g("proposed_j1", "pretrained", "held-out↔seen"),
                     g("b1", "final", "held-out↔seen"), g("b2_ktpformer", "final", "held-out↔seen"),
                     g("proposed_j1", "final", "held-out↔seen"), g("proposed_j1", "final", "seen↔seen")])
    R.table(["Stratum", "random init", "after CVDC", "B1", "B2-KTP", "Prop λ1", "Prop λ1 seen↔seen"], rows)
    rows = []
    keep = VD[(VD.method == "proposed_j1") | ((VD.method == "b1") & (VD.comparison == "held-out↔seen − seen↔seen"))]
    for _, r in keep.iterrows():
        rows.append([r.stratum, SHORT.get(r.method, r.method), r.comparison, ci(r, "delta")])
    if rows:
        R.w("Paired VCI differences (Proposed; and B1's held-out↔seen − seen↔seen; all comparisons in "
            "`perex_crossview/vci_deltas.csv`):")
        R.w()
        R.table(["Stratum", "method", "comparison", "Δ VCI"], rows)
    R.w("**View stability of the distance to the PT reference** (synced expert performances, rep level, ICC(1); "
        "held-out↔seen = one member in the view the model never saw):")
    R.w()
    rows = []
    for ex in list(dict.fromkeys(DS.exercise)):
        d = DS[(DS.exercise == ex) & (DS.setting == "cross-view")]
        cells = []
        for meth in ("b1", "b2_ktpformer", "proposed_j1"):
            parts = []
            for var in ("held-out↔seen", "seen↔seen"):
                r = d[(d.method == meth) & (d.variant == var)]
                parts.append(f2(r.icc.iloc[0]) if len(r) else "–")
            cells.append(" · ".join(parts))
        dd = DSD[(DSD.exercise == ex) & (DSD.setting == "cross-view") & (DSD.variant == "held-out↔seen") &
                 (DSD.method == "proposed_j1")]
        rows.append([ex, *cells, ci(dd.iloc[0], "delta") if len(dd) else "–"])
    R.table(["Exercise", "B1 held-out↔seen · seen↔seen", "B2-KTP", "Prop λ1", "Prop λ1 − B1 (held-out↔seen)"], rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise.yaml")
    ap.add_argument("--crossview-tag", default="perex_crossview")
    ap.add_argument("--step1", help="step-1 tables folder (default <results>/final_E01_E04/step1)")
    a = ap.parse_args()
    cfg = load_config(a.config if Path(a.config).exists() else REPO / a.config)
    res = resolve(cfg["eval"]["results_dir"])
    s1 = resolve(a.step1) if a.step1 else res / "final_E01_E04" / "step1"
    s9, xv = res / "perex_gkf5", res / a.crossview_tag
    p4 = resolve(cfg["outputs"]["phase1"]).parent / "phase4"
    rd = lambda d, f: pd.read_csv(d / f) if (d / f).exists() else None  # noqa: E731
    S, D, V, DV, G = (rd(s1, f) for f in ("score_metrics.csv", "paired_deltas.csv", "vci.csv",
                                           "distance_view_stability.csv", "group_probe_per_exercise.csv"))
    S9, D9, V9, DV9, G9 = (rd(s9, f) for f in ("score_metrics.csv", "paired_deltas.csv", "vci.csv",
                                               "distance_view_stability.csv", "group_probe_per_exercise.csv"))
    NF = pd.read_csv(s9 / "noise_floor.csv").set_index("exercise")
    X = {k: rd(xv, f) for k, f in (("mq", "mq_score_metrics.csv"), ("mqd", "mq_score_deltas.csv"), ("vci", "vci.csv"),
                                    ("vcid", "vci_deltas.csv"), ("dvs", "distance_view_stability.csv"),
                                    ("dvsd", "distance_view_stability_deltas.csv"))}
    A, CA, RR = rd(p4, "agreement.csv"), rd(p4 / "calibration", "agreement.csv"), rd(p4 / "calibration", "rule_rates.csv")
    pooled = ["experts, E01–E04", "non-experts, E01–E04"]
    R = Report()
    w = R.w
    w("# Tables: final scope E01–E04 (D18), E05–E09 as a supplementary appendix")
    w()
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/final/report_final.py` from the step-1 tables "
      f"(`{s1.relative_to(REPO).as_posix()}`), the cross-view tables (`{xv.relative_to(REPO).as_posix()}`) and Module "
      f"III (`{p4.relative_to(REPO).as_posix()}`). Within groups, video level, 95% subject-bootstrap CIs; **bold** = a paired CI that "
      "excludes 0; ⚠ = fewer than 15 subjects.")
    w()
    w("## 1. Score prediction within groups (step 1: one model per exercise, GroupKFold-5)")
    w()
    w("**Total score**, video head, RMSE / Spearman ρ (pooled rows: ρ of the scores centred per exercise, i.e. the "
      "ranking within exercises); group-only = the within-group mean of the training fold; noise floor = RMSE between "
      "the views of one performance (experts):")
    w()
    score_tables(R, S, D, NF, FINAL, pooled, "score")
    w("**Movement-quality score (MQ, D13)**, video head:")
    w()
    score_tables(R, S, D, NF, FINAL, pooled, "MQ")
    w("## 2. View consistency of the embeddings (step 1)")
    w()
    w("**Within-group VCI** (centred 256-d, held-out subjects; same rep from another view vs other groups):")
    w()
    vci_table(R, V, FINAL)
    w("**View stability of the distance to the PT reference** (synced expert performances, rep level, ICC(1) of the "
      "standardised distance; Δ vs B1 paired):")
    w()
    distance_table(R, DV, FINAL)
    w("## 3. Cross-view protocol (train on two views, test on the third)")
    w()
    crossview_tables(R, X)
    w("## 4. Expert / non-expert group probe (E11)")
    w()
    w("AUC of a linear probe for the group label on frozen video embeddings [95% CI]:")
    w()
    group_probe_table(R, G, FINAL, "pooled E01–E04")
    w("## 5. Module III (rule-based assessment against the PT demo)")
    w()
    w("Agreement of the Module III set score with the physiotherapists within groups (Spearman ρ [95% CI]; E03 after "
      "the M18 calibration):")
    w()
    module3_table(R, A, FINAL)
    if CA is not None and RR is not None:
        w("**E03 range-rule calibration (M18, post-hoc):**")
        w()
        rows = []
        for ver in RR[RR.exercise == "E03"].version.unique():
            r = RR[(RR.exercise == "E03") & (RR.version == ver) & (RR.rule == "lateral_rotation_range_insufficient")]
            if len(r):
                r = r.iloc[0]
                rows.append([ver, f"{100 * r.experts_ge95:.0f}%", f"{100 * r.experts_lt80:.0f}%",
                             f"{100 * r.non_experts:.0f}%"])
        R.table(["version", "reps flagged: experts ≥ 95", "experts < 80", "non-experts"], rows)
        rows = [[f"E03 {r.group}", r.target, f2(r.before), f2(r.after), ci(r, "delta")]
                for _, r in CA[CA.exercise == "E03"].iterrows()]
        w("Agreement with the physiotherapists before (PT-demo range target) and after (experts' range target):")
        w()
        R.table(["stratum", "target", "ρ before", "ρ after", "Δ [95% CI]"], rows)
    w("---")
    w()
    w("# Appendix A (supplementary, not part of the final scope): E05–E09")
    w()
    w("Step 1 and Module III only (no cross-view runs, nothing further run for these exercises, D18). From the "
      "nine-exercise step-1 report (`outputs/full/results/perex_gkf5/`; its pooled strata cover all nine exercises "
      "and are not repeated here). ⚠ marks strata with fewer than 15 subjects (E07, E08, E09 experts; E09 "
      "non-experts).")
    w()
    w("**A.1 Total score**, video head:")
    w()
    score_tables(R, S9, D9, NF, APPX, [], "score")
    w("**A.2 MQ**, video head:")
    w()
    score_tables(R, S9, D9, NF, APPX, [], "MQ")
    w("**A.3 Within-group VCI** (centred):")
    w()
    vci_table(R, V9, APPX)
    w("**A.4 View stability of the distance to the PT reference:**")
    w()
    distance_table(R, DV9, APPX)
    w("**A.5 Group probe** (AUC):")
    w()
    group_probe_table(R, G9[G9.exercise.isin(APPX)], APPX, "")
    w("**A.6 Module III agreement** (Spearman ρ):")
    w()
    module3_table(R, A, APPX)
    out = res / "final_E01_E04"
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(R.L) + "\n"
    (out / "tables.md").write_text(text, encoding="utf-8")
    (REPO / "reports").mkdir(exist_ok=True)  # the committed copy, linked from reports/FINAL_REPORT_E01_E04.md
    (REPO / "reports" / "final_E01_E04_tables.md").write_text(text, encoding="utf-8")
    print("written", out / "tables.md", "and reports/final_E01_E04_tables.md")


if __name__ == "__main__":
    main()
