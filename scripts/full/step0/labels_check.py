"""Label structure per exercise and group (no models): item activity under the T14 rule (off the stratum's modal value
in videos of >= 5 subjects and SD >= 0.05), the share of the within-group total-score variance that comes from the
repetition items PF3 + PF4 (pilot: 50% E01 experts, 61% E02), whether PF1/PF2/CF1-CF3 are identical across the views
of one expert performance (pilot E3: yes), and the score spread. Total = sum of the 7 judge-mean items / 7 x 100.

usage: python -I scripts/full/step0/labels_check.py --out outputs/step0_full
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ITEMS = ["PF1", "PF2", "PF3", "PF4", "CF1", "CF2", "CF3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/step0_full")
    a = ap.parse_args()
    inv = pd.read_csv(Path(a.out) / "inventory.csv")
    rows = []
    for (ex, grp), g in inv.groupby(["exercise", "group"]):
        r = dict(exercise=ex, group=grp, videos=len(g), subjects=g.subject.nunique(), score_sd=round(g.score.std(), 1),
                 share_100=round(float((g.score >= 99.99).mean()), 2))
        active = []
        for c in ITEMS:
            mode = g[c].round(3).mode().iloc[0]
            off_subj = g.loc[g[c].round(3) != mode, "subject"].nunique()
            if c != "PF3" and off_subj >= 5 and g[c].std() >= 0.05:
                active.append(c)
        r["active_items"] = " ".join(active)
        tot = 100 * g[ITEMS].sum(axis=1) / 7
        rep = 100 * (g.PF3 + g.PF4) / 7
        var_t = tot.var()
        r["rep_items_var_share"] = round(float(np.cov(tot, rep)[0, 1] / var_t), 2) if var_t > 0 else np.nan
        if grp == "expert":
            multi = g[g.n_views_group >= 2]
            same = {c: float(multi.groupby("group_id")[c].nunique().eq(1).mean()) for c in ITEMS}
            r["items_identical_across_views"] = " ".join(c for c in ITEMS if same[c] >= 0.999) or "-"
            r["groups_total_differs"] = int((multi.groupby("group_id").score.nunique() > 1).sum())
            r["multi_view_groups"] = multi.group_id.nunique()
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(Path(a.out) / "labels_check.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
