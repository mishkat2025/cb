"""Classify every multi-view group of the new exercises as synced / not-synced / unknown (Step 0 method, S1-S3).

Order of evidence: (1) audio: F+L+R groups whose three GCC-PHAT lags (+-3 s) are additive within 20 ms and not at
the window edge -> synced (high); (2) creation_time: recording windows that cannot overlap -> not-synced (high),
every pair |dC| <= 3 s -> synced (medium); (3) motion (pose sample only): every pair confirmed by the rep-timing
test -> synced (medium); otherwise unknown. Also reports the creation-time controls (same subject, different
variation), the LR / missing-creation-time copy checks and the per-exercise audio statistics.

usage: python -I scripts/step0_full/sync_classify.py --out outputs/step0_full
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/step0_full")
    a = ap.parse_args()
    D = Path(a.out)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    p = pd.read_csv(D / "av_pairs.csv")
    vids = pd.read_csv(D / "av_videos.csv")
    mo_path = D / "motion_pairs_sample.csv"
    mo = pd.read_csv(mo_path) if mo_path.exists() else pd.DataFrame()

    print("=== containers: phone model by exercise x view x group (expert P01-P24)")
    vids["group"] = np.where(vids.pid.str[1:].astype(int) <= 24, "expert", "non-expert")
    print(vids.groupby(["ex", "group", "angle"]).model.agg(lambda s: s.value_counts().head(3).to_dict()).to_string())
    print("decode errors:", vids.loc[vids.get("error", pd.Series(dtype=str)).notna(), ["file", "error"]].values.tolist()
          if "error" in vids else "none")

    lr = p[p.kind == "lr_check"].sort_values("audio_peak_ratio", ascending=False).groupby(
        ["ex", "pid", "var_a", "view_a"]).head(1)
    print(f"\n=== LR videos: {len(lr)}; audio copy of another variation (peak ratio > 10): "
          f"{lr.loc[lr.audio_peak_ratio > 10, ['ex', 'pid', 'view_a', 'var_b', 'audio_peak_ratio']].values.tolist() or 'none'}")

    s = p[p.kind == "same_group"].copy()
    s["group_id"] = s.ex + "_" + s.pid + "_V" + s.var_a
    if len(mo):
        m = mo[mo.kind == "same_group"][["group_id", "view_a", "view_b", "motion_confirmed", "resid_sd_lin_ms"]]
        s = s.merge(m, on=["group_id", "view_a", "view_b"], how="left")
    else:
        s["motion_confirmed"], s["resid_sd_lin_ms"] = np.nan, np.nan
    rows = []
    for gid, g in s.groupby("group_id"):
        views = sorted(set(g.view_a) | set(g.view_b))
        lag = {(x, y): l for x, y, l in zip(g.view_a, g.view_b, g.audio_lag3_s)}
        raw = {(x, y): l for x, y, l in zip(g.view_a, g.view_b, g.audio_lag3_raw_s)}
        tri = np.nan
        if views == ["F", "L", "R"] and np.isfinite(list(lag.values())).all() and len(lag) == 3:
            tri = 1000 * abs(lag[("F", "L")] + lag[("L", "R")] - lag[("F", "R")])
        edge = any(abs(v) > 2.9 or not np.isfinite(v) for v in raw.values())
        apart = (g.d_creation_s.abs() > (g.dur_a + g.dur_b)).any()
        close = (g.d_creation_s.abs() <= 3).all()
        # creation_time looks like the END of the recording on these phones (see report): start times agree
        close_start = ((g.d_creation_s - (g.dur_b - g.dur_a)).abs() <= 3).all()
        tested = g.motion_confirmed.notna().sum()
        mot_all = tested == len(g) and g.motion_confirmed.fillna(False).astype(bool).all()
        # recording windows that cannot overlap override audio: identical sound cues in separate takes give
        # additive audio lags too (E03 P59 FL, non-expert)
        if tri <= 20 and not edge and not apart:
            cls, method = "synced", "audio"
        elif apart:
            cls, method = "not-synced", "creation_time"
        elif close:
            cls, method = "synced", "creation_time"
        elif close_start:
            cls, method = "synced", "creation_time_start"
        elif mot_all:
            cls, method = "synced", "motion"
        else:
            cls, method = "unknown", "-"
        rows.append(dict(group_id=gid, ex=g.ex.iloc[0], pid=g.pid.iloc[0], var=g.var_a.iloc[0], views="".join(views),
                         group="expert" if int(g.pid.iloc[0][1:]) <= 24 else "non-expert", cls=cls, method=method,
                         max_abs_dC_s=float(g.d_creation_s.abs().max()), audio_triangle_err_ms=tri,
                         motion_tested=int(tested), motion_confirmed=int(g.motion_confirmed.fillna(False).sum()),
                         **{f"lag_{v}_vs_F_s": lag.get(("F", v), np.nan) for v in ("L", "R")}))
    G = pd.DataFrame(rows)
    G.to_csv(D / "sync_groups_classified.csv", index=False)
    print("\n=== group classes (all multi-view groups)")
    print(G.groupby(["ex", "group", "cls", "method"]).size().unstack(["cls", "method"], fill_value=0).to_string())
    e = G[G.group == "expert"]
    print("\nexpert pairs |dC| <= 3 s:", s[s.pid.str[1:].astype(int) <= 24].groupby("ex").d_creation_s.apply(
        lambda d: f"{int((d.abs() <= 3).sum())}/{len(d)} (max {d.abs().max():.1f} s)").to_dict())
    print("share of expert FLR groups with audio triangle error <= 20 ms:",
          e[e.audio_triangle_err_ms.notna()].groupby("ex").audio_triangle_err_ms.apply(
              lambda x: f"{(x <= 20).mean():.2f} (median {x.median():.0f} ms)").to_dict())
    au = G[G.method == "audio"]
    if len(au):
        print("audio-synced groups |lag vs F| median %.2f s, max %.2f s" % (
            np.nanmedian(np.abs(au[["lag_L_vs_F_s", "lag_R_vs_F_s"]].values)),
            np.nanmax(np.abs(au[["lag_L_vs_F_s", "lag_R_vs_F_s"]].values))))
    ne = s[s.pid.str[1:].astype(int) > 24]
    if len(ne):
        print("non-expert same-group pairs: |dC| s by exercise:",
              ne.groupby("ex").d_creation_s.apply(lambda d: d.abs().round(0).tolist()).to_dict())
    c = p[p.kind == "control"]
    ds = (c.d_creation_s - (c.dur_b - c.dur_a)).abs()
    print(f"\ncontrols (same subject, adjacent variations): {len(c)} pairs; |dC| <= 3 s: "
          f"{int((c.d_creation_s.abs() <= 3).sum())}; start-time difference <= 3 s: {int((ds <= 3).sum())}; "
          f"min |dC| {c.d_creation_s.abs().min():.1f} s, min start diff {ds.min():.1f} s; median "
          f"{c.d_creation_s.abs().median() / 60:.1f} min")
    a = s.merge(G[["group_id", "method"]], on="group_id")
    a = a[a.method == "audio"]
    if len(a):
        r_start = (a.d_creation_s + a.audio_lag3_s).abs()                       # creation = start: dC = -lag
        r_end = (a.d_creation_s - (a.dur_b - a.dur_a) + a.audio_lag3_s).abs()  # creation = end
        print("audio-synced pairs, |dC residual| (s) median / p90 / max: creation = start "
              f"{r_start.median():.1f} / {r_start.quantile(.9):.1f} / {r_start.max():.1f}; creation = end "
              f"{r_end.median():.1f} / {r_end.quantile(.9):.1f} / {r_end.max():.1f}; end fits better in "
              f"{(r_end < r_start).mean():.0%}")
    if len(mo):
        smp = G[G.motion_tested > 0]
        print("\nsampled groups: class vs motion evidence")
        print(smp.groupby(["ex", "cls", "method"]).agg(groups=("group_id", "size"),
                                                       pairs_confirmed=("motion_confirmed", "sum"),
                                                       pairs_tested=("motion_tested", "sum")).to_string())


if __name__ == "__main__":
    main()
