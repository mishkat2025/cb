"""Phase 1 visual checks, skeleton-only (D10): skeleton montages with the primary angle and detected reps,
rep spot-check grids per exercise, aligned multi-view angle curves for sync groups, PT reference videos.

usage: python -m cvdc.viz.checks [--config C] [--per-exercise 16]
Writes PNGs to outputs/checks/.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from ..assess.angles import UNITS  # noqa: E402
from ..config import exercise_cfg, load_config, resolve  # noqa: E402
from ..data.reps import video_angles  # noqa: E402
from .skeleton import COL_L, COL_R, draw_skeleton  # noqa: E402

VIEW_COL = {"F": "#2f855a", "L": "#2b6cb0", "R": "#c05621"}


def load_pre(path):
    with np.load(path) as z:
        return {k: z[k] for k in ("t", "xy", "world", "gap_s")}


def angle_axes(ax, pre, reps, side, pa, title=None):
    """The exercise's rep signal for both arms (bold = moving arm), reps shaded, peaks marked."""
    t = pre["t"]
    sig = video_angles(pre, pa)
    for s, c in (("left", COL_L), ("right", COL_R)):
        ax.plot(t, sig[s], color=c, lw=1.6 if s == side else 0.8, alpha=1 if s == side else 0.5)
    for r in reps.itertuples():
        ax.axvspan(t[r.start], t[r.end], color="#a0aec0", alpha=0.25, lw=0)
        ax.plot(t[r.peak], r.peak_val, "v", color="k", ms=4)
    if pa["name"] == "shoulder_elevation":
        ax.set_ylim(-5, 185)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel(f"{pa['name']} ({UNITS[pa['name']]})", fontsize=7)
    ax.set_xlabel("time (s)")
    if title:
        ax.set_title(title, fontsize=9)


def skeleton_check(pre, reps, side, pa, title, out, n=8):
    if len(reps):
        r0 = reps.iloc[0]
        r1 = reps.iloc[min(1, len(reps) - 1)]
        frames = np.linspace(r0.start, r1.end, n).round().astype(int)
    else:
        frames = np.linspace(0, len(pre["t"]) - 1, n).round().astype(int)
    fig = plt.figure(figsize=(16, 6.2))
    gs = GridSpec(2, n, height_ratios=[1.5, 1], figure=fig)
    sel = pre["xy"][frames]
    finite = sel[np.isfinite(sel).all(-1)]
    lo, hi = np.nanpercentile(finite, 1, 0) - 0.3, np.nanpercentile(finite, 99, 0) + 0.3
    for k, f in enumerate(frames):
        ax = fig.add_subplot(gs[0, k])
        draw_skeleton(ax, pre["xy"][f], pre["gap_s"][f].astype(float))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(hi[1], lo[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={pre['t'][f]:.1f}s", fontsize=8)
    angle_axes(fig.add_subplot(gs[1, :]), pre, reps, side, pa)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def rep_grid(rows, reps, pre_paths, pa, title, out, ncols=4):
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.4 * nrows), squeeze=False)
    for ax, r in zip(axs.ravel(), rows.itertuples()):
        pre = load_pre(pre_paths[r.video_id])
        angle_axes(ax, pre, reps[reps.video_id == r.video_id], r.side, pa,
                   f"{r.video_id}  score {r.score:.0f}  PF4 {r.PF4:.2f}  reps {r.n_reps}")
        ax.set_xlabel("")
    for ax in axs.ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def sync_check(group, idx, reps, pa, out, fps):
    """The rep signal of the moving arm per view, on the aligned clock (t - offset_view); shaded spans = the
    group's shared reps (identical in every aligned view) or, for unaligned views, nothing."""
    vids = idx[idx.group_id == group.group_id]
    fig, ax = plt.subplots(figsize=(12, 3))
    rg = reps[reps.group_id == group.group_id]
    for r in vids.itertuples():
        off = getattr(group, f"offset_{r.view}_s")
        rv = rg[(rg.video_id == r.video_id) & rg.rep_source.isin(["best_view", "transfer"])]
        if np.isfinite(off) and len(rv):
            for x in rv.itertuples():
                ax.axvspan(x.start / fps - off, x.end / fps - off, color="#a0aec0", alpha=0.12, lw=0)
    for r in vids.itertuples():
        pre = load_pre(resolve(r.pre_path))
        off = getattr(group, f"offset_{r.view}_s")
        sh = 0.0 if not np.isfinite(off) else off
        a = video_angles(pre, pa)
        s = max(a, key=lambda k: np.nanpercentile(a[k], 95) - np.nanpercentile(a[k], 5))
        ax.plot(pre["t"] - sh, a[s], color=VIEW_COL[r.view], lw=1.2,
                label=f"{r.view} ({s}, offset {sh:+.3f}s)" if np.isfinite(off) else f"{r.view} (not aligned)")
    ax.set_title(f"{group.group_id}: {group.cls} by {group.method} ({group.confidence}); "
                 f"motion-confirmed pairs {group.motion_confirmed_pairs}/{group.n_pairs}", fontsize=9)
    ax.set_xlabel("aligned time (s)")
    ax.set_ylabel(f"{pa['name']} ({UNITS[pa['name']]})", fontsize=7)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--per-exercise", type=int, default=16)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = resolve(cfg["outputs"]["checks"])
    out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    V = pd.read_csv(resolve(cfg["outputs"]["phase1"]) / "reps_per_video.csv")
    V = V.merge(idx[["video_id", "pre_path"]], on="video_id")
    reps = pd.read_csv(resolve("data/reps.csv"))
    pre_paths = {r.video_id: resolve(r.pre_path) for r in idx.itertuples()}
    rng = np.random.default_rng(0)
    n_png = 0
    for ex, g in V.groupby("exercise"):
        pa = exercise_cfg(cfg, ex)["primary_angle"]
        # skeleton montages: one expert F+L+R group (FL), low light, occlusion, low resolution, a non-expert
        picks = []
        flr = g[(g.group == "expert") & (g.variation == "FL")]
        full = flr.group_id.value_counts()
        if len(full):
            picks += list(flr[flr.group_id == full.index[full == full.max()][0]].video_id)
        for var in ("LL", "O", "LR"):
            c = g[(g.group == "expert") & (g.variation == var)]
            if len(c):
                picks.append(c.iloc[rng.integers(len(c))].video_id)
        c = g[g.group == "non-expert"]
        if len(c):
            picks.append(c.iloc[rng.integers(len(c))].video_id)
        for v in picks:
            r = g[g.video_id == v].iloc[0]
            skeleton_check(load_pre(pre_paths[v]), reps[reps.video_id == v], r.side, pa,
                           f"{v} | {r.group} | score {r.score:.0f} | {r.n_reps} reps ({r.side} arm) | skeleton only",
                           out / f"skeleton_{v}.png")
            n_png += 1
        # rep spot-check grids: random expert and non-expert videos
        for grp, k in (("expert", a.per_exercise), ("non-expert", min(8, a.per_exercise))):
            c = g[g.group == grp]
            if len(c):
                rows = c.iloc[rng.choice(len(c), min(k, len(c)), replace=False)].sort_values("video_id")
                rep_grid(rows, reps, pre_paths, pa, f"{ex} {grp}: rep signal (bold = moving arm), reps shaded",
                         out / f"reps_{ex}_{grp}.png")
                n_png += 1
        # the videos with the most and the fewest reps among experts (outlier check)
        e = g[g.group == "expert"].sort_values("n_reps")
        if len(e) >= 8:
            rep_grid(pd.concat([e.head(4), e.tail(4)]), reps, pre_paths, pa, f"{ex} expert: fewest / most reps",
                     out / f"reps_{ex}_extremes.png")
            n_png += 1

    rep_path = resolve(cfg["sync"]["report_csv"])
    if rep_path.exists():
        G = pd.read_csv(rep_path)
        for ex, g in G.groupby("exercise"):
            for (cls, method), gg in g.groupby(["cls", "method"]):
                for grp in gg.sample(min(2, len(gg)), random_state=0).itertuples():
                    sync_check(grp, idx, reps, exercise_cfg(cfg, ex)["primary_angle"],
                               out / f"sync_{grp.group_id}_{cls}_{method}.png", cfg["preprocess"]["fps"])
                    n_png += 1

    ref = pd.read_csv(resolve(cfg["data"]["ref_index_csv"]))
    rp = resolve("data/reps_ref.csv")
    rref = pd.read_csv(rp) if rp.exists() and rp.stat().st_size > 2 else pd.DataFrame()
    for r in ref.itertuples():
        p = resolve(r.pre_path)
        if p.exists():
            rr = rref[(rref.ref_id == r.ref_id) & (rref.exercise == r.exercise)] if len(rref) else rref
            side = rr.side.iloc[0] if len(rr) else "right"
            skeleton_check(load_pre(p), rr, side, exercise_cfg(cfg, r.exercise)["primary_angle"],
                           f"PT reference {r.file} ({r.exercise} rep settings) | {len(rr)} reps | skeleton only",
                           out / f"skeleton_ref_{r.exercise}_{r.ref_id}.png")
            n_png += 1
    print(f"wrote {n_png} PNGs to {out}")


if __name__ == "__main__":
    main()
