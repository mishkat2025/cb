"""Label diagnostics and video-level baselines for Phase 3 step (a2) (E10, E16, E20, T14).

  video_table       one row per video with >= 1 training rep (the evaluated videos, E15): labels, stratum
                    (exercise x expert / non-expert) and the set-level features
  set_features      per video: the set rep count (Phase 1 ``n_reps_set``, best view, P6b), the mean rep duration of
                    its training reps and the mean range of motion of the exercise's main angle over those reps
                    (``assess.angle``: E01 3D shoulder elevation, E02 3D horizontal adduction, M7a); cached in
                    data/cache/set_features.csv
  item_activity     T14: per stratum and EAAQ item, the spread of the labels and whether the item varies within
                    the group (off its modal value in videos of >= ``min_subjects`` subjects)
  score_decomposition  the total score is (100 / 7) x the sum of the 7 items, so each item's covariance with the
                    score splits the within-group score variance exactly into item shares
  video_baselines   out-of-fold predictions per video on the same folds as the models, fitted on the training
                    fold's videos (each video once, as the T6 weighting): predict-the-mean, group-only, duration-only
                    (E16, now video-weighted) and the within-group regressions rep-count-only, duration-within,
                    ROM-within and set-features (all three features; E20)
"""
import numpy as np
import pandas as pd

from ..assess.angles import horizontal_adduction, shoulder_elevation
from ..config import data_path, resolve

ALL_ITEMS = ["PF1", "PF2", "PF3", "PF4", "CF1", "CF2", "CF3"]
SET_FEATURES = ["n_reps_set", "rep_duration_s", "rom_deg"]
STRATA = [("E01 expert", "E01", "expert"), ("E02 expert", "E02", "expert"), ("E01 non-expert", "E01", "non-expert")]
VIDEO_BASELINES = {
    "base_mean": None, "base_group": None, "base_duration": None,
    "base_repcount": ["n_reps_set"], "base_duration_within": ["rep_duration_s"], "base_rom_within": ["rom_deg"],
    "base_setfeat": SET_FEATURES,
}


def stratum_name(exercise, group):
    return f"{exercise} {group}"


def main_angle(world, cfg_ex, side):
    a = cfg_ex["assess"]["angle"]
    if a == "shoulder_elevation":
        return shoulder_elevation(world, side)
    if a == "horizontal_adduction":
        return horizontal_adduction(world, side, cfg_ex["assess"].get("angle_min_proj", 0.0))
    raise ValueError(f"no main-angle rule for {a!r}")


def angle_range(a, name):
    """ROM of one rep's angle values (deg). Shoulder elevation lives in [0, 180]: max - min. Horizontal adduction
    is an arctan2 angle that wraps at +-180 deg, so its ROM is the circular range: 360 - the largest gap between
    the sorted angles (the smallest arc that holds every value)."""
    if name != "horizontal_adduction":
        return float(a.max() - a.min())
    s = np.sort(np.mod(a, 360.0))
    gaps = np.diff(np.r_[s, s[0] + 360.0])
    return float(360.0 - gaps.max())


def set_features(cfg, refresh=False):
    """Per-video set features (see module docstring), cached in data/cache/set_features.csv."""
    path = data_path(cfg, "set_features_csv")
    if path.exists() and not refresh:
        return pd.read_csv(path)
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"])).set_index("video_id")
    reps = pd.read_csv(data_path(cfg, "reps_csv"))
    rpv = pd.read_csv(resolve(cfg["outputs"]["phase1"]) / "reps_per_video.csv").set_index("video_id")
    rows = []
    for vid, r in reps.groupby("video_id", sort=False):
        ex = r.exercise.iloc[0]
        w = np.load(resolve(idx.loc[vid, "pre_path"]))["world"].astype(float)
        roms, defined = [], []
        for x in r.itertuples():
            a = main_angle(w[x.start:x.end + 1], cfg["exercises"][ex], x.side)
            ok = np.isfinite(a)
            defined.append(ok.mean())
            roms.append(angle_range(a[ok], cfg["exercises"][ex]["assess"]["angle"]) if ok.sum() >= 2 else np.nan)
        rows.append(dict(video_id=vid, n_reps_set=int(rpv.loc[vid, "n_reps_set"]), n_train_reps=len(r),
                         rep_duration_s=float(r.duration_s.mean()),
                         rom_deg=float(np.nanmean(roms)) if np.isfinite(roms).any() else np.nan,
                         rom_reps_defined=float(np.isfinite(roms).mean()), angle_defined_frac=float(np.mean(defined)),
                         main_angle=cfg["exercises"][ex]["assess"]["angle"]))
    out = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def video_table(cfg, feats=None):
    """Evaluated videos (>= 1 training rep) with labels, stratum and set features."""
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    feats = set_features(cfg) if feats is None else feats
    v = feats.merge(idx[["video_id", "exercise", "subject", "group", "view", "variation", "group_id", "score",
                         "quality", *ALL_ITEMS]], on="video_id", how="left", validate="one_to_one")
    v["stratum"] = [stratum_name(e, g) for e, g in zip(v.exercise, v.group)]
    return v


def item_activity(v, items=ALL_ITEMS, min_subjects=5, min_sd=0.05):
    """T14: one row per (stratum, item): mean, SD (0-1), SD in score points ((100 / 7) x SD), videos and subjects
    off the stratum's modal value, and ``varies`` = off-mode in >= min_subjects subjects and SD >= min_sd."""
    rows = []
    for (ex, g), d in v.groupby(["exercise", "group"]):
        for k in items:
            x = d[k].round(4)
            off = x != x.mode().iloc[0]
            sd = float(d[k].std(ddof=0))
            ns = int(d.loc[off, "subject"].nunique())
            rows.append(dict(stratum=stratum_name(ex, g), exercise=ex, group=g, item=k, videos=len(d),
                             subjects=d.subject.nunique(), mean=float(d[k].mean()), sd=sd, sd_points=100 / 7 * sd,
                             n_values=int(x.nunique()), off_mode_videos=int(off.sum()), off_mode_subjects=ns,
                             varies=bool(ns >= min_subjects and sd >= min_sd)))
    return pd.DataFrame(rows)


def active_items(v, items, min_subjects=5, min_sd=0.05):
    """{(exercise, group): [items that vary within the group]} (T14)."""
    a = item_activity(v, items, min_subjects, min_sd)
    return {(e, g): list(d[d.varies].item) for (e, g), d in a.groupby(["exercise", "group"])}


def score_decomposition(v):
    """Per stratum: score variance and, per item k, share_k = Cov(score, (100/7) item_k) / Var(score) (the shares
    sum to 1 because score = (100/7) sum_k item_k) and the single-item R^2 (squared correlation)."""
    rows = []
    for (ex, g), d in v.groupby(["exercise", "group"]):
        var = float(d.score.var(ddof=0))
        recon = float(np.abs(d.score - 100 / 7 * d[ALL_ITEMS].sum(axis=1)).max())
        for k in ALL_ITEMS:
            c = float(np.mean((d.score - d.score.mean()) * (100 / 7) * (d[k] - d[k].mean())))
            r2 = float(np.corrcoef(d.score, d[k])[0, 1] ** 2) if d[k].std() > 0 else np.nan
            rows.append(dict(stratum=stratum_name(ex, g), item=k, score_var=var, score_sd=np.sqrt(var),
                             share=c / var, r2_single=r2, max_reconstruction_error=recon))
    return pd.DataFrame(rows)


def _ols(X, y, Xt):
    """Least squares with intercept on standardised columns; columns constant in the training data are dropped."""
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-9
    A = np.c_[np.ones(len(X)), (X[:, keep] - mu[keep]) / sd[keep]]
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return np.c_[np.ones(len(Xt)), (Xt[:, keep] - mu[keep]) / sd[keep]] @ beta


def video_baselines(v, plans, targets):
    """Out-of-fold predictions for every test video of every plan (dict with 'fold', 'train', 'test' subject
    lists). Training videos = all videos of the training fold (inner validation included, E16). Features with NaN
    (ROM of a video whose main angle is never defined) are imputed with the training stratum's median.
    Returns a long frame: video_id, fold, method, <target>_pred."""
    out = []
    for p in plans:
        tr, te = v[v.subject.isin(p["train"])], v[v.subject.isin(p["test"])]
        for name, feats in VIDEO_BASELINES.items():
            d = te[["video_id"]].copy()
            d["fold"], d["method"] = p["fold"], name
            for t in targets:
                pred = np.full(len(te), np.nan)
                if name == "base_mean":
                    pred[:] = tr[t].mean()
                elif name == "base_duration":  # E16: per exercise, both groups
                    for ex, de in tr.groupby("exercise"):
                        sel = (te.exercise == ex).to_numpy()
                        pred[sel] = _ols(de[["rep_duration_s"]].to_numpy(), de[t].to_numpy(),
                                         te.loc[sel, ["rep_duration_s"]].to_numpy())
                else:  # within group: per exercise x group
                    for (ex, g), de in tr.groupby(["exercise", "group"]):
                        sel = ((te.exercise == ex) & (te.group == g)).to_numpy()
                        if not sel.any():
                            continue
                        if feats is None:
                            pred[sel] = de[t].mean()
                            continue
                        Xtr, Xte = de[feats].to_numpy(float), te.loc[sel, feats].to_numpy(float)
                        med = np.nanmedian(Xtr, 0)
                        Xtr = np.where(np.isfinite(Xtr), Xtr, med)
                        Xte = np.where(np.isfinite(Xte), Xte, med)
                        pred[sel] = _ols(Xtr, de[t].to_numpy(float), Xte)
                d[f"{t}_pred"] = pred
            out.append(d)
    return pd.concat(out, ignore_index=True)
