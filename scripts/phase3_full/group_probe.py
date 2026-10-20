"""E11 (supervisor warning, fourth review): do the embeddings encode the expert / non-expert group more than they order
the scores within a group? Full scope, GroupKFold-10 (runs/full/<tag>, cvdc.train.run).

Per encoder and fold, every rep of every subject is embedded with the fold's checkpoint (z, 256-d, plain packs) and
a video embedding = the mean of its reps' z. Probes are linear, trained on the fold's training subjects' videos and
scored on its test subjects' videos (pooled over the folds, as E1); features are standardised with the training
videos; the regularisation is chosen by a subject-grouped 5-fold CV inside the training subjects.
  group probe   logistic regression expert vs non-expert (class-balanced weights), one per exercise that has both
                groups (E01, E03, E05, E08, E09); out-of-fold probability -> balanced accuracy (threshold 0.5) and AUC,
                per exercise and pooled over the five
  view-matched  the same on front-view FL videos only, where both groups were filmed in the same view and light
  score probe   ridge regression of the total score within each exercise x group stratum; out-of-fold Spearman and
                R2 against the within-group training mean (R2 > 0 = better than the group-only baseline); pooled
                within group after subtracting each video's within-group training mean from label and prediction
Encoders: B1, B2-MediaPipe, B2-KTPFormer, Proposed (lambda 1.0) final checkpoints; the frozen CVDC encoder
(Proposed alternative = the encoder after CVDC); the random-init encoder of the 2D input (seed of the fold), as a
floor. 95% CIs: subject bootstrap within expert / non-expert (E17).

usage: python scripts/phase3_full/group_probe.py [--config configs/data_full.yaml] [--tag full_gkf10] [--boot 2000]
Writes group_probe_summary.csv, group_probe_per_exercise.csv, group_probe_predictions.csv, score_probe.csv to
<eval.results_dir>/<tag>/.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.dataset import PackData  # noqa: E402
from cvdc.eval.metrics import interval, item_weights, regression, subject_draws, wranks  # noqa: E402
from cvdc.models.model import CVDCModel  # noqa: E402
from cvdc.train.run import method_config  # noqa: E402
from cvdc.train.trainer import fold_plan, forward_all, get_device, seed_everything  # noqa: E402

ENCODERS = [("init_xy", "random-init encoder (2D input)", "b1", "init"),
            ("b1", "B1 (2D xy, scratch)", "b1", "final"),
            ("b2_mediapipe", "B2-MediaPipe (world xyz, scratch)", "b2_mediapipe", "final"),
            ("b2_ktpformer", "B2-KTPFormer (lifted xyz, scratch)", "b2_ktpformer", "final"),
            ("proposed_j1", "Proposed (CVDC + fine-tune, λ = 1.0)", "proposed_j1", "final"),
            ("proposed_lp", "Proposed alt. = encoder after CVDC (frozen)", "proposed_lp", "final")]
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
ALPHA_GRID = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]


def video_embeddings(model, data, cfg):
    z = forward_all(model, data, torch.arange(len(data), device=data.device), cfg, heads=False)["z"]
    nv = len(data.vmeta)
    s = torch.zeros(nv, z.shape[1], device=z.device).index_add_(0, data.video_of, z)
    return (s / data.v_nreps[:, None].float()).cpu().numpy()


def load_encoder(cfg, data, root, method, stage, fold):
    c = method_config(cfg, method)
    seed = cfg["train"]["seed"] + fold
    if stage == "init":  # as in cvdc.train.run: the same seed, the same construction order
        seed_everything(seed)
        return CVDCModel(c).to(data.device), c
    model = CVDCModel(c).to(data.device)
    ck = torch.load(root / method / f"fold{fold}" / "finetune" / "best.pt", map_location=data.device,
                    weights_only=True)
    model.load_state_dict(ck["model"])
    return model, c


def grouped_cv(est, grid, X, y, groups, scoring, stratified=False):
    """Pick the regularisation by a subject-grouped 5-fold CV inside the training subjects (stratified by class for
    the group probe, so every inner fold holds both groups); too few subjects for a CV -> the middle grid value."""
    if stratified:
        k = min(5, min(len(set(groups[y == c])) for c in (0, 1)))
        cv = StratifiedGroupKFold(k) if k >= 2 else None
    else:
        k = min(5, len(set(groups)))
        cv = GroupKFold(k) if k >= 2 else None
    if cv is None:
        (name, vals), = grid.items()
        return est.set_params(**{name: vals[len(vals) // 2]}).fit(X, y), {name: vals[len(vals) // 2]}
    gs = GridSearchCV(est, grid, scoring=scoring, cv=cv, refit=True)
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_, gs.best_params_


def standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def group_probe(E, vm, plans, view_matched=False):
    rows = []
    for p in plans:
        tr_m, te_m = vm.subject.isin(p["train"]).to_numpy(), vm.subject.isin(p["test"]).to_numpy()
        for ex, d in vm.groupby("exercise"):
            if d.group.nunique() < 2:
                continue
            sel = (vm.exercise == ex).to_numpy()
            if view_matched:
                sel = sel & ((vm.view == "F") & (vm.variation == "FL")).to_numpy()
            tr, te = np.flatnonzero(sel & tr_m), np.flatnonzero(sel & te_m)
            if not len(te) or vm.group.iloc[tr].nunique() < 2:
                continue
            ytr = (vm.group.iloc[tr] == "non-expert").to_numpy(int)
            Xtr, Xte = standardise(E[tr], E[te])
            est, bp = grouped_cv(LogisticRegression(class_weight="balanced", max_iter=5000), {"C": C_GRID}, Xtr, ytr,
                                 vm.subject.iloc[tr].to_numpy(), "roc_auc", stratified=True)
            prob = est.predict_proba(Xte)[:, 1]
            out = vm.iloc[te][["video_id", "subject", "group", "exercise", "view", "variation"]].copy()
            out["fold"], out["p_nonexpert"], out["C"] = p["fold"], prob, bp["C"]
            rows.append(out)
    return pd.concat(rows, ignore_index=True)


def score_probe(E, vm, plans):
    rows = []
    for p in plans:
        tr_m, te_m = vm.subject.isin(p["train"]).to_numpy(), vm.subject.isin(p["test"]).to_numpy()
        for (ex, g), d in vm.groupby(["exercise", "group"]):
            sel = ((vm.exercise == ex) & (vm.group == g)).to_numpy()
            tr, te = np.flatnonzero(sel & tr_m), np.flatnonzero(sel & te_m)
            if not len(te) or vm.subject.iloc[tr].nunique() < 2:
                continue
            ytr = vm.score.iloc[tr].to_numpy(float)
            Xtr, Xte = standardise(E[tr], E[te])
            est, bp = grouped_cv(Ridge(), {"alpha": ALPHA_GRID}, Xtr, ytr, vm.subject.iloc[tr].to_numpy(),
                                 "neg_mean_squared_error")
            out = vm.iloc[te][["video_id", "subject", "group", "exercise", "view", "score"]].copy()
            out["fold"], out["pred"], out["train_mean"], out["alpha"] = p["fold"], est.predict(Xte), ytr.mean(), bp["alpha"]
            rows.append(out)
    return pd.concat(rows, ignore_index=True)


def weighted_auc(y, s, W):
    """AUC (non-expert = positive) per row of W via the weighted Mann-Whitney U on weighted mid-ranks."""
    R = wranks(np.asarray(s, float), W)
    pos = np.asarray(y, bool)
    wp, wn = (W * pos).sum(1), (W * ~pos).sum(1)
    u = (W * pos * R).sum(1) - wp * (wp + 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return u / (wp * wn)


def weighted_bacc(y, s, W):
    pos, hit = np.asarray(y, bool), np.asarray(s) >= 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        tpr = (W * (pos & hit)).sum(1) / (W * pos).sum(1)
        tnr = (W * (~pos & ~hit)).sum(1) / (W * ~pos).sum(1)
    return (tpr + tnr) / 2


def draws(d, n_boot, seed):
    subj = d.drop_duplicates("subject")[["subject", "group"]].sort_values("subject")
    M = subject_draws(subj.subject.to_numpy(), subj.group.to_numpy(), n_boot, seed)
    return np.vstack([np.ones(len(d)), item_weights(d.subject.to_numpy(), subj.subject.to_numpy(), M)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data_full.yaml")
    ap.add_argument("--model-config")
    ap.add_argument("--tag", default="full_gkf10")
    ap.add_argument("--boot", type=int)
    ap.add_argument("--device", help="override train.device (e.g. cpu while a training job holds the GPU)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    t0 = time.time()
    cfg = load_config(a.config, a.model_config)
    if a.device:
        cfg["train"]["device"] = a.device
        if a.device == "cpu":
            torch.set_num_threads(4)  # leave the other cores to a training job
    ec = cfg["eval"]
    n_boot, ci, bseed = a.boot or ec["bootstrap"]["n"], ec["bootstrap"]["ci"], ec["bootstrap"]["seed"]
    root = resolve(cfg["train"]["runs_dir"]) / a.tag
    out = resolve(ec["results_dir"]) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    smoke = a.tag.endswith("_smoke")
    data = PackData(cfg, "rep", get_device(cfg), lift="ktpformer")
    vm = data.vmeta.reset_index(drop=True)
    plans = fold_plan(cfg, data.meta, smoke=smoke)
    plans = [p for p in plans if (root / "b1" / f"fold{p['fold']}" / "done.json").exists()]
    gp_rows, sp_rows = [], []
    for key, label, method, stage in ENCODERS:
        if not all((root / method / f"fold{p['fold']}" / "done.json").exists() for p in plans):
            print(f"skip {key}: not every fold is finished", flush=True)
            continue
        E = {}
        for p in plans:
            model, c = load_encoder(cfg, data, root, method, stage, p["fold"])
            E[p["fold"]] = video_embeddings(model, data, c)
            del model
        # probes per fold use that fold's encoder for its training and test videos
        g, gv, s = [], [], []
        for p in plans:
            g.append(group_probe(E[p["fold"]], vm, [p]))
            gv.append(group_probe(E[p["fold"]], vm, [p], view_matched=True))
            s.append(score_probe(E[p["fold"]], vm, [p]))
        for df, kind in ((pd.concat(g), "all"), (pd.concat(gv), "view_matched")):
            df["encoder"], df["kind"] = key, kind
            gp_rows.append(df)
        s = pd.concat(s)
        s["encoder"] = key
        sp_rows.append(s)
        print(f"{key}: probes done ({time.time() - t0:.0f} s)", flush=True)
    GP, SP = pd.concat(gp_rows, ignore_index=True), pd.concat(sp_rows, ignore_index=True)
    GP.to_csv(out / "group_probe_predictions.csv", index=False)
    SP.to_csv(out / "score_probe_predictions.csv", index=False)

    # ---- metrics with subject-bootstrap CIs; the same draws for every encoder (paired)
    per_ex, summ, sprow = [], [], []
    labels = {k: lab for k, lab, _, _ in ENCODERS}
    for key in [k for k, *_ in ENCODERS if k in set(GP.encoder)]:
        row = dict(encoder=key, label=labels[key])
        for kind, pre in (("all", ""), ("view_matched", "vm_")):
            d = GP[(GP.encoder == key) & (GP.kind == kind)].sort_values("video_id")
            for ex in ["pooled", *sorted(d.exercise.unique())]:
                de = d if ex == "pooled" else d[d.exercise == ex]
                W = draws(de, n_boot, bseed)
                y = (de.group == "non-expert").to_numpy()
                auc, bacc = weighted_auc(y, de.p_nonexpert.to_numpy(), W), weighted_bacc(y, de.p_nonexpert.to_numpy(), W)
                r = dict(encoder=key, kind=kind, exercise=ex, videos=len(de), experts=int(de[~y].subject.nunique()),
                         non_experts=int(de[y].subject.nunique()), auc=auc[0], bacc=bacc[0])
                r["auc_lo"], r["auc_hi"] = interval(auc[1:], ci)
                r["bacc_lo"], r["bacc_hi"] = interval(bacc[1:], ci)
                per_ex.append(r)
                if ex == "pooled":
                    row.update({f"{pre}auc": r["auc"], f"{pre}auc_lo": r["auc_lo"], f"{pre}auc_hi": r["auc_hi"],
                                f"{pre}bacc": r["bacc"], f"{pre}bacc_lo": r["bacc_lo"], f"{pre}bacc_hi": r["bacc_hi"],
                                f"{pre}videos": len(de)})
        ds = SP[SP.encoder == key].sort_values("video_id")
        for (ex, g), de in [(("all", "expert"), ds[ds.group == "expert"]), (("all", "non-expert"),
                                                                            ds[ds.group == "non-expert"])] + \
                list(ds.groupby(["exercise", "group"])):
            W = draws(de, n_boot, bseed)
            yc, pc = de.score.to_numpy() - de.train_mean.to_numpy(), de.pred.to_numpy() - de.train_mean.to_numpy()
            rho = regression(yc, pc, W)["spearman"] if ex == "all" else regression(de.score.to_numpy(),
                                                                                  de.pred.to_numpy(), W)["spearman"]
            sse, sse0 = (W * (de.score.to_numpy() - de.pred.to_numpy()) ** 2).sum(1), (W * yc ** 2).sum(1)
            r2 = 1 - sse / sse0
            r = dict(encoder=key, exercise=ex, group=g, videos=len(de), subjects=de.subject.nunique(), rho=rho[0],
                     r2=r2[0], rmse=float(np.sqrt(sse[0] / len(de))), rmse_group_mean=float(np.sqrt(sse0[0] / len(de))))
            r["rho_lo"], r["rho_hi"] = interval(rho[1:], ci)
            r["r2_lo"], r["r2_hi"] = interval(r2[1:], ci)
            sprow.append(r)
            if ex == "all":
                pre = "sp_exp_" if g == "expert" else "sp_non_"
                row.update({f"{pre}rho": r["rho"], f"{pre}rho_lo": r["rho_lo"], f"{pre}rho_hi": r["rho_hi"],
                            f"{pre}r2": r["r2"], f"{pre}r2_lo": r["r2_lo"], f"{pre}r2_hi": r["r2_hi"]})
        summ.append(row)
    pd.DataFrame(per_ex).to_csv(out / "group_probe_per_exercise.csv", index=False)
    pd.DataFrame(sprow).to_csv(out / "score_probe.csv", index=False)
    pd.DataFrame(summ).to_csv(out / "group_probe_summary.csv", index=False)
    print(pd.DataFrame(summ)[["encoder", "auc", "bacc", "vm_auc", "vm_bacc", "sp_exp_rho", "sp_exp_r2", "sp_non_rho",
                              "sp_non_r2"]].round(3).to_string())
    print(f"written to {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
