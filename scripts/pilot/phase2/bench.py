"""Step-time benchmark (bf16 vs fp32) and the runtime estimate for Phase 3 (E2, C7).

Measures on the real data path (CVDC batches, rep- and video-level supervised steps, evaluation forward), then
counts batches per epoch for the actual folds: GroupKFold-5 (stratified by expert / non-expert) and LOSO. The
estimate multiplies these by the epoch budget given on the command line. Run it on an idle GPU.

  GroupKFold-5              CVDC pretraining per fold + fine-tuning per fold
  LOSO, per-fold encoder    CVDC pretraining once per GroupKFold fold; every LOSO subject is fine-tuned from the
                            encoder of the fold that held it out (C7, no leakage)
  LOSO, full (reference)    CVDC pretraining for every LOSO fold
Rep-level LOSO has 50 folds: subject P39's only clip has no detected rep (it stays in video-level runs).

usage: python scripts/pilot/phase2/bench.py [--cvdc-epochs 100] [--sup-epochs 60] [--steps 30]
Writes outputs/phase2/bench_steps.csv, runtime_folds.csv and runtime_estimate.csv.
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.augment import augment  # noqa: E402
from cvdc.data.cvdc_batch import CVDCBatcher  # noqa: E402
from cvdc.data.dataset import PackData  # noqa: E402
from cvdc.eval.splits import check_no_leakage, fold_of, group_kfold, loso  # noqa: E402
from cvdc.losses.cvdc import cvdc_loss  # noqa: E402
from cvdc.losses.supervised import SupervisedLoss, TargetScaler  # noqa: E402
from cvdc.models.model import CVDCModel  # noqa: E402

SUP_BS, VID_BS, EVAL_BS = 256, 32, 2048


def timed(fn, n, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    return float(np.median(ts))


def ctx(amp):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp == "bf16")


def bench_steps(cfg, rep, vid, n):
    rows = []
    batcher = CVDCBatcher(rep, cfg, seed=0)
    it = iter(batcher)
    batches = [next(it) for _ in range(8)]
    t0 = time.time()
    for _ in range(8):
        next(it)
    rows.append(dict(step="cvdc batch building (CPU+GPU)", amp="-", items=len(batches[0]["x"]),
                     ms=1000 * (time.time() - t0) / 8))
    lc = cfg["cvdc"]["loss"]
    items_rep = CVDCModel(cfg).heads.items
    items_vid = CVDCModel(cfg, unit="video").heads.items
    for amp in ("bf16", "fp32"):
        torch.manual_seed(0)
        m = CVDCModel(cfg).cuda()
        opt = torch.optim.AdamW(m.parameters(), 1e-3)
        k = [0]

        def cvdc_step():
            b = batches[k[0] % len(batches)]
            k[0] += 1
            with ctx(amp):
                o = m(b["x"], project=True, heads=False)
            loss = cvdc_loss(o["proj"], b["set_ids"], b["W"], b["group_ids"], b["exercise"], lc["temperature"],
                             lc["negatives"], lc["same_group_other_rep"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        rows.append(dict(step="cvdc step (fwd+bwd+opt)", amp=amp, items=len(batches[0]["x"]), ms=1000 * timed(cvdc_step, n)))

        for unit, data, bs, items in (("rep", rep, SUP_BS, items_rep), ("video", vid, VID_BS, items_vid)):
            torch.manual_seed(0)
            m = CVDCModel(cfg, unit=unit).cuda()
            opt = torch.optim.AdamW(m.parameters(), 1e-3)
            allidx = torch.arange(len(data), device="cuda")
            sup = SupervisedLoss(items, cfg["supervised"]["weights"], TargetScaler().fit(data.targets(allidx, items)))
            ac = {**cfg["cvdc"]["augment"], "share_temporal_crop": False}
            g = torch.Generator(device="cuda").manual_seed(0)

            def sup_step():
                it_ = allidx[torch.randint(len(data), (bs,), device="cuda", generator=g)]
                x = augment(data.inputs(it_, "xy"), ac, 2, generator=g).permute(0, 3, 1, 2).contiguous()
                y = data.targets(it_, items)
                with ctx(amp):
                    o = m(x, exercise=data.exercise[it_])
                loss, _ = sup(o, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            rows.append(dict(step=f"supervised {unit} step (aug+fwd+bwd+opt)", amp=amp, items=bs, ms=1000 * timed(sup_step, n)))

            @torch.no_grad()
            def eval_step():
                m.eval()
                ebs = EVAL_BS if unit == "rep" else 256
                x = data.inputs(allidx[:ebs], "xy").permute(0, 3, 1, 2).contiguous()
                with ctx(amp):
                    m(x, exercise=data.exercise[allidx[:ebs]])
                m.train()

            rows.append(dict(step=f"eval {unit} forward", amp=amp, items=EVAL_BS if unit == "rep" else 256,
                             ms=1000 * timed(eval_step, max(5, n // 3))))
    try:
        import triton  # noqa: F401
        rows.append(dict(step="torch.compile", amp="-", items=0, ms=np.nan, note="triton available, not benchmarked"))
    except ImportError:
        rows.append(dict(step="torch.compile", amp="-", items=0, ms=np.nan,
                         note="unavailable: no triton on this Windows install"))
    return pd.DataFrame(rows)


def fold_counts(cfg, rep, vid, splits, name):
    rows = []
    for i, (tr, te) in enumerate(splits):
        mask = rep.subject_mask(tr)
        n_sets = len(CVDCBatcher(rep, cfg, allowed=mask, seed=0).sets)
        n_tr = int(mask.sum())
        n_te = len(rep) - n_tr
        v_tr = int(vid.subject_mask(tr).sum())
        rows.append(dict(protocol=name, fold=i, train_subjects=len(tr), test_subjects=len(te), train_reps=n_tr,
                         test_reps=n_te, cvdc_sets=n_sets,
                         cvdc_batches=int(np.ceil(n_sets / cfg["cvdc"]["sampler"]["batch_sets"])),
                         sup_batches=int(np.ceil(n_tr / SUP_BS)), train_videos=v_tr, test_videos=len(vid) - v_tr,
                         vid_batches=int(np.ceil(v_tr / VID_BS))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cvdc-epochs", type=int, default=100)
    ap.add_argument("--sup-epochs", type=int, default=60)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config()
    out = resolve("outputs/phase2")
    rep = PackData(cfg, "rep", "cuda")
    vid = PackData(cfg, "video", "cuda")
    print(f"GPU {torch.cuda.get_device_name()}, torch {torch.__version__}")

    S = bench_steps(cfg, rep, vid, a.steps)
    S.to_csv(out / "bench_steps.csv", index=False)
    print(S.round(1).to_string(index=False))
    ms = {(r.step, r.amp): r.ms for r in S.itertuples()}

    gk = group_kfold(rep.meta, 5, a.seed)
    lo = loso(rep.meta)
    subs = rep.meta.subject.unique()
    check_no_leakage(gk, subs)
    check_no_leakage(lo, subs)
    F = pd.concat([fold_counts(cfg, rep, vid, gk, "groupkfold5"), fold_counts(cfg, rep, vid, lo, "loso")])
    F.to_csv(out / "runtime_folds.csv", index=False)
    print(F.groupby("protocol")[["train_subjects", "train_reps", "cvdc_sets", "cvdc_batches", "sup_batches",
                                 "vid_batches"]].agg(["min", "mean", "max"]).round(0).to_string())

    amp = "bf16"
    t_cvdc = ms[("cvdc step (fwd+bwd+opt)", amp)] / 1000 + ms[("cvdc batch building (CPU+GPU)", "-")] / 1000
    t_sup = ms[("supervised rep step (aug+fwd+bwd+opt)", amp)] / 1000
    t_vid = ms[("supervised video step (aug+fwd+bwd+opt)", amp)] / 1000
    t_eval = ms[("eval rep forward", amp)] / 1000 / EVAL_BS
    t_eval_v = ms[("eval video forward", amp)] / 1000 / 256

    def cvdc_s(r):
        return a.cvdc_epochs * r.cvdc_batches * t_cvdc

    def sup_s(r, unit="rep"):  # training epochs + one evaluation pass of the held-out items per epoch
        if unit == "rep":
            return a.sup_epochs * (r.sup_batches * t_sup + r.test_reps * t_eval)
        return a.sup_epochs * (r.vid_batches * t_vid + r.test_videos * t_eval_v)

    G, L = F[F.protocol == "groupkfold5"], F[F.protocol == "loso"]
    nl = len(L)  # rep level: subjects with at least one rep (P39's only clip has none, so 50 of 51)
    pre_gk = sum(cvdc_s(r) for r in G.itertuples())
    rows = []
    for unit in ("rep", "video"):
        ft_gk = sum(sup_s(r, unit) for r in G.itertuples())
        ft_lo = sum(sup_s(r, unit) for r in L.itertuples())
        pre_lo = sum(cvdc_s(r) for r in L.itertuples())
        rows += [
            dict(unit=unit, protocol="GroupKFold-5", method="B1 / B2 (each; supervised only)", hours=ft_gk / 3600),
            dict(unit=unit, protocol="GroupKFold-5", method="Proposed (CVDC pretrain + fine-tune)", hours=(pre_gk + ft_gk) / 3600),
            dict(unit=unit, protocol=f"LOSO-{nl}", method="B1 / B2 (each; supervised only)", hours=ft_lo / 3600),
            dict(unit=unit, protocol=f"LOSO-{nl}, per-fold encoder (C7)", method=f"Proposed (5 pretrains + {nl} fine-tunes)",
                 hours=(pre_gk + ft_lo) / 3600),
            dict(unit=unit, protocol=f"LOSO-{nl}, full (reference)", method=f"Proposed ({nl} pretrains + {nl} fine-tunes)",
                 hours=(pre_lo + ft_lo) / 3600),
        ]
    R = pd.DataFrame(rows)
    gk_set = R[(R.unit == "rep") & (R.protocol == "GroupKFold-5")].set_index("method").hours
    lo_set = R[(R.unit == "rep") & R.protocol.str.startswith("LOSO")].set_index("method").hours
    summary = {
        "main table, rep level, LOSO with per-fold encoder (B1 + B2-MediaPipe + B2-KTPFormer + Proposed)":
            3 * lo_set["B1 / B2 (each; supervised only)"] + lo_set[f"Proposed (5 pretrains + {nl} fine-tunes)"],
        "one CVDC ablation setting on GroupKFold-5 (pretrain + fine-tune)": gk_set["Proposed (CVDC pretrain + fine-tune)"],
        "CVDC pretraining alone, one GroupKFold fold": pre_gk / 5 / 3600,
        "fine-tuning alone, one LOSO fold (rep level)": sum(sup_s(r) for r in L.itertuples()) / len(L) / 3600,
    }
    for k, v in summary.items():
        rows.append(dict(unit="rep", protocol="summary", method=k, hours=v))
    R = pd.DataFrame(rows)
    R["assumptions"] = (f"bf16; CVDC {a.cvdc_epochs} epochs x {cfg['cvdc']['sampler']['batch_sets']} sets/batch; "
                        f"fine-tune {a.sup_epochs} epochs, batch {SUP_BS} reps / {VID_BS} videos, 1 eval pass per epoch; "
                        f"no early stopping")
    R.to_csv(out / "runtime_estimate.csv", index=False)
    print(f"\nRuntime estimate (hours; {R.assumptions.iloc[0]}):")
    print(R[["unit", "protocol", "method", "hours"]].round(2).to_string(index=False))
    print("LOSO shortcut fold map (subject -> GroupKFold fold):", fold_of(gk))


if __name__ == "__main__":
    main()
