"""Phase 5 B report: CVDC pair-source ablation on the held-out within-group VCI (C5, E25), GroupKFold-5.

Settings (``ablation.sources``): (a) sync only, (b) + audio, (c) + rep-matched, (d) + synthetic = all sources (the
step (a) pretraining, reused), (d') all without rep-matched pairs, (e) synthetic only. For each, the CVDC encoder after
pretraining (no fine-tuning) is scored by the VCI on the held-out test subjects of its fold (E13, E19: centred 256-d
primary, raw and projection; same-rep view pairs from data/cvdc_members.csv, whatever the training sources), within
E01 expert, E02 expert and E01 non-expert (the separate stratum C5 asks for, D12), pooled over the 5 folds, with
subject-bootstrap CIs (E17) and paired differences (same draws) between settings.

Pair counts per source: for every setting, the positive pairs one epoch of CVDC training holds (unordered pairs with
weight > 0 in the batch weight matrices, A7): real pairs by source and pairs involving a synthetic view, per stratum,
over all subjects and per fold's inner training subjects (mean over folds). Synthetic views are drawn at random per
epoch, so their count is one epoch's draw with a fixed seed.

usage: python scripts/pilot/phase5/report_pair_sources.py [--boot 2000] [--smoke]
Writes CSVs and summary.md to outputs/results/phase5_sources/.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pilot" / "phase3"))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.cvdc_batch import CVDCBatcher, positive_weights  # noqa: E402
from cvdc.data.dataset import PackData  # noqa: E402
from cvdc.eval.consistency import pooled_vci  # noqa: E402
from cvdc.train.trainer import fold_plan  # noqa: E402
from report_phase3a import fci, fmt  # noqa: E402

STRATA = [("E01 expert", "E01", "expert"), ("E02 expert", "E02", "expert"), ("E01 non-expert", "E01", "non-expert"),
          ("pooled", None, None)]
LABEL = {"sync": "(a) sync only", "sync_audio": "(b) + audio", "sync_audio_rep": "(c) + rep-matched",
         "all": "(d) + synthetic (all)", "all_no_rep": "(d′) all − rep-matched", "synthetic": "(e) synthetic only"}
STEPS = [("sync_audio", "sync", "+ audio"), ("sync_audio_rep", "sync_audio", "+ rep-matched"),
         ("all", "sync_audio_rep", "+ synthetic"), ("all", "all_no_rep", "rep-matched, given synthetic"),
         ("synthetic", "all", "synthetic only − all"), ("all_no_rep", "sync_audio", "+ synthetic (without rep)")]


def run_dir(cfg, setting, fold, smoke):
    ab = cfg["ablation"]
    if setting in ab.get("reuse", {}) and not smoke:
        return resolve(cfg["train"]["runs_dir"]) / ab["reuse"][setting] / "proposed" / f"fold{fold}"
    return resolve(cfg["train"]["runs_dir"]) / (ab["runs_tag"] + ("_smoke" if smoke else "")) / setting / f"fold{fold}"


def check_reuse(cfg, d, sources):
    """The reused step (a) pretraining must have used the setting's sources and today's CVDC settings."""
    import yaml

    c = yaml.safe_load((d / "cvdc" / "config.yaml").read_text(encoding="utf-8"))
    if sorted(c["cvdc"]["sources"]) != sorted(sources):
        raise RuntimeError(f"{d}: sources {c['cvdc']['sources']} vs {sources}")
    for k in ("loss", "sampler", "synthetic", "augment", "pairs"):
        if c["cvdc"][k] != cfg["cvdc"][k]:
            raise RuntimeError(f"{d}: cvdc.{k} differs from the current config")
    if c["model"] != {**cfg["model"], "input": c["model"]["input"]} or c["sync"]["pair_weights"] != cfg["sync"]["pair_weights"]:
        raise RuntimeError(f"{d}: model or pair weights differ from the current config")


def epoch_pairs(b, members, pair_src, seed=0):
    """Positive pairs (unordered, weight > 0) in one epoch of batcher ``b``, per stratum and kind (real source or
    'synthetic'). Within-set weights do not depend on the other sets of a batch, so sets are counted one by one with
    the batcher's own component and synthetic-source draws."""
    rng = np.random.default_rng(seed)
    strat = (members.exercise + " " + members.group).to_numpy()
    out = Counter()
    for comps, _, sid in b.sets:
        comp = comps[rng.integers(len(comps))] if len(comps) > 1 else comps[0]
        src = list(comp) + [comp[rng.integers(len(comp))] for _ in range(b.n_syn)]
        W = positive_weights(src, [0] * len(src), len(comp), b.edge_w, b.w["synthetic"])
        st = strat[comp[0]]
        for i in range(len(src)):
            for j in range(i + 1, len(src)):
                if W[i, j] <= 0:
                    continue
                kind = "synthetic" if i >= len(comp) or j >= len(comp) else pair_src[(src[i], src[j])]
                out[(st, kind)] += 1
        out[(st, "sets")] += 1
        out[(st, "real_items")] += len(comp)
    return out


def pair_count_table(cfg, data, plans, settings, members):
    pairs = pd.read_csv(resolve(cfg["cvdc"]["pairs"]["pairs_csv"]))
    pair_src = {}
    for a, c, s in zip(pairs.item_a, pairs.item_b, pairs.pair_source):
        pair_src[(a, c)] = pair_src[(c, a)] = s
    rows = []
    for s in settings:
        c = {**cfg, "cvdc": {**cfg["cvdc"], "sources": cfg["ablation"]["sources"][s]}}
        scopes = [("all subjects", None)] + [(f"fold {p['fold']} inner train", p["inner_train"]) for p in plans]
        for scope, subs in scopes:
            allowed = None if subs is None else data.subject_mask(subs)
            b = CVDCBatcher(data, c, allowed=allowed, seed=0)
            for (st, kind), n in epoch_pairs(b, members, pair_src).items():
                rows.append(dict(setting=s, scope=scope, stratum=st, kind=kind, n=n))
    T = pd.DataFrame(rows)
    T = T.pivot_table(index=["setting", "scope", "stratum"], columns="kind", values="n", aggfunc="sum",
                      fill_value=0).reset_index()
    for k in ("sync", "audio", "rep", "synthetic", "sets", "real_items"):
        if k not in T:
            T[k] = 0
    T["positive_pairs"] = T[["sync", "audio", "rep", "synthetic"]].sum(axis=1)
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--boot", type=int)
    ap.add_argument("--smoke", action="store_true", help="the smoke runs (phase5_sources_smoke): plumbing only")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    ec, ab = cfg["eval"], cfg["ablation"]
    n_boot, ci, seed = a.boot or ec["bootstrap"]["n"], ec["bootstrap"]["ci"], ec["bootstrap"]["seed"]
    out = resolve(ec["results_dir"]) / (ab["runs_tag"] + ("_smoke" if a.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    settings = list(ab["sources"])
    data = PackData(cfg, "rep", "cpu")
    plans = fold_plan(cfg, data.meta, smoke=a.smoke)
    members = pd.read_csv(resolve(cfg["cvdc"]["pairs"]["members_csv"]))
    gcode = pd.factorize(members.group_id)[0]

    snaps, runs = [], []
    for s in settings:
        for p in plans:
            d = run_dir(cfg, s, p["fold"], a.smoke)
            if not (d / "embeddings_pretrained.npz").exists():
                if a.smoke and s in ab.get("reuse", {}):
                    continue
                raise RuntimeError(f"{d}: no pretrained embeddings")
            if s in ab.get("reuse", {}) and not a.smoke:
                check_reuse(cfg, d, ab["sources"][s])
            info = json.loads((d / "cvdc" / "run_info.json").read_text(encoding="utf-8"))
            for k in ("inner_train", "inner_val", "test"):
                assert sorted(info[k]) == sorted(p[k]), (s, p["fold"], k)
            sm = json.loads((d / "cvdc" / "summary.json").read_text(encoding="utf-8"))
            runs.append(dict(setting=s, fold=p["fold"], best_epoch=sm["best_epoch"], stopped_epoch=sm["stopped_epoch"],
                             best_val=sm["best_val"], wall_s=sm["wall_s"], train_sets=info["batcher_train"]["sets"],
                             **{f"train_{k}_pairs": v for k, v in info["batcher_train"]["real_pairs_by_source"].items()}))
            snaps.append(((s, "pretrained"), p["fold"], d / "embeddings_pretrained.npz"))
            if s == settings[0]:
                snaps.append((("init", "init"), p["fold"], d / "embeddings_init.npz"))
    R = pd.DataFrame(runs)
    R.to_csv(out / "pretraining_runs.csv", index=False)
    # the random-init encoder is the same for every setting (seed train.seed + fold): check one fold
    if not a.smoke:
        for s in settings[1:]:
            with np.load(run_dir(cfg, settings[0], 0, False) / "embeddings_init.npz") as x, \
                    np.load(run_dir(cfg, s, 0, False) / "embeddings_init.npz") as y:
                assert np.abs(x["z"] - y["z"]).max() < 0.05 * max(1.0, np.abs(x["z"]).max()), s

    rows, fv = pooled_vci(snaps, members, gcode, STRATA, n_boot, seed, ci)
    V = pd.DataFrame([{**{k: r[k] for k in r if k not in ("key", "boot", "present")}, "setting": r["key"][0]}
                      for r in rows])
    V.to_csv(out / "vci.csv", index=False)
    fv.assign(setting=[k[0] for k in fv.key]).drop(columns="key").to_csv(out / "vci_per_fold.csv", index=False)
    boot = {(r["key"][0], r["stratum"], r["variant"]): (r["boot"], r["present"], r["vci"]) for r in rows}
    deltas = []
    for sname, _, _ in STRATA:
        for var in ("centred", "raw", "proj"):
            comps = STEPS + [(s, "init", "pretraining gain") for s in settings]
            for x, y, what in comps:
                bx, by = boot.get((x, sname, var)), boot.get((y, sname, var))
                if bx is None or by is None or bx[1] != by[1]:
                    continue
                lo, hi = (np.nan, np.nan) if a.smoke else np.quantile(bx[0] - by[0], [(1 - ci) / 2, (1 + ci) / 2])
                deltas.append(dict(stratum=sname, variant=var, setting=x, vs=y, comparison=what, delta=bx[2] - by[2],
                                   delta_lo=lo, delta_hi=hi))
    D = pd.DataFrame(deltas)
    D.to_csv(out / "vci_deltas.csv", index=False)

    C = pair_count_table(cfg, data, plans, settings, members)
    C.to_csv(out / "pair_counts.csv", index=False)
    write_summary(out, cfg, settings, V, D, C, R, n_boot, a.smoke)
    print(f"report written to {out} in {time.time() - t0:.0f} s")


def write_summary(out, cfg, settings, V, D, C, R, n_boot, smoke):
    L = []
    w = L.append
    strata = [s for s, _, _ in STRATA]
    w("# Phase 5 B: CVDC pair-source ablation, held-out within-group VCI (GroupKFold-5)" +
      (" (SMOKE RUN: plumbing check, not a result)" if smoke else ""))
    w("")
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/pilot/phase5/report_pair_sources.py`. CVDC pretraining only "
      "(no fine-tuning), same folds, inner validation splits, seeds, budget (≤ 100 epochs), early stopping (T8a), "
      "schedule, augmentations and synthetic-view settings as the step (a) pretraining; only `cvdc.sources` changes. "
      "(d) is the step (a) pretraining itself (reused after checking its configuration). VCI = centred 256-d (E13) on "
      "the held-out test subjects, same-rep view pairs of every source (the evaluation does not depend on the training "
      f"sources), pooled over the 5 folds; 95% CIs from the subject bootstrap ({n_boot} draws, the same for every "
      "setting, so differences are paired). The pooled row mixes exercises and is inflated by exercise separation "
      "(E13); it is secondary.")
    w("")
    w("## Pair counts per source (one training epoch, all subjects)")
    w("")
    w("Positive pairs (unordered, weight > 0, A7) in one epoch: real pairs by source (weights sync 1.0, audio 0.8, rep "
      "0.5) and pairs with a synthetic view (2 per rep set per epoch, drawn at random). Rep sets = sets with ≥ 2 items "
      "(real or synthetic) that the sampler can use. Per-fold training counts (inner training subjects only) are in "
      "`pair_counts.csv`.")
    w("")
    w("| Setting | stratum | rep sets | real items | sync | audio | rep | synthetic | all positives |")
    w("|---|---|---|---|---|---|---|---|---|")
    A = C[C.scope == "all subjects"]
    for s in settings:
        for st in ["E01 expert", "E02 expert", "E01 non-expert"]:
            r = A[(A.setting == s) & (A.stratum == st)]
            if r.empty:
                w(f"| {LABEL[s]} | {st} | 0 | 0 | 0 | 0 | 0 | 0 | 0 |")
                continue
            r = r.iloc[0]
            w(f"| {LABEL[s]} | {st} | {int(r.sets)} | {int(r.real_items)} | {int(r.sync)} | {int(r.audio)} | "
              f"{int(r.rep)} | {int(r.synthetic)} | {int(r.positive_pairs)} |")
    F = C[C.scope != "all subjects"].groupby("setting")[["sets", "sync", "audio", "rep", "synthetic"]].sum() / \
        max(1, C[C.scope != "all subjects"].scope.nunique())
    w("")
    w("Mean per fold, inner training subjects (all strata): " + "; ".join(
        f"{LABEL[s]} {F.loc[s, 'sets']:.0f} sets, {F.loc[s, 'sync']:.0f} / {F.loc[s, 'audio']:.0f} / "
        f"{F.loc[s, 'rep']:.0f} / {F.loc[s, 'synthetic']:.0f} (sync / audio / rep / synthetic)" for s in settings
        if s in F.index) + ".")
    w("")
    w("## VCI after CVDC pretraining (centred 256-d), held-out subjects")
    w("")
    w("VCI [95% CI] (same-rep view pairs).")
    w("")
    w("| Setting | " + " | ".join(strata) + " |")
    w("|---|" + "---|" * len(strata))
    for s in ["init"] + settings:
        r = V[(V.setting == s) & (V.variant == "centred")]
        if r.empty:
            continue
        cells = []
        for st in strata:
            x = r[r.stratum == st]
            cells.append("–" if x.empty else f"{fci(x.vci.iloc[0], x.vci_lo.iloc[0], x.vci_hi.iloc[0], 3)} ({x.same_pairs.iloc[0]})")
        w(f"| {'random init' if s == 'init' else LABEL[s]} | " + " | ".join(cells) + " |")
    w("")
    w("Paired differences (centred VCI):")
    w("")
    w("| Comparison | " + " | ".join(strata) + " |")
    w("|---|" + "---|" * len(strata))
    for x, y, what in STEPS:
        cells = []
        for st in strata:
            r = D[(D.setting == x) & (D.vs == y) & (D.stratum == st) & (D.variant == "centred")]
            cells.append("–" if r.empty else fci(r.delta.iloc[0], r.delta_lo.iloc[0], r.delta_hi.iloc[0], 3))
        w(f"| {LABEL[x]} vs {LABEL[y]} ({what}) | " + " | ".join(cells) + " |")
    w("")
    w("Raw and projection variants, per-fold values and the gains over the random-init encoder: `vci.csv`, "
      "`vci_per_fold.csv`, `vci_deltas.csv`.")
    w("")
    w("## Pretraining runs")
    w("")
    w("| Setting | kept epochs (folds) | stopped | wall time (5 folds) | training sets per fold |")
    w("|---|---|---|---|---|")
    for s in settings:
        r = R[R.setting == s]
        if r.empty:
            continue
        w(f"| {LABEL[s]} | {', '.join(str(int(x)) for x in r.best_epoch)} | {', '.join(str(int(x)) for x in r.stopped_epoch)} "
          f"| {r.wall_s.sum() / 60:.0f} min | {r.train_sets.min()}–{r.train_sets.max()} |")
    w("")
    w("The validation loss differs between settings (different positives), so it is not compared across settings.")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
