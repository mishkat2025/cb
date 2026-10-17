"""Subject-independent splits (E1): every rep, view and video of a subject falls in the same fold.

  group_kfold  k folds over subjects, stratified by expert / non-expert so each fold holds both groups in
               proportion (development and ablations; also the CVDC pretraining folds of the LOSO shortcut, C7)
  loso         leave-one-subject-out (final B1 / B2 / Proposed table)
Each split is a list of (train_subjects, test_subjects) of sorted subject keys.
"""
import numpy as np
import pandas as pd


def subject_table(meta):
    """One row per subject with its group (expert / non-expert)."""
    return meta.drop_duplicates("subject")[["subject", "group"]].sort_values("subject").reset_index(drop=True)


def group_kfold(meta, k=5, seed=0):
    """Stratified subject folds: subjects of each group are shuffled and dealt round-robin over the k folds."""
    subj = subject_table(meta)
    rng = np.random.default_rng(seed)
    fold = {}
    offset = 0
    for g, d in subj.groupby("group"):
        s = list(rng.permutation(d.subject.to_numpy()))
        for i, sid in enumerate(s):
            fold[sid] = (i + offset) % k
        offset += len(s)  # continue the deal so small groups do not all start at fold 0
    allsub = set(subj.subject)
    return [(sorted(allsub - {s for s, f in fold.items() if f == i}), sorted(s for s, f in fold.items() if f == i))
            for i in range(k)]


def loso(meta):
    subs = list(subject_table(meta).subject)
    return [(sorted(set(subs) - {s}), [s]) for s in subs]


def fold_of(splits):
    """subject -> index of the split in which it is held out (for the LOSO per-fold-encoder shortcut, C7)."""
    return {s: i for i, (_, test) in enumerate(splits) for s in test}


def check_no_leakage(splits, all_subjects):
    """Raise if a subject is in both train and test of a split, or if the test sets do not partition the subjects."""
    seen = []
    for tr, te in splits:
        if set(tr) & set(te):
            raise AssertionError(f"subjects in train and test: {sorted(set(tr) & set(te))}")
        if set(tr) | set(te) != set(all_subjects):
            raise AssertionError("a split does not cover every subject")
        seen += te
    if sorted(seen) != sorted(all_subjects):
        raise AssertionError("test sets do not partition the subjects")
    return True


def items_mask(meta, subjects):
    return meta.subject.isin(set(subjects)).to_numpy()


def describe(splits, meta):
    rows = []
    for i, (tr, te) in enumerate(splits):
        m = meta[meta.subject.isin(te)]
        rows.append(dict(fold=i, train_subjects=len(tr), test_subjects=len(te),
                         test_experts=int((m.drop_duplicates("subject").group == "expert").sum()),
                         train_items=int(items_mask(meta, tr).sum()), test_items=len(m)))
    return pd.DataFrame(rows)
