"""How many reps each view's recording of one performance actually contains (Phase 4, recording-noise test).

The phones of a synced group start and stop at different times (P6b), so an edge rep can be cut off in one view. The
reps come from the best view's own segmentation and are placed on the group's aligned clock
(aligned time = t - offset_view, outputs/sync_report.csv).
"""


def counts_per_view(own_best, off_best, views, fps, tol_s):
    """{view: (contained, half)} for the best view's reps on the aligned clock; views: {view: (offset_s, n_frames)}."""
    out = {}
    for v, (off_v, n_v) in views.items():
        lo, hi = -off_v, (n_v - 1) / fps - off_v  # the view's recording on the aligned clock
        full = half = 0
        for r in own_best.itertuples():
            a0, a1 = r.start / fps - off_best, r.end / fps - off_best
            if a0 >= lo - tol_s and a1 <= hi + tol_s:
                full += 1
            if max(0.0, min(a1, hi) - max(a0, lo)) >= 0.5 * (a1 - a0):
                half += 1
        out[v] = (full, half)
    return out
