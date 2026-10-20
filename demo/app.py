"""Phase 6 Streamlit app (spec §10): upload a video (or pick a dataset video by its fields) -> the per-rep report
(Module III score, quality, errors with joint / frames / deviation; the selected exercise's model score and distance to
the PT reference, T17), angle-vs-template plots, the video's position in the embedding space and the JSON download.

Only keypoint-derived numbers and plots are shown; no frame is displayed or stored (D10). An uploaded file is written
to a temporary file for decoding and deleted right after.

usage: streamlit run demo/app.py
"""
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve, video_root  # noqa: E402
from cvdc.data.video import iter_frames  # noqa: E402
from demo.pipeline import ExerciseModel, OnlineSession, PoseStream, dataset_subject  # noqa: E402

CONFIG = REPO / "configs" / "per_exercise.yaml"


@st.cache_resource
def config():
    return load_config(CONFIG)


def run(path, cfg, ex, view, method, subject, progress):
    model, note = None, ""
    if method != "none (Module III only)":
        try:
            model = ExerciseModel(cfg, ex, method, None, subject, "cpu")
            note = model.warning or ""
        except FileNotFoundError as e:
            note = f"no checkpoint: {e}"
    pose = PoseStream(cfg, hands=ex in (cfg["pose"].get("hands_for") or []))
    sess = OnlineSession(cfg, ex, view, model)
    try:
        frames = list(iter_frames(path, cfg["pose"]["max_side"], False))
        for i, (t, rgb, _) in enumerate(frames):
            pose.process(t, rgb)
            if i % 15 == 0:
                progress.progress(min(1.0, (i + 1) / len(frames)), text=f"pose {i + 1}/{len(frames)} frames")
        progress.progress(1.0, text="assessing")
        sess.update(pose, final=True)
    finally:
        pose.close()
    return sess, model, note


def template_plot(sess, k, view):
    d = sess.debug[k]
    chans = sess.ref["channels"]
    fig, axes = plt.subplots(1, len(chans), figsize=(3.2 * len(chans), 2.6), squeeze=False)
    band = sess.ac["band_k"]
    for ax, c in zip(axes[0], chans):
        tgt = d["cmp"]["target"][c]
        sd = sess.lib[view]["sigma"][c] * band
        x = np.linspace(0, 100, len(tgt))
        ax.fill_between(x, tgt - sd, tgt + sd, color="0.85", label=f"band ±{band:g} SD")
        ax.plot(x, tgt, "k-", lw=1.5, label="target")
        ax.plot(x, d["aligned"]["curves"][c], "C1-", lw=1.5, label="rep (DTW-aligned)")
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("template time (%)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("deg", fontsize=8)
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    return fig


def embedding_plot(cfg, model, sess):
    """PCA of the fold's held-out embeddings (centred z, coloured by the video score) with this video's reps and the
    PT reference."""
    run = resolve(cfg["train"]["runs_dir"]) / "perex_gkf5" / model.ex / model.method / f"fold{model.fold}"
    with np.load(run / "embeddings_final.npz") as z:
        Z, items = z["z"].astype(np.float64), z["item"]
    reps = pd.read_csv(resolve(cfg["data"]["reps_csv"]))
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"])).set_index("video_id")
    score = idx.score.reindex(reps.video_id.iloc[items]).to_numpy()
    c = model.centre
    A = Z - c
    _, _, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
    P = lambda X: (np.atleast_2d(X) - c - A.mean(0)) @ Vt[:2].T  # noqa: E731
    fig, ax = plt.subplots(figsize=(5, 4))
    p = P(Z)
    sc = ax.scatter(p[:, 0], p[:, 1], c=score, cmap="viridis", s=6, alpha=0.5)
    fig.colorbar(sc, ax=ax, label="physiotherapist score (held-out reps)")
    if sess.zs:
        q = P(np.stack(sess.zs))
        ax.scatter(q[:, 0], q[:, 1], c="C3", s=40, marker="x", label="this video's reps")
    r = P(model.ref[sess.side or "right"])
    ax.scatter(r[:, 0], r[:, 1], c="k", s=120, marker="*", label="PT reference")
    ax.legend(fontsize=7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{model.ex} {model.method} fold {model.fold}", fontsize=9)
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="CVDC assessment", layout="wide")
    cfg = config()
    st.title("CVDC physiotherapy assessment (Module III + per-exercise model)")
    st.caption("Keypoints only: no video frame is shown or stored. Raw Module III output, no coaching text.")
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    with st.sidebar:
        mode = st.radio("video", ["upload", "dataset video"])
        if mode == "dataset video":
            ex = st.selectbox("exercise", sorted(idx.exercise.unique()))
            d = idx[idx.exercise == ex]
            subj = st.selectbox("subject", sorted(d.subject.unique()))
            d = d[d.subject == subj]
            view = st.selectbox("view", sorted(d.view.unique()))
            var = st.selectbox("variation", sorted(d[d.view == view].variation.unique()))
            row = d[(d.view == view) & (d.variation == var)].iloc[0]
            st.write(f"`{row.video_id}`: score {row.score}")
            up = None
        else:
            ex = st.selectbox("exercise (one model per exercise)", sorted(idx.exercise.unique()))
            view = st.selectbox("camera view", ["F", "L", "R"])
            up = st.file_uploader("video", type=["mp4", "mov", "avi", "mkv"])
            row = None
        method = st.selectbox("model", [cfg["eval"]["proposed_default"], "b1", "proposed_lp", "none (Module III only)"])
        go = st.button("assess", type="primary")
    if go:
        progress = st.progress(0.0, text="starting")
        if row is not None:
            res = run(video_root(cfg) / row.rel_path, cfg, ex, view, method, row.subject, progress)
            name = row.video_id
        elif up is not None:
            with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as f:
                f.write(up.getbuffer())
                tmp = Path(f.name)
            try:
                res = run(tmp, cfg, ex, view, method, dataset_subject(cfg, Path(up.name).stem), progress)
            finally:
                tmp.unlink(missing_ok=True)  # the upload is not kept
            name = Path(up.name).stem
        else:
            st.warning("upload a video first")
            return
        progress.empty()
        st.session_state["result"] = (*res, name, view)  # widgets rerun the script; keep the result
    if "result" not in st.session_state:
        st.info("Choose a video and press **assess**.")
        return
    sess, model, note, name, view = st.session_state["result"]
    rep = sess.set_report(name)
    if note:
        st.warning(note)
    c1, c2, c3 = st.columns(3)
    c1.metric("reps", rep["n_reps"])
    c2.metric("Module III set score", rep["set_score"], rep["set_quality"])
    if rep.get("model", {}).get("set_score") is not None:
        c3.metric(f"model set score ({model.method}, fold {model.fold})", round(rep["model"]["set_score"], 1),
                  f"MQ {rep['model']['movement_quality']:.1f}" if rep["model"].get("movement_quality") else None)
    if rep["set_errors"]:
        st.write("set errors:", rep["set_errors"])
    rows = [dict(rep=r["rep"] + 1, start_s=r["time_s"][0], end_s=r["time_s"][1], score=r["score"],
                 quality=r["quality"], model_score=r.get("model_score"),
                 distance=(r.get("embedding_distance") or {}).get("cosine_centred"),
                 errors=", ".join(f"{e['type']} ({e.get('joint', '')}, {e['severity']})" for e in r["errors"]))
            for r in rep["reps"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    if rep["reps"]:
        k = st.slider("rep", 1, len(rep["reps"]), 1) - 1
        st.json(rep["reps"][k]["errors"], expanded=False)
        st.pyplot(template_plot(sess, k, view))
    if model is not None:
        st.pyplot(embedding_plot(cfg, model, sess))
    st.download_button("download JSON", json.dumps(rep, indent=1, default=float), file_name=f"{name}_report.json",
                       mime="application/json")


main()
