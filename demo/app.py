"""Phase 6 Streamlit app (spec §10; X5, X9, X10): pick a dataset video from a readable, filterable table (or upload
one) -> the four-panel video (X9: patient | PT reference / estimated 3D | CVDC embedding; faces blurred, D10), saved to
<outputs>/demo/app/ and played in the page, and the per-rep report in plain language: the assessment, which is always
Module III's score and verdict (X11, X12: the proposal's pipeline only: Module III aligns and compares in the Proposed
CVDC model's embedding space, M19 + M20; the rule-based Module III only as a fallback when the checkpoint is missing;
no model menu), every problem as a sentence (joint, time, measured / target, severity), the
measurements and how the score was made, the patient-vs-PT and angle-vs-template plots, and, in a separate section
labelled supplementary, the network's own learned score, EAAQ items, distance to the PT reference and embedding plot
(the instrument used to compare the models, not the pipeline's output), the raw JSON and a glossary.

Video frames appear only in the four-panel export, which blurs faces (D10, X9). An uploaded file is written to a
temporary folder for decoding and the export, and deleted right after.

usage: streamlit run demo/app.py
"""
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
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
from demo import readable as R  # noqa: E402
from demo.pipeline import ExerciseModel, OnlineSession, PoseStream, dataset_subject  # noqa: E402

CONFIG = REPO / "configs" / "per_exercise.yaml"
TABLE_COLS = ["Participant", "Group", "Gender", "Camera", "Condition", "PT score", "PT verdict", "File"]
VERDICT_COLOR = {"correct": "green", "partially_correct": "orange", "incorrect": "red"}
SEVERITY_COLOR = {"mild": "yellow", "moderate": "orange", "severe": "red"}


@st.cache_resource
def config():
    return load_config(CONFIG)


@st.cache_data
def video_table(index_csv):
    """The dataset index with readable columns, sorted by participant, camera and condition."""
    idx = pd.read_csv(index_csv)
    t = pd.DataFrame({
        "video_id": idx.video_id, "exercise": idx.exercise, "subject": idx.subject, "view": idx.view,
        "variation": idx.variation, "quality": idx.quality, "rel_path": idx.rel_path,
        "Participant": idx.subject, "Group": idx.group.map(R.GROUPS).fillna(idx.group),
        "Gender": idx.gender.map(R.GENDERS).fillna(idx.gender), "Camera": idx.view.map(R.CAMERAS).fillna(idx.view),
        "Condition": idx.variation.map(R.CONDITIONS).fillna(idx.variation), "PT score": idx.score,
        "PT verdict": idx.quality.map(R.QUALITY).fillna(idx.quality), "File": idx.file})
    order = {k: i for i, k in enumerate([*R.CAMERAS, *R.CONDITIONS])}
    return t.sort_values(["subject", "view", "variation"], key=lambda s: s if s.name == "subject"
                         else s.map(order).fillna(99)).reset_index(drop=True)


def verdict(q):
    return R.label(R.QUALITY, q)


def plain(x):
    """JSON-safe copy (numpy numbers -> float) for st.json."""
    return json.loads(json.dumps(x, default=float))


UNITS = {"deg": "°", "deg/s": " °/s", "x": "×", "s": " s", "": ""}


def num(v, digits=3):
    """A number as short text for st.table (which would otherwise pad floats with zeros)."""
    return "–" if v is None else f"{float(v):.{digits}g}" if abs(float(v)) < 1 else f"{float(v):.1f}"


def table(rows, alt):
    """A small static table that wraps long text (st.table)."""
    st.table(pd.DataFrame(rows), hide_index=True, alt=alt)


def pyplot(fig, alt):
    st.pyplot(fig, alt=alt)
    plt.close(fig)


# ------------------------------------------------------------------ assessment and the four-panel export


def load_model(cfg, ex, subject):
    """The proposal's pipeline only (X12): the Proposed CVDC model of the fold that held the subject out. Without a
    trained checkpoint the app falls back to the rule-based Module III and says so (not offered as a choice)."""
    try:
        model = ExerciseModel(cfg, ex, cfg["eval"]["proposed_default"], None, subject, "cpu")
        return model, model.warning or ""
    except FileNotFoundError as e:
        return None, (f"The trained CVDC model is missing, so this run uses the rule-based Module III only, without "
                      f"the network ({e}).")


def assess(path, cfg, ex, view, model, bar):
    """Pose on every frame, then Module III (+ the model) on the whole set: the batch result of X5."""
    pose = PoseStream(cfg, hands=ex in (cfg["pose"].get("hands_for") or []))
    sess = OnlineSession(cfg, ex, view, model)
    try:
        frames = list(iter_frames(path, cfg["pose"]["max_side"], False))
        for i, (t, rgb, _) in enumerate(frames):
            pose.process(t, rgb)
            if i % 15 == 0:
                bar.progress(min(1.0, (i + 1) / len(frames)), text=f"Finding the body in frame {i + 1} of {len(frames)}")
        bar.progress(1.0, text="Scoring the repetitions")
        sess.update(pose, final=True)
    finally:
        pose.close()
    return sess


def start_export(src, ex, view, model, ghost, out):
    """demo/live.py --export4 in a separate process (X9), written to a .part file and renamed when it succeeds."""
    out.parent.mkdir(parents=True, exist_ok=True)
    part, log = out.with_name(out.stem + ".part.mp4"), out.with_suffix(".log")
    cmd = [sys.executable, str(REPO / "demo" / "live.py"), "--config", str(CONFIG), "--source", str(src),
           "--exercise", ex, "--view", view, "--export4", str(part)]
    cmd += ["--method", model.method, "--fold", str(model.fold)] if model is not None else ["--no-model"]
    if ghost:
        cmd.append("--ghost")
    f = open(log, "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT,
                         env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
    return {"proc": p, "log_file": f, "part": part, "log": log, "out": out, "t0": time.time()}


def finish_export(job, status):
    p = job["proc"]
    while p.poll() is None:
        text = job["log"].read_text(encoding="utf-8", errors="replace")
        n = max([int(x) for x in re.findall(r"^rep (\d+)", text, re.M)] or [0])
        status.update(label=f"Making the four-panel video: {time.time() - job['t0']:.0f} s so far, "
                            f"{n} repetitions drawn")
        time.sleep(1.0)
    job["log_file"].close()
    if p.returncode == 0 and job["part"].exists():
        job["part"].replace(job["out"])
        return job["out"], None
    job["part"].unlink(missing_ok=True)
    tail = job["log"].read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
    return None, "\n".join(tail)


EXPORT_VERSION = "x11"  # in the file name, so a video made before the X11 / X12 assessment is never reused


def video_name(name, model, ghost):
    return (f"{name}__{model.method if model is not None else 'module3'}{'__ghost' if ghost else ''}"
            f"_4panel_{EXPORT_VERSION}.mp4")


# ------------------------------------------------------------------ plots


def template_plot(sess, k, view):
    d = sess.debug[k]
    chans = sess.ref["channels"]
    fig, axes = plt.subplots(1, len(chans), figsize=(3.4 * len(chans), 2.8), squeeze=False)
    band = sess.ac["band_k"]
    for ax, c in zip(axes[0], chans):
        tgt = d["cmp"]["target"][c]
        sd = sess.lib[view]["sigma"][c] * band
        x = np.linspace(0, 100, len(tgt))
        ax.fill_between(x, tgt - sd, tgt + sd, color="0.85", label=f"allowed range (±{band:g} expert SD)")
        ax.plot(x, tgt, "k-", lw=1.5, label="target")
        ax.plot(x, d["aligned"]["curves"][c], "C1-", lw=1.5, label="this repetition")
        ax.set_title(R.label(R.CHANNELS, c), fontsize=9)
        ax.set_xlabel("movement progress (%)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("angle (degrees)", fontsize=8)
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
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    p = P(Z)
    sc = ax.scatter(p[:, 0], p[:, 1], c=score, cmap="viridis", s=6, alpha=0.5)
    fig.colorbar(sc, ax=ax, label="physiotherapists' score (held-out repetitions)")
    if sess.zs:
        q = P(np.stack(sess.zs))
        ax.scatter(q[:, 0], q[:, 1], c="C3", s=40, marker="x", label="this video's repetitions")
        for i, (x, y) in enumerate(q):
            ax.annotate(str(i + 1), (x, y), fontsize=7, color="C3", xytext=(3, 3), textcoords="offset points")
    r = P(model.ref[sess.side or "right"])
    ax.scatter(r[:, 0], r[:, 1], c="k", s=120, marker="*", label="physiotherapist's demo")
    ax.legend(fontsize=7)
    ax.set_xlabel("main direction 1")
    ax.set_ylabel("main direction 2")
    ax.set_title(f"{R.EXERCISES.get(model.ex, model.ex)} · {R.MODELS.get(model.method, (model.method,))[0]} · "
                 f"fold {model.fold}", fontsize=9)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ picking a video


def pick_dataset_video(cfg):
    t = video_table(str(resolve(cfg["data"]["index_csv"])))
    ex = st.selectbox("Exercise", sorted(t.exercise.unique()), format_func=R.exercise)
    d = t[t.exercise == ex]
    with st.container(horizontal=True, gap="large"):
        cams = st.pills("Camera", [c for c in R.CAMERAS.values() if c in set(d.Camera)], selection_mode="multi",
                        key=f"cam_{ex}")
        groups = st.pills("Participant group", [g for g in R.GROUPS.values() if g in set(d.Group)],
                          selection_mode="multi", key=f"grp_{ex}")
    conds = st.pills("Recording condition", [c for c in R.CONDITIONS.values() if c in set(d.Condition)],
                     selection_mode="multi", key=f"cond_{ex}")
    subs = st.multiselect("Participants", sorted(d.Participant.unique()), placeholder="All participants",
                          key=f"sub_{ex}")
    for col, sel in (("Camera", cams), ("Group", groups), ("Condition", conds), ("Participant", subs)):
        if sel:
            d = d[d[col].isin(sel)]
    d = d.reset_index(drop=True)
    st.caption(f"{len(d)} videos. A filter with nothing selected shows everything. Tick a row to choose that video.")
    key = "table_" + hashlib.md5(repr((ex, cams, groups, conds, subs)).encode()).hexdigest()[:10]
    ev = st.dataframe(
        d, hide_index=True, on_select="rerun", selection_mode="single-row", key=key, height=330,
        column_order=TABLE_COLS, alt=f"Dataset videos of {R.exercise(ex)}",
        column_config={
            "Participant": st.column_config.TextColumn(help="Participant ID in the dataset"),
            "Group": st.column_config.TextColumn(help="Expert or non-expert participant"),
            "Camera": st.column_config.TextColumn(help="Which phone recorded the video"),
            "Condition": st.column_config.TextColumn(help="Lighting, camera shake, hidden body parts or low "
                                                          "resolution during recording"),
            "PT score": st.column_config.ProgressColumn("PT score", min_value=0, max_value=100, format="%.1f",
                                                        help="Average score of three physiotherapists (0 to 100)"),
            "PT verdict": st.column_config.TextColumn(help=R.QUALITY_RULE),
            "File": st.column_config.TextColumn(help="Original file name"),
        })
    rows = ev.selection.rows
    return ex, (d.iloc[rows[0]] if rows else None)


def describe_row(ex, row):
    return (f"**{R.exercise(ex)}** · participant **{row.Participant}** ({row.Group.lower()}, {row.Gender.lower()})"
            f" · **{row.Camera}** camera · {row.Condition}")


# ------------------------------------------------------------------ results


def points_lost(r, ac):
    """Points lost per score part (M11): 100 * w_g * min(1, X_g / d_zero) / sum of w over the defined parts."""
    X = r["measures"].get("distance_sd") or {}
    sc = ac["score"]
    w = {g: sc["weights"].get(g, 1.0) for g, x in X.items() if x is not None and np.isfinite(x)}
    tot = sum(w.values()) or 1.0
    return [{"Part of the score": R.label(R.SCORE_PARTS, g), "Distance from the experts (SD)": round(X[g], 2),
             "Points lost (of 100)": round(100 * w[g] * min(1.0, X[g] / sc["d_zero"]) / tot, 1)} for g in w]


def problem_card(e, fps):
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(R.label(R.SEVERITY, e.get("severity")), color=SEVERITY_COLOR.get(e.get("severity"), "gray"))
            st.markdown(f"**{R.error_title(e)}**")
        st.markdown(R.error_sentence(e, fps))
        if e.get("excess_sd") is not None:
            st.caption(f"{e['excess_sd']:.2f} expert SD beyond the allowed range · rule `{e['type']}`")


def show_summary(res, rep, model, fps):
    ex, row = res["ex"], res["row"]
    with st.container(horizontal=True):
        exp = rep.get("expected_reps")
        st.metric("Repetitions found" + (f" (asks for {exp})" if exp else ""), f"{rep['n_reps']}", border=True,
                  help="Repetitions the app found in the video" + (f"; the exercise asks for {exp}." if exp else "."))
        mode = res["sess"].mode
        st.metric("Assessment score", f"{rep['set_score']:.1f}", border=True,
                  help=f"{R.ASSESSMENT[mode]}: each repetition compared with the physiotherapist's demo, then "
                       "averaged. 0 to 100. This is the system's result.")
        if row is not None and pd.notna(row["PT score"]):
            st.metric("Physiotherapists' score", f"{row['PT score']:.1f}", border=True,
                      help="Average of three physiotherapists for this video. 0 to 100.")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown("Assessment verdict:")
        st.badge(verdict(rep["set_quality"]), color=VERDICT_COLOR.get(rep["set_quality"], "gray"))
        if row is not None and isinstance(row.quality, str):
            st.markdown("Physiotherapists' verdict:")
            st.badge(verdict(row.quality), color=VERDICT_COLOR.get(row.quality, "gray"))
    # the same in words
    s = [f"The app found **{rep['n_reps']} repetitions**"
         + (f" (the exercise asks for {rep['expected_reps']})" if rep.get("expected_reps") else "") + "."]
    how = ("lined each one up with the physiotherapist's demo in the CVDC model's camera-independent "
           f"representation (fold {model.fold}) and compared the joint angles, the angular speed and the movement "
           "pattern" if mode == "completed" else "compared each one's joint angles with the physiotherapist's demo")
    s.append(f"The assessment, {R.ASSESSMENT_SHORT[mode]}, {how}, and gave "
             f"**{rep['set_score']:.1f} / 100 ({verdict(rep['set_quality']).lower()})**.")
    if row is not None and pd.notna(row["PT score"]):
        s.append(f"The three physiotherapists gave this video **{row['PT score']:.1f} / 100 "
                 f"({verdict(row.quality).lower()})**.")
    st.markdown(" ".join(s))
    st.caption("In the project's tests this score agreed only weakly with the physiotherapists (Spearman about 0.15 "
               "to 0.4) and did not predict their score better than the average of the participant's group, so treat "
               "it as an example, not a validated clinical score.")
    if rep.get("model", {}).get("set_score") is not None:
        st.caption("The network's own learned score is in the **supplementary** section at the end of the page.")


def show_problem_overview(rep, fps):
    if rep["set_errors"]:
        st.markdown("**Problems with the whole set**")
        for e in rep["set_errors"]:
            problem_card(e, fps)
    counts = {}
    for r in rep["reps"]:
        for t in {R.error_title(e) for e in r["errors"]}:
            counts[t] = counts.get(t, 0) + 1
    st.markdown("**Problems in the repetitions**")
    if not counts:
        st.success("No problems found in any repetition: every checked angle, the range and the speed stayed within "
                   "the experts' range.")
    else:
        n = len(rep["reps"])
        st.markdown("\n".join(f"- **{t}** in {c} of {n} repetitions" for t, c in
                              sorted(counts.items(), key=lambda x: -x[1])))


def show_rep_table(rep, fps):
    rows = [{"Rep": r["rep"] + 1, "Time": f"{r['time_s'][0]:.1f}–{r['time_s'][1]:.1f} s",
             "Score": r["score"], "Verdict": verdict(r["quality"]), "Suppl. model score": r.get("model_score"),
             "Distance to PT": (r.get("embedding_distance") or {}).get("cosine_centred"),
             "Problems": "; ".join(f"{R.error_title(e)} ({R.label(R.SEVERITY, e['severity']).lower()})"
                                   for e in r["errors"]) or "none"}
            for r in rep["reps"]]
    df = pd.DataFrame(rows)
    if df["Suppl. model score"].isna().all():
        df = df.drop(columns=["Suppl. model score", "Distance to PT"])
    st.dataframe(df, hide_index=True, alt="Per-repetition results", column_config={
        "Rep": st.column_config.NumberColumn(help="Repetition number"),
        "Time": st.column_config.TextColumn(help="Start and end of the repetition in the video"),
        "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f",
                                                 help="The assessment of this repetition: Module III's score (0 to 100)"),
        "Verdict": st.column_config.TextColumn(help=R.QUALITY_RULE),
        "Suppl. model score": st.column_config.NumberColumn(
            format="%.1f", help="Supplementary: the network's own learned score of this repetition, used to compare "
                                "the models; not the assessment. A raw prediction, so it can land slightly above 100 "
                                "or below 0."),
        "Distance to PT": st.column_config.NumberColumn(
            format="%.2f", help="How different this repetition looks to the model compared with the physiotherapist's "
                                "demo: 0 = the same, larger = less alike (about 0 to 1.5)."),
        "Problems": st.column_config.TextColumn(width="large", help="Problems found by the assessment (Module III)"),
    })


def show_rep_detail(cfg, res, rep, sess, fps):
    st.subheader("One repetition in detail")
    reps = rep["reps"]
    k = st.selectbox("Repetition", range(len(reps)), key=f"rep_{res['name']}", format_func=lambda i: (
        f"Rep {i + 1} · {reps[i]['time_s'][0]:.1f}–{reps[i]['time_s'][1]:.1f} s · score {reps[i]['score']:.1f} "
        f"({verdict(reps[i]['quality']).lower()}) · {len(reps[i]['errors']) or 'no'} "
        f"problem{'' if len(reps[i]['errors']) == 1 else 's'}"))
    r = reps[k]
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Problems in this repetition**")
        if not r["errors"]:
            st.success("No problems: every checked angle, the range and the speed stayed within the experts' range.")
        for e in r["errors"]:
            problem_card(e, fps)
        st.markdown("**Measurements**")
        m = r["measures"]
        rows = [{"Measurement": R.MEASURES[key][0], "Value": f"{m[key]:g}{UNITS[R.MEASURES[key][1]]}",
                 "What it means": R.MEASURES[key][2]} for key in R.MEASURES if m.get(key) is not None]
        table(rows, "Measurements of this repetition")
    with right:
        st.markdown("**How the score was made**")
        parts = points_lost(r, sess.ac)
        table([{k: (f"{v:g}" if isinstance(v, float) else v) for k, v in p.items()} for p in parts], "Score breakdown")
        lost = sum(p["Points lost (of 100)"] for p in parts)
        st.caption(f"Each part is compared with the high-scoring experts and loses points the further away it is: "
                   f"0 SD loses nothing, {sess.ac['score']['d_zero']:g} SD or more loses that part's whole share. "
                   f"Score = 100 − {lost:.1f} ≈ {r['score']:.1f} (small differences are rounding).")
        dev = m.get("angle_dev_deg") or {}
        if dev:
            st.markdown("**Average difference from the target, per angle**")
            table([{"Angle": R.label(R.CHANNELS, c), "Average difference": "–" if v is None else f"{v:g}°"}
                   for c, v in dev.items()], "Angle differences")
        if r.get("model_score") is not None:
            st.markdown("**Supplementary: the network's own outputs for this repetition**")
            st.caption("Not part of the assessment: the learned score head used to compare the models.")
            rows = [{"Question": "Score (0 to 100)", "Model's estimate": num(r["model_score"])}]
            rows += [{"Question": R.label(R.ITEMS, i) + " (0 to 1)", "Model's estimate": num(v)}
                     for i, v in (r.get("model_items") or {}).items()]
            if r.get("embedding_distance"):
                rows.append({"Question": "Distance to the physiotherapist's demo (0 = the same)",
                             "Model's estimate": num(r["embedding_distance"]["cosine_centred"])})
            table(rows, "Model outputs for this repetition")
    from demo.reference import ReferencePlayer, compare_figure
    st.markdown("**The patient against the physiotherapist at 5 moments of the repetition**")
    pyplot(compare_figure(sess, ReferencePlayer(cfg, sess.ex, sess.ref, sess.ac), k, res["view"]),
           "Patient and physiotherapist skeletons at 5 moments of the repetition")
    st.caption("Black = the patient, cyan = the physiotherapist at the matching moment of the demo, both scaled to the "
               "same body size. Numbers at the joints: patient minus target in degrees; green = inside the allowed "
               "range, red = outside.")
    st.markdown("**Each angle through the repetition**")
    pyplot(template_plot(sess, k, res["view"]), "Each angle of the repetition against its target and allowed range")
    st.caption("The repetition (orange) is stretched in time to line up with the demo, so 0% is the start and 100% the "
               "end of the movement. Grey = the range the high-scoring experts stay in.")
    with st.expander("Raw data of this repetition (JSON)"):
        st.json(plain(r))


def show_model(cfg, rep, model, sess):
    st.subheader("Supplementary: the network's own score (for comparing the models)")
    info = rep["model"]
    st.info("Not the assessment. The proposal's pipeline outputs Module III's score (above). This network also has a "
            "learned score head, trained on the physiotherapists' scores; the project uses it to compare the "
            "networks (B1, B2, Proposed) on the same task. In the tests it did not predict the physiotherapists' "
            "score better than the average of the participant's group.", icon=":material/info:")
    with st.container(horizontal=True):
        st.metric("Network's set score", f"{info['set_score']:.1f}", border=True,
                  help=f"{R.MODELS[model.method][0]}, fold {model.fold}. 0 to 100.")
        if info.get("movement_quality") is not None:
            st.metric("Network's movement quality", f"{info['movement_quality']:.1f}", border=True,
                      help="Its estimate of the posture and control questions only, leaving out the repetition "
                           "count. 0 to 100.")
    left, right = st.columns(2, gap="large")
    with left:
        held = info.get("test_subjects") or []
        st.markdown(f"**{R.MODELS[model.method][0]}**, fold {model.fold} of 5. Participants this model never saw "
                    f"during training: {', '.join(held) or 'unknown'}.")
        rows = [{"Question": R.label(R.ITEMS, i), "Model's estimate (0 to 1)": num(v)}
                for i, v in (info.get("items") or {}).items()]
        if rows:
            table(rows, "Model's answers to the physiotherapists' form")
        st.caption(R.ITEMS_NOTE + " Movement quality = 100 × the average of PF1, PF2, CF1, CF2 and CF3.")
        st.caption(f"Model file: `{info.get('checkpoint', '')}`")
    with right:
        pyplot(embedding_plot(cfg, model, sess), "Map of the model's view of the repetitions and the demo")
        st.caption("A map of how the model sees repetitions (its 256 numbers squeezed to 2). Each small dot is a "
                   "repetition of people the model was tested on, coloured by its score; red crosses are this "
                   "video's repetitions; the star is the physiotherapist's demo. Closer to the star = more like the "
                   "demo. The map keeps about half of the information, so distances here are only a rough guide.")


def show_glossary():
    with st.expander("Glossary: what the words mean"):
        st.markdown(
            f"- **Assessment (Module III):** every repetition is lined up in time with the physiotherapist's demo "
            f"and compared with it: joint angles, range, speed and symmetry, each against the range the "
            f"high-scoring experts stay in. The lining up happens in the CVDC model's camera-independent "
            f"representation, and the angular speed and the movement pattern are scored too (the proposal's "
            f"pipeline).\n"
            f"- **Supplementary network score:** the CVDC network's own score, trained on the physiotherapists' "
            f"scores (one network per exercise); the project uses it to compare networks, not as the assessment.\n"
            f"- **Verdict:** {R.QUALITY_RULE}\n"
            f"- **Severity:** {R.SEVERITY_RULE}\n"
            f"- **SD (standard deviation):** how much the high-scoring experts vary. 1 SD away = an ordinary "
            f"difference between experts; 3 SD = far outside what experts do.\n"
            f"- **Fold:** the models were trained 5 times, each time leaving out a different set of participants for "
            f"testing. The app uses the fold that left this participant out, so the score is fair.\n"
            f"- **Distance to PT:** how different a repetition looks to the model compared with the physiotherapist's "
            f"demo (0 = the same).\n"
            f"- **Questionnaire items:** {R.ITEMS_NOTE}\n"
            f"- **Cameras:** Front, Left side, Right side. **Conditions:** "
            + ", ".join(R.CONDITIONS[c] for c in R.CONDITIONS if c != "unknown") + ".")


def show_video(res):
    if res.get("video"):
        st.video(str(res["video"]), alt="Four-panel video: patient, physiotherapist reference, 3D view and the "
                                        "model's view")
        st.caption(f"Saved at `{res['video']}`" + (" (made earlier and reused)" if res.get("reused") else ""))
        if hasattr(os, "startfile") and st.button("Open the folder", icon=":material/folder_open:"):
            os.startfile(Path(res["video"]).parent)  # noqa: S606 (local app: opens the output folder)
        with st.expander("What the four panels show"):
            st.markdown(R.PANELS)
    elif res.get("video_error"):
        st.error("The four-panel video could not be made. The last lines of its log:")
        st.code(res["video_error"])


def show_results(cfg, res):
    sess, model, note = res["sess"], res["model"], res["note"]
    rep = sess.set_report(res["name"])
    fps = float(sess.fps)
    st.divider()
    st.header("Results")
    st.markdown(res["title"])
    if note:
        if note.startswith("LEAKAGE"):
            st.warning(note, title="The model saw this participant during training, so its score is not a fair test")
        else:
            st.warning(note)
    if not rep["reps"]:
        st.warning("No repetitions were found in this video, so there is nothing to score.")
    has_video = bool(res.get("video") or res.get("video_error"))
    if has_video:
        vid, main = st.columns([2, 3], gap="large")
        with vid:
            show_video(res)
    else:
        main = st.container()
    with main:
        show_summary(res, rep, model, fps)
        show_problem_overview(rep, fps)
    if rep["reps"]:
        st.subheader("All repetitions")
        show_rep_table(rep, fps)
        st.divider()
        show_rep_detail(cfg, res, rep, sess, fps)
    if model is not None and rep.get("model", {}).get("set_score") is not None:
        st.divider()
        show_model(cfg, rep, model, sess)
    st.divider()
    with st.container(horizontal=True):
        st.download_button("Download the report (JSON)", json.dumps(rep, indent=1, default=float),
                           file_name=f"{res['name']}_report.json", mime="application/json",
                           icon=":material/download:")
    with st.expander("Raw report of the whole set (JSON)"):
        st.json(plain(rep), expanded=False)
    show_glossary()


# ------------------------------------------------------------------ page


def main():
    st.set_page_config(page_title="CVDC assessment", page_icon=":material/accessibility_new:", layout="wide")
    cfg = config()
    with st.sidebar:
        st.header("Settings")
        st.caption("Pipeline (the proposal's): 2D pose → ST-GCN with CVDC → Module III → score and the problems "
                   "found. The comparison networks are part of the evaluation (the final report), not of this demo.")
        make_video = st.toggle("Make the four-panel video", value=True,
                               help="Takes about a minute. Saved in outputs/full/demo/app/ and played on this page. "
                                    "Faces are blurred.")
        ghost = st.toggle("Draw the physiotherapist on the patient", value=True,
                          help="Front-camera videos only: the physiotherapist's skeleton as a faint cyan outline on "
                               "the patient during each repetition.")
    st.title("Physiotherapy exercise assessment")
    st.caption("Choose a video, press Assess, and read the report: Module III's assessment against the "
               "physiotherapist's demo (with the Proposed CVDC model, the proposal's pipeline), the four-panel video, "
               "and, separately, the network's own supplementary score.")
    source = st.segmented_control("Video source", ["dataset", "upload"], required=True, default="dataset",
                                  format_func={"dataset": "From the dataset", "upload": "Upload my own video"}.get)
    row = up = None
    if source == "dataset":
        ex, row = pick_dataset_video(cfg)
        view = row.view if row is not None else None
        if row is not None:
            with st.container(border=True):
                st.markdown("Selected: " + describe_row(ex, row))
                st.markdown(f"Physiotherapists' score **{row['PT score']:.1f} / 100** ({verdict(row.quality).lower()})"
                            f" · file `{row.File}`")
            if not (video_root(cfg) / row.rel_path).exists():
                st.error(f"This video is not on this computer (looked for `{video_root(cfg) / row.rel_path}`). Put "
                         "the MobiPhysio dataset next to this folder (`../dataset/videos/E01(Abduction)/...`) or "
                         "inside it (`dataset/videos/...`), or set `MOBIPHYSIO_VIDEO_ROOT` (see the README). Or "
                         "choose **Upload my own video**.")
                row = None
        else:
            st.info("Tick a row in the table to choose a video.")
    else:
        t = video_table(str(resolve(cfg["data"]["index_csv"])))
        ex = st.selectbox("Exercise (one model per exercise)", sorted(t.exercise.unique()), format_func=R.exercise)
        view = st.segmented_control("Camera position", list(R.CAMERAS), required=True, default="F",
                                    format_func=R.CAMERAS.get)
        up = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])
    go = st.button("Assess this video", type="primary", icon=":material/play_arrow:",
                   disabled=row is None and up is None)
    if go:
        with contextlib.ExitStack() as stack, st.status("Assessing the video…", expanded=True) as status:
            if row is not None:
                src, name, subject = video_root(cfg) / row.rel_path, row.video_id, row.subject
                title, prow = describe_row(ex, row), row
            else:
                tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
                src = tmp / Path(up.name).name  # the original name, so a dataset file is recognised
                src.write_bytes(up.getbuffer())
                name = f"upload_{src.stem}_{hashlib.md5(up.getbuffer()).hexdigest()[:8]}"
                subject = dataset_subject(cfg, src.stem)
                t = video_table(str(resolve(cfg["data"]["index_csv"])))
                match = t[t.video_id == src.stem]
                prow = match.iloc[0] if not match.empty else None
                title = (f"**{R.exercise(ex)}** · uploaded file `{up.name}` · **{R.CAMERAS[view]}** camera"
                         + (f" (dataset video of participant {subject})" if subject else ""))
            model, note = load_model(cfg, ex, subject)
            st.write(f"Model: {R.MODELS[model.method][0]}, fold {model.fold}" if model is not None
                     else "Model: missing (rule-based Module III only)")
            out = resolve(cfg["outputs"]["phase1"]).parent / "demo" / "app" / video_name(name, model, ghost)
            job, video, reused = None, None, False
            if make_video and out.exists():
                video, reused = out, True
            elif make_video:
                job = start_export(src, ex, view, model, ghost, out)
                st.write("Four-panel video: started in the background")
            bar = st.progress(0.0, text="Starting")
            sess = assess(src, cfg, ex, view, model, bar)
            st.write(f"Assessment done: {len(sess.entries)} repetitions")
            video_error = None
            if job is not None:
                video, video_error = finish_export(job, status)
            status.update(label="Done", state="complete" if not video_error else "error", expanded=False)
        st.session_state["result"] = dict(sess=sess, model=model, note=note, name=name, view=view, ex=ex, row=prow,
                                          subject=subject, title=title, video=video, reused=reused,
                                          video_error=video_error)
    if "result" in st.session_state:
        show_results(cfg, st.session_state["result"])


main()
