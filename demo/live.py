"""Phase 6 live demo (spec §10; X1-X4, T17): a video file played at real-time speed, frame by frame, like a camera
stream (no physical webcam on this machine, X1; ``--source 0`` opens a webcam where one exists).

The window shows the frame with the skeleton overlay (hand landmarks for E06; joints named by the last rep's errors in
red), the exercise and camera view, the rep counter, after each rep its Module III score, quality and raw errors
(type · joint · deviation · frames), the model's rep score and the distance to the PT reference when a checkpoint is
loaded, the set score at the end (the model's video head if loaded, else Module III), a live plot of the main angle
and the last rep against its template band, FPS and the loaded checkpoint. No coaching text (spec scope). The side
panel shows the PT reference (X6, X7): the PT's own rep from rest to rest, at rest while the patient rests (nothing
compared), phase-aligned during a rep with the live angle differences per joint; ``--ghost`` draws it on the patient
during reps.

Keys: q quit, r reset the set, s save the session report (JSON + CSV) to <outputs>/demo/.

usage: python -m demo.live --source VIDEO.mp4 --exercise E03 [--view F|L|R] [--no-model] [--method proposed_j1]
       [--fold K] [--save-mp4 out.mp4] [--save-mode blur|skeleton] [--headless] [--max-seconds S]
       python -m demo.live --select E03 P05 F FL [...]      # a dataset video by its parsed fields (X2)
       python -m demo.live --list E03                        # dataset videos of an exercise
One model per exercise (T17): the exercise must be given (or comes from the selected dataset video); there is no
exercise head to detect it. For a dataset video the model fold is the one that held its subject out; otherwise the
demo prints a leakage warning (X2). ``--save-mp4`` writes the annotated output with faces blurred (D10), or with
``--save-mode skeleton`` the overlay on a black frame.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve, video_root  # noqa: E402
from cvdc.data.extract import HAND_SIDES  # noqa: E402
from cvdc.data.skeleton import MP_DRAW_EDGES, J  # noqa: E402
from cvdc.data.video import iter_frames  # noqa: E402
from demo.pipeline import ExerciseModel, OnlineSession, PoseStream, dataset_subject, save_report  # noqa: E402

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
GROUP_JOINTS = {"trunk": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"], "knees": ["left_knee", "right_knee"],
                "hips": ["left_hip", "right_hip"]}
GREEN, RED, WHITE, YELLOW, GREY = (60, 220, 60), (40, 40, 235), (255, 255, 255), (0, 215, 255), (150, 150, 150)


def faulty_joints(entry):
    out = set()
    for e in (entry or {}).get("errors", []):
        for name in GROUP_JOINTS.get(e.get("joint", ""), [e.get("joint", "")]):
            if name in J:
                out.add(J[name])
    return out


def draw_pose(img, lm, hands, bad):
    """Skeleton (coco17 edges) and hand landmarks on a BGR image; landmarks normalised to the frame."""
    H, W = img.shape[:2]
    if lm is None:
        return
    p = np.c_[lm[:, 0] * W, lm[:, 1] * H]
    for a, b in MP_DRAW_EDGES:
        if np.isfinite(p[[a, b]]).all():
            cv2.line(img, tuple(int(v) for v in p[a]), tuple(int(v) for v in p[b]),
                     RED if (a in bad or b in bad) else GREEN, 3, cv2.LINE_AA)
    for j in {x for e in MP_DRAW_EDGES for x in e}:
        if np.isfinite(p[j]).all():
            cv2.circle(img, tuple(int(v) for v in p[j]), 7 if j in bad else 4, RED if j in bad else WHITE, -1,
                       cv2.LINE_AA)
    for h in hands if hands is not None else []:
        if not np.isfinite(h[:, :2]).all():
            continue
        q = np.c_[h[:, 0] * W, h[:, 1] * H]
        for a, b in HAND_EDGES:
            cv2.line(img, tuple(int(v) for v in q[a]), tuple(int(v) for v in q[b]), YELLOW, 1, cv2.LINE_AA)


def blur_faces(img, lm, state=None):
    """Gaussian blur over the head box from the face landmarks (nose, eyes, ears) for saved output (D10). A frame
    without face landmarks reuses the last head box (``state``); with no box yet the whole frame is blurred, so a
    saved frame never shows an unblurred face."""
    H, W = img.shape[:2]
    state = {} if state is None else state
    pts = np.empty((0, 2))
    if lm is not None:
        pts = np.array([[lm[j, 0] * W, lm[j, 1] * H] for j in range(11) if np.isfinite(lm[j, :2]).all()])
    if len(pts) >= 2:
        c, r = pts.mean(0), max(np.ptp(pts, 0).max() * 1.4, 0.06 * max(H, W))
        state["box"] = (max(0, int(c[0] - r)), max(0, int(c[1] - r * 1.3)), min(W, int(c[0] + r)), min(H, int(c[1] + r)),
                        max(31, (int(r) // 2) * 2 + 1))
    if "box" not in state:
        return cv2.GaussianBlur(img, (61, 61), 0)
    x0, y0, x1, y1, k = state["box"]
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (k, k), 0)
    return img


CYAN = (230, 200, 60)
from demo.reference import SHORTC  # noqa: E402  (short channel names for the labels, X6)


def joint_px(lm, name, W, H):
    if lm is None:
        return None
    if name == "mid_hip":
        p = (lm[J["left_hip"], :2] + lm[J["right_hip"], :2]) / 2
    else:
        p = lm[J[name], :2]
    return None if not np.isfinite(p).all() else np.array([p[0] * W, p[1] * H])


def draw_angle_labels(img, lm, diffs, side):
    """The live angle difference (deg, patient - reference target) at each channel's joint; green inside the band,
    red outside (X6)."""
    from demo.reference import CHANNEL_JOINT
    H, W = img.shape[:2]
    o = "left" if side == "right" else "right"
    for c, (d, ok, _, _) in diffs.items():
        jn, row = CHANNEL_JOINT.get(c, (None, 0))
        p = joint_px(lm, jn.format(s=side, o=o), W, H) if jn else None
        if p is None:
            continue
        text(img, f"{SHORTC.get(c, c)} {d:+.0f}", (int(p[0]) + 10, int(p[1]) - 6 + 18 * row), 0.5, GREEN if ok else RED)


def body_geometry(lm, W, H):
    """(mid-hip px, torso length px, shoulder width px) of one frame's landmarks, or None."""
    mh = joint_px(lm, "mid_hip", W, H)
    if mh is None:
        return None
    ls, rs = joint_px(lm, "left_shoulder", W, H), joint_px(lm, "right_shoulder", W, H)
    if ls is None or rs is None:
        return None
    return mh, float(np.linalg.norm((ls + rs) / 2 - mh)), float(np.linalg.norm(ls - rs))


def draw_ghost(img, mid_hip, ref_xy, scale, label):
    """The PT reference skeleton as a faint ghost on the patient: centred on the mid-hip and scaled so that its
    shoulder width matches the patient's (X7; the caller falls back to the torso length in side-on poses)."""
    H = img.shape[0]
    p = mid_hip + ref_xy * scale
    layer = img.copy()
    for a, b in MP_DRAW_EDGES:
        if np.isfinite(p[[a, b]]).all():
            cv2.line(layer, tuple(int(v) for v in p[a]), tuple(int(v) for v in p[b]), CYAN, 5, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.45, img, 0.55, 0, img)
    text(img, label, (10, H - 34), 0.5, CYAN)


def reference_panel(H, PW, ref_xy, k, player, mirrored, patient_view, diffs, state, torso_px=None, hip_y=None,
                    j=None):
    """Side panel (X6, X7): the PT's own keypoints at displayed frame k of its rest -> peak -> rest rep, scaled to the
    patient by torso length and centred on the mid-hip (shrunk if the rep would leave the panel), its source and view,
    the patient's state and, during a rep, the per-channel live angles (patient / target / difference)."""
    panel = np.full((H, PW, 3), 22, np.uint8)
    X = player.xy
    reach_x = float(np.abs(X[..., 0]).max()) + 0.15
    up, down = float(-X[..., 1].min()) + 0.15, float(X[..., 1].max()) + 0.1
    top, bottom = 150, H - 10  # the rep must fit between the header and the panel's bottom
    s = min(torso_px or 0.17 * H, (PW / 2 - 6) / reach_x, (bottom - top) / (up + down))
    y0 = float(np.clip(hip_y if hip_y is not None else 0.55 * H, top + up * s, bottom - down * s))
    p = np.array([PW / 2, y0]) + ref_xy * s
    for a, b in MP_DRAW_EDGES:
        if np.isfinite(p[[a, b]]).all():
            cv2.line(panel, tuple(int(v) for v in p[a]), tuple(int(v) for v in p[b]), CYAN, 3, cv2.LINE_AA)
    text(panel, "PT reference (its own rep, time-aligned)", (8, 22), 0.5)
    text(panel, f"{player.file} | {player.view} view" + (" | mirrored" if mirrored else ""), (8, 42), 0.45, GREY)
    j = player.template_frame(k) if j is None else j
    label = None
    if state == "rep":
        line = (f"patient in a rep | PT frame {k + 1}/{player.n}, template {j + 1}/{player.n_template}" if j >= 0
                else f"patient in a rep | PT frame {k + 1}/{player.n} (rest lead-in / return: not compared)")
        text(panel, line, (8, 62), 0.42, YELLOW)
        label = player.return_label if k >= player.tail[0] else None
    else:
        text(panel, "patient at rest | PT at rest | not compared", (8, 62), 0.42, GREY)
        label = player.rest_label
    y = 84
    for part in (label.replace(" (", "\n(").split("\n") if label else []):  # X8: E03's rest pose / return
        text(panel, part, (8, y), 0.42, CYAN)
        y += 18
    if patient_view != "F" or player.view != "front":
        for line in (f"patient view {patient_view}, reference {player.view} view:",
                     "compare the 3D-angle differences, not the picture"):
            text(panel, line, (8, y), 0.42, YELLOW)
            y += 18
    if diffs:
        y = int(0.80 * H) - 18 * max(0, len(diffs) - 6)
        text(panel, "channel: patient / target  (diff)", (8, y), 0.42, GREY)
        for ch, (d, ok, v, t) in diffs.items():
            y += 18
            text(panel, f"{SHORTC.get(ch, ch)}: {v:.0f} / {t:.0f}  ({d:+.0f})", (8, y), 0.45, GREEN if ok else RED)
    return panel


def text(img, s, org, scale=0.6, color=WHITE, thick=1):
    """Text with a dark outline drawn as offset copies of the same stroke (OpenCV 5's text renderer spaces a thicker
    stroke wider, so a thick black copy no longer sits under the text)."""
    x, y = org
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)):
        cv2.putText(img, s, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def plot_inset(img, x0, y0, w, h, series, title, ylim=None):
    """series: [(values (n,), BGR colour, thickness)] drawn on a shared x range in a dark box."""
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (25, 25, 25), -1)
    vals = np.concatenate([np.asarray(s[0], float) for s in series if len(s[0])]) if series else np.array([])
    vals = vals[np.isfinite(vals)]
    text(img, title, (x0 + 6, y0 + 16), 0.45, GREY)
    if not len(vals):
        return
    lo, hi = ylim or (float(vals.min()), float(vals.max()))
    hi = hi if hi > lo + 1e-6 else lo + 1.0
    for v, col, th in series:
        v = np.asarray(v, float)
        if len(v) < 2:
            continue
        xs = x0 + 4 + np.arange(len(v)) * (w - 8) / (len(v) - 1)
        ys = y0 + h - 6 - (v - lo) / (hi - lo) * (h - 28)
        ok = np.isfinite(ys)
        pts = np.c_[xs[ok], ys[ok]].astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, col, th, cv2.LINE_AA)
    text(img, f"{hi:.0f}", (x0 + w - 34, y0 + 30), 0.4, GREY)
    text(img, f"{lo:.0f}", (x0 + w - 34, y0 + h - 8), 0.4, GREY)


def error_lines(entry, n=4):
    out = []
    for e in (entry or {}).get("errors", [])[:n]:
        dev = e.get("deviation_deg", e.get("deviation_s"))
        unit = "deg" if "deviation_deg" in e else "s"
        fr = e.get("frames")
        out.append(f"{e['type']} . {e.get('joint', '')} . {dev:+.1f} {unit}" + (f" . frames {fr[0]}-{fr[1]}" if fr else "")
                   + f" ({e['severity']})")
    return out


def frames_from(source, max_side):
    if str(source).isdigit():  # webcam (none on this machine, X1)
        cap = cv2.VideoCapture(int(source))
        t0 = time.time()
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            s = min(1.0, max_side / max(rgb.shape[:2]))
            if s < 1:
                rgb = cv2.resize(rgb, (int(rgb.shape[1] * s) // 2 * 2, int(rgb.shape[0] * s) // 2 * 2))
            yield time.time() - t0, rgb, {"orig_hw": rgb.shape[:2]}
        cap.release()
        return
    yield from iter_frames(source, max_side, False)


def source_fps(src):
    """The video's average frame rate, so a saved video plays back at normal speed (30 for a webcam)."""
    if str(src).isdigit():
        return 30.0
    try:
        import av
        with av.open(str(src)) as c:
            r = c.streams.video[0].average_rate
            return float(r) if r else 30.0
    except Exception:  # noqa: BLE001 (an unreadable rate falls back to 30 fps)
        return 30.0


def even_size(img, max_side=1920):
    """Downscale so the longer side is <= max_side, with even dimensions (mp4)."""
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    nw, nh = int(w * s) // 2 * 2, int(h * s) // 2 * 2
    return img if (nw, nh) == (w, h) else cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def pick_dataset_video(cfg, ex, subject, view, variation):
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    r = idx[(idx.exercise == ex) & (idx.subject == subject) & (idx.view == view) & (idx.variation == variation)]
    if r.empty:
        raise SystemExit(f"no dataset video {ex} {subject} {view} {variation}")
    r = r.iloc[0]
    return video_root(cfg) / r.rel_path, r.video_id, r.view


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise.yaml")
    ap.add_argument("--source", help="video file, or a webcam index (0)")
    ap.add_argument("--select", nargs=4, metavar=("EXERCISE", "SUBJECT", "VIEW", "VARIATION"))
    ap.add_argument("--list", metavar="EXERCISE")
    ap.add_argument("--exercise")
    ap.add_argument("--view", choices=["F", "L", "R"])
    ap.add_argument("--no-model", action="store_true", help="Module III only (works without any checkpoint, X3)")
    ap.add_argument("--method", default=None, help="default eval.proposed_default (proposed_j1)")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--tag", default="perex_gkf5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-mp4")
    ap.add_argument("--save-mode", choices=["blur", "skeleton"], default="blur")
    ap.add_argument("--headless", action="store_true", help="no window (testing); with --save-mp4 for the output")
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--no-realtime", action="store_true", help="process as fast as possible")
    ap.add_argument("--no-reference", action="store_true", help="no reference side panel / angle labels (X6)")
    ap.add_argument("--ghost", action="store_true", help="draw the PT reference as a faint ghost on the patient (X6)")
    ap.add_argument("--ghost-any-view", action="store_true",
                    help="allow the ghost on side-view videos too (labelled: the reference is a front-view clip)")
    ap.add_argument("--export4", metavar="OUT.mp4",
                    help="render offline a 2 x 2 video: patient | PT reference / estimated 3D | CVDC embedding (X9)")
    a = ap.parse_args()
    if a.export4:  # offline rendering; written at the source's frame rate, so it plays back at normal speed
        a.headless, a.no_realtime = True, True
    cfg = load_config(a.config if Path(a.config).exists() else REPO / a.config)  # runs from any folder
    if a.list:
        idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
        d = idx[idx.exercise == a.list][["video_id", "subject", "group", "view", "variation", "score"]]
        print(d.to_string(index=False))
        return
    subject, video_id = None, None
    if a.select:
        src, video_id, view = pick_dataset_video(cfg, *a.select)
        ex = a.select[0]
        subject = a.select[1]
        a.view = a.view or view
    else:
        if not a.source or not a.exercise:
            raise SystemExit("give --source and --exercise (one model per exercise, T17), or --select")
        src, ex = a.source, a.exercise.upper()
        video_id = Path(str(src)).stem
        subject = dataset_subject(cfg, video_id)
    view = a.view or cfg["demo"]["default_view"]
    model = None
    if not a.no_model:
        try:
            model = ExerciseModel(cfg, ex, a.method or cfg["eval"]["proposed_default"], a.fold, subject, a.device,
                                  a.tag)
            print(f"model: {model.checkpoint}")
            if model.warning:
                print("WARNING:", model.warning)
        except FileNotFoundError as e:
            print(f"no model ({e}); Module III only")
    hands = ex in (cfg["pose"].get("hands_for") or [])
    pose = PoseStream(cfg, hands=hands)
    sess = OnlineSession(cfg, ex, view, model)
    out_dir = resolve(cfg["outputs"]["phase1"]).parent / "demo"
    writer = None
    win = f"CVDC demo {ex} ({view})"
    t_wall0, n_frames, fps_t, fps = time.time(), 0, time.time(), 0.0
    last_entry, last_update, trace = None, 0.0, []
    # X6, X7: the PT reference (its own rest -> peak -> rest rep), at rest while the patient rests, phase-aligned
    # to the patient during a rep (demo/reference.py LiveReference)
    live_ref = player = None
    if not a.no_reference:
        from demo.reference import LiveReference
        live_ref = LiveReference(cfg, sess, view)
        player = live_ref.player
        ghost_on = a.ghost and (view == "F" or a.ghost_any_view)
        if a.ghost and not ghost_on:
            print(f"ghost off: the {view} view is not the reference's {player.view} view (--ghost-any-view to force)")
    geo_hist, blur_state = [], {}
    out_fps = source_fps(src)
    writer4 = sk3d = emb = None
    if a.export4:  # X9: the four-panel export (demo/export4.py)
        from demo.export4 import EmbeddingView, Skeleton3DView, compose4, normalise3d
        from demo.reference import mirror33
        if player is None:
            raise SystemExit("--export4 needs the PT reference (drop --no-reference)")
        sk3d = Skeleton3DView()
        emb = EmbeddingView(cfg, model) if model is not None else None
    try:
        for t, rgb, geom in frames_from(src, cfg["pose"]["max_side"]):
            if a.max_seconds and t > a.max_seconds:
                break
            lm = pose.process(t, rgb)
            hand = pose.hand2d[-1] if hands else None
            w = pose.world[-1]
            trace.append(sess.live_angle(w))
            g = body_geometry(lm, rgb.shape[1], rgb.shape[0]) if lm is not None else None
            if g is not None:
                geo_hist = (geo_hist + [(g[1], g[2], g[0][1])])[-90:]  # torso, shoulder width, mid-hip y (px)
            if t - last_update >= 0.5:
                last_update = t
                for e in sess.update(pose):
                    last_entry = e
                    print(f"rep {e['rep'] + 1}{' (updated)' if e.get('updated') else ''}: score {e['score']} "
                          f"({e['quality']})" + (f", model {e['model_score']}" if "model_score" in e else "")
                          + (f", distance {e['embedding_distance']['cosine_centred']}" if e.get("embedding_distance") else "")
                          + "".join(f"\n    {x}" for x in error_lines(e, 9)))
                if live_ref is not None:
                    live_ref.after_update(sess, pose)
            rs = live_ref.frame(pose) if live_ref is not None else {"in_rep": False, "k": 0, "diffs": {}, "side": None}
            # ---------------- drawing
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            canvas = np.zeros_like(bgr) if (a.save_mode == "skeleton" and a.save_mp4) else bgr.copy()
            draw_pose(canvas, lm, hand, faulty_joints(last_entry))
            H, W = canvas.shape[:2]
            n_frames += 1
            if time.time() - fps_t >= 1.0:
                fps, n_frames, fps_t = n_frames / (time.time() - fps_t), 0, time.time()
            y = 24
            for s in (f"{ex} {cfg['exercises'][ex]['name']} | view {view} | t {t:5.1f} s | "
                      + ("offline render" if a.export4 else f"{fps:4.1f} FPS"),
                      f"reps: {len(sess.entries)}" + (" | in a rep" if rs["in_rep"] else " | at rest")
                      + (f" | model: {model.method} fold {model.fold}" if model else
                         " | Module III only (no checkpoint)")):
                text(canvas, s, (10, y))
                y += 24
            if last_entry:
                text(canvas, f"rep {last_entry['rep'] + 1}: {last_entry['score']} ({last_entry['quality']})"
                     + (f" | model {last_entry['model_score']}" if "model_score" in last_entry else "")
                     + (f" | dist {last_entry['embedding_distance']['cosine_centred']}"
                        if last_entry.get("embedding_distance") else ""), (10, y), 0.6, YELLOW)
                y += 22
                for s in error_lines(last_entry):
                    text(canvas, s, (10, y), 0.5, RED)
                    y += 20
            if model and model.warning:
                text(canvas, model.warning[:90], (10, H - 12), 0.45, RED)
            # live plot: main angle over the last 6 s; last rep vs template target and band
            pw, ph = min(300, W // 2 - 20), 110
            plot_inset(canvas, W - pw - 10, 10, pw, ph, [(np.asarray(trace[-180:]), YELLOW, 2)],
                       f"{sess.ac['angle']} (live, deg)")
            if sess.debug:
                d = sess.debug[-1]
                main = sess.ref["main"]
                tgt = d["cmp"]["target"][main]
                sd = sess.lib[view]["sigma"][main] * sess.ac["band_k"]
                plot_inset(canvas, W - pw - 10, 20 + ph, pw, ph,
                           [(tgt - sd, GREY, 1), (tgt + sd, GREY, 1), (tgt, WHITE, 2),
                            (d["aligned"]["curves"][main], YELLOW, 2)], "last rep vs template (band)")
            panel = None
            if player is not None:
                ref_xy, mirrored = player.frame(rs["k"], rs["side"])
                gm = np.median(geo_hist, 0) if geo_hist else None
                if ghost_on and rs["in_rep"] and g is not None and gm is not None:  # X7: only during a rep
                    # mid-hip + shoulder width (within 25% of the torso scale); the torso length when either
                    # skeleton's shoulders are foreshortened (side-on patient, E04's oblique demo)
                    scale = (float(np.clip(gm[1] / player.shoulder_w, 0.8 * gm[0], 1.25 * gm[0]))
                             if player.view == "front" and gm[1] >= 0.35 * gm[0] else gm[0])
                    draw_ghost(canvas, g[0], ref_xy, scale,
                               f"ghost: PT reference ({player.view} view)" + (" - not this camera's view" if
                                                                             view != "F" else ""))
                draw_angle_labels(canvas, lm, rs["diffs"], rs["side"] or "right")
                panel = reference_panel(H, max(260, int(0.55 * H)), ref_xy, rs["k"], player, mirrored, view,
                                        rs["diffs"], "rep" if rs["in_rep"] else "rest",
                                        gm[0] if gm is not None else None, gm[2] if gm is not None else None,
                                        rs.get("j"))
            saved = blur_faces(canvas.copy(), lm, blur_state) if (writer is not None or a.save_mp4 or a.export4) \
                and a.save_mode == "blur" else canvas
            if a.export4:  # X9: 2 x 2, every panel the size of the video frame
                p2 = reference_panel(H, W, ref_xy, rs["k"], player, mirrored, view, rs["diffs"],
                                     "rep" if rs["in_rep"] else "rest", gm[0] if gm is not None else None,
                                     gm[2] if gm is not None else None, rs.get("j"))
                Pw = np.asarray(pose.world[-3:], float)
                Pw = Pw[np.isfinite(Pw).all(axis=(1, 2))]
                pt3 = player.world[int(np.clip(rs["k"], 0, player.n - 1))]
                p3 = sk3d.draw(W, H, normalise3d(np.median(Pw, 0)) if len(Pw) else None,
                               normalise3d(mirror33(pt3) if mirrored else pt3), t, mirrored)
                if emb is not None:
                    p4 = emb.draw(W, H, sess, rs["side"] or sess.side or player.side)
                else:
                    p4 = np.full((H, W, 3), 22, np.uint8)
                    text(p4, "CVDC embedding: no model loaded (Module III only)", (10, 24), 0.55, GREY)
                grid = even_size(compose4(saved, p2, p3, p4))
                if writer4 is None:
                    writer4 = cv2.VideoWriter(str(a.export4), cv2.VideoWriter_fourcc(*"mp4v"), out_fps,
                                              (grid.shape[1], grid.shape[0]))
                writer4.write(grid)
            if panel is not None:
                canvas, saved = np.hstack([canvas, panel]), np.hstack([saved, panel])
            if writer is None and a.save_mp4:
                writer = cv2.VideoWriter(str(a.save_mp4), cv2.VideoWriter_fourcc(*"mp4v"), out_fps,
                                         (canvas.shape[1], canvas.shape[0]))
            if writer is not None:
                writer.write(saved)
            if not a.headless:
                cv2.imshow(win, canvas)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("r"):
                    pose.reset()
                    sess.reset()
                    last_entry, trace, geo_hist = None, [], []
                    if live_ref is not None:
                        live_ref.reset()
                if k == ord("s"):
                    print("saved", save_report(sess.set_report(video_id or ""), out_dir, f"{ex}_{video_id}"))
            if not a.no_realtime:  # real-time playback: wait until the frame's own time
                lag = t - (time.time() - t_wall0)
                if lag > 0:
                    time.sleep(lag)
        for e in sess.update(pose, final=True):
            last_entry = e
            print(f"rep {e['rep'] + 1}{' (updated)' if e.get('updated') else ''} (final pass): score {e['score']} "
                  f"({e['quality']})" + (f", model {e['model_score']}" if "model_score" in e else ""))
    finally:
        pose.close()
        if writer is not None:
            writer.release()
        if writer4 is not None:
            writer4.release()
            print("four-panel video:", a.export4)
        if not a.headless:
            cv2.destroyAllWindows()
    rep = sess.set_report(video_id or "")
    ms = rep.get("model", {}).get("set_score")
    print(f"set: {rep['n_reps']} reps | Module III set score {rep['set_score']} ({rep['set_quality']})"
          + (f" | model set score {ms:.1f}, MQ {rep['model'].get('movement_quality', float('nan')):.1f}" if ms is not None else "")
          + (f" | set errors {rep['set_errors']}" if rep["set_errors"] else ""))
    print("report:", save_report(rep, out_dir, f"{ex}_{video_id}"))


if __name__ == "__main__":
    main()
