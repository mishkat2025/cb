"""Phase 6 four-panel export (user, 2026-10-20; X9): ``demo/live.py --export4 out.mp4`` renders a video offline and
writes it at the source's frame rate, so it plays back at normal speed. The 2 x 2 layout:

  1 top-left      the patient video with the 2D skeleton (the last rep's error joints in red), faces blurred (D10)
  2 top-right     the PT reference panel of the live demo (X7, X8: its own keypoints, rest / rep logic, labels)
  3 bottom-left   Skeleton3DView: the patient's and the PT's MediaPipe world landmarks, each centred on its mid-hip
                  and scaled by its 3D torso length, the PT at the frame the reference panel shows (time-aligned,
                  mirrored for the other side), seen by a slowly turning orthographic camera; "estimated 3D (single
                  camera)": MediaPipe's world landmarks are a single-view estimate, not a measurement
  4 bottom-right  EmbeddingView: the Proposed model's embedding (the CVDC-trained encoder, 256-d) of every completed
                  rep and the PT demo's reference embedding (E5; mirrored for the other side), projected to 2D by a
                  PCA fitted on that fold's training embeddings (the fold that held the subject out; centred on
                  their mean, as the cosine distance is, E13); the training reps as a faint cloud coloured by
                  score, the patient's reps as a trail, a gauge of the latest rep's centred cosine distance to the
                  reference, and its scores: Module III's (the assessment, X11) and the model's own (supplementary)

The live window stays two-panel. CPU only. The PCA (and the training projections) are cached per exercise, method
and fold in <outputs>/demo/pca/.
"""
import math

import cv2
import numpy as np
import torch

from cvdc.config import resolve
from cvdc.data.skeleton import MP_DRAW_EDGES, J

GREEN, RED, WHITE, YELLOW, GREY, CYAN = (60, 220, 60), (40, 40, 235), (255, 255, 255), (0, 215, 255), (150, 150, 150), (230, 200, 60)
BG = (22, 22, 22)


def put(img, s, org, scale=0.5, color=WHITE):
    """Text with an outline of offset copies (as demo/live.py: OpenCV 5 spaces a thicker stroke wider)."""
    x, y = org
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(img, s, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def compose4(p1, p2, p3, p4):
    """The 2 x 2 grid (all panels the same size)."""
    return np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])


# ----------------------------------------------------------------------------------------------- panel 3
def normalise3d(P):
    """World landmarks (33, 3) -> centred on the mid-hip, in units of the 3D torso length (mid-shoulder to mid-hip)."""
    P = np.asarray(P, float)
    mh = (P[J["left_hip"]] + P[J["right_hip"]]) / 2
    ms = (P[J["left_shoulder"]] + P[J["right_shoulder"]]) / 2
    return (P - mh) / max(float(np.linalg.norm(ms - mh)), 1e-6)


class Skeleton3DView:
    def __init__(self, deg_per_s=12.0, tilt_deg=15.0):
        self.w, self.tilt = math.radians(deg_per_s), math.radians(tilt_deg)

    def project(self, P, theta, c, s):
        """Orthographic view of torso-unit points P (n, 3) (MediaPipe axes: x, y down, z depth) from a camera turned
        by theta about the vertical axis and tilted down by self.tilt."""
        ct, st = math.cos(theta), math.sin(theta)
        x = ct * P[:, 0] + st * P[:, 2]
        z = -st * P[:, 0] + ct * P[:, 2]
        y = math.cos(self.tilt) * P[:, 1] - math.sin(self.tilt) * z
        return np.c_[c[0] + s * x, c[1] + s * y]

    def draw(self, W, H, patient, pt, t, mirrored):
        img = np.full((H, W, 3), BG, np.uint8)
        theta = self.w * t
        s = 0.2 * min(H, 1.4 * W)
        c = (W / 2, 0.42 * H)
        # floor grid under the feet (the lowest ankle / heel / toe of either skeleton), 3 x 3 torso units
        feet = [J[n] for n in ("left_ankle", "right_ankle", "left_heel", "right_heel", "left_foot_index",
                               "right_foot_index") if n in J]
        ys = [float(np.max(P[feet, 1])) for P in (patient, pt) if P is not None and np.isfinite(P[feet]).all()]
        fy = max(ys) if ys else 2.4
        g = np.linspace(-1.5, 1.5, 7)
        for a in g:
            for seg in (np.c_[np.full(7, a), np.full(7, fy), g], np.c_[g, np.full(7, fy), np.full(7, a)]):
                q = self.project(seg, theta, c, s).astype(np.int32)
                cv2.polylines(img, [q], False, (60, 60, 60), 1, cv2.LINE_AA)
        for P, col, lw in ((pt, CYAN, 5), (patient, GREEN, 3)):
            if P is None or not np.isfinite(P).all():
                continue
            q = self.project(P, theta, c, s)
            for a, b in MP_DRAW_EDGES:
                cv2.line(img, tuple(int(v) for v in q[a]), tuple(int(v) for v in q[b]), col, lw, cv2.LINE_AA)
        put(img, "estimated 3D (single camera)", (10, 24), 0.6)
        put(img, "MediaPipe world landmarks, centred on the mid-hip, torso-scaled", (10, 46), 0.42, GREY)
        put(img, "green: patient | cyan: PT, time-aligned" + (" (mirrored)" if mirrored else ""), (10, 66), 0.42, GREY)
        put(img, f"camera {math.degrees(theta) % 360:3.0f} deg", (10, H - 12), 0.42, GREY)
        return img


# ----------------------------------------------------------------------------------------------- panel 4
class EmbeddingView:
    def __init__(self, cfg, model, max_points=3000):
        self.model = model
        out = resolve(cfg["outputs"]["phase1"]).parent / "demo" / "pca"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{model.ex}_{model.method}_fold{model.fold}.npz"
        if path.exists():
            with np.load(path) as z:
                self.mean, self.comp, self.train_xy, self.train_score, self.var = (
                    z["mean"], z["comp"], z["train_xy"], z["train_score"], z["var"])
        else:
            Z, score = self._train_embeddings(cfg)
            self.mean = Z.mean(0)
            _, sv, vt = np.linalg.svd(Z - self.mean, full_matrices=False)
            self.comp = vt[:2]
            self.var = (sv[:2] ** 2 / (sv ** 2).sum()).astype(np.float64)
            keep = np.random.default_rng(0).permutation(len(Z))[:max_points]
            self.train_xy, self.train_score = (Z[keep] - self.mean) @ self.comp.T, score[keep]
            np.savez(path, mean=self.mean, comp=self.comp, train_xy=self.train_xy, train_score=self.train_score,
                     var=self.var)
        self.ref_xy = {s: (v - self.mean) @ self.comp.T for s, v in model.ref.items()}
        lo, hi = np.percentile(self.train_xy, [1, 99], axis=0)  # the view: the training cloud and the references
        R = np.stack(list(self.ref_xy.values()))
        lo, hi = np.minimum(lo, R.min(0)), np.maximum(hi, R.max(0))
        pad = 0.08 * (hi - lo)
        self.lo, self.hi = lo - pad, hi + pad
        self._key, self._img = None, None

    @torch.no_grad()
    def _train_embeddings(self, cfg):
        """The model's embedding z of every training rep of its fold (the packed inputs the model was trained on)."""
        from cvdc.data.dataset import PackData
        from cvdc.train.trainer import forward_all, items_of
        m = self.model
        data = PackData(cfg, "rep", "cpu")
        data.set_scope([m.ex])
        items = items_of(data, m.plan["train"])
        z = forward_all(m.model, data, items, m.mc, heads=False)["z"]
        return z.float().cpu().numpy().astype(np.float64), data.score[items].cpu().numpy()

    def _to_px(self, p, box):
        x0, y0, x1, y1 = box
        span = np.maximum(self.hi - self.lo, 1e-9)
        u = (np.asarray(p, float) - self.lo) / span
        return np.c_[x0 + u[..., 0] * (x1 - x0), y1 - u[..., 1] * (y1 - y0)] if np.ndim(p) > 1 else \
            np.array([x0 + u[0] * (x1 - x0), y1 - u[1] * (y1 - y0)])

    def draw(self, W, H, sess, side):
        key = (W, H, side, len(sess.zs), tuple(e.get("model_score") for e in sess.entries))
        if key == self._key:
            return self._img
        img = np.full((H, W, 3), BG, np.uint8)
        m = self.model
        put(img, "CVDC embedding (Proposed model)", (10, 24), 0.6)
        put(img, f"{m.ex} {m.method} fold {m.fold} | PCA of the fold's training reps "
                 f"({100 * self.var.sum():.0f}% of the variance)", (10, 46), 0.42, GREY)
        side_len = min(W - 20, H - 220)
        box = (int((W - side_len) / 2), 70, int((W + side_len) / 2), 70 + side_len)
        cv2.rectangle(img, box[:2], box[2:], (45, 45, 45), 1)
        cl = np.clip(self.train_xy, self.lo, self.hi)
        sc = np.clip((self.train_score - 30) / 70, 0, 1)
        for (x, y), f in zip(self._to_px(cl, box), sc):
            cv2.circle(img, (int(x), int(y)), 1, (int(70 + 40 * f), int(60 + 110 * f), int(150 - 90 * f)), -1)
        r = self._to_px(np.clip(self.ref_xy.get(side, next(iter(self.ref_xy.values()))), self.lo, self.hi), box)
        cv2.drawMarker(img, (int(r[0]), int(r[1])), CYAN, cv2.MARKER_STAR, 22, 2, cv2.LINE_AA)
        put(img, "PT", (int(r[0]) + 10, int(r[1]) - 8), 0.5, CYAN)
        zs = [z for z in sess.zs if z is not None]
        if zs:
            P = self._to_px(np.clip((np.stack(zs) - self.mean) @ self.comp.T, self.lo, self.hi), box)
            n = len(P)
            for i in range(1, n):
                a = 0.25 + 0.75 * i / max(n - 1, 1)
                cv2.line(img, tuple(int(v) for v in P[i - 1]), tuple(int(v) for v in P[i]),
                         tuple(int(a * c) for c in YELLOW), 1, cv2.LINE_AA)
            for i, p in enumerate(P):
                last = i == n - 1
                cv2.circle(img, (int(p[0]), int(p[1])), 7 if last else 4, YELLOW if last else (0, 140, 170), -1,
                           cv2.LINE_AA)
                put(img, str(i + 1), (int(p[0]) + 6, int(p[1]) + 14), 0.4, YELLOW if last else GREY)
        put(img, "dots: training reps (red = low score, green = high) | yellow: this patient's reps",
            (10, box[3] + 18), 0.38, GREY)
        # gauge and scores of the latest rep
        y = box[3] + 48
        e = sess.entries[-1] if sess.entries else None
        d = (e or {}).get("embedding_distance", {}) or {}
        dist = d.get("cosine_centred")
        put(img, "distance to the PT reference (centred cosine)", (10, y), 0.45)
        gx0, gx1, gy = 14, W - 60, y + 18
        for i in range(gx1 - gx0):
            f = i / (gx1 - gx0)
            cv2.line(img, (gx0 + i, gy), (gx0 + i, gy + 14), (40, int(200 * (1 - f) + 40), int(60 + 180 * f)), 1)
        for v in (0.0, 0.5, 1.0, 1.5):
            x = int(gx0 + v / 1.5 * (gx1 - gx0))
            put(img, f"{v:.1f}", (x - 10, gy + 32), 0.38, GREY)
        if dist is not None:
            x = int(gx0 + min(dist, 1.5) / 1.5 * (gx1 - gx0))
            cv2.rectangle(img, (x - 3, gy - 5), (x + 3, gy + 19), WHITE, -1)
            put(img, f"{dist:.2f}", (gx1 + 6, gy + 13), 0.5, WHITE)
        y = gy + 62
        if e is not None:
            put(img, f"rep {e['rep'] + 1}: assessment (Module III) {e['score']} ({e['quality']}) | "
                     f"suppl. model {e.get('model_score', float('nan')):.1f}", (10, y), 0.55, YELLOW)
        else:
            put(img, "no completed rep yet", (10, y), 0.5, GREY)
        # the trail as a table: the latest reps that fit
        rows = max(0, (H - (y + 40)) // 20)
        if sess.entries and rows:
            y += 32
            put(img, "rep   Module III   suppl. model   distance", (10, y), 0.45, GREY)
            for e in sess.entries[-rows:]:
                y += 20
                dd = (e.get("embedding_distance") or {}).get("cosine_centred")
                put(img, f"{e['rep'] + 1:>3}   {str(e['score']):>10}   {e.get('model_score', float('nan')):12.1f}   "
                         + (f"{dd:8.2f}" if dd is not None else "       -"), (10, y), 0.45,
                    YELLOW if e is sess.entries[-1] else WHITE)
        self._key, self._img = key, img
        return img
