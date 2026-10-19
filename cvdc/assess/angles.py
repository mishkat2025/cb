"""Joint-angle math on keypoint arrays (MediaPipe 33-landmark indexing).

All functions take arrays shaped (..., 33, C) with C = 2 (image) or 3 (world) and return degrees.
Rep segmentation (Phase 1) uses ``shoulder_elevation`` for E01 and the non-angle ``wrist_path_pc1`` for E02;
Module III (Phase 4) assesses joint angles only: E01 ``shoulder_elevation``, E02 ``horizontal_adduction``
(M7; the full per-exercise angle set is added in Phase 4).
"""
import numpy as np

from ..data.skeleton import J

SIDES = {"right": ("right_shoulder", "right_elbow", "right_wrist", "right_hip"),
         "left": ("left_shoulder", "left_elbow", "left_wrist", "left_hip")}


def angle_between(u, v):
    """Unsigned angle between vectors (last axis), degrees in [0, 180]. NaN where a vector is zero/NaN."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    nu, nv = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.sum(u * v, axis=-1) / (nu * nv)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def joint_angle(P, a, b, c):
    """Angle at joint b formed by a-b-c (e.g. elbow flexion: shoulder-elbow-wrist; 180 = straight)."""
    return angle_between(P[..., J[a], :] - P[..., J[b], :], P[..., J[c], :] - P[..., J[b], :])


def mid(P, a, b):
    return (P[..., J[a], :] + P[..., J[b], :]) / 2


def shoulder_elevation(P, side="right"):
    """Angle between the upper arm (shoulder->elbow) and the trunk line (mid-shoulder->mid-hip).

    0 deg = arm hanging along the trunk, 90 = horizontal, 180 = overhead. Independent of the plane of
    elevation, so it measures abduction in the frontal plane and flexion in the sagittal plane alike.
    """
    sh, el = SIDES[side][0], SIDES[side][1]
    trunk = mid(P, "left_hip", "right_hip") - mid(P, "left_shoulder", "right_shoulder")
    return angle_between(P[..., J[el], :] - P[..., J[sh], :], trunk)


def trunk_frame(P):
    """Per-frame unit axes of the trunk: lat (left -> right shoulder), up (hips -> shoulders, orthogonal to
    lat) and fwd = up x lat. Independent of the camera and of the landmark axis conventions."""
    lat = P[..., J["right_shoulder"], :] - P[..., J["left_shoulder"], :]
    lat = lat / np.linalg.norm(lat, axis=-1, keepdims=True)
    up = mid(P, "left_shoulder", "right_shoulder") - mid(P, "left_hip", "right_hip")
    up = up - np.sum(up * lat, axis=-1, keepdims=True) * lat
    up = up / np.linalg.norm(up, axis=-1, keepdims=True)
    return lat, up, np.cross(up, lat)


def arm_frontal_angle(P, side="right", distal="wrist"):
    """Signed arm angle in the trunk's frontal plane (3D only): shoulder->distal vector projected on
    (outward, up); 0 = hanging along the trunk, + = abducted outward, - = adducted across the midline."""
    lat, up, _ = trunk_frame(P)
    d = P[..., J[f"{side}_{distal}"], :] - P[..., J[f"{side}_shoulder"], :]
    out = lat if side == "right" else -lat
    return np.degrees(np.arctan2(np.sum(d * out, axis=-1), -np.sum(d * up, axis=-1)))


def horizontal_adduction(P, side="right", min_proj=0.0):
    """3D horizontal adduction: the upper arm (shoulder->elbow) projected on the trunk's transverse plane,
    angle from the outward shoulder line, positive toward the front. 0 = straight out to the side,
    90 = straight forward, > 90 = across the body (toward the midline), < 0 = behind the shoulder line.
    The angle is unstable when the arm hangs nearly vertical; frames whose projected length is below
    ``min_proj`` x the upper-arm length (min_proj = sin of the elevation from vertical) are NaN."""
    lat, up, fwd = trunk_frame(P)
    u = P[..., J[f"{side}_elbow"], :] - P[..., J[f"{side}_shoulder"], :]
    out = lat if side == "right" else -lat
    a, f = np.sum(u * out, axis=-1), np.sum(u * fwd, axis=-1)
    ang = np.degrees(np.arctan2(f, a))
    if min_proj > 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.hypot(a, f) / np.linalg.norm(u, axis=-1)
        ang = np.where(frac >= min_proj, ang, np.nan)
    return ang


def other(side):
    return "left" if side == "right" else "right"


def elbow_extension(P, side="right"):
    """Elbow angle shoulder-elbow-wrist: 180 = straight arm, smaller = bent."""
    return joint_angle(P, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")


def contralateral_elevation(P, side="right"):
    """Shoulder elevation of the arm that should stay at the side (the other arm of ``side``)."""
    return shoulder_elevation(P, other(side))


def trunk_lateral_lean(P, side="right"):
    """Signed lateral trunk lean (3D): 90 - the angle between the trunk axis (mid-hip -> mid-shoulder) and the hip
    line pointing to ``side`` (other hip -> ``side`` hip). 0 = trunk perpendicular to the hip line; + = leaning
    toward ``side``. Forward flexion and axial rotation keep the trunk perpendicular to the hip line, so only the
    lateral (frontal-plane) component moves it."""
    trunk = mid(P, "left_shoulder", "right_shoulder") - mid(P, "left_hip", "right_hip")
    hips = P[..., J[f"{side}_hip"], :] - P[..., J[f"{other(side)}_hip"], :]
    return 90.0 - angle_between(trunk, hips)


def shoulder_asymmetry(P, side="right"):
    """Left/right shoulder-level asymmetry (3D): elevation of the ``side`` shoulder over the other one along the trunk
    axis, as the angle of the shoulder line (other -> ``side`` shoulder) above the plane perpendicular to the trunk
    axis. 0 = level shoulders; + = the ``side`` shoulder is hitched up (shrug). A trunk lean tilts the trunk axis with
    the shoulder line, so it does not change this angle."""
    trunk = mid(P, "left_shoulder", "right_shoulder") - mid(P, "left_hip", "right_hip")
    sh = P[..., J[f"{side}_shoulder"], :] - P[..., J[f"{other(side)}_shoulder"], :]
    return 90.0 - angle_between(sh, trunk)


def wrist_path_pc1(P, side="right"):
    """Not an angle: the wrist position relative to the shoulder in the trunk frame (torso units after
    preprocessing), projected on its first principal component over the video, oriented so that moving the
    wrist toward the body midline (adduction) is positive. Rep signal of E02, whose cross-body movement
    changes direction with the camera view."""
    lat, up, fwd = trunk_frame(P)
    d = P[..., J[f"{side}_wrist"], :] - P[..., J[f"{side}_shoulder"], :]
    out = lat if side == "right" else -lat
    r = np.stack([np.sum(d * out, -1), np.sum(d * up, -1), np.sum(d * fwd, -1)], -1)  # outward, up, forward
    ok = np.isfinite(r).all(-1)
    s = np.full(len(r), np.nan)
    if ok.sum() < 3:
        return s
    z = r[ok] - r[ok].mean(0)
    s[ok] = z @ np.linalg.svd(z, full_matrices=False)[2][0]
    return -s if np.corrcoef(s[ok], r[ok, 0])[0, 1] > 0 else s  # outward motion -> negative


PRIMARY_ANGLES = {"shoulder_elevation": shoulder_elevation, "arm_frontal_angle": arm_frontal_angle,
                  "horizontal_adduction": horizontal_adduction, "wrist_path_pc1": wrist_path_pc1}
UNITS = {"shoulder_elevation": "deg", "arm_frontal_angle": "deg", "horizontal_adduction": "deg",
         "wrist_path_pc1": "torso"}
# Module III channels (Phase 4, M7): every function takes (P, side) with side = the moving arm, so a left-arm
# performance is assessed as its mirror image (P8).
ASSESS_ANGLES = {"shoulder_elevation": shoulder_elevation, "horizontal_adduction": horizontal_adduction,
                 "elbow_extension": elbow_extension, "contralateral_elevation": contralateral_elevation,
                 "trunk_lateral_lean": trunk_lateral_lean, "shoulder_asymmetry": shoulder_asymmetry}


def primary_angle(P, name, side):
    return PRIMARY_ANGLES[name](P, side)
