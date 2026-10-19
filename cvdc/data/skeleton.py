"""MediaPipe Pose landmark names, joint sets and skeleton edges (P4).

The cache keeps all 33 MediaPipe landmarks; a joint set is selected when data is loaded.
``coco17`` keeps nose, eyes, ears, shoulders, elbows, wrists, hips, knees and ankles. Its graph is the
standard ST-GCN COCO layout (mmaction2/pyskl ``coco``: 16 edges, centre = nose), given here in joint-set
order; DRAW_EDGES adds the shoulder and hip lines for renders.
"""
MP33 = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
        "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
        "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"]
J = {n: i for i, n in enumerate(MP33)}

JOINT_SETS = {
    "mp33": list(range(33)),
    "coco17": [J[n] for n in ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                              "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
                              "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]],
}

# ST-GCN COCO layout in coco17 order (child, parent): mmaction2 / pyskl 'coco', centre joint 0 (nose)
COCO17_EDGES = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (9, 7), (7, 5), (10, 8), (8, 6),
                (5, 0), (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]
COCO17_CENTER = 0
COCO17_DRAW_EDGES = COCO17_EDGES + [(5, 6), (11, 12)]

# MediaPipe-index edges for drawing the body (face points other than nose/eyes/ears omitted)
MP_DRAW_EDGES = [(JOINT_SETS["coco17"][a], JOINT_SETS["coco17"][b]) for a, b in COCO17_DRAW_EDGES]
LEFT = {i for i, n in enumerate(MP33) if n.startswith("left")}
RIGHT = {i for i, n in enumerate(MP33) if n.startswith("right")}
COCO17_MIRROR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]  # left <-> right in coco17 order


def mirror_coco17(X):
    """Mirror image of root-centred coco17 keypoints (..., 17, C): x -> -x and left / right joints swapped (a left-arm
    performance seen as a right-arm one, P8)."""
    import numpy as np

    Y = np.asarray(X)[..., COCO17_MIRROR, :].copy()
    Y[..., 0] = -Y[..., 0]
    return Y
