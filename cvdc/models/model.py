"""The full network: shared ST-GCN encoder -> embedding z; projection head (CVDC); task heads (supervised).

B1 = input xy without CVDC; B2 = input xyz (MediaPipe world landmarks) or xyz_ktpformer (KTPFormer lifted 3D)
without CVDC; Proposed = input xy with CVDC pretraining. All use this class with the same encoder size (Phase 3
trains them). With ``heads.video_head.enabled`` (step a2, T13) the model also has a set-level video head on the
pooled rep embeddings of each video (``video``).
"""
import torch.nn as nn

from ..config import exercise_codes
from .heads import ProjectionHead, TaskHeads, VideoHead
from .stgcn import STGCN

INPUT_CHANNELS = {"xy": 2, "xyv": 3, "xyz": 3, "xyz_ktpformer": 3}


class CVDCModel(nn.Module):
    def __init__(self, cfg, unit="rep"):
        super().__init__()
        mc, hc = cfg["model"], cfg["heads"]
        self.input = mc["input"]
        self.encoder = STGCN.from_config(mc, INPUT_CHANNELS[self.input])
        self.projection = ProjectionHead(mc["embed_dim"], **mc["projection"])
        self.exercises = exercise_codes(cfg)
        self.heads = TaskHeads(mc["embed_dim"], len(self.exercises), hc[f"targets_{unit}"],
                               ex_dim=mc["exercise_embedding"], hidden=hc["hidden"], dropout=hc["dropout"],
                               n_quality=len(cfg["labels"]["quality_names"]))
        # T13: set-level head over the rep embeddings of a video (rep-level training only). Built after the encoder
        # and the rep heads, so their random initialisation for a given seed is the same as in step (a).
        vc = hc.get("video_head") or {}
        self.video_head = None
        if vc.get("enabled") and unit == "rep":
            self.video_head = VideoHead(mc["embed_dim"], len(self.exercises), vc["targets"], vc["pooling"],
                                        vc["attn_hidden"], vc["count_feature"], vc["exercise_embedding"])

    def forward(self, x, exercise=None, project=False, heads=True):
        z = self.encoder(x)
        out = {"z": z}
        if project:
            out["proj"] = self.projection(z)
        if heads:
            out.update(self.heads(z, exercise))
        return out

    def video(self, z, seg, n, count=None, exercise=None):
        """Video head outputs for rep embeddings z grouped by seg (T13)."""
        if self.video_head is None:
            raise RuntimeError("heads.video_head is not enabled")
        return self.video_head(z, seg, n, count, exercise)
