"""The full network: shared ST-GCN encoder -> embedding z; projection head (CVDC); task heads (supervised).

B1 = input xy without CVDC; B2 = input xyz (MediaPipe world landmarks) or xyz_ktpformer (KTPFormer lifted 3D)
without CVDC; Proposed = input xy with CVDC pretraining. All use this class with the same encoder size (Phase 3
trains them).
"""
import torch.nn as nn

from ..config import exercise_codes
from .heads import ProjectionHead, TaskHeads
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

    def forward(self, x, exercise=None, project=False, heads=True):
        z = self.encoder(x)
        out = {"z": z}
        if project:
            out["proj"] = self.projection(z)
        if heads:
            out.update(self.heads(z, exercise))
        return out
