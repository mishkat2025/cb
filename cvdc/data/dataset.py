"""Network inputs and labels as tensors: the fixed-length packs (cvdc/data/pack.py) joined with the video labels
of data/index.csv. Rep level (default, T1): every rep inherits its video's score, items and quality class as a
weak label. Video level: whole videos resampled to T_video.

Input modes (``model.input``, A1): ``xy`` (2D keypoints), ``xyv`` (2D + visibility), ``xyz`` (3D MediaPipe world
landmarks, B2) or ``xyz_ktpformer`` (3D lifted by KTPFormer from the 2D keypoints, B2; A10). The pack is small
(7,095 reps x 64 frames x 17 joints), so it is held on the device.
"""
import numpy as np
import pandas as pd
import torch

from ..config import exercise_codes, resolve

ITEMS = ["PF1", "PF2", "PF4", "CF1", "CF2", "CF3"]  # D5: PF3 is not a target


LIFTED = {"xyz_ktpformer": "ktpformer"}


def to_input(xy, vis, world, mode, lifted=None):
    """(N, T, V, *) arrays/tensors -> (N, T, V, C) for the input mode (``lifted``: the lifter's 3D)."""
    if mode == "xy":
        return xy
    if mode == "xyv":
        return torch.cat([xy, vis[..., None]], -1)
    if mode == "xyz":
        return world
    if mode in LIFTED:
        if lifted is None:
            raise ValueError(f"model.input {mode!r} needs the lifted pack: python -m cvdc.data.pack --lift {LIFTED[mode]}")
        return lifted
    raise ValueError(f"model.input {mode!r}")


class PackData:
    def __init__(self, cfg, unit="rep", device="cpu", lift=None):
        """lift: also load the lifted 3D pack of this lifter (e.g. 'ktpformer'); None = only if
        ``model.input`` needs it."""
        pc = cfg["preprocess"]
        T = pc["T_rep"] if unit == "rep" else pc["T_video"]
        name = f"{'reps' if unit == 'rep' else 'videos'}_{pc['joint_set']}_T{T}"
        pdir = resolve(pc["pack_dir"])
        meta = pd.read_csv(pdir / f"{name}.csv")
        if unit == "rep":
            reps = pd.read_csv(resolve("data/reps.csv"))
            if not (reps.video_id.values == meta.video_id.values).all():
                raise RuntimeError("data/reps.csv and the rep pack are out of sync")
            meta = reps
        idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
        lab = ["exercise", "subject", "group", "view", "variation", "gender", "group_id", "score", "quality", *ITEMS]
        meta = meta.drop(columns=[c for c in lab if c in meta.columns and c != "video_id"]).merge(
            idx[["video_id", *lab]], on="video_id", how="left", validate="many_to_one")
        self.unit, self.T, self.device, self.name = unit, T, device, name
        self.meta = meta.reset_index(drop=True)
        with np.load(pdir / f"{name}.npz") as z:
            self.xy, self.vis, self.world = (torch.from_numpy(z[k]).to(device) for k in ("xy", "vis", "world"))
        lift = lift or LIFTED.get(cfg["model"]["input"])
        self.lifted = None
        if lift:
            with np.load(pdir / f"{name}_{lift}.npz") as z:
                self.lifted = torch.from_numpy(z["world"]).to(device)
            if self.lifted.shape != self.world.shape:
                raise RuntimeError(f"lifted pack {name}_{lift} does not match the main pack")
        codes = exercise_codes(cfg)
        qn = cfg["labels"]["quality_names"]
        m = self.meta
        self.exercise = torch.tensor(m.exercise.map(codes.index).to_numpy(), device=device)
        self.quality = torch.tensor(m.quality.map(lambda q: qn.index(q) if q in qn else -1).to_numpy(), device=device)
        self.score = torch.tensor(m.score.to_numpy(np.float32), device=device)
        self.items = {k: torch.tensor(m[k].to_numpy(np.float32), device=device) for k in ITEMS}
        self.group_code = torch.tensor(pd.factorize(m.group_id)[0], device=device)  # (exercise, subject, variation)

    def __len__(self):
        return len(self.meta)

    def inputs(self, idx, mode):
        return to_input(self.xy[idx], self.vis[idx], self.world[idx], mode,
                        None if self.lifted is None else self.lifted[idx])

    def targets(self, idx, items):
        return {"score": self.score[idx], "quality": self.quality[idx], "exercise": self.exercise[idx],
                "items": torch.stack([self.items[k][idx] for k in items], -1) if items else None}

    def subject_mask(self, subjects):
        return torch.tensor(self.meta.subject.isin(set(subjects)).to_numpy(), device=self.device)
