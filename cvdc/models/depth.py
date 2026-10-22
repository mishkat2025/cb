"""DepthAnything V2 (Small) wrapper: relative depth at the skeleton's joints (proposal D7.5 Module II: "investigate
established monocular pose/depth estimation methods, such as DepthAnything and KTPFormer").

Model: Depth-Anything-V2-Small (Yang et al., NeurIPS 2024), code github.com/DepthAnything/Depth-Anything-V2 (commit
a561b84, Apache-2.0; a local clone in ``depth.code_dir``, gitignored like KTPFormer), weights
depth-anything/Depth-Anything-V2-Small ``depth_anything_v2_vits.pth`` (Apache-2.0; the Base / Large / Giant weights are
CC-BY-NC-4.0 and are not used). It predicts relative inverse depth (disparity: larger = closer), defined per image up
to scale and shift. Frames are resized so the short side is ``depth.input_size`` (both sides multiples of 14),
normalised with the ImageNet statistics as in the reference code, and the map is sampled bilinearly at every joint.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from ..config import resolve

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _sdpa_attention():
    """DINOv2's attention (third_party code) computes softmax(q k^T) v by hand when xFormers is missing (it is, on this
    Windows install); route it through torch's fused scaled_dot_product_attention instead: the same function, 2-3x
    faster here (checked against the original on real frames, depth_extract.py --check)."""
    from depth_anything_v2.dinov2_layers import attention as A

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

    A.Attention.forward = forward
    A.MemEffAttention.forward = forward


class DepthAnything:
    def __init__(self, dc, device, fused_attention=True):
        sys.path.insert(0, str(resolve(dc["code_dir"])))
        from depth_anything_v2.dpt import DepthAnythingV2

        if fused_attention:
            _sdpa_attention()
        self.model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        sd = torch.load(resolve(dc["weights"]), map_location="cpu", weights_only=True)
        self.model.load_state_dict(sd)
        self.model.to(device).eval()
        self.dev, self.size = device, int(dc["input_size"])
        self.mean, self.std = MEAN.to(device), STD.to(device)

    def target_hw(self, h, w):
        s = self.size / min(h, w)
        return max(14, int(round(h * s / 14)) * 14), max(14, int(round(w * s / 14)) * 14)

    @torch.no_grad()
    def __call__(self, frames):
        """frames: (B, H, W, 3) uint8 RGB numpy or tensor -> (B, h, w) float32 disparity on the device."""
        x = torch.as_tensor(frames).to(self.dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        h, w = self.target_hw(*x.shape[2:])
        x = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
        x = (x - self.mean) / self.std
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.dev.type == "cuda"):
            d = self.model(x)
        return d.float()

    @staticmethod
    def sample(disp, uv):
        """disp: (B, h, w); uv: (B, J, 2) normalised image coordinates in [0, 1] (NaN allowed) -> (B, J)."""
        g = torch.as_tensor(uv, dtype=torch.float32, device=disp.device) * 2 - 1
        ok = torch.isfinite(g).all(-1)
        g = torch.nan_to_num(g, 0.0)[:, :, None, :]  # (B, J, 1, 2), x then y
        v = F.grid_sample(disp[:, None], g, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]
        return torch.where(ok, v, torch.full_like(v, float("nan")))

    @staticmethod
    def spread(disp):
        """(B,) interquartile range of each disparity map (a per-image scale; every 4th pixel)."""
        q = torch.quantile(disp.flatten(1)[:, ::4], torch.tensor([0.25, 0.75], device=disp.device), dim=1)
        return q[1] - q[0]


def to_numpy(x):
    return x.detach().cpu().numpy().astype(np.float32)
