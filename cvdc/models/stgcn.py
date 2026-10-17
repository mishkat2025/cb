"""ST-GCN encoder (Yan et al. 2018, "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action
Recognition"), following the reference implementation (github.com/yysijie/st-gcn, net/st_gcn.py) without its
classifier: input BatchNorm over (joint, channel), 10 ST-GCN blocks (spatial graph convolution with the
partitioned adjacency, then a temporal convolution, with residuals and learnable edge importance), global
average pooling over time and joints -> embedding (default 256-d).

Views share the encoder: every view of a sample is a separate item in the batch and goes through the same
weights (C1); there is no per-view parameter.

Input x: (N, C, T, V) with C = 2 (x, y), 3 (x, y, visibility) or 3 (x, y, z), T frames, V joints.
"""
import torch
import torch.nn as nn

from .graph import Graph


class GraphConv(nn.Module):
    """Spatial graph convolution: a 1x1 convolution to K x C_out channels, then a sum over the K partition
    subsets of the adjacency: out[n, c, t, w] = sum_k sum_v x_k[n, c, t, v] A[k, v, w]."""

    def __init__(self, c_in, c_out, k):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(c_in, c_out * k, kernel_size=1)

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.shape
        x = x.view(n, self.k, kc // self.k, t, v)
        return torch.einsum("nkctv,kvw->nctw", x, A).contiguous()


class STGCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k, t_kernel=9, stride=1, dropout=0.0, residual=True):
        super().__init__()
        pad = (t_kernel - 1) // 2
        self.gcn = GraphConv(c_in, c_out, k)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, (t_kernel, 1), (stride, 1), (pad, 0)),
            nn.BatchNorm2d(c_out), nn.Dropout(dropout, inplace=True))
        if not residual:
            self.residual = None
        elif c_in == c_out and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(nn.Conv2d(c_in, c_out, 1, (stride, 1)), nn.BatchNorm2d(c_out))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = 0 if self.residual is None else self.residual(x)
        return self.relu(self.tcn(self.gcn(x, A)) + res)


class STGCN(nn.Module):
    """ST-GCN encoder: (N, C, T, V) -> (N, embed_dim)."""

    def __init__(self, in_channels=2, graph=None, channels=(64, 64, 64, 64, 128, 128, 128, 256, 256, 256),
                 strides=(1, 1, 1, 1, 2, 1, 1, 2, 1, 1), temporal_kernel=9, dropout=0.0, edge_importance=True,
                 embed_dim=256):
        super().__init__()
        graph = graph or Graph()
        A = torch.tensor(graph.A, dtype=torch.float32)
        self.register_buffer("A", A)
        K, V = A.shape[0], A.shape[1]
        self.in_channels, self.num_joints = in_channels, V
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        widths = [in_channels, *channels]
        self.blocks = nn.ModuleList(
            STGCNBlock(widths[i], widths[i + 1], K, temporal_kernel, strides[i],
                       dropout=0.0 if i == 0 else dropout, residual=i > 0)
            for i in range(len(channels)))
        self.edge_importance = nn.ParameterList(
            nn.Parameter(torch.ones_like(A), requires_grad=edge_importance) for _ in self.blocks)
        self.out = nn.Identity() if embed_dim == channels[-1] else nn.Linear(channels[-1], embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        N, C, T, V = x.shape
        x = self.data_bn(x.permute(0, 3, 1, 2).reshape(N, V * C, T))
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        for block, imp in zip(self.blocks, self.edge_importance):
            x = block(x, self.A * imp)
        return self.out(x.mean(dim=(2, 3)))

    @classmethod
    def from_config(cls, mc, in_channels):
        g = mc["graph"]
        s = mc["stgcn"]
        return cls(in_channels=in_channels, graph=Graph(g["layout"], g["strategy"], g["max_hop"]),
                   channels=s["channels"], strides=s["strides"], temporal_kernel=s["temporal_kernel"],
                   dropout=s["dropout"], edge_importance=s["edge_importance"], embed_dim=mc["embed_dim"])
