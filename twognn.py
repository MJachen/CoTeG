import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv


# ===================== GNN 分支 =====================

class GNNBranch(nn.Module):
    def __init__(self, in_dim=16, hid=64, out_dim=32, num_classes=2):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hid)
        )

        self.conv1 = SAGEConv(hid, hid)
        self.conv2 = SAGEConv(hid, out_dim)

        self.norm1 = nn.LayerNorm(hid)
        self.norm2 = nn.LayerNorm(out_dim)

        self.cls = nn.Linear(out_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.mlp(x)

        h = F.relu(self.norm1(self.conv1(x, edge_index)))
        h = F.relu(self.norm2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=0.5, training=self.training)
        logits = self.cls(h)
        return logits,h


# =========================
# Gate Fusion
# =========================
class GateFusion(nn.Module):
    def __init__(self, dim_t, dim_g, hidden):
        super().__init__()
        self.proj_t = nn.Linear(dim_t, hidden)
        self.proj_g = nn.Linear(dim_g, hidden)
        self.gate   = nn.Linear(hidden * 2, 1)

    def forward(self, ft, fg):
        ft = self.proj_t(ft)
        fg = self.proj_g(fg)
        alpha = torch.sigmoid(self.gate(torch.cat([ft, fg], dim=1)))
        fused = alpha * ft + (1 - alpha) * fg
        return fused, alpha


class FusionNet(nn.Module):
    def __init__(self, dim_t, dim_g, hidden, num_classes):
        super().__init__()
        self.fusion = GateFusion(dim_t, dim_g, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, ft, fg):
        fused, gate = self.fusion(ft, fg)
        return self.head(fused), gate