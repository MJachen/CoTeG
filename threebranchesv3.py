import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler


from zerotoolfunc import *
from twognn import *
from twotrlv2 import ConvTRLNet
from metrics import *
from modality_groups import MODALITY_GROUPS, group_to_omega_row
from project_paths import find_train_pack_path, find_xfull_path, get_pack_dir

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ==============================
# 基本配置
# ==============================
torch.set_default_dtype(torch.float32)
torch.manual_seed(seed)


class TensorDataset3D(Dataset):
    def __init__(self, X_cpu, y_cpu, indices, base_idx_cpu=None):
        self.X = X_cpu
        self.y = y_cpu
        self.idx = indices.long()
        self.base = None if base_idx_cpu is None else base_idx_cpu.long()

    def __len__(self):
        return self.idx.numel()

    def __getitem__(self, k):
        i = int(self.idx[k])
        if self.base is None:
            return self.X[i], self.y[i], torch.tensor(-1, dtype=torch.long)
        return self.X[i], self.y[i], self.base[i]


class CompletedTrainDataset3D(Dataset):
    def __init__(self, x_full_ncdhw, x_hat_ncdhw, y_cpu, omega_rows, augment=True):
        self.x_full = x_full_ncdhw.float().contiguous()
        self.x_hat = x_hat_ncdhw.float().contiguous()
        self.y = y_cpu.long().contiguous()
        self.omega = omega_rows.float().view(omega_rows.shape[0], omega_rows.shape[1], 1, 1, 1)
        self.augment = bool(augment)
        self.num_groups = int(self.omega.shape[0]) if self.augment else 1
        self.labels = self.y.repeat_interleave(self.num_groups)
        self.base_idx = torch.arange(self.y.numel(), dtype=torch.long).repeat_interleave(self.num_groups)
        self.input_spatial = tuple(self.x_full.shape[2:])

    def __len__(self):
        return int(self.y.numel() * self.num_groups)

    def __getitem__(self, idx):
        base_i = int(idx) // self.num_groups
        group_i = int(idx) % self.num_groups
        omega = self.omega[group_i]
        x = self.x_full[base_i] * omega + self.x_hat[base_i] * (1.0 - omega)
        return x, self.y[base_i], torch.tensor(base_i, dtype=torch.long)


def make_modality_omega_rows():
    return torch.tensor([group_to_omega_row(present) for _, present in MODALITY_GROUPS], dtype=torch.float32)


def load_train_full_ncdhw(pack_dir, fac_train):
    if "idx_train_global" not in fac_train:
        raise KeyError("tucker_factors_train.pt missing idx_train_global; rerun GRNNTCtrain.py patient mode.")
    full = torch.load(find_xfull_path(pack_dir), map_location="cpu")
    x_full_all = full["X_full"].float()
    idx_train = fac_train["idx_train_global"].long()
    return x_full_all.index_select(0, idx_train).permute(0, 1, 4, 2, 3).contiguous()


def make_class_weights(y, num_classes, device):
    bc = torch.bincount(y.long().cpu(), minlength=num_classes).float()
    weights = bc.sum() / (num_classes * bc.clamp_min(1.0))
    return weights.to(device)



def train_tensor_branch_trl(
        X_tensor,          # (N,3,64,64,64) on CPU
        y,                 # (N,) on CPU
        num_classes,
        idx_train=None,    # 训练索引（增强样本）
        base_idx_cpu=None, # (N,) 同病例id，用于一致性正则（可选）
        epochs=200,
        lr=1e-4,
        batch_size=32,
        device=None,
        lambda_cons=0.1,   # 一致性正则权重
        focal_alpha=0.75,
        focal_gamma=2.0,
):
    if device is None:
        device = get_device()

    y = y.long().contiguous()

    if isinstance(X_tensor, Dataset):
        ds = X_tensor
        labels_for_sampler = ds.labels.long()
        input_spatial = ds.input_spatial
        loader_workers = int(os.environ.get("TENSOR_NUM_WORKERS", "0"))
    else:
        X_tensor = X_tensor.float().contiguous()
        if idx_train is None:
            idx_train = torch.arange(X_tensor.shape[0])
        ds = TensorDataset3D(X_tensor, y, idx_train, base_idx_cpu=base_idx_cpu)
        labels_for_sampler = y[idx_train].long()
        input_spatial = tuple(X_tensor.shape[2:])
        loader_workers = int(os.environ.get("TENSOR_NUM_WORKERS", "2"))

    # --------- (1) WeightedRandomSampler: 让每个 batch 类别更均衡 ----------
    y_sub = labels_for_sampler.cpu().numpy()
    class_count = np.bincount(y_sub, minlength=num_classes).astype(np.float32)
    class_weight = 1.0 / np.clip(class_count, 1.0, None)
    sample_weights = class_weight[y_sub]  # 每个样本一个权重

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,          # 用 sampler 就不要 shuffle
        num_workers=loader_workers,
        pin_memory=True,
        drop_last=False
    )

    model = ConvTRLNet(
        n_outputs=num_classes,
        ranks=(16, 4, 4, 4, 8),
        input_spatial=input_spatial,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # --------- (2) Focal Loss 替换 weighted CE ----------
    criterion = FocalLoss(alpha=make_class_weights(labels_for_sampler, num_classes, device), gamma=focal_gamma)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_n = 0

        for xb, yb, bid in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            bid = bid.to(device, non_blocking=True)  # base id

            optimizer.zero_grad()
            logits, z = model(xb, return_embed=True)   # z: (B,dim)

            cls_loss = criterion(logits, yb)

            # --------- (3) 同病例一致性正则（同 base_idx 的 embedding 应一致） ----------
            cons_loss = torch.tensor(0.0, device=device)
            if (bid >= 0).any():
                # 只在 batch 内有重复 base id 时才算
                # 简单稳妥：遍历 unique id（batch_size=32 可接受）
                uniq = torch.unique(bid)
                acc = 0.0
                cnt = 0
                for u in uniq:
                    if u.item() < 0:
                        continue
                    mask = (bid == u)
                    if mask.sum() >= 2:
                        zg = z[mask]                 # (k,dim)
                        mu = zg.mean(dim=0, keepdim=True)
                        acc = acc + (zg - mu).pow(2).mean()
                        cnt += 1
                if cnt > 0:
                    cons_loss = acc / cnt

            loss = cls_loss + lambda_cons * cons_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * yb.size(0)
            total_correct += int((logits.argmax(1) == yb).sum().item())
            total_n += int(yb.size(0))

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"[TRL Tensor Branch] Epoch {epoch:03d} | "
                  f"Loss={total_loss/max(total_n,1):.4e} | Acc={100*total_correct/max(total_n,1):.2f}% | "
                  f"lambda_cons={lambda_cons}")

    tensor_branch_out = os.environ.get("TENSOR_BRANCH_OUT", "./tensor_branch.pth")
    torch.save(model.state_dict(), tensor_branch_out)
    print("Saved", tensor_branch_out)
    return model


# ===================== 训练：GNN 分支 =====================

def train_gnn_branch(U_S, A, y, num_classes, epochs=200, lr=1e-3, device=None):
    if device is None:
        device = get_device()

    U_S = U_S.to(device)
    y = y.to(device)
    A = A.to(device)

    model = GNNBranch(
        in_dim=U_S.shape[1],
        hid=64,
        out_dim=32,
        num_classes=num_classes
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # class weights 用训练集统计
    criterion = nn.CrossEntropyLoss(weight=make_class_weights(y, num_classes, device))

    model.train()
    edge_index = A.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(U_S, edge_index)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            acc = accuracy_from_logits(logits, y)
            print(f"[GNN Branch]   Epoch {epoch:03d}, "
                  f"Loss={loss.item():.4f}, Acc={acc * 100:.2f}%")

    gnn_branch_out = os.environ.get("GNN_BRANCH_OUT", "./gnn_branch.pth")
    torch.save(model.state_dict(), gnn_branch_out)
    print("GNN branch saved as", gnn_branch_out)
    return model


# ===================== 主程序 =====================

if __name__ == "__main__":
    device = get_device()
    print("Using device:", device)
    PACK_DIR = get_pack_dir()
    TENSOR_EPOCHS = int(os.environ.get("TENSOR_BRANCH_EPOCHS", os.environ.get("BRANCH_EPOCHS", "200")))
    GNN_EPOCHS = int(os.environ.get("GNN_BRANCH_EPOCHS", os.environ.get("BRANCH_EPOCHS", "200")))
    TENSOR_BATCH_SIZE = int(os.environ.get("TENSOR_BATCH_SIZE", "32"))

    # ====== 加载数据 ======
    data = torch.load(os.path.join(PACK_DIR, "tucker_factors_train.pt"), map_location="cpu")
    U_N_all = data["U_N"]
    G = data["G"]
    X_true = torch.load(os.path.join(PACK_DIR, "X_hat_train_5d.pt"), map_location="cpu")
    y_train = data.get("y_train", None)
    pack = None
    train_pack_path = find_train_pack_path(PACK_DIR)
    if y_train is None and os.path.exists(train_pack_path):
        pack = torch.load(train_pack_path, map_location="cpu")
        y_train = pack["y"]
    if y_train is None:
        raise KeyError("Cannot find training labels. Expected y_train in tucker_factors_train.pt.")
    y_train = y_train.long()

    # 如果文件里有 base_idx 就用；没有就按每3条为同一病例推断
    base_idx = data.get("base_idx", None)
    if base_idx is None and pack is not None:
        base_idx = pack.get("base_idx", None)
    if base_idx is not None:
        base_idx = base_idx.long()

    X_hat_ncdhw = X_true.permute(4, 3, 2, 0, 1).contiguous()  # (N,4,D,H,W)
    if X_hat_ncdhw.size(0) != y_train.numel() or U_N_all.size(0) != y_train.numel():
        raise ValueError(
            f"Train N mismatch: X={X_hat_ncdhw.size(0)} U_N={U_N_all.size(0)} y={y_train.numel()}"
        )

    num_classes = int(y_train.max().item()) + 1
    # --- 特征标准化必须用 train 的统计量 ---
    mean = U_N_all.mean(0, keepdim=True)
    std = U_N_all.std(0, keepdim=True).clamp_min(1e-6)
    U_N_std = (U_N_all - mean) / std
    torch.save(
        {
            "u_mean": mean.cpu(),
            "u_std": std.cpu(),
            "train_modality_aug": os.environ.get("TRAIN_MODALITY_AUG", "1") != "0",
            "groups": [{"name": name, "present": present} for name, present in MODALITY_GROUPS],
            "tensor_input_spatial": tuple(X_hat_ncdhw.shape[2:]),
        },
        os.path.join(PACK_DIR, "branch_meta.pt"),
    )

    # ====== 图也只能基于训练集构建 ======
    edge_index_train = knn_graph_pytorch(U_N_std, k=10)

    train_modality_aug = os.environ.get("TRAIN_MODALITY_AUG", "1") != "0"
    if train_modality_aug:
        X_full_ncdhw = load_train_full_ncdhw(PACK_DIR, data)
        tensor_train_data = CompletedTrainDataset3D(
            X_full_ncdhw,
            X_hat_ncdhw,
            y_train,
            make_modality_omega_rows(),
            augment=True,
        )
        print("Tensor branch train data: on-the-fly modality groups, samples=", len(tensor_train_data))
    else:
        tensor_train_data = X_hat_ncdhw
        print("Tensor branch train data: patient-only X_hat samples=", X_hat_ncdhw.size(0))

    # ====== 预训练两个分支 ======
    gnn_model = train_gnn_branch(U_N_std, edge_index_train, y_train, num_classes, epochs=GNN_EPOCHS, lr=1e-4, device=device)
    tensor_model = train_tensor_branch_trl(
        X_tensor=tensor_train_data,
        y=y_train,
        num_classes=num_classes,
        base_idx_cpu=base_idx,
        epochs=TENSOR_EPOCHS,
        lr=1e-4,
        batch_size=TENSOR_BATCH_SIZE,
        device=device,
        lambda_cons=0.1,  # 可以先用 0.05~0.2 搜一下
        focal_alpha=0.75,
        focal_gamma=2.0
    )


