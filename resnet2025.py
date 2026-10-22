import os
import json
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
)
from project_paths import find_xfull_path, get_pack_dir

# =========================
# MONAI ResNet
# =========================
try:
    from monai.networks.nets import ResNet
except ImportError:
    raise ImportError("请先安装 MONAI: pip install monai")

# =========================
# Reproducibility
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==========================================================
# LRTC (HaLRTC) —— 直接复用你 resnetLRTC.py 的实现风格 :contentReference[oaicite:2]{index=2}
# ==========================================================
def _unfold(x: torch.Tensor, mode: int) -> torch.Tensor:
    dims = list(range(x.ndim))
    perm = [mode] + [d for d in dims if d != mode]
    x_perm = x.permute(perm).contiguous()
    return x_perm.view(x.shape[mode], -1)

def _fold(mat: torch.Tensor, mode: int, shape) -> torch.Tensor:
    ndim = len(shape)
    dims = list(range(ndim))
    perm = [mode] + [d for d in dims if d != mode]
    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i
    x = mat.view([shape[mode]] + [shape[d] for d in dims if d != mode])
    return x.permute(inv_perm).contiguous()

def _svt(mat: torch.Tensor, tau: float) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    S_thr = torch.clamp(S - tau, min=0.0)
    return (U * S_thr.unsqueeze(0)) @ Vh

def lrtc_halrtc(
    X_obs: torch.Tensor,     # (C,H,W,D) 缺失处随意（我们用0）
    Omega: torch.Tensor,     # (C,H,W,D) 1观测 0缺失
    alpha=None,
    rho: float = 1.0,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> torch.Tensor:
    assert X_obs.ndim == 4 and Omega.shape == X_obs.shape, f"{X_obs.shape}, {Omega.shape}"
    shape = tuple(X_obs.shape)
    K = X_obs.ndim  # 4

    if alpha is None:
        alpha = [1.0 / K] * K
    else:
        assert len(alpha) == K

    X = X_obs.clone()
    M = [X.clone() for _ in range(K)]
    Y = [torch.zeros_like(X) for _ in range(K)]

    for _ in range(max_iter):
        X_prev = X

        for k in range(K):
            Zk = X + (1.0 / rho) * Y[k]
            Mk = _svt(_unfold(Zk, k), tau=alpha[k] / rho)
            M[k] = _fold(Mk, k, shape)

        X = sum(M[k] - (1.0 / rho) * Y[k] for k in range(K)) / K
        X = Omega * X_obs + (1.0 - Omega) * X

        for k in range(K):
            Y[k] = Y[k] + rho * (X - M[k])

        rel = (X - X_prev).norm() / (X_prev.norm().clamp_min(1e-8))
        if rel.item() < tol:
            break

    return X

def lrtc_complete_batch(
    x_miss: torch.Tensor,   # (B,C,H,W,D)
    omega: torch.Tensor,    # (B,C,1,1,1) 或 (B,C,H,W,D)
    max_iter: int = 20,
    rho: float = 1.0,
) -> torch.Tensor:
    if omega.ndim == 5 and omega.shape[2:] == (1, 1, 1):
        omega_full = omega.expand(-1, -1, x_miss.shape[2], x_miss.shape[3], x_miss.shape[4])
    else:
        omega_full = omega

    B = x_miss.shape[0]
    out = torch.empty_like(x_miss)

    # baseline：逐样本补全（最稳定、最接近你原脚本） :contentReference[oaicite:3]{index=3}
    for b in range(B):
        Xb = x_miss[b]
        Ob = omega_full[b]
        out[b] = lrtc_halrtc(Xb, Ob, rho=rho, max_iter=max_iter, tol=1e-4)
    return out

# =========================
# Missing 模态生成
# =========================
def apply_random_missing_k_with_omega(x: torch.Tensor, missing_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (B,C,H,W,D)
    missing_k: 每样本丢 k 个模态（永不全丢）
    返回：
      x_miss: (B,C,H,W,D) 缺失模态置0
      omega : (B,C,1,1,1) 1=观测模态, 0=缺失模态
    """
    B, C = x.shape[:2]
    x_miss = x.clone()
    omega = torch.ones((B, C, 1, 1, 1), device=x.device, dtype=x.dtype)

    if missing_k <= 0:
        return x_miss, omega

    k = min(int(missing_k), C - 1)
    for b in range(B):
        drop_idx = torch.randperm(C, device=x.device)[:k]
        x_miss[b, drop_idx] = 0.0
        omega[b, drop_idx] = 0.0
    return x_miss, omega

# =========================
# Mean 补全（复用你 resnetmean.py 的思路 :contentReference[oaicite:4]{index=4}）
# =========================
def mean_impute(x_miss: torch.Tensor, omega: torch.Tensor, channel_means: torch.Tensor) -> torch.Tensor:
    # omega: (B,C,1,1,1)
    B, C = x_miss.shape[:2]
    out = x_miss.clone()
    miss = (omega == 0).view(B, C)  # (B,C)
    for b in range(B):
        for c in range(C):
            if miss[b, c]:
                out[b, c] = channel_means[c]
    return out

# =========================
# Metrics
# =========================
@dataclass
class Metrics:
    acc: float
    bal_acc: float
    f1_pos: float
    f1_macro: float
    auc: float
    ap: float
    sen: float
    spe: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "ACC": self.acc,
            "BACC": self.bal_acc,
            "F1": self.f1_pos,
            "F1M": self.f1_macro,
            "AUC": self.auc,
            "AP": self.ap,
            "SEN": self.sen,
            "SPE": self.spe,
        }

def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob[:, 1]))
    except Exception:
        return float("nan")

def _safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_prob[:, 1]))
    except Exception:
        return float("nan")

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Metrics:
    acc = float(accuracy_score(y_true, y_pred))
    f1_pos = float(f1_score(y_true, y_pred, average="binary", pos_label=1))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    sen = float(recall_score(y_true, y_pred, pos_label=1))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spe = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    bal_acc = float(0.5 * (sen + spe)) if (not np.isnan(spe)) else float("nan")
    auc = _safe_auc(y_true, y_prob)
    ap = _safe_ap(y_true, y_prob)
    return Metrics(acc=acc, bal_acc=bal_acc, f1_pos=f1_pos, f1_macro=f1_macro, auc=auc, ap=ap, sen=sen, spe=spe)

# =========================
# Sampler
# =========================
def build_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels)
    inv = 1.0 / np.maximum(counts, 1)
    sample_weights = inv[labels].astype(np.float64)
    sample_weights = torch.from_numpy(sample_weights)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# =========================
# 计算 train 通道均值
# =========================
@torch.no_grad()
def compute_channel_mean(train_loader: DataLoader, device: torch.device) -> torch.Tensor:
    sums = None
    counts = None
    for x, _ in train_loader:
        x = x.to(device)  # (B,C,H,W,D)
        B, C = x.shape[:2]
        vox = x.shape[2] * x.shape[3] * x.shape[4]

        if sums is None:
            sums = torch.zeros(C, device=device, dtype=torch.float32)
            counts = torch.zeros(C, device=device, dtype=torch.float32)

        sums += x.sum(dim=(0, 2, 3, 4))
        counts += float(B * vox)

    return sums / counts.clamp_min(1.0)

# =========================
# 评估：固定 missing_k，多次重复
# =========================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    method: str,
    missing_k: int,
    channel_means: Optional[torch.Tensor],
    lrtc_iter: int,
    repeats: int = 10,
) -> Tuple[Metrics, Metrics]:

    model.eval()
    runs: List[Metrics] = []

    for _ in range(repeats):
        y_true_all: List[int] = []
        y_pred_all: List[int] = []
        y_prob_all: List[np.ndarray] = []

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            x_miss, omega = apply_random_missing_k_with_omega(x, missing_k)

            if method == "zero":
                x_in = x_miss
            elif method == "mean":
                assert channel_means is not None
                x_in = mean_impute(x_miss, omega, channel_means)
            elif method == "lrtc":
                # LRTC 在 GPU 上做 SVD 可能更快/更稳（取决于环境）
                x_in = lrtc_complete_batch(x_miss, omega, max_iter=lrtc_iter, rho=1.0)
            else:
                raise ValueError("method must be zero/mean/lrtc")

            logits = model(x_in)
            prob = torch.softmax(logits, dim=1)
            p1 = prob[:, 1]
            y_pred = (p1 >= 0.5).long()

            y_true_all.extend(y.cpu().numpy().tolist())
            y_pred_all.extend(y_pred.cpu().numpy().tolist())
            y_prob_all.append(prob.cpu().numpy())

        y_true_np = np.asarray(y_true_all, dtype=np.int64)
        y_pred_np = np.asarray(y_pred_all, dtype=np.int64)
        y_prob_np = np.concatenate(y_prob_all, axis=0)
        runs.append(compute_metrics(y_true_np, y_pred_np, y_prob_np))

    keys = list(runs[0].as_dict().keys())
    vals = {k: np.array([m.as_dict()[k] for m in runs], dtype=np.float64) for k in keys}

    def nm(key): return float(np.nanmean(vals[key]))
    def ns(key): return float(np.nanstd(vals[key]))

    mean = Metrics(acc=nm("ACC"), bal_acc=nm("BACC"), f1_pos=nm("F1"), f1_macro=nm("F1M"),
                   auc=nm("AUC"), ap=nm("AP"), sen=nm("SEN"), spe=nm("SPE"))
    std  = Metrics(acc=ns("ACC"), bal_acc=ns("BACC"), f1_pos=ns("F1"), f1_macro=ns("F1M"),
                   auc=ns("AUC"), ap=ns("AP"), sen=ns("SEN"), spe=ns("SPE"))
    return mean, std

# =========================
# 训练
# =========================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    method: str,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    channel_means: Optional[torch.Tensor],
    lrtc_iter_train: int,
    train_missing_ks: Sequence[int],
):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # 训练时随机采样 missing_k（例如 [0,0,0,1,1,2,3] 与你脚本一致 :contentReference[oaicite:5]{index=5}）
        missing_k = int(random.choice(train_missing_ks))
        x_miss, omega = apply_random_missing_k_with_omega(x, missing_k)

        if method == "zero":
            x_in = x_miss
        elif method == "mean":
            assert channel_means is not None
            x_in = mean_impute(x_miss, omega, channel_means)
        elif method == "lrtc":
            x_in = lrtc_complete_batch(x_miss, omega, max_iter=lrtc_iter_train, rho=1.0)
        else:
            raise ValueError("method must be zero/mean/lrtc")

        logits = model(x_in)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))

# =========================
# Main
# =========================
def main():
    # ======= 你只需要改这里 =======
    PACK_DIR = get_pack_dir()
    METHOD = "lrtc"     # zero / mean / lrtc
    BATCH_SIZE = 4
    EPOCHS = 30
    LR = 1e-4
    USE_WEIGHTED_SAMPLER = True

    # 训练时的缺失强度分布（与你原脚本保持一致风格） :contentReference[oaicite:6]{index=6}
    TRAIN_MISSING_KS = (0, 0, 0, 1, 1, 2, 3)

    # LRTC 迭代次数（baseline 重要控制变量）
    LRTC_ITER_TRAIN = 5    # 训练时建议小一点，否则太慢
    LRTC_ITER_EVAL  = 20   # 测试时可用 20（与你脚本一致） :contentReference[oaicite:7]{index=7}
    EVAL_REPEATS = 10

    # =================================
    # 1) Load X_full + split indices
    # =================================
    xfull_path = find_xfull_path(PACK_DIR)
    split_path = os.path.join(PACK_DIR, "split.json")

    full = torch.load(xfull_path, map_location="cpu")
    X = full["X_full"].float()  # (N,4,H,W,D)
    Y = full["y"].long()

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    idx_train = np.array(split["idx_train"], dtype=np.int64)
    idx_val   = np.array(split["idx_val"], dtype=np.int64)
    idx_test  = np.array(split["idx_test"], dtype=np.int64)

    # 二分类检查
    uniq = torch.unique(Y)
    if len(uniq) != 2:
        raise RuntimeError(f"当前标签不是二分类：unique={uniq.tolist()}，请检查 pack。")

    dataset = TensorDataset(X, Y)
    train_ds = Subset(dataset, idx_train.tolist())
    val_ds   = Subset(dataset, idx_val.tolist())
    test_ds  = Subset(dataset, idx_test.tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # class weights
    y_train_np = Y[idx_train].numpy()
    counts = np.bincount(y_train_np, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.sum() * 2
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # loaders
    if USE_WEIGHTED_SAMPLER:
        sampler = build_weighted_sampler(y_train_np)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # =================================
    # 2) channel_means for mean baseline
    # =================================
    channel_means = None
    if METHOD == "mean":
        channel_means = compute_channel_mean(train_loader, device)
        print("Channel means:", channel_means.detach().cpu().numpy().tolist())

    # =================================
    # 3) Build model
    # =================================
    model = ResNet(
        block="basic",
        layers=[2, 2, 2, 2],
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=4,
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # =================================
    # 4) Train + select best by Val BACC
    # =================================
    best_bacc = -1.0
    best_path = os.path.join(PACK_DIR, f"best_resnet_{METHOD}.pt")

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            method=METHOD,
            criterion=criterion,
            optimizer=optimizer,
            channel_means=channel_means,
            lrtc_iter_train=LRTC_ITER_TRAIN,
            train_missing_ks=TRAIN_MISSING_KS,
        )

        val_mean, _ = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            method=METHOD,
            missing_k=0,
            channel_means=channel_means,
            lrtc_iter=LRTC_ITER_EVAL,
            repeats=1,
        )

        if not np.isnan(val_mean.bal_acc) and val_mean.bal_acc > best_bacc:
            best_bacc = val_mean.bal_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "bacc": best_bacc}, best_path)

        print(
            f"Epoch {epoch:03d} | TrainLoss {tr_loss:.4f} | "
            f"Val(0-miss) ACC {val_mean.acc:.4f} BACC {val_mean.bal_acc:.4f} "
            f"F1 {val_mean.f1_pos:.4f} AUC {val_mean.auc:.4f} | "
            f"BestBACC {best_bacc:.4f}"
        )

    # load best
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"\nLoaded best checkpoint: {best_path} (epoch={ckpt.get('epoch')}, bacc={ckpt.get('bacc')})")

    # =================================
    # 5) Final test under missing_k=0..3
    # =================================
    print("\n--- Final Evaluation on Test (missing_k=0..3) ---")
    print(f"Method = {METHOD} | repeats={EVAL_REPEATS} | LRTC(eval_iter)={LRTC_ITER_EVAL}")
    header = (
        f"{'missing_k':>9} | {'ACC':>18} | {'BACC':>18} | {'F1(pos)':>18} | {'F1(macro)':>18} | "
        f"{'AUC':>18} | {'AP':>18} | {'SEN':>18} | {'SPE':>18}"
    )
    print(header)
    print("-" * len(header))

    def fmt(m, s):
        if np.isnan(m) or np.isnan(s):
            return "nan"
        return f"{m:.4f}±{s:.4f}"

    for k in [0, 1, 2, 3]:
        mean_m, std_m = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            method=METHOD,
            missing_k=k,
            channel_means=channel_means,
            lrtc_iter=LRTC_ITER_EVAL,
            repeats=EVAL_REPEATS,
        )
        print(
            f"{k:>9d} | "
            f"{fmt(mean_m.acc, std_m.acc):>18} | "
            f"{fmt(mean_m.bal_acc, std_m.bal_acc):>18} | "
            f"{fmt(mean_m.f1_pos, std_m.f1_pos):>18} | "
            f"{fmt(mean_m.f1_macro, std_m.f1_macro):>18} | "
            f"{fmt(mean_m.auc, std_m.auc):>18} | "
            f"{fmt(mean_m.ap, std_m.ap):>18} | "
            f"{fmt(mean_m.sen, std_m.sen):>18} | "
            f"{fmt(mean_m.spe, std_m.spe):>18}"
        )

if __name__ == "__main__":
    main()
