import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors

from monai.networks.nets import ResNet
from project_paths import find_xfull_path, get_pack_dir

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


# =========================
# Metrics
# =========================
def compute_metrics_binary(y_true, y_pred, y_prob_pos):
    """
    二分类指标（默认 label=1 为阳性类：MET）
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob_pos = np.asarray(y_prob_pos, dtype=np.float64)

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    try:
        auc = float(roc_auc_score(y_true, y_prob_pos))
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    return {"ACC": acc, "F1_macro": f1_macro, "AUC": auc, "SENS": sens, "SPEC": spec}


# =========================
# Missing 模态生成：每样本丢 k 个模态，永不全丢
# =========================
def apply_missing_k(x, k: int):
    """
    x: (B,4,H,W,D)
    return:
      x_miss: (B,4,H,W,D) 缺失模态置0
      mask:   (B,4) 1=observed, 0=missing
    """
    B, C = x.shape[:2]
    x_miss = x.clone()
    mask = torch.ones((B, C), device=x.device, dtype=torch.float32)

    k = int(k)
    k = max(0, min(k, C - 1))
    if k == 0:
        return x_miss, mask

    for b in range(B):
        drop = torch.randperm(C, device=x.device)[:k]
        x_miss[b, drop] = 0.0
        mask[b, drop] = 0.0
    return x_miss, mask


# =========================
# KNN Imputation
# =========================
class KNNImputer3D:
    """
    用训练集(完整样本)做 KNN，为缺失模态做邻居均值填充。
    特征：每模态体素均值 (N,4)
    """
    def __init__(self, X_train_full: torch.Tensor, k_neighbors: int = 5):
        """
        X_train_full: (N_train,4,H,W,D) torch.FloatTensor (CPU/或GPU都可，但内部会转CPU算特征)
        """
        self.X_train = X_train_full.cpu().float()  # 保持在CPU，避免knn阶段频繁GPU<->CPU
        self.k = int(k_neighbors)

        feats = self.X_train.mean(dim=(2, 3, 4))  # (N_train,4)
        self.features = feats.numpy()

        self.nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        self.nn.fit(self.features)

    def impute(self, x_miss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x_miss: (B,4,H,W,D)  (建议CPU)
        mask:   (B,4) float {0,1} (CPU)
        return: (B,4,H,W,D)
        """
        x_miss_cpu = x_miss.cpu().float()
        mask_cpu = mask.cpu().float()
        B = x_miss_cpu.shape[0]

        out = x_miss_cpu.clone()

        feat = x_miss_cpu.mean(dim=(2, 3, 4)).numpy()  # (B,4)
        _, idxs = self.nn.kneighbors(feat)             # (B,k)

        for b in range(B):
            miss_c = (mask_cpu[b] == 0).nonzero(as_tuple=False).view(-1).tolist()
            if len(miss_c) == 0:
                continue
            for c in miss_c:
                neighbors = self.X_train[idxs[b], c]   # (k,H,W,D)
                out[b, c] = neighbors.mean(dim=0)

        return out


# =========================
# Train/Eval
# =========================
def train_one_epoch(model, loader, device, optimizer, criterion, knn_imputer, train_missing_ks):
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        mk = int(random.choice(train_missing_ks))
        xb_miss, mask = apply_missing_k(xb, mk)

        # KNN补全在CPU上做（稳定/节省GPU）
        xb_imp = knn_imputer.impute(xb_miss.detach().cpu(), mask.detach().cpu()).to(device)

        logits = model(xb_imp)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_fixed_missing_k(model, loader, device, knn_imputer, missing_k: int):
    model.eval()
    all_true, all_pred, all_prob1 = [], [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        xb_miss, mask = apply_missing_k(xb, missing_k)
        xb_imp = knn_imputer.impute(xb_miss.detach().cpu(), mask.detach().cpu()).to(device)

        logits = model(xb_imp)
        prob = torch.softmax(logits, dim=1)
        p1 = prob[:, 1]
        pred = torch.argmax(prob, dim=1)

        all_true.extend(yb.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
        all_prob1.extend(p1.cpu().numpy().tolist())

    return compute_metrics_binary(all_true, all_pred, all_prob1)


def main():
    PACK_DIR = get_pack_dir()
    XFULL_PATH = find_xfull_path(PACK_DIR)
    SPLIT_PATH = os.path.join(PACK_DIR, "split.json")

    # ---- Hyperparams ----
    BATCH_SIZE = 4
    EPOCHS = 30
    LR = 1e-4
    KNN_K = 5

    # 训练时随机缺失强度分布（可改）
    TRAIN_MISSING_KS = (0, 0, 0, 1, 1, 2, 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- Load data ----
    full = torch.load(XFULL_PATH, map_location="cpu")
    X = full["X_full"].float()  # (N,4,H,W,D)
    y = full["y"].long()        # (N,)

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    idx_train = np.array(split["idx_train"], dtype=np.int64)
    idx_val   = np.array(split["idx_val"], dtype=np.int64)
    idx_test  = np.array(split["idx_test"], dtype=np.int64)

    train_ds = TensorDataset(X[idx_train], y[idx_train])
    val_ds   = TensorDataset(X[idx_val], y[idx_val])
    test_ds  = TensorDataset(X[idx_test], y[idx_test])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ---- Build KNN imputer from full train set (no missing) ----
    knn_imputer = KNNImputer3D(X[idx_train], k_neighbors=KNN_K)

    # ---- Model ----
    model = ResNet(
        block="basic",
        layers=[2, 2, 2, 2],
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=4,
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ---- Train ----
    best_val_auc = -1.0
    best_path = os.path.join(PACK_DIR, "best_resnet_knn.pt")

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, device, optimizer, criterion, knn_imputer, TRAIN_MISSING_KS)

        # 简单用 val missing_k=0 选模型（可按需要换为macro-F1或BACC）
        val_metrics = evaluate_fixed_missing_k(model, val_loader, device, knn_imputer, missing_k=0)

        if (not np.isnan(val_metrics["AUC"])) and val_metrics["AUC"] > best_val_auc:
            best_val_auc = val_metrics["AUC"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_auc": best_val_auc}, best_path)

        print(
            f"Epoch {epoch:03d} | TrainLoss {tr_loss:.4f} | "
            f"Val(miss0) ACC {val_metrics['ACC']:.4f} F1M {val_metrics['F1_macro']:.4f} "
            f"AUC {val_metrics['AUC']:.4f} SENS {val_metrics['SENS']:.4f} SPEC {val_metrics['SPEC']:.4f} | "
            f"BestValAUC {best_val_auc:.4f}"
        )

    # ---- Load best ----
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"\nLoaded best checkpoint: {best_path} (epoch={ckpt.get('epoch')}, val_auc={ckpt.get('val_auc')})")

    # ---- Test: missing_k 0..3 ----
    print("\nTest results (ResNet + KNN imputation)")
    print("missing_k\tACC\tF1_macro\tAUC\tSENS\tSPEC")
    for mk in [0, 1, 2, 3]:
        m = evaluate_fixed_missing_k(model, test_loader, device, knn_imputer, missing_k=mk)
        print(f"{mk}\t{m['ACC']:.4f}\t{m['F1_macro']:.4f}\t{m['AUC']:.4f}\t{m['SENS']:.4f}\t{m['SPEC']:.4f}")


if __name__ == "__main__":
    main()
