import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

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

    return {
        "ACC": acc,
        "F1_macro": f1_macro,
        "AUC": auc,
        "SENS": sens,
        "SPEC": spec,
    }


# =========================
# Missing 模态生成（ModDrop）
# =========================
def apply_missing_k(x, k: int):
    """
    x: (B,4,H,W,D)
    return:
      x_miss: (B,4,H,W,D)
    """
    B, C = x.shape[:2]
    x_miss = x.clone()

    k = max(0, min(k, C - 1))
    if k == 0:
        return x_miss

    for b in range(B):
        drop = torch.randperm(C, device=x.device)[:k]
        x_miss[b, drop] = 0.0

    return x_miss


# =========================
# Train/Eval
# =========================
def train_one_epoch(model, loader, device, optimizer, criterion, train_missing_ks):
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        mk = int(random.choice(train_missing_ks))
        xb_miss = apply_missing_k(xb, mk)

        logits = model(xb_miss)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_fixed_missing_k(model, loader, device, missing_k):
    model.eval()
    all_true, all_pred, all_prob1 = [], [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        xb_miss = apply_missing_k(xb, missing_k)

        logits = model(xb_miss)
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

    # -------- Hyperparams --------
    BATCH_SIZE = 4
    EPOCHS = 30
    LR = 1e-4

    # 训练时随机缺失策略
    TRAIN_MISSING_KS = (0, 0, 0, 1, 1, 2, 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -------- Load Data --------
    full = torch.load(XFULL_PATH, map_location="cpu")
    X = full["X_full"].float()
    y = full["y"].long()

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    idx_train = np.array(split["idx_train"], dtype=np.int64)
    idx_val   = np.array(split["idx_val"], dtype=np.int64)
    idx_test  = np.array(split["idx_test"], dtype=np.int64)

    train_ds = TensorDataset(X[idx_train], y[idx_train])
    val_ds   = TensorDataset(X[idx_val], y[idx_val])
    test_ds  = TensorDataset(X[idx_test], y[idx_test])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # -------- Model --------
    model = ResNet(
        block="basic",
        layers=[2,2,2,2],
        block_inplanes=[64,128,256,512],
        spatial_dims=3,
        n_input_channels=4,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_auc = -1.0
    best_path = os.path.join(PACK_DIR, "best_resnet_moddrop.pt")

    # -------- Train --------
    for epoch in range(1, EPOCHS+1):

        tr_loss = train_one_epoch(model, train_loader, device, optimizer, criterion, TRAIN_MISSING_KS)

        val_metrics = evaluate_fixed_missing_k(model, val_loader, device, missing_k=0)

        if not np.isnan(val_metrics["AUC"]) and val_metrics["AUC"] > best_val_auc:
            best_val_auc = val_metrics["AUC"]
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch {epoch:03d} | "
            f"TrainLoss {tr_loss:.4f} | "
            f"Val ACC {val_metrics['ACC']:.4f} "
            f"F1M {val_metrics['F1_macro']:.4f} "
            f"AUC {val_metrics['AUC']:.4f} "
            f"SENS {val_metrics['SENS']:.4f} "
            f"SPEC {val_metrics['SPEC']:.4f} | "
            f"BestAUC {best_val_auc:.4f}"
        )

    # -------- Load best --------
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))

    # -------- Test --------
    print("\nTest results (ResNet + ModDrop)")
    print("missing_k\tACC\tF1_macro\tAUC\tSENS\tSPEC")

    for mk in [0,1,2,3]:
        m = evaluate_fixed_missing_k(model, test_loader, device, mk)
        print(f"{mk}\t{m['ACC']:.4f}\t{m['F1_macro']:.4f}\t{m['AUC']:.4f}\t{m['SENS']:.4f}\t{m['SPEC']:.4f}")


if __name__ == "__main__":
    main()
