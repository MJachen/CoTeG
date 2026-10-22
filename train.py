import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import BrainTumor3DDataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

class Tiny3DCNN(nn.Module):
    def __init__(self, in_channels=4, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),   # /2

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),   # /4

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),   # /8
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # (B,64,1,1,1)
            nn.Flatten(),             # (B,64)
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += x.size(0)

    return total_loss / n, correct / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        preds = torch.argmax(probs, dim=1)

        all_labels.extend(y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 1. Accuracy
    acc = accuracy_score(all_labels, all_preds)

    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # 3. Precision / Recall / F1
    report = classification_report(
        all_labels,
        all_preds,
        target_names=["GLI", "MEN", "MET"],
        digits=4
    )

    # 4. AUC (One-vs-Rest)
    auc = roc_auc_score(
        all_labels,
        all_probs,
        multi_class="ovr"
    )

    return acc, cm, report, auc

# ========== main ==========
root = "/home/cjc/brats2025"
dataset = BrainTumor3DDataset(root_dir=root)

# 简单划分：8:2
n_total = len(dataset)
n_train = int(n_total * 0.8)
n_val = n_total - n_train
train_set, val_set = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Tiny3DCNN(in_channels=4, num_classes=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

for epoch in range(1, 21):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)

    val_acc, cm, report, auc = evaluate(model, val_loader, device)

    print(f"\nEpoch {epoch}")
    print(f"Train Acc: {tr_acc:.4f}")
    print(f"Val Acc: {val_acc:.4f}")
    print(f"Val AUC: {auc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)