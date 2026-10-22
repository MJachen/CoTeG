import glob
import json
import os
import random
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from modality_groups import MODALITY_GROUPS, group_to_omega_row
from project_paths import find_train_pack_path, find_xfull_path, get_pack_dir, get_weight_path
from twognn import GNNBranch
from twotrlv2 import ConvTRLNet
from zerotoolfunc import knn_graph_pytorch


SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


def _parse_int_tuple(value, default):
    if not value:
        return default
    parts = tuple(int(x.strip()) for x in value.split(","))
    if len(parts) != len(default):
        raise ValueError(f"Expected {len(default)} comma-separated integers, got {value!r}")
    return parts


def _parse_float_list(value):
    if not value:
        return None
    return [float(x.strip()) for x in value.split(",") if x.strip()]


PACK_DIR = get_pack_dir()
TRAIN_AUG_PATH = find_train_pack_path(PACK_DIR)
XHAT_TRAIN_PATH = os.environ.get("XHAT_TRAIN_PATH", os.path.join(PACK_DIR, "X_hat_train_5d.pt"))
TUCKER_FACTORS_TRAIN = os.environ.get("TUCKER_FACTORS_TRAIN", os.path.join(PACK_DIR, "tucker_factors_train.pt"))
COMP_DIR = PACK_DIR

TENSOR_BRANCH_W = get_weight_path("TENSOR_BRANCH_W", "./tensor_branch.pth", None)
GNN_BRANCH_W = get_weight_path("GNN_BRANCH_W", "./gnn_branch.pth", None)
TENSOR_RANKS = _parse_int_tuple(os.environ.get("TENSOR_RANKS"), (16, 4, 4, 4, 8))
BRANCH_META_PATH = os.environ.get("BRANCH_META_PATH", os.path.join(PACK_DIR, "branch_meta.pt"))

CPU_IMAGE_DTYPE = (
    torch.float16
    if os.environ.get("FUSION_CPU_IMAGE_DTYPE", "float32").lower() in {"float16", "fp16", "half"}
    else torch.float32
)
FUSION_HIDDEN = int(os.environ.get("FUSION_HIDDEN", "32"))
FUSION_DROPOUT = float(os.environ.get("FUSION_DROPOUT", "0.4"))
FUSION_LR = float(os.environ.get("FUSION_LR", "3e-4"))
FUSION_WEIGHT_DECAY = float(os.environ.get("FUSION_WEIGHT_DECAY", "1e-3"))
FUSION_BALANCED_SAMPLER = os.environ.get("FUSION_BALANCED_SAMPLER", "1") != "0"
FUSION_MODE = os.environ.get("FUSION_MODE", "moe").strip().lower()
FUSION_USE_MASK = os.environ.get("FUSION_USE_MASK", "1") != "0"
FUSION_EARLY_STOP_PATIENCE = int(os.environ.get("FUSION_EARLY_STOP_PATIENCE", "40"))
FUSION_EARLY_STOP_MIN_DELTA = float(os.environ.get("FUSION_EARLY_STOP_MIN_DELTA", "1e-4"))
FUSION_CLASS_WEIGHT_MULT = _parse_float_list(os.environ.get("FUSION_CLASS_WEIGHT_MULT"))
FUSION_MONITOR = os.environ.get("FUSION_MONITOR", "bac").strip().lower()

OUT_DIR = os.environ.get("FUSION_OUT_DIR", "./fusion_out_multiclass")
BEST_FUSION_PATH = os.path.join(OUT_DIR, "best_fusion_head.pth")
METRICS_JSON = os.path.join(OUT_DIR, "metrics_modality_groups.json")


class MoEFusion(nn.Module):
    def __init__(self, dim_t, dim_g, hidden, num_classes, p=0.3, mask_dim=4):
        super().__init__()
        self.mask_dim = int(mask_dim)
        self.proj_t = nn.Sequential(nn.Linear(dim_t, hidden), nn.LayerNorm(hidden))
        self.proj_g = nn.Sequential(nn.Linear(dim_g, hidden), nn.LayerNorm(hidden))
        self.expert_t = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(True), nn.Dropout(p), nn.Linear(64, num_classes))
        self.expert_g = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(True), nn.Dropout(p), nn.Linear(64, num_classes))
        self.gate = nn.Sequential(nn.Linear(hidden * 4 + self.mask_dim, 32), nn.ReLU(True), nn.Linear(32, 1))

    def forward(self, ft, fg, mask=None):
        ft = self.proj_t(ft)
        fg = self.proj_g(fg)
        z = torch.cat([ft, fg, (ft - fg).abs(), ft * fg], dim=1)
        if self.mask_dim > 0:
            if mask is None:
                mask = torch.zeros((z.size(0), self.mask_dim), dtype=z.dtype, device=z.device)
            z = torch.cat([z, mask.to(device=z.device, dtype=z.dtype)], dim=1)
        alpha = torch.sigmoid(self.gate(z))
        logits = alpha * self.expert_t(ft) + (1.0 - alpha) * self.expert_g(fg)
        return logits, alpha


class ConcatFusion(nn.Module):
    def __init__(self, dim_t, dim_g, hidden, num_classes, p=0.3, mask_dim=4):
        super().__init__()
        self.mask_dim = int(mask_dim)
        in_dim = int(dim_t) + int(dim_g) + self.mask_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(True),
            nn.Dropout(p),
            nn.Linear(hidden, 64),
            nn.ReLU(True),
            nn.Dropout(p),
            nn.Linear(64, num_classes),
        )

    def forward(self, ft, fg, mask=None):
        xs = [ft, fg]
        if self.mask_dim > 0:
            if mask is None:
                mask = torch.zeros((ft.size(0), self.mask_dim), dtype=ft.dtype, device=ft.device)
            xs.append(mask.to(device=ft.device, dtype=ft.dtype))
        logits = self.head(torch.cat(xs, dim=1))
        alpha = torch.full((ft.size(0), 1), float("nan"), dtype=ft.dtype, device=ft.device)
        return logits, alpha


class TensorOnlyHead(nn.Module):
    def __init__(self, dim_t, hidden, num_classes, p=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_t, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(True),
            nn.Dropout(p),
            nn.Linear(hidden, 64),
            nn.ReLU(True),
            nn.Dropout(p),
            nn.Linear(64, num_classes),
        )

    def forward(self, ft, fg, mask=None):
        logits = self.head(ft)
        alpha = torch.ones((ft.size(0), 1), dtype=ft.dtype, device=ft.device)
        return logits, alpha


class GNNOnlyHead(nn.Module):
    def __init__(self, dim_g, hidden, num_classes, p=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_g, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(True),
            nn.Dropout(p),
            nn.Linear(hidden, 64),
            nn.ReLU(True),
            nn.Dropout(p),
            nn.Linear(64, num_classes),
        )

    def forward(self, ft, fg, mask=None):
        logits = self.head(fg)
        alpha = torch.zeros((fg.size(0), 1), dtype=fg.dtype, device=fg.device)
        return logits, alpha


def build_fusion_model(mode, dim_t, dim_g, num_classes):
    mask_dim = 4 if FUSION_USE_MASK else 0
    if mode == "moe":
        return MoEFusion(dim_t, dim_g, FUSION_HIDDEN, num_classes, p=FUSION_DROPOUT, mask_dim=mask_dim)
    if mode == "concat":
        return ConcatFusion(dim_t, dim_g, FUSION_HIDDEN, num_classes, p=FUSION_DROPOUT, mask_dim=mask_dim)
    if mode == "tensor_only":
        return TensorOnlyHead(dim_t, FUSION_HIDDEN, num_classes, p=FUSION_DROPOUT)
    if mode == "gnn_only":
        return GNNOnlyHead(dim_g, FUSION_HIDDEN, num_classes, p=FUSION_DROPOUT)
    raise ValueError("FUSION_MODE must be one of: moe, concat, tensor_only, gnn_only")


def make_modality_omega_rows():
    return torch.tensor([group_to_omega_row(present) for _, present in MODALITY_GROUPS], dtype=torch.float32)


def standardize_u(U, mean, std):
    return (U.float() - mean.float()) / std.float().clamp_min(1e-6)


def ensure_ncdhw(x):
    assert x.ndim == 5 and x.shape[1] == 4, f"Expected (N,4,D,H,W), got {tuple(x.shape)}"
    return x


def load_train_full_ncdhw(fac_train):
    if "idx_train_global" not in fac_train:
        raise KeyError("tucker_factors_train.pt missing idx_train_global; rerun GRNNTCtrain.py patient mode.")
    full = torch.load(find_xfull_path(PACK_DIR), map_location="cpu")
    x_full_all = full["X_full"].to(dtype=CPU_IMAGE_DTYPE)
    idx_train = fac_train["idx_train_global"].long()
    return x_full_all.index_select(0, idx_train).permute(0, 1, 4, 2, 3).contiguous()


def load_label_names():
    config_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(PACK_DIR)), "config.json"),
        os.path.join(PACK_DIR, "config.json"),
    ]
    label_map = None
    for path in config_candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            label_map = payload.get("label_map")
            if isinstance(label_map, dict):
                break
    if not isinstance(label_map, dict):
        return None
    out = [None] * len(label_map)
    for name, idx in label_map.items():
        try:
            out[int(idx)] = str(name)
        except Exception:
            return None
    return out


class CompletedTrainDataset3D(torch.utils.data.Dataset):
    def __init__(self, x_full_ncdhw, x_hat_ncdhw, y_cpu, omega_rows):
        self.x_full = x_full_ncdhw.to(dtype=CPU_IMAGE_DTYPE).contiguous()
        self.x_hat = x_hat_ncdhw.to(dtype=CPU_IMAGE_DTYPE).contiguous()
        self.y = y_cpu.long().contiguous()
        self.omega = omega_rows.float().view(omega_rows.shape[0], omega_rows.shape[1], 1, 1, 1)
        self.num_groups = int(self.omega.shape[0])
        self.labels = self.y.repeat_interleave(self.num_groups)

    def __len__(self):
        return int(self.y.numel() * self.num_groups)

    def __getitem__(self, idx):
        base_i = int(idx) // self.num_groups
        group_i = int(idx) % self.num_groups
        omega = self.omega[group_i]
        x = self.x_full[base_i] * omega + self.x_hat[base_i] * (1.0 - omega)
        return x, self.y[base_i], omega.view(-1), torch.tensor(base_i, dtype=torch.long)


@torch.no_grad()
def extract_tensor_embeddings_from_dataset(model, ds, batch_size=2):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    zs, ys, masks, base_idx = [], [], [], []
    for xb, yb, mb, bid in loader:
        xb = xb.to(DEVICE).float()
        _, z = model(xb, return_embed=True)
        zs.append(z.detach().cpu())
        ys.append(yb.long().cpu())
        masks.append(mb.float().cpu())
        base_idx.append(bid.long().cpu())
    return torch.cat(zs, dim=0), torch.cat(ys, dim=0), torch.cat(masks, dim=0), torch.cat(base_idx, dim=0)


@torch.no_grad()
def extract_tensor_embeddings_from_nmhwd(model, x_cpu, batch_size=2):
    model.eval()
    zs = []
    for s0 in range(0, x_cpu.size(0), batch_size):
        xb = x_cpu[s0:s0 + batch_size].permute(0, 1, 4, 2, 3).contiguous().to(DEVICE).float()
        _, z = model(xb, return_embed=True)
        zs.append(z.detach().cpu())
        del xb, z
    return torch.cat(zs, dim=0)


@torch.no_grad()
def run_gnn_embeddings(gnn_model, U_cpu, knn_k=10):
    gnn_model.eval()
    U = U_cpu.to(DEVICE).float()
    edge_index = knn_graph_pytorch(U, k=knn_k).to(DEVICE)
    _, z = gnn_model(U, edge_index)
    return z.detach().cpu()


def get_paths(split, group_name):
    return (
        os.path.join(COMP_DIR, f"{split}_{group_name}_completed.pt"),
        os.path.join(COMP_DIR, f"{split}_{group_name}_completed_tucker_factors.pt"),
    )


def discover_completed_groups():
    vals = glob.glob(os.path.join(COMP_DIR, "val_*_completed.pt"))
    groups = []
    for path in sorted(vals):
        base = os.path.basename(path)
        group = base[len("val_"):-len("_completed.pt")]
        if os.path.exists(os.path.join(COMP_DIR, f"test_{group}_completed.pt")):
            groups.append(group)
    if not groups:
        raise FileNotFoundError(f"No completed val/test packs found in {COMP_DIR}")
    return groups


def load_eval_group_embeddings(split, group, tensor_branch, gnn_branch, u_mean, u_std, tensor_batch_size):
    comp_pack_path, tucker_path = get_paths(split, group)
    pack = torch.load(comp_pack_path, map_location="cpu")
    if "X_hat" not in pack:
        raise KeyError(f"{comp_pack_path} missing key 'X_hat'")
    x_hat = pack["X_hat"].to(dtype=CPU_IMAGE_DTYPE)
    y = pack["y"].long()
    idx = pack.get("idx", torch.arange(y.numel(), dtype=torch.long)).long()
    mask = pack["Omega"].float().view(pack["Omega"].shape[0], pack["Omega"].shape[1])

    fac = torch.load(tucker_path, map_location="cpu")
    if "U_N" not in fac:
        raise KeyError(f"{tucker_path} missing key 'U_N'")
    U = standardize_u(fac["U_N"].float(), u_mean, u_std)

    print(f"[Embedding] {split.upper()} {group}: X={tuple(x_hat.shape)} U={tuple(U.shape)} y={tuple(y.shape)}")
    zt = extract_tensor_embeddings_from_nmhwd(tensor_branch, x_hat, batch_size=tensor_batch_size)
    zg = run_gnn_embeddings(gnn_branch, U, knn_k=10)
    out = {"zt": zt, "zg": zg, "y": y, "idx": idx, "mask": mask}

    del pack, x_hat, fac, U
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def make_class_weights(y, num_classes):
    bc = torch.bincount(y.long().cpu(), minlength=num_classes).float()
    weights = bc.sum() / (num_classes * bc.clamp_min(1.0))
    mult = torch.ones(num_classes, dtype=torch.float32)
    if FUSION_CLASS_WEIGHT_MULT and len(FUSION_CLASS_WEIGHT_MULT) >= num_classes:
        mult = torch.tensor(FUSION_CLASS_WEIGHT_MULT[:num_classes], dtype=torch.float32)
    return (weights * mult).to(DEVICE)


def make_balanced_sampler(labels, num_classes):
    y_np = labels.long().cpu().numpy()
    class_count = np.bincount(y_np, minlength=num_classes).astype(np.float32)
    class_weight = 1.0 / np.clip(class_count, 1.0, None)
    return WeightedRandomSampler(
        weights=torch.from_numpy(class_weight[y_np]),
        num_samples=len(y_np),
        replacement=True,
    )


@torch.no_grad()
def get_probs(fusion, zt_cpu, zg_cpu, mask_cpu=None):
    fusion.eval()
    mask = None if mask_cpu is None else mask_cpu.to(DEVICE).float()
    logits, _ = fusion(zt_cpu.to(DEVICE).float(), zg_cpu.to(DEVICE).float(), mask)
    return torch.softmax(logits, dim=1).detach().cpu().numpy()


def multiclass_metrics(y_true, probs, num_classes):
    y_true = np.asarray(y_true).astype(np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    pred = probs.argmax(axis=1).astype(np.int64)
    out = {
        "n": int(len(y_true)),
        "ACC": float(accuracy_score(y_true, pred)),
        "BAC": float(balanced_accuracy_score(y_true, pred)),
        "F1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "F1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "Precision_macro": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=list(range(num_classes))).astype(int).tolist(),
    }
    try:
        if num_classes == 2:
            out["AUC"] = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            out["AUC_macro_ovr"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
            out["AUC_weighted_ovr"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="weighted"))
    except Exception:
        if num_classes == 2:
            out["AUC"] = float("nan")
        else:
            out["AUC_macro_ovr"] = float("nan")
            out["AUC_weighted_ovr"] = float("nan")
    return out, pred


def score_for_selection(metrics_rows):
    vals = []
    key = {
        "bac": "BAC",
        "macro_f1": "F1_macro",
        "f1_macro": "F1_macro",
        "auc": "AUC_macro_ovr",
    }.get(FUSION_MONITOR, "BAC")
    for row in metrics_rows:
        v = row.get(key, float("nan"))
        if not (np.isnan(v) or np.isinf(v)):
            vals.append(v)
    return float(np.mean(vals)) if vals else -1.0


def collect_sample_table(split, group, y, probs, idx, class_names):
    pred = probs.argmax(axis=1).astype(np.int64)
    df = pd.DataFrame(
        {
            "split": split,
            "idx": idx,
            "modality_group": group,
            "y_true": y,
            "y_pred": pred,
            "correct": pred == y,
        }
    )
    for c in range(probs.shape[1]):
        name = class_names[c] if class_names and c < len(class_names) else str(c)
        df[f"prob_{name}"] = probs[:, c]
    return df


def evaluate_split(split, groups, data_by_group, fusion, num_classes, class_names):
    by_group = {}
    tables = []
    y_all, p_all = [], []
    for group in groups:
        data = data_by_group[group]
        y = data["y"].cpu().numpy().astype(np.int64)
        probs = get_probs(fusion, data["zt"], data["zg"], data["mask"])
        rep, _ = multiclass_metrics(y, probs, num_classes)
        by_group[group] = rep
        tables.append(collect_sample_table(split, group, y, probs, data["idx"].cpu().numpy(), class_names))
        y_all.append(y)
        p_all.append(probs)

    y_all = np.concatenate(y_all, axis=0)
    p_all = np.concatenate(p_all, axis=0)
    overall_concat, _ = multiclass_metrics(y_all, p_all, num_classes)

    def mean_key(key):
        vals = []
        for group in groups:
            v = by_group[group].get(key, float("nan"))
            if not (np.isnan(v) or np.isinf(v)):
                vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    overall_group_mean = {
        "n": int(sum(by_group[g]["n"] for g in groups)),
        "ACC": mean_key("ACC"),
        "BAC": mean_key("BAC"),
        "F1_macro": mean_key("F1_macro"),
        "F1_weighted": mean_key("F1_weighted"),
        "Precision_macro": mean_key("Precision_macro"),
        "Recall_macro": mean_key("Recall_macro"),
        "AUC_macro_ovr": mean_key("AUC_macro_ovr"),
        "AUC_weighted_ovr": mean_key("AUC_weighted_ovr"),
    }
    sample_path = os.path.join(OUT_DIR, f"{split}_sample_level_results_argmax.csv")
    pd.concat(tables, ignore_index=True).to_csv(sample_path, index=False)
    print("Saved sample predictions:", sample_path)
    return overall_concat, overall_group_mean, by_group


def print_multiclass_report(title, overall, by_group):
    print("\n" + "=" * 80)
    print(title)
    print(
        "Overall: "
        f"ACC={overall.get('ACC', float('nan')):.4f} "
        f"BAC={overall.get('BAC', float('nan')):.4f} "
        f"F1_macro={overall.get('F1_macro', float('nan')):.4f} "
        f"F1_weighted={overall.get('F1_weighted', float('nan')):.4f}"
    )
    print("By group:")
    for group, rep in by_group.items():
        print(
            f"  group={group} n={rep['n']} "
            f"ACC={rep.get('ACC', float('nan')):.4f} "
            f"BAC={rep.get('BAC', float('nan')):.4f} "
            f"F1_macro={rep.get('F1_macro', float('nan')):.4f} "
            f"F1_weighted={rep.get('F1_weighted', float('nan')):.4f}"
        )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    X_hat = torch.load(XHAT_TRAIN_PATH, map_location="cpu").to(dtype=CPU_IMAGE_DTYPE)
    Xtr = ensure_ncdhw(X_hat.permute(4, 3, 2, 0, 1).contiguous())
    fac_train = torch.load(TUCKER_FACTORS_TRAIN, map_location="cpu")
    Utr = fac_train["U_N"].float()
    ytr = fac_train.get("y_train", None)
    if ytr is None and os.path.exists(TRAIN_AUG_PATH):
        train_aug = torch.load(TRAIN_AUG_PATH, map_location="cpu")
        ytr = train_aug["y"]
    if ytr is None:
        raise KeyError("Cannot find training labels. Expected y_train in tucker_factors_train.pt.")
    ytr = ytr.long()
    num_classes = int(ytr.max().item()) + 1
    class_names = load_label_names()

    print("Train Q loaded:", "Xtr", tuple(Xtr.shape), "Utr", tuple(Utr.shape), "ytr", tuple(ytr.shape))
    print("Num classes:", num_classes, "class names:", class_names)

    if os.path.exists(BRANCH_META_PATH):
        branch_meta = torch.load(BRANCH_META_PATH, map_location="cpu")
        u_mean = branch_meta["u_mean"].float()
        u_std = branch_meta["u_std"].float()
    else:
        print("[Warn] branch_meta.pt not found; computing U_N mean/std from train factors.")
        u_mean = Utr.mean(0, keepdim=True)
        u_std = Utr.std(0, keepdim=True).clamp_min(1e-6)
    Utr_std = standardize_u(Utr, u_mean, u_std)

    tensor_branch = ConvTRLNet(n_outputs=num_classes, ranks=TENSOR_RANKS, input_spatial=tuple(Xtr.shape[2:])).to(DEVICE)
    tensor_branch.load_state_dict(torch.load(TENSOR_BRANCH_W, map_location=DEVICE))
    tensor_branch.eval()
    for p in tensor_branch.parameters():
        p.requires_grad = False

    gnn_branch = GNNBranch(in_dim=Utr_std.shape[1], hid=64, out_dim=32, num_classes=num_classes).to(DEVICE)
    gnn_branch.load_state_dict(torch.load(GNN_BRANCH_W, map_location=DEVICE), strict=False)
    gnn_branch.eval()
    for p in gnn_branch.parameters():
        p.requires_grad = False

    groups = discover_completed_groups()
    print("Discovered modality groups:", groups)

    train_modality_aug = os.environ.get("TRAIN_MODALITY_AUG", "1") != "0"
    if train_modality_aug:
        print("Building on-the-fly train modality groups for fusion...")
        x_full_ncdhw = load_train_full_ncdhw(fac_train)
        train_ds = CompletedTrainDataset3D(x_full_ncdhw, Xtr, ytr, make_modality_omega_rows())
        zt_tr, ytr_fusion, mask_tr, base_idx_tr = extract_tensor_embeddings_from_dataset(
            tensor_branch,
            train_ds,
            batch_size=int(os.environ.get("FUSION_TENSOR_EMBED_BATCH_SIZE", "2")),
        )
        zg_base = run_gnn_embeddings(gnn_branch, Utr_std, knn_k=10)
        zg_tr = zg_base.index_select(0, base_idx_tr)
    else:
        zt_tr = extract_tensor_embeddings_from_nmhwd(tensor_branch, Xtr.permute(0, 1, 3, 4, 2).contiguous())
        zg_tr = run_gnn_embeddings(gnn_branch, Utr_std, knn_k=10)
        ytr_fusion = ytr
        mask_tr = torch.ones((ytr.numel(), 4), dtype=torch.float32)

    dim_t = zt_tr.shape[1]
    dim_g = zg_tr.shape[1]
    print(f"Embedding dims: tensor={dim_t}, gnn={dim_g}")
    del Xtr, X_hat
    if train_modality_aug:
        del x_full_ncdhw, train_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_tensor_batch_size = int(os.environ.get("EVAL_TENSOR_EMBED_BATCH_SIZE", os.environ.get("FUSION_TENSOR_EMBED_BATCH_SIZE", "2")))
    val_data, test_data = {}, {}
    for group in groups:
        val_data[group] = load_eval_group_embeddings("val", group, tensor_branch, gnn_branch, u_mean, u_std, eval_tensor_batch_size)
        test_data[group] = load_eval_group_embeddings("test", group, tensor_branch, gnn_branch, u_mean, u_std, eval_tensor_batch_size)

    fusion = build_fusion_model(FUSION_MODE, dim_t, dim_g, num_classes).to(DEVICE)
    class_weights = make_class_weights(ytr_fusion, num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(fusion.parameters(), lr=FUSION_LR, weight_decay=FUSION_WEIGHT_DECAY)

    ds = TensorDataset(zt_tr, zg_tr, mask_tr, ytr_fusion)
    sampler = make_balanced_sampler(ytr_fusion, num_classes) if FUSION_BALANCED_SAMPLER else None
    dl = DataLoader(
        ds,
        batch_size=int(os.environ.get("FUSION_BATCH_SIZE", "128")),
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
    )

    print(
        "Fusion config:",
        {
            "mode": FUSION_MODE,
            "use_mask": FUSION_USE_MASK,
            "hidden": FUSION_HIDDEN,
            "dropout": FUSION_DROPOUT,
            "lr": FUSION_LR,
            "weight_decay": FUSION_WEIGHT_DECAY,
            "balanced_sampler": FUSION_BALANCED_SAMPLER,
            "class_weights": [float(x) for x in class_weights.detach().cpu().tolist()],
            "monitor": FUSION_MONITOR,
        },
    )

    best_score = -1e9
    best_state = None
    best_epoch = -1
    epochs_without_improve = 0
    history = []
    epochs = int(os.environ.get("FUSION_EPOCHS", "200"))

    print("Start training multiclass fusion head...")
    for ep in range(epochs):
        fusion.train()
        for za, zb, mb, yb in dl:
            za = za.to(DEVICE).float()
            zb = zb.to(DEVICE).float()
            mb = mb.to(DEVICE).float()
            yb = yb.to(DEVICE).long()
            logits, _ = fusion(za, zb, mb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_rows = []
        for group in groups:
            probs = get_probs(fusion, val_data[group]["zt"], val_data[group]["zg"], val_data[group]["mask"])
            rep, _ = multiclass_metrics(val_data[group]["y"].cpu().numpy(), probs, num_classes)
            val_rows.append(rep)
        score = score_for_selection(val_rows)
        history.append({"epoch": ep, "score": score})

        if ep % 20 == 0 or ep == epochs - 1:
            print(f"[Epoch {ep:03d}] val_{FUSION_MONITOR}={score:.4f}")

        if score > best_score + FUSION_EARLY_STOP_MIN_DELTA:
            best_score = score
            best_state = {k: v.cpu() for k, v in fusion.state_dict().items()}
            best_epoch = ep
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if FUSION_EARLY_STOP_PATIENCE > 0 and epochs_without_improve >= FUSION_EARLY_STOP_PATIENCE:
            print(
                f"[EarlyStop] no val improvement for {FUSION_EARLY_STOP_PATIENCE} epochs; "
                f"best_epoch={best_epoch:03d}, best_score={best_score:.4f}"
            )
            break

    if best_state is None:
        raise RuntimeError("Fusion training did not produce a best state.")
    torch.save(best_state, BEST_FUSION_PATH)
    print("Saved best fusion head:", BEST_FUSION_PATH, "best_score:", best_score, "best_epoch:", best_epoch)
    pd.DataFrame(history).to_csv(os.path.join(OUT_DIR, "fusion_training_history.csv"), index=False)

    fusion.load_state_dict(torch.load(BEST_FUSION_PATH, map_location=DEVICE))
    fusion.eval()

    val_concat, val_group_mean, val_by_group = evaluate_split("val", groups, val_data, fusion, num_classes, class_names)
    test_concat, test_group_mean, test_by_group = evaluate_split("test", groups, test_data, fusion, num_classes, class_names)
    print_multiclass_report("VAL (all modality groups, argmax)", val_concat, val_by_group)
    print_multiclass_report("TEST (all modality groups, argmax)", test_concat, test_by_group)
    print("\n[TEST group mean over modality groups]:", test_group_mean)

    metrics = {
        "pack_dir": PACK_DIR,
        "out_dir": OUT_DIR,
        "num_classes": num_classes,
        "class_names": class_names,
        "fusion_config": {
            "mode": FUSION_MODE,
            "use_mask": FUSION_USE_MASK,
            "hidden": FUSION_HIDDEN,
            "dropout": FUSION_DROPOUT,
            "lr": FUSION_LR,
            "weight_decay": FUSION_WEIGHT_DECAY,
            "balanced_sampler": FUSION_BALANCED_SAMPLER,
            "class_weights": [float(x) for x in class_weights.detach().cpu().tolist()],
            "monitor": FUSION_MONITOR,
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
        },
        "val": {
            "overall_concat": val_concat,
            "overall_group_mean": val_group_mean,
            "by_group": val_by_group,
        },
        "test": {
            "overall_concat": test_concat,
            "overall_group_mean": test_group_mean,
            "by_group": test_by_group,
        },
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics json:", METRICS_JSON)


if __name__ == "__main__":
    main()
