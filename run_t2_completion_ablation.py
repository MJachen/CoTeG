"""Run T2: population-level versus per-subject tensor completion.

The experiment keeps the split, 15 modality masks, tensor classifier,
classifier seed, and threshold protocol fixed across all three arms:

1. per-subject spatial x modality completion;
2. population Tucker completion without subject-graph regularization;
3. full CoTeG population Tucker completion with subject-graph regularization.

Example
-------
E:/cjj/anaconda3/envs/tensor/python.exe run_t2_completion_ablation.py \
    --config configs/t2_smoke.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Dataset

from modality_groups import MODALITY_GROUPS, MODALITY_ORDER, group_to_omega_row
from t2_completion import (
    ARM_FULL_COTEG,
    ARM_ORDER,
    ARM_PER_SUBJECT,
    ARM_POPULATION_NO_GRAPH,
    CompletionConfig,
    fit_per_subject_training_completion,
    fit_population_training_basis,
    infer_per_subject_completion,
    infer_population_completion,
    set_deterministic_seed,
    tensor_sha256,
    training_capacity,
)
from twotrlv2 import ConvTRLNet


PRIMARY_METRICS = ("BAC", "AUC")
SEVERE_GROUPS = {name for name, present in MODALITY_GROUPS if len(present) == 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a T2 JSON config.")
    parser.add_argument("--output-dir", default=None, help="Override config output_dir.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Override config device.")
    parser.add_argument("--dry-run", action="store_true", help="Validate protocol and write manifest without fitting.")
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (config_path.parent / path).resolve()


def atomic_json_dump(payload: Dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    tmp.replace(path)


def stable_hash_json(payload) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_completion_config(raw: Dict, seed: int) -> CompletionConfig:
    return CompletionConfig(
        rank_s=int(raw["rank_s"]),
        rank_m=int(raw["rank_m"]),
        rank_n=int(raw["rank_n"]),
        train_iterations=int(raw["train_iterations"]),
        eval_iterations=int(raw["eval_iterations"]),
        lr_train=float(raw["lr_train"]),
        lr_eval=float(raw["lr_eval"]),
        lambda_space=float(raw.get("lambda_space", 1e-3)),
        lambda_mod=float(raw.get("lambda_mod", 5e-4)),
        lambda_subject=float(raw.get("lambda_subject", 5e-4)),
        lambda_tv=float(raw.get("lambda_tv", 0.0)),
        tv_max_pairs=int(raw.get("tv_max_pairs", 0)),
        chunk_s=int(raw.get("chunk_s", 2048)),
        knn_k=int(raw.get("knn_k", 10)),
        graph_sigma=float(raw.get("graph_sigma", 1.0)),
        seed=int(seed),
        per_subject_batch_size=int(raw.get("per_subject_batch_size", 1)),
        log_every=int(raw.get("log_every", 0)),
    )


def validate_protocol(
    x_full: torch.Tensor,
    y: torch.Tensor,
    split: Dict,
    groups: Sequence[Tuple[str, Sequence[str]]],
    completion_cfg: CompletionConfig,
) -> Dict[str, torch.Tensor]:
    if x_full.ndim != 5:
        raise ValueError(f"X_full must have shape (N,M,H,W,D), got {tuple(x_full.shape)}")
    if int(x_full.shape[1]) != len(MODALITY_ORDER):
        raise ValueError(f"Expected {len(MODALITY_ORDER)} modalities, got {x_full.shape[1]}")
    if y.ndim != 1 or y.numel() != x_full.shape[0]:
        raise ValueError("y must be one-dimensional and aligned with X_full.")
    if sorted(torch.unique(y).tolist()) != [0, 1]:
        raise ValueError("T2 currently requires binary labels encoded as 0/1.")
    if min(x_full.shape[2:]) < 8 or any(int(v) % 8 for v in x_full.shape[2:]):
        raise ValueError("Spatial dimensions must be >=8 and divisible by 8 for ConvTRLNet.")
    if completion_cfg.rank_m > x_full.shape[1]:
        raise ValueError("rank_m cannot exceed the number of modalities.")
    if completion_cfg.lambda_subject <= 0:
        raise ValueError("lambda_subject must be > 0 so full_coteg is distinct from the no-graph arm.")

    indices = {}
    seen = set()
    n = int(x_full.shape[0])
    for key in ("idx_train", "idx_val", "idx_test"):
        if key not in split:
            raise KeyError(f"split.json is missing {key}")
        idx = torch.tensor(split[key], dtype=torch.long)
        if idx.numel() == 0:
            raise ValueError(f"{key} is empty.")
        if int(idx.min()) < 0 or int(idx.max()) >= n:
            raise IndexError(f"{key} contains an index outside [0,{n - 1}].")
        overlap = seen.intersection(idx.tolist())
        if overlap:
            raise ValueError(f"Split leakage: {key} overlaps earlier splits at {sorted(overlap)}")
        seen.update(idx.tolist())
        indices[key] = idx

    names = [name for name, _ in groups]
    if len(names) != 15 or len(set(names)) != 15:
        raise ValueError("The primary T2 protocol requires all 15 unique non-empty modality combinations.")
    expected = {name for name, _ in MODALITY_GROUPS}
    if set(names) != expected:
        raise ValueError(f"T2 groups differ from the canonical 15 groups: {sorted(set(names) ^ expected)}")
    return indices


class CompletionAugmentedTrainDataset(Dataset):
    def __init__(self, x_full_nmhwd: torch.Tensor, x_hat_nmhwd: torch.Tensor, y: torch.Tensor, omega_gm: torch.Tensor):
        self.x_full = x_full_nmhwd.float().contiguous()
        self.x_hat = x_hat_nmhwd.float().contiguous()
        self.y = y.long().contiguous()
        self.omega = omega_gm.float().view(omega_gm.shape[0], omega_gm.shape[1], 1, 1, 1)
        self.groups = int(omega_gm.shape[0])

    def __len__(self) -> int:
        return int(self.y.numel()) * self.groups

    def __getitem__(self, index: int):
        subject = int(index) // self.groups
        group = int(index) % self.groups
        omega = self.omega[group]
        x = self.x_full[subject] * omega + self.x_hat[subject] * (1.0 - omega)
        # Repository pack: M,H,W,D; Conv3D: M,D,H,W.
        return x.permute(0, 3, 1, 2).contiguous(), self.y[subject]


def class_weights(y: torch.Tensor, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(y.long().cpu(), minlength=2).float()
    return (counts.sum() / (2.0 * counts.clamp_min(1.0))).to(device)


def train_common_classifier(
    x_full: torch.Tensor,
    x_hat: torch.Tensor,
    y: torch.Tensor,
    omega_gm: torch.Tensor,
    cfg: Dict,
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, Dict]:
    set_deterministic_seed(seed)
    dataset = CompletionAugmentedTrainDataset(x_full, x_hat, y, omega_gm)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    spatial_dhw = (int(x_full.shape[4]), int(x_full.shape[2]), int(x_full.shape[3]))
    ranks = tuple(int(v) for v in cfg.get("trl_ranks", [16, 4, 4, 4, 8]))
    model = ConvTRLNet(n_outputs=2, ranks=ranks, input_spatial=spatial_dhw).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y, device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )
    start = time.perf_counter()
    peak_cuda = 0
    history = []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(yb.numel())
            total += int(yb.numel())
            correct += int((logits.argmax(dim=1) == yb).sum().item())
        if device.type == "cuda":
            peak_cuda = max(peak_cuda, int(torch.cuda.max_memory_allocated(device)))
        row = {
            "epoch": epoch,
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
        }
        history.append(row)
        if epoch == 0 or epoch + 1 == int(cfg["epochs"]) or (epoch + 1) % int(cfg.get("log_every", 20)) == 0:
            print(f"  classifier epoch={epoch:03d} loss={row['loss']:.4e} acc={row['accuracy']:.4f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model.eval(), {
        "seconds": time.perf_counter() - start,
        "peak_cuda_allocated_bytes": peak_cuda,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "history": history,
    }


@torch.no_grad()
def predict_probabilities(model: nn.Module, completed_nmhwd: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    probabilities = []
    for start in range(0, int(completed_nmhwd.shape[0]), batch_size):
        x = completed_nmhwd[start : start + batch_size].permute(0, 1, 4, 2, 3).contiguous()
        logits = model(x.to(device, non_blocking=True).float())
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    return torch.cat(probabilities).numpy()


def safe_auc(y: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else float("nan")


def safe_bac(y: np.ndarray, probability: np.ndarray, threshold: float) -> float:
    return float(balanced_accuracy_score(y, (probability >= threshold).astype(np.int64)))


def select_macro_bac_threshold(y_by_group: Dict[str, np.ndarray], p_by_group: Dict[str, np.ndarray], grid: Iterable[float]):
    best = None
    for threshold in grid:
        values = [safe_bac(y_by_group[g], p_by_group[g], float(threshold)) for g in y_by_group]
        score = float(np.mean(values))
        candidate = (score, -abs(float(threshold) - 0.5), -float(threshold), float(threshold))
        if best is None or candidate > best:
            best = candidate
    return float(best[-1]), float(best[0])


def bounding_box(mask: np.ndarray) -> Tuple[slice, ...]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(slice(None) for _ in range(mask.ndim))
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def completion_quality_rows(
    x_full_nmhwd: torch.Tensor,
    completed_nmhwd: torch.Tensor,
    omega_m: torch.Tensor,
    brain_threshold: float,
) -> List[Dict]:
    rows = []
    missing = torch.where(omega_m == 0)[0].tolist()
    if not missing:
        return rows
    x_full = x_full_nmhwd.float().numpy()
    completed = completed_nmhwd.float().numpy()
    for subject in range(x_full.shape[0]):
        brain = x_full[subject].max(axis=0) > brain_threshold
        bbox = bounding_box(brain)
        for modality in missing:
            gt = x_full[subject, modality]
            pred = completed[subject, modality]
            gt_crop = gt[bbox]
            pred_crop = pred[bbox]
            data_range = float(gt_crop.max() - gt_crop.min()) if gt_crop.size else 0.0
            ssim = float("nan")
            if gt_crop.size and min(gt_crop.shape) >= 7 and data_range > 0:
                ssim = float(structural_similarity(gt_crop, pred_crop, data_range=data_range))
            rows.append(
                {
                    "subject_local_index": subject,
                    "missing_modality": MODALITY_ORDER[modality],
                    "MAE_all_voxels": float(np.mean(np.abs(pred - gt))),
                    "MAE_brain": float(np.mean(np.abs(pred[brain] - gt[brain]))) if brain.any() else float("nan"),
                    "SSIM_brain_bbox": ssim,
                }
            )
    return rows


def nanmean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(array)) if array.size and np.isfinite(array).any() else float("nan")


def aggregate_quality(rows: List[Dict]) -> Dict[str, float]:
    return {
        "MAE_all_voxels": nanmean(row["MAE_all_voxels"] for row in rows),
        "MAE_brain": nanmean(row["MAE_brain"] for row in rows),
        "SSIM_brain_bbox": nanmean(row["SSIM_brain_bbox"] for row in rows),
    }


def build_summary(arm: str, combination_rows: List[Dict]) -> Dict:
    all_rows = [row for row in combination_rows if row["arm"] == arm]
    severe = [row for row in all_rows if row["group"] in SEVERE_GROUPS]
    missing = [row for row in all_rows if row["num_available"] < len(MODALITY_ORDER)]
    return {
        "arm": arm,
        "macro15_BAC": nanmean(row["BAC"] for row in all_rows),
        "macro15_AUC": nanmean(row["AUC"] for row in all_rows),
        "severe_k1_macro_BAC": nanmean(row["BAC"] for row in severe),
        "severe_k1_macro_AUC": nanmean(row["AUC"] for row in severe),
        "missing_groups_MAE_brain": nanmean(row["MAE_brain"] for row in missing),
        "missing_groups_SSIM_brain_bbox": nanmean(row["SSIM_brain_bbox"] for row in missing),
    }


def decision_from_summary(summary_rows: List[Dict]) -> Dict:
    indexed = {row["arm"]: row for row in summary_rows}
    full = indexed[ARM_FULL_COTEG]
    comparisons = {}
    passes = True
    for baseline in (ARM_PER_SUBJECT, ARM_POPULATION_NO_GRAPH):
        deltas = {
            metric: float(full[f"macro15_{metric}"] - indexed[baseline][f"macro15_{metric}"])
            for metric in PRIMARY_METRICS
        }
        comparisons[baseline] = deltas
        passes = passes and all(value > 0.0 for value in deltas.values())
    return {
        "criterion": "Full CoTeG must strictly exceed both baselines on 15-combination macro BAC and macro AUC.",
        "passes_primary_criterion": bool(passes),
        "deltas_full_minus_baseline": comparisons,
        "scope_note": "A smoke run validates execution only and is not inferential evidence.",
    }


def environment_manifest(device: torch.device) -> Dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    seed = int(config.get("seed", 42))
    output_dir = Path(args.output_dir or config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_device = args.device or config.get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu explicitly if intended.")
    device = torch.device(requested_device)
    set_deterministic_seed(seed)

    xfull_path = resolve_path(config["data"]["x_full_path"], config_path)
    split_path = resolve_path(config["data"]["split_path"], config_path)
    pack = torch.load(xfull_path, map_location="cpu")
    if "X_full" not in pack or "y" not in pack:
        raise KeyError("The full-data pack must contain X_full and y.")
    x_full = pack["X_full"].float().contiguous()
    y_all = pack["y"].long().contiguous()
    split = load_json(split_path)
    groups = [(name, present) for name, present in MODALITY_GROUPS]
    completion_cfg = build_completion_config(config["completion"], seed)
    indices = validate_protocol(x_full, y_all, split, groups, completion_cfg)
    omega_gm = torch.tensor([group_to_omega_row(present) for _, present in groups], dtype=torch.float32)

    mask_hashes = {}
    for split_name, idx in indices.items():
        mask_hashes[split_name] = {}
        for group_index, (group_name, _) in enumerate(groups):
            payload = torch.cat(
                [idx.view(-1, 1).float(), omega_gm[group_index].view(1, -1).expand(idx.numel(), -1)], dim=1
            )
            mask_hashes[split_name][group_name] = tensor_sha256(payload)

    capacity_rows = []
    train_shape = (indices["idx_train"].numel(), *x_full.shape[1:])
    for arm in ARM_ORDER:
        capacity_rows.append({"arm": arm, "scope": "train", **training_capacity(arm, train_shape, completion_cfg)})

    manifest = {
        "experiment": "T2_population_vs_per_subject_completion",
        "status": "DRY_RUN_VALIDATED" if args.dry_run else "RUNNING",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_path": str(config_path),
        "config_sha256": stable_hash_json(config),
        "input": {
            "x_full_path": str(xfull_path),
            "split_path": str(split_path),
            "shape": list(x_full.shape),
            "x_full_sha256": tensor_sha256(x_full),
            "labels_sha256": tensor_sha256(y_all),
            "split_sha256": stable_hash_json(split),
        },
        "environment": environment_manifest(device),
        "groups": [{"name": name, "present": list(present)} for name, present in groups],
        "mask_hashes": mask_hashes,
        "invariants": {
            "same_split": True,
            "same_15_masks": True,
            "same_classifier_architecture": "ConvTRLNet",
            "same_classifier_seed": seed,
            "same_threshold_rule": "single validation threshold maximizing macro BAC over 15 groups",
            "population_arm_only_difference": "lambda_subject is zero versus configured non-zero value",
            "graph_classifier_used": False,
        },
        "capacity": capacity_rows,
    }
    atomic_json_dump(manifest, output_dir / "protocol_manifest.json")
    pd.DataFrame(capacity_rows).to_csv(output_dir / "capacity.csv", index=False)
    print(f"Protocol validated. Output: {output_dir}", flush=True)
    if args.dry_run:
        return

    x_train = x_full.index_select(0, indices["idx_train"])
    y_train = y_all.index_select(0, indices["idx_train"])
    split_tensors = {
        "val": (x_full.index_select(0, indices["idx_val"]), y_all.index_select(0, indices["idx_val"])),
        "test": (x_full.index_select(0, indices["idx_test"]), y_all.index_select(0, indices["idx_test"])),
    }
    combination_rows: List[Dict] = []
    quality_detail_rows: List[Dict] = []
    prediction_rows: List[Dict] = []
    resource_rows: List[Dict] = []
    classifier_rows: List[Dict] = []
    checkpoint_dir = output_dir / "checkpoints"
    if bool(config.get("save_checkpoints", False)):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for arm_index, arm in enumerate(ARM_ORDER):
        print(f"\n[T2] Arm {arm_index + 1}/3: {arm}", flush=True)
        basis = None
        if arm == ARM_PER_SUBJECT:
            xhat_train, train_stats = fit_per_subject_training_completion(x_train, omega_gm, completion_cfg, device)
        else:
            use_graph = arm == ARM_FULL_COTEG
            basis, xhat_train, train_stats = fit_population_training_basis(
                x_train, omega_gm, completion_cfg, device, use_subject_graph=use_graph
            )
        resource_rows.append(
            {
                "arm": arm,
                "stage": "completion_train",
                "split": "train",
                "group": "cyclic_15",
                **train_stats.__dict__,
            }
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        classifier, classifier_info = train_common_classifier(
            x_train,
            xhat_train,
            y_train,
            omega_gm,
            config["classifier"],
            seed,
            device,
        )
        classifier_rows.append(
            {
                "arm": arm,
                "seconds": classifier_info["seconds"],
                "peak_cuda_allocated_bytes": classifier_info["peak_cuda_allocated_bytes"],
                "parameters": classifier_info["parameters"],
            }
        )
        pd.DataFrame(classifier_info["history"]).to_csv(output_dir / f"classifier_history_{arm}.csv", index=False)
        if bool(config.get("save_checkpoints", False)):
            torch.save(classifier.state_dict(), checkpoint_dir / f"classifier_{arm}.pth")

        probabilities = {"val": {}, "test": {}}
        labels = {"val": {}, "test": {}}
        quality_by_group = {"test": {}}
        inference_capacity_recorded = False
        for split_offset, split_name in enumerate(("val", "test")):
            x_split, y_split = split_tensors[split_name]
            for group_index, (group_name, present) in enumerate(groups):
                print(f"  {split_name}/{group_name}", flush=True)
                omega_m = omega_gm[group_index]
                # Identical inference initialization across arms; arm identity
                # must not change the random seed in this strict comparison.
                eval_seed = seed + 1000 * split_offset + group_index
                if arm == ARM_PER_SUBJECT:
                    completed, stats, rep_params, rep_bytes = infer_per_subject_completion(
                        x_split, omega_m, completion_cfg, device, eval_seed
                    )
                else:
                    completed, stats, rep_params, rep_bytes = infer_population_completion(
                        basis,
                        x_split,
                        omega_m,
                        completion_cfg,
                        device,
                        use_subject_graph=(arm == ARM_FULL_COTEG),
                        seed=eval_seed,
                    )
                resource_rows.append(
                    {
                        "arm": arm,
                        "stage": "completion_inference",
                        "split": split_name,
                        "group": group_name,
                        **stats.__dict__,
                    }
                )
                if not inference_capacity_recorded and split_name == "test":
                    capacity_rows.append(
                        {
                            "arm": arm,
                            "scope": f"test_cohort_n{len(y_split)}",
                            "rank_s": completion_cfg.rank_s,
                            "rank_m": completion_cfg.rank_m,
                            "rank_n": 1 if arm == ARM_PER_SUBJECT else completion_cfg.rank_n,
                            "parameters_per_subject": -1,
                            "parameters_total": rep_params,
                            "factor_storage_bytes": rep_bytes,
                        }
                    )
                    inference_capacity_recorded = True
                prob = predict_probabilities(
                    classifier,
                    completed,
                    int(config["classifier"].get("eval_batch_size", config["classifier"]["batch_size"])),
                    device,
                )
                probabilities[split_name][group_name] = prob
                labels[split_name][group_name] = y_split.numpy()
                for local_index, (label, probability) in enumerate(zip(y_split.tolist(), prob.tolist())):
                    prediction_rows.append(
                        {
                            "arm": arm,
                            "split": split_name,
                            "group": group_name,
                            "global_index": int(indices[f"idx_{split_name}"][local_index]),
                            "y_true": int(label),
                            "probability": float(probability),
                        }
                    )
                if split_name == "test":
                    detail = completion_quality_rows(
                        x_split,
                        completed,
                        omega_m,
                        float(config.get("brain_threshold", 1e-6)),
                    )
                    for row in detail:
                        quality_detail_rows.append({"arm": arm, "group": group_name, **row})
                    quality_by_group["test"][group_name] = aggregate_quality(detail)
                del completed
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        threshold_cfg = config.get("threshold", {})
        grid = np.arange(
            float(threshold_cfg.get("min", 0.05)),
            float(threshold_cfg.get("max", 0.95)) + 0.5 * float(threshold_cfg.get("step", 0.01)),
            float(threshold_cfg.get("step", 0.01)),
        )
        threshold, val_macro_bac = select_macro_bac_threshold(labels["val"], probabilities["val"], grid)
        print(f"  selected validation threshold={threshold:.3f}, val macro BAC={val_macro_bac:.4f}", flush=True)
        for group_name, present in groups:
            y_test = labels["test"][group_name]
            p_test = probabilities["test"][group_name]
            quality = quality_by_group["test"][group_name]
            combination_rows.append(
                {
                    "arm": arm,
                    "group": group_name,
                    "present_modalities": "+".join(present),
                    "num_available": len(present),
                    "n": len(y_test),
                    "threshold": threshold,
                    "BAC": safe_bac(y_test, p_test, threshold),
                    "AUC": safe_auc(y_test, p_test),
                    **quality,
                }
            )
        del classifier, xhat_train, basis
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = [build_summary(arm, combination_rows) for arm in ARM_ORDER]
    classifier_by_arm = {row["arm"]: row for row in classifier_rows}
    for row in summary_rows:
        arm_resources = [item for item in resource_rows if item["arm"] == row["arm"]]
        row["completion_train_seconds"] = sum(
            item["seconds"] for item in arm_resources if item["stage"] == "completion_train"
        )
        row["completion_val_seconds"] = sum(
            item["seconds"]
            for item in arm_resources
            if item["stage"] == "completion_inference" and item["split"] == "val"
        )
        row["completion_test_seconds"] = sum(
            item["seconds"]
            for item in arm_resources
            if item["stage"] == "completion_inference" and item["split"] == "test"
        )
        row["completion_peak_cpu_rss_bytes"] = max(item["peak_cpu_rss_bytes"] for item in arm_resources)
        row["completion_peak_cuda_allocated_bytes"] = max(
            item["peak_cuda_allocated_bytes"] for item in arm_resources
        )
        row["classifier_train_seconds"] = classifier_by_arm[row["arm"]]["seconds"]
        row["classifier_peak_cuda_allocated_bytes"] = classifier_by_arm[row["arm"]][
            "peak_cuda_allocated_bytes"
        ]
    decision = decision_from_summary(summary_rows)
    pd.DataFrame(combination_rows).to_csv(output_dir / "per_combination.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(quality_detail_rows).to_csv(output_dir / "completion_quality_detail.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "sample_predictions.csv", index=False)
    pd.DataFrame(resource_rows).to_csv(output_dir / "resource_usage.csv", index=False)
    pd.DataFrame(classifier_rows).to_csv(output_dir / "classifier_resources.csv", index=False)
    pd.DataFrame(capacity_rows).to_csv(output_dir / "capacity.csv", index=False)
    atomic_json_dump(decision, output_dir / "decision.json")
    manifest["status"] = "COMPLETED"
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["capacity"] = capacity_rows
    manifest["primary_decision"] = decision
    atomic_json_dump(manifest, output_dir / "protocol_manifest.json")
    print("\nT2 completed.", flush=True)
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
