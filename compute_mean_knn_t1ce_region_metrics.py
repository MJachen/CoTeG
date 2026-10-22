import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from compute_t1ce_region_metrics import regional_metrics
from predataset import center_crop_or_pad3d, make_folds, minmax01


MODALITIES = ("t1", "t1ce", "t2", "flair")
T1CE_INDEX = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct Mean/KNN T1CE baselines and compute regional MAE/SSIM."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--label-xlsx", required=True)
    parser.add_argument("--completed-pack", required=True)
    parser.add_argument("--seg-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--case-id", default="unknown")
    parser.add_argument("--case-col", default="BraTS2021")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--brain-threshold", type=float, default=1e-6)
    parser.add_argument("--data-range", type=float, default=1.0)
    parser.add_argument("--ssim-win-size", type=int, default=7)
    return parser.parse_args()


def modality_path(data_root, case_id, modality):
    path = Path(data_root) / case_id / f"{case_id}_{modality}.nii.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_normalized_volume(path, shape):
    volume = nib.load(str(path)).get_fdata().astype(np.float32)
    volume = center_crop_or_pad3d(volume, shape)
    return minmax01(volume)


def reference_ids(args):
    df = pd.read_excel(args.label_xlsx)
    if args.case_col not in df.columns or args.label_col not in df.columns:
        raise KeyError(
            f"Expected columns {args.case_col!r}, {args.label_col!r}; got {list(df.columns)}"
        )
    df = df[[args.case_col, args.label_col]].dropna().reset_index(drop=True)
    names = df[args.case_col].astype(str).to_numpy()
    labels = df[args.label_col].astype(int).to_numpy()

    folds = list(make_folds(labels, args.n_splits, args.val_ratio, args.seed))
    if args.fold < 0 or args.fold >= len(folds):
        raise ValueError(f"fold must be in [0, {len(folds) - 1}]")
    _, idx_train, _, _ = folds[args.fold]
    selected = names[idx_train].tolist()
    selected = [name for name in selected if name != args.case_id]
    return selected


def build_mean_and_knn(args, gt_all):
    shape = tuple(gt_all.shape[1:])
    names = reference_ids(args)
    if args.knn_k <= 0 or args.knn_k > len(names):
        raise ValueError(f"knn-k must be in [1, {len(names)}]")

    target_feature = gt_all.mean(axis=(1, 2, 3)).astype(np.float64)
    target_feature[T1CE_INDEX] = 0.0

    t1ce_sum = np.zeros(shape, dtype=np.float64)
    features = np.zeros((len(names), len(MODALITIES)), dtype=np.float64)
    for i, case_id in enumerate(names):
        for mod_idx, modality in enumerate(MODALITIES):
            volume = load_normalized_volume(
                modality_path(args.data_root, case_id, modality), shape
            )
            features[i, mod_idx] = float(volume.mean())
            if mod_idx == T1CE_INDEX:
                t1ce_sum += volume
        if (i + 1) % 25 == 0 or (i + 1) == len(names):
            print(f"Loaded reference cases: {i + 1}/{len(names)}", flush=True)

    mean_t1ce = (t1ce_sum / len(names)).astype(np.float32)

    # Match the current KNN.py baseline: Euclidean distance over four global
    # modality means, with the missing T1CE query feature set to zero.
    distances = np.linalg.norm(features - target_feature[None, :], axis=1)
    neighbor_indices = np.argsort(distances)[: args.knn_k]
    neighbor_names = [names[i] for i in neighbor_indices]

    knn_sum = np.zeros(shape, dtype=np.float64)
    for case_id in neighbor_names:
        knn_sum += load_normalized_volume(
            modality_path(args.data_root, case_id, "t1ce"), shape
        )
    knn_t1ce = (knn_sum / args.knn_k).astype(np.float32)

    neighbors = pd.DataFrame(
        {
            "rank": np.arange(1, args.knn_k + 1),
            "case_id": neighbor_names,
            "distance": distances[neighbor_indices],
        }
    )
    return mean_t1ce, knn_t1ce, neighbors, len(names)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pack = torch.load(args.completed_pack, map_location="cpu")
    gt_all = pack["X_full"][0].float().numpy()
    if gt_all.shape[0] != 4:
        raise ValueError(f"Expected four modalities, got {gt_all.shape}")
    gt = gt_all[T1CE_INDEX]

    seg_raw = nib.load(args.seg_path).get_fdata().astype(np.float32)
    seg = center_crop_or_pad3d(seg_raw, gt.shape)
    seg = np.rint(seg).astype(np.uint8)
    masks = {
        "whole_brain": gt_all.max(axis=0) > args.brain_threshold,
        "tumor_region": seg > 0,
        "enhancing_region": seg == 4,
    }

    mean_t1ce, knn_t1ce, neighbors, reference_count = build_mean_and_knn(
        args, gt_all
    )
    methods = {"Mean": mean_t1ce, "KNN": knn_t1ce}

    rows = []
    total_voxels = int(gt.size)
    for method, completed in methods.items():
        abs_error = np.abs(completed - gt)
        for region, mask in masks.items():
            completed_metrics = regional_metrics(
                gt, completed, mask, args.data_range, args.ssim_win_size
            )
            abs_error_metrics = regional_metrics(
                gt, abs_error, mask, args.data_range, args.ssim_win_size
            )
            rows.append(
                {
                    "case": args.case_id,
                    "modality": "T1CE",
                    "method": method,
                    "region": region,
                    "voxel_count": int(mask.sum()),
                    "volume_fraction": float(mask.sum() / total_voxels),
                    "MAE_abs_error_vs_GT": abs_error_metrics["MAE"],
                    "SSIM_abs_error_vs_GT": abs_error_metrics["SSIM"],
                    "MAE_completed_vs_GT": completed_metrics["MAE"],
                    "SSIM_completed_vs_GT": completed_metrics["SSIM"],
                    "reference_case_count": reference_count,
                    "SSIM_data_range": args.data_range,
                    "SSIM_win_size": args.ssim_win_size,
                }
            )

    metrics = pd.DataFrame(rows)
    csv_path = out_dir / f"{args.case_id}_mean_knn_t1ce_region_mae_ssim.csv"
    xlsx_path = out_dir / f"{args.case_id}_mean_knn_t1ce_region_mae_ssim.xlsx"
    volume_path = out_dir / f"{args.case_id}_mean_knn_t1ce_volumes.pt"
    metrics.to_csv(csv_path, index=False, encoding="utf-8-sig")
    torch.save(
        {
            "case": args.case_id,
            "modality": "T1CE",
            "GT": torch.from_numpy(gt),
            "Mean": torch.from_numpy(mean_t1ce),
            "KNN": torch.from_numpy(knn_t1ce),
            "Mean_abs_error": torch.from_numpy(np.abs(mean_t1ce - gt)),
            "KNN_abs_error": torch.from_numpy(np.abs(knn_t1ce - gt)),
            "reference_case_count": reference_count,
            "knn_neighbors": neighbors.to_dict(orient="records"),
        },
        volume_path,
    )

    metadata = pd.DataFrame(
        [
            {"field": "reference_split", "value": f"fold{args.fold} train"},
            {"field": "reference_case_count", "value": reference_count},
            {"field": "mean_definition", "value": "voxel-wise T1CE mean template"},
            {
                "field": "knn_definition",
                "value": "KNN.py-compatible global modality mean features",
            },
            {"field": "knn_k", "value": args.knn_k},
            {
                "field": "regional_SSIM_definition",
                "value": "mean of the 3D local SSIM map inside each ROI",
            },
        ]
    )
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        metrics.to_excel(writer, index=False, sheet_name="regional_metrics")
        neighbors.to_excel(writer, index=False, sheet_name="knn_neighbors")
        metadata.to_excel(writer, index=False, sheet_name="metadata")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved XLSX: {xlsx_path}")
    print(f"Saved volumes: {volume_path}")
    print(metrics.to_string(index=False))
    print("KNN neighbors:")
    print(neighbors.to_string(index=False))


if __name__ == "__main__":
    main()
