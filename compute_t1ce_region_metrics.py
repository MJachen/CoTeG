import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity

from predataset import center_crop_or_pad3d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute regional MAE/SSIM for a completed T1CE volume."
    )
    parser.add_argument("--completed-pack", required=True)
    parser.add_argument("--seg-path", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--case-id", default="unknown")
    parser.add_argument("--brain-threshold", type=float, default=1e-6)
    parser.add_argument("--data-range", type=float, default=1.0)
    parser.add_argument("--ssim-win-size", type=int, default=7)
    return parser.parse_args()


def regional_metrics(gt, candidate, mask, data_range, win_size):
    voxel_count = int(mask.sum())
    if voxel_count == 0:
        return {"MAE": np.nan, "SSIM": np.nan}

    _, ssim_map = structural_similarity(
        gt.astype(np.float32),
        candidate.astype(np.float32),
        data_range=data_range,
        win_size=win_size,
        full=True,
    )
    return {
        "MAE": float(np.mean(np.abs(candidate[mask] - gt[mask]))),
        "SSIM": float(np.mean(ssim_map[mask])),
    }


def main():
    args = parse_args()
    completed_path = Path(args.completed_pack)
    seg_path = Path(args.seg_path)
    out_dir = Path(args.out_dir) if args.out_dir else completed_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pack = torch.load(completed_path, map_location="cpu")
    gt_all = pack["X_full"][0].float().numpy()
    if gt_all.shape[0] != 4:
        raise ValueError(f"Expected four modalities, got shape {gt_all.shape}")

    # Modality order is T1, T1CE, T2, FLAIR.
    gt = gt_all[1]
    completed = np.clip(pack["X_hat"][0, 1].float().numpy(), 0.0, 1.0)
    abs_error = np.abs(completed - gt)

    seg_raw = nib.load(str(seg_path)).get_fdata().astype(np.float32)
    seg = center_crop_or_pad3d(seg_raw, gt.shape)
    seg = np.rint(seg).astype(np.uint8)

    masks = {
        "whole_brain": gt_all.max(axis=0) > args.brain_threshold,
        "tumor_region": seg > 0,
        "enhancing_region": seg == 4,
    }

    total_voxels = int(gt.size)
    rows = []
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
                "region": region,
                "voxel_count": int(mask.sum()),
                "volume_fraction": float(mask.sum() / total_voxels),
                "MAE_abs_error_vs_GT": abs_error_metrics["MAE"],
                "SSIM_abs_error_vs_GT": abs_error_metrics["SSIM"],
                "MAE_completed_vs_GT": completed_metrics["MAE"],
                "SSIM_completed_vs_GT": completed_metrics["SSIM"],
                "SSIM_data_range": args.data_range,
                "SSIM_win_size": args.ssim_win_size,
            }
        )

    df = pd.DataFrame(rows)
    stem = f"{args.case_id}_t1ce_region_mae_ssim"
    csv_path = out_dir / f"{stem}.csv"
    xlsx_path = out_dir / f"{stem}.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    metadata = pd.DataFrame(
        [
            {"field": "completed_pack", "value": str(completed_path)},
            {"field": "segmentation", "value": str(seg_path)},
            {"field": "image_shape", "value": "x".join(map(str, gt.shape))},
            {"field": "segmentation_labels", "value": ",".join(map(str, np.unique(seg)))},
            {
                "field": "regional_SSIM_definition",
                "value": "mean of the 3D local SSIM map inside each ROI",
            },
        ]
    )
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="regional_metrics")
        metadata.to_excel(writer, index=False, sheet_name="metadata")

    print(f"Segmentation labels after preprocessing: {np.unique(seg, return_counts=True)}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved XLSX: {xlsx_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
