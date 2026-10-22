import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity


MODALITIES = [("t1", "T1"), ("t1ce", "T1CE"), ("t2", "T2"), ("flair", "FLAIR")]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute MAE and SSIM between completed missing modalities and GT."
    )
    parser.add_argument(
        "--pack-dir",
        required=True,
        help="Directory containing missing_<modality>_completed.pt files.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to the parent of --pack-dir.",
    )
    parser.add_argument("--case-id", default="unknown")
    parser.add_argument("--slice-index", type=int, default=45)
    parser.add_argument("--brain-threshold", type=float, default=1e-6)
    parser.add_argument("--data-range", type=float, default=1.0)
    return parser.parse_args()


def bbox_from_mask(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(slice(None) for _ in range(mask.ndim))
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def safe_ssim(gt, pred, data_range):
    if min(gt.shape) < 7:
        return np.nan
    return float(
        structural_similarity(
            gt.astype(np.float32),
            pred.astype(np.float32),
            data_range=data_range,
        )
    )


def main():
    args = parse_args()
    pack_dir = Path(args.pack_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pack_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    mod_to_idx = {key: i for i, (key, _) in enumerate(MODALITIES)}

    for key, name in MODALITIES:
        path = pack_dir / f"missing_{key}_completed.pt"
        if not path.exists():
            rows.append({"case": args.case_id, "missing_modality": name, "error": str(path)})
            continue

        pack = torch.load(path, map_location="cpu")
        idx = mod_to_idx[key]
        gt_all = pack["X_full"][0].float().numpy()
        gt = gt_all[idx]
        pred = pack["X_hat"][0, idx].float().numpy()
        pred = np.clip(pred, 0.0, 1.0)

        brain_mask_3d = gt_all.max(axis=0) > args.brain_threshold
        bbox3 = bbox_from_mask(brain_mask_3d)
        gt_crop = gt[bbox3]
        pred_crop = pred[bbox3]

        z = args.slice_index
        if z < 0 or z >= gt.shape[2]:
            raise ValueError(f"--slice-index must be in [0, {gt.shape[2] - 1}], got {z}")
        slice_mask = brain_mask_3d[:, :, z]
        bbox2 = bbox_from_mask(slice_mask)
        gt_z = gt[:, :, z]
        pred_z = pred[:, :, z]
        abs_error = np.abs(pred - gt)
        abs_error_z = abs_error[:, :, z]
        gt_z_crop = gt_z[bbox2]
        pred_z_crop = pred_z[bbox2]
        abs_error_z_crop = abs_error_z[bbox2]

        rows.append(
            {
                "case": args.case_id,
                "missing_modality": name,
                "completed_file": path.name,
                "image_shape": "x".join(str(v) for v in gt.shape),
                "axial_slice_z": z,
                "MAE_volume_all_voxels": float(np.mean(np.abs(pred - gt))),
                "SSIM_volume_all_voxels": safe_ssim(gt, pred, args.data_range),
                "MAE_brain_bbox": float(np.mean(np.abs(pred_crop - gt_crop))),
                "SSIM_brain_bbox": safe_ssim(gt_crop, pred_crop, args.data_range),
                "MAE_axial_z45_brain_bbox": float(np.mean(np.abs(pred_z_crop - gt_z_crop))),
                "SSIM_axial_z45_brain_bbox": safe_ssim(gt_z_crop, pred_z_crop, args.data_range),
                "MAE_abs_error_vs_GT_volume": float(np.mean(np.abs(abs_error - gt))),
                "SSIM_abs_error_vs_GT_volume": safe_ssim(gt, abs_error, args.data_range),
                "MAE_abs_error_vs_GT_brain_bbox": float(np.mean(np.abs(abs_error[bbox3] - gt_crop))),
                "SSIM_abs_error_vs_GT_brain_bbox": safe_ssim(gt_crop, abs_error[bbox3], args.data_range),
                "MAE_abs_error_vs_GT_axial_z45_brain_bbox": float(
                    np.mean(np.abs(abs_error_z_crop - gt_z_crop))
                ),
                "SSIM_abs_error_vs_GT_axial_z45_brain_bbox": safe_ssim(
                    gt_z_crop, abs_error_z_crop, args.data_range
                ),
                "GT_min": float(gt.min()),
                "GT_max": float(gt.max()),
                "completed_min": float(pred.min()),
                "completed_max": float(pred.max()),
                "abs_error_min": float(abs_error.min()),
                "abs_error_max": float(abs_error.max()),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "completion_quality_mae_ssim.csv"
    xlsx_path = out_dir / "completion_quality_mae_ssim.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="MAE_SSIM")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved XLSX: {xlsx_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
