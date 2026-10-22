import argparse
import glob
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

import GRNNTCeval
from modality_groups import MODALITY_ORDER, group_to_omega_row
from predataset import center_crop_or_pad3d, minmax01
from project_paths import get_pack_dir


MODALITY_FILE_KEYS = {
    "T1": ("t1", "t1n"),
    "T1CE": ("t1ce", "t1c"),
    "T2": ("t2", "t2w"),
    "FLAIR": ("flair", "t2f"),
}


def parse_shape(values):
    if len(values) != 3:
        raise argparse.ArgumentTypeError("image size must contain exactly three integers")
    return tuple(int(v) for v in values)


def build_argparser():
    p = argparse.ArgumentParser(
        description="Smoke visualization for one-case one-missing-modality completion."
    )
    p.add_argument("--data-root", default=os.environ.get("BRATS_DATA_ROOT", r"E:\brats2021"))
    p.add_argument("--case-id", default=os.environ.get("CASE_ID", "BraTS2021_0000"))
    p.add_argument(
        "--factors-path",
        default=os.environ.get("FACTORS_PATH", os.path.join(get_pack_dir(), "fold0", "tucker_factors_train.pt")),
        help="Train Tucker factors from GRNNTCtrain.py.",
    )
    p.add_argument("--out-dir", default=os.environ.get("VIS_OUT_DIR", "./single_case_completion_vis"))
    p.add_argument("--image-size", nargs=3, default=os.environ.get("OUT_SHAPE", "128,128,96").split(","))
    p.add_argument("--max-iter", type=int, default=int(os.environ.get("VIS_EVAL_MAX_ITER", "80")))
    p.add_argument("--chunk-s", type=int, default=int(os.environ.get("VIS_EVAL_CHUNK_S", "16384")))
    p.add_argument("--lr", type=float, default=float(os.environ.get("VIS_EVAL_LR", "1e-3")))
    p.add_argument("--lambda-subj", type=float, default=float(os.environ.get("VIS_EVAL_LAMBDA_SUBJ", "5e-4")))
    p.add_argument(
        "--slice-index",
        type=int,
        default=os.environ.get("VIS_SLICE_INDEX", None),
        help="Axial slice index. Default: slice with largest tumor-mask area.",
    )
    p.add_argument("--dpi", type=int, default=int(os.environ.get("VIS_DPI", "180")))
    return p


def find_case_dir(data_root, case_id):
    if os.path.isdir(case_id):
        return case_id

    direct = os.path.join(data_root, case_id)
    if os.path.isdir(direct):
        return direct

    wanted = case_id.lower()
    variants = {wanted, wanted.replace("brats2021_", "brats21_")}
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if not os.path.isdir(path):
            continue
        lname = name.lower()
        if lname in variants or wanted in lname or lname.endswith(wanted[-4:]):
            return path

    hits = sorted(glob.glob(os.path.join(data_root, "**", case_id), recursive=True))
    hits = [h for h in hits if os.path.isdir(h)]
    if hits:
        return hits[0]

    raise FileNotFoundError(f"Cannot find case {case_id!r} under {data_root!r}")


def find_first_file(case_dir, keys):
    patterns = []
    for key in keys:
        patterns.extend(
            [
                os.path.join(case_dir, f"*_{key}.nii*"),
                os.path.join(case_dir, f"*-{key}.nii*"),
                os.path.join(case_dir, f"*{key}.nii*"),
            ]
        )
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"[{case_dir}] no file matched keys: {keys}")


def find_seg_file(case_dir):
    return find_first_file(case_dir, ("seg", "mask", "label", "tumor"))


def load_case(case_dir, out_shape):
    vols = []
    files = {}
    for mod in MODALITY_ORDER:
        fp = find_first_file(case_dir, MODALITY_FILE_KEYS[mod])
        files[mod] = fp
        x = nib.load(fp).get_fdata().astype(np.float32)
        x = center_crop_or_pad3d(x, out_shape)
        vols.append(minmax01(x))
    x_full = np.stack(vols, axis=0)

    seg_path = find_seg_file(case_dir)
    seg = nib.load(seg_path).get_fdata()
    seg = center_crop_or_pad3d(seg.astype(np.float32), out_shape)
    tumor_mask = seg > 0
    return x_full, tumor_mask, files, seg_path


def largest_axial_slice(mask_hwd):
    areas = mask_hwd.sum(axis=(0, 1))
    if float(areas.max()) <= 0:
        return int(mask_hwd.shape[2] // 2)
    return int(np.argmax(areas))


def normalize_img(x):
    x = np.asarray(x, dtype=np.float32)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def render_missing_modality(out_path, case_id, missing_mod, x_full, x_miss, x_hat, mask, slice_idx, dpi):
    mod_i = MODALITY_ORDER.index(missing_mod)
    gt = x_full[mod_i, :, :, slice_idx]
    miss = x_miss[mod_i, :, :, slice_idx]
    hat = x_hat[0, mod_i, :, :, slice_idx]
    err = np.abs(hat - gt)
    tumor = mask[:, :, slice_idx].astype(np.float32)

    titles = ["Ground truth", "Input with modality missing", "Completed", "Abs error"]
    images = [gt, miss, hat, err]
    cmaps = ["gray", "gray", "gray", "magma"]

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), squeeze=False)
    for i, ax in enumerate(axes[0]):
        ax.imshow(normalize_img(images[i]), cmap=cmaps[i])
        if tumor.max() > 0:
            ax.contour(tumor, levels=[0.5], colors=["#ff3b30"], linewidths=0.8)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{case_id} | missing {missing_mod} | axial z={slice_idx}", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_eval_pack(out_path, x_full_np, missing_mod):
    present = [m for m in MODALITY_ORDER if m != missing_mod]
    omega = torch.tensor(group_to_omega_row(present), dtype=torch.float32).view(1, 4, 1, 1, 1)
    x_full = torch.from_numpy(x_full_np[None]).float()
    x_miss = x_full * omega.expand_as(x_full)
    pack = {
        "X_miss": x_miss,
        "Omega": omega,
        "X_full": x_full,
        "y": torch.tensor([0], dtype=torch.long),
        "idx": torch.tensor([0], dtype=torch.long),
        "modality_group": "missing_" + missing_mod.lower(),
        "present_modalities": present,
        "mods": MODALITY_ORDER,
    }
    torch.save(pack, out_path)


def main():
    args = build_argparser().parse_args()
    args.image_size = parse_shape(args.image_size)

    if not os.path.exists(args.factors_path):
        raise FileNotFoundError(f"Factors not found: {args.factors_path}")

    case_dir = find_case_dir(args.data_root, args.case_id)
    case_name = os.path.basename(case_dir)
    print("Case dir:", case_dir)
    print("Factors:", args.factors_path)
    print("Output dir:", args.out_dir)
    print("Image size:", args.image_size)

    x_full, tumor_mask, files, seg_path = load_case(case_dir, args.image_size)
    slice_idx = largest_axial_slice(tumor_mask) if args.slice_index is None else int(args.slice_index)
    if not (0 <= slice_idx < args.image_size[2]):
        raise ValueError(f"slice-index must be in [0, {args.image_size[2] - 1}], got {slice_idx}")
    print("Modality files:")
    for mod in MODALITY_ORDER:
        print(f"  {mod}: {files[mod]}")
    print("Seg/mask:", seg_path)
    print("Axial visualization slice:", slice_idx)

    GRNNTCeval.MAX_ITER = int(args.max_iter)
    GRNNTCeval.CHUNK_S = int(args.chunk_s)
    GRNNTCeval.LR = float(args.lr)
    GRNNTCeval.LAMBDA_SUBJ = float(args.lambda_subj)

    smoke_dir = os.path.join(args.out_dir, case_name, "packs")
    vis_dir = os.path.join(args.out_dir, case_name, "png")
    os.makedirs(smoke_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    for missing_mod in MODALITY_ORDER:
        tag = "missing_" + missing_mod.lower()
        eval_pack = os.path.join(smoke_dir, f"{tag}.pt")
        completed_pack = os.path.join(smoke_dir, f"{tag}_completed.pt")
        make_eval_pack(eval_pack, x_full, missing_mod)

        print("\n" + "=" * 80)
        print(f"Completing case={case_name}, missing={missing_mod}")
        GRNNTCeval.run_one(eval_pack, args.factors_path, completed_pack)

        out = torch.load(completed_pack, map_location="cpu")
        x_miss = out["X_miss"][0].float().numpy()
        x_hat = out["X_hat"].float().numpy()
        png_path = os.path.join(vis_dir, f"{case_name}_{tag}_axial_z{slice_idx}.png")
        render_missing_modality(
            png_path,
            case_name,
            missing_mod,
            x_full,
            x_miss,
            x_hat,
            tumor_mask,
            slice_idx,
            dpi=args.dpi,
        )
        print("Saved visualization:", png_path)

    print("\nDone.")
    print("Visualizations:", vis_dir)


if __name__ == "__main__":
    main()
