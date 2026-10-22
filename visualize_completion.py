import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from project_paths import find_xfull_path, get_pack_dir


def build_argparser():
    p = argparse.ArgumentParser(description="Visualize completed multimodal MRI packs.")
    p.add_argument("--pack-dir", default=get_pack_dir())
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--groups", nargs="*", default=None, help="Modality groups to visualize. Default: all completed packs.")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--num-cases", type=int, default=2)
    p.add_argument("--axis", choices=["H", "W", "D"], default="D")
    p.add_argument("--slice-index", type=int, default=None)
    return p


def discover_completed_packs(pack_dir, split, groups):
    if groups:
        paths = [os.path.join(pack_dir, f"{split}_{g}_completed.pt") for g in groups]
    else:
        paths = sorted(glob.glob(os.path.join(pack_dir, f"{split}_*_completed.pt")))
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing completed packs: {missing[:5]}")
    return paths


def slice_3d(volume, axis, slice_index):
    if axis == "H":
        return volume[slice_index, :, :]
    if axis == "W":
        return volume[:, slice_index, :]
    return volume[:, :, slice_index]


def normalize_img(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def render_case(out_path, x_full, x_miss, x_hat, omega, mods, axis, slice_index):
    rows = len(mods)
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)
    col_titles = ["GT", "Missing input", "Completed", "Abs error"]

    for r, mod in enumerate(mods):
        gt = slice_3d(x_full[r], axis, slice_index)
        miss = slice_3d(x_miss[r], axis, slice_index)
        hat = slice_3d(x_hat[r], axis, slice_index)
        err = np.abs(hat - gt)
        images = [gt, miss, hat, err]
        observed = bool(omega[r] > 0.5)
        for c, img in enumerate(images):
            ax = axes[r][c]
            ax.imshow(normalize_img(img), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c])
            if c == 0:
                ax.set_ylabel(f"{mod}\n{'obs' if observed else 'miss'}")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = build_argparser().parse_args()
    out_dir = args.out_dir or os.path.join(args.pack_dir, "visualizations", args.split)
    xfull = torch.load(find_xfull_path(args.pack_dir), map_location="cpu")
    x_full_all = xfull["X_full"].float()
    mods = xfull.get("mods", ["T1", "T1CE", "T2", "FLAIR"])

    for pack_path in discover_completed_packs(args.pack_dir, args.split, args.groups):
        pack = torch.load(pack_path, map_location="cpu")
        group = pack.get("modality_group", os.path.basename(pack_path).replace(f"{args.split}_", "").replace("_completed.pt", ""))
        x_miss = pack["X_miss"].float()
        x_hat = pack["X_hat"].float()
        omega = pack["Omega"].float().view(x_hat.shape[0], x_hat.shape[1])
        idx = pack["idx"].long()

        n = min(args.num_cases, x_hat.shape[0])
        for i in range(n):
            global_idx = int(idx[i])
            h, w, d = x_hat.shape[2:]
            default_slice = {"H": h // 2, "W": w // 2, "D": d // 2}[args.axis]
            slice_index = default_slice if args.slice_index is None else args.slice_index
            out_path = os.path.join(out_dir, str(group), f"{global_idx}_{args.axis}{slice_index}.png")
            render_case(
                out_path,
                x_full_all[global_idx].numpy(),
                x_miss[i].numpy(),
                x_hat[i].numpy(),
                omega[i].numpy(),
                mods,
                args.axis,
                slice_index,
            )
            print("Saved", out_path)


if __name__ == "__main__":
    main()
