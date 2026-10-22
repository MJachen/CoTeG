import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from modality_groups import MODALITY_ORDER


def build_argparser():
    p = argparse.ArgumentParser(description="Render one-case completion comparison as a PDF grid.")
    p.add_argument("--pack-dir", required=True, help="Directory containing missing_*_completed.pt files.")
    p.add_argument("--out-pdf", default=None)
    p.add_argument("--case-id", default="BraTS2021_00000")
    p.add_argument("--slice-index", type=int, default=45)
    p.add_argument(
        "--columns",
        choices=["gt_completed_error", "gt_error"],
        default="gt_completed_error",
        help="Columns to render in the PDF.",
    )
    p.add_argument(
        "--black-error-background",
        action="store_true",
        help="Set abs-error pixels outside the brain mask to zero.",
    )
    p.add_argument("--brain-threshold", type=float, default=1e-6)
    p.add_argument("--rotate-k", type=int, default=0, help="np.rot90 k before optional flips.")
    p.add_argument("--flip-lr", action="store_true", help="Flip left-right after rotation.")
    p.add_argument("--flip-ud", action="store_true", help="Flip up-down after rotation.")
    p.add_argument("--error-cmap", default="gray")
    p.add_argument("--dpi", type=int, default=300)
    return p


def orient(img, rotate_k=0, flip_lr=False, flip_ud=False):
    out = np.asarray(img)
    if rotate_k:
        out = np.rot90(out, k=int(rotate_k))
    if flip_lr:
        out = np.fliplr(out)
    if flip_ud:
        out = np.flipud(out)
    return out


def robust_window(img):
    img = np.asarray(img, dtype=np.float32)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1, 99])
    if hi - lo < 1e-8:
        return float(finite.min()), float(finite.max() + 1e-6)
    return float(lo), float(hi)


def render(args):
    rows = []
    for mod in MODALITY_ORDER:
        tag = "missing_" + mod.lower()
        path = os.path.join(args.pack_dir, f"{tag}_completed.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        pack = torch.load(path, map_location="cpu")
        mod_i = MODALITY_ORDER.index(mod)
        x_full = pack["X_full"].float()[0]
        gt = x_full[mod_i, :, :, args.slice_index].numpy()
        comp = pack["X_hat"].float()[0, mod_i, :, :, args.slice_index].numpy()
        err = np.abs(comp - gt)
        brain_mask = (x_full[:, :, :, args.slice_index].amax(dim=0).numpy() > args.brain_threshold)
        if args.black_error_background:
            err = err.copy()
            err[~brain_mask] = 0.0
        rows.append((mod, gt, comp, err))

    out_pdf = args.out_pdf or os.path.join(args.pack_dir, f"{args.case_id}_completion_comparison_z{args.slice_index}.pdf")
    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)

    col_titles = ["GT", "Completed", "Abs error"]
    if args.columns == "gt_error":
        col_titles = ["GT", "Abs error"]

    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=len(col_titles),
        figsize=(5.8 if len(col_titles) == 2 else 8.2, 10.4),
        squeeze=False,
        constrained_layout=False,
    )
    fig.suptitle(f"{args.case_id} | completion comparison | axial slice z={args.slice_index}", fontsize=15, y=0.985)
    for c, title in enumerate(col_titles):
        axes[0][c].set_title(title, fontsize=12, pad=6)

    for r, (mod, gt, comp, err) in enumerate(rows):
        gt_o = orient(gt, args.rotate_k, args.flip_lr, args.flip_ud)
        comp_o = orient(comp, args.rotate_k, args.flip_lr, args.flip_ud)
        err_o = orient(err, args.rotate_k, args.flip_lr, args.flip_ud)

        vmin, vmax = robust_window(gt_o)
        err_vmin, err_vmax = 0.0, max(float(np.percentile(err_o, 99)), 1e-6)
        if args.columns == "gt_error":
            images = [
                (gt_o, "gray", vmin, vmax),
                (err_o, args.error_cmap, err_vmin, err_vmax),
            ]
        else:
            images = [
                (gt_o, "gray", vmin, vmax),
                (comp_o, "gray", vmin, vmax),
                (err_o, args.error_cmap, err_vmin, err_vmax),
            ]

        for c, (img, cmap, lo, hi) in enumerate(images):
            ax = axes[r][c]
            ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
        axes[r][0].set_ylabel(f"Missing {mod}", fontsize=11, rotation=90, labelpad=10)

    plt.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.02, wspace=0.08, hspace=0.08)
    fig.savefig(out_pdf, format="pdf", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(out_pdf)


def main():
    render(build_argparser().parse_args())


if __name__ == "__main__":
    main()
