import argparse
import glob
import json
import os
import shutil


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Create a zero-filled image-completion ablation pack. "
            "Completed image tensors are replaced by X_miss, while Tucker factor files are copied."
        )
    )
    p.add_argument("--src-pack-dir", required=True, help="Source fold directory with full completed packs.")
    p.add_argument("--out-pack-dir", required=True, help="Output fold directory for zero-filled ablation.")
    p.add_argument(
        "--train-zero-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="dtype for zero X_hat_train_5d.pt. float16 saves disk space and is cast to float during training.",
    )
    p.add_argument(
        "--copy-xfull",
        action="store_true",
        help="Copy parent X_full_4mod.pt into the output parent directory. This can be very large.",
    )
    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_if_exists(src, dst):
    if os.path.exists(src):
        ensure_dir(os.path.dirname(dst))
        shutil.copy2(src, dst)
        print("copied:", src, "->", dst)


def save_zero_train_xhat(src_pack_dir, out_pack_dir, dtype):
    import torch

    src_xhat = os.path.join(src_pack_dir, "X_hat_train_5d.pt")
    if not os.path.exists(src_xhat):
        raise FileNotFoundError(f"Missing source train completion: {src_xhat}")
    xhat = torch.load(src_xhat, map_location="cpu")
    zero_dtype = torch.float16 if dtype == "float16" else torch.float32
    zero = torch.zeros(tuple(xhat.shape), dtype=zero_dtype)
    out_xhat = os.path.join(out_pack_dir, "X_hat_train_5d.pt")
    torch.save(zero, out_xhat)
    print("saved zero train X_hat:", out_xhat, tuple(zero.shape), zero.dtype)


def convert_completed_packs(src_pack_dir, out_pack_dir):
    import torch

    paths = sorted(glob.glob(os.path.join(src_pack_dir, "val_*_completed.pt")))
    paths += sorted(glob.glob(os.path.join(src_pack_dir, "test_*_completed.pt")))
    if not paths:
        raise FileNotFoundError(f"No val/test completed packs found in {src_pack_dir}")

    for src in paths:
        base = os.path.basename(src)
        pack = torch.load(src, map_location="cpu")
        if "X_miss" not in pack:
            raise KeyError(f"{src} missing key X_miss")
        pack["X_hat"] = pack["X_miss"].clone()
        out = os.path.join(out_pack_dir, base)
        torch.save(pack, out)
        print("saved zero completed pack:", out)

        fac_src = src.replace(".pt", "_tucker_factors.pt")
        fac_out = out.replace(".pt", "_tucker_factors.pt")
        copy_if_exists(fac_src, fac_out)


def main():
    args = parse_args()
    src_pack_dir = os.path.abspath(args.src_pack_dir)
    out_pack_dir = os.path.abspath(args.out_pack_dir)
    ensure_dir(out_pack_dir)

    for name in ["split.json", "train_groups.json", "branch_meta.pt", "tucker_factors_train.pt"]:
        copy_if_exists(os.path.join(src_pack_dir, name), os.path.join(out_pack_dir, name))

    save_zero_train_xhat(src_pack_dir, out_pack_dir, args.train_zero_dtype)
    convert_completed_packs(src_pack_dir, out_pack_dir)

    src_parent = os.path.dirname(src_pack_dir)
    out_parent = os.path.dirname(out_pack_dir)
    copy_if_exists(os.path.join(src_parent, "config.json"), os.path.join(out_parent, "config.json"))
    if args.copy_xfull:
        for name in ["X_full_4mod.pt", "X_full_4mod_gli_met.pt"]:
            src = os.path.join(src_parent, name)
            if os.path.exists(src):
                copy_if_exists(src, os.path.join(out_parent, name))
                break

    meta = {
        "source_pack_dir": src_pack_dir,
        "note": "Ablation pack: X_hat is replaced by zero-filled X_miss for val/test; train X_hat is all zeros.",
        "requires_x_full_path_for_training": not args.copy_xfull,
    }
    meta_path = os.path.join(out_pack_dir, "zero_completion_ablation.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("saved metadata:", meta_path)
    if not args.copy_xfull:
        print("When training branches, set X_FULL_PATH to the source parent X_full_4mod.pt.")


if __name__ == "__main__":
    main()
