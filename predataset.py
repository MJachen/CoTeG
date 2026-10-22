import argparse
import json
import os
import glob

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from modality_groups import MODALITY_GROUPS, MODALITY_ORDER, group_to_omega_row, modalities_to_file_keys


def parse_shape(values):
    if len(values) != 3:
        raise argparse.ArgumentTypeError("image size must contain exactly three integers")
    return tuple(int(v) for v in values)


def build_argparser():
    p = argparse.ArgumentParser(description="Build BraTS multimodal missing-modality packs.")
    p.add_argument("--data-root", default=os.environ.get("BRATS_DATA_ROOT", r"E:\brats2021"))
    p.add_argument("--label-xlsx", default=os.environ.get("BRATS_LABEL_XLSX", r"E:\EXPS\tensorexps\brats2021_label2020.xlsx"))
    p.add_argument("--case-col", default=os.environ.get("BRATS_CASE_COL", "BraTS2021"))
    p.add_argument("--label-col", default=os.environ.get("BRATS_LABEL_COL", "label"))
    p.add_argument("--out-dir", default=os.environ.get("PACK_DIR", "./brats2021_hgg_lgg_packs"))
    p.add_argument("--image-size", nargs=3, default=os.environ.get("OUT_SHAPE", "64,64,64").split(","))
    p.add_argument("--n-splits", type=int, default=int(os.environ.get("N_SPLITS", "5")))
    p.add_argument("--val-ratio", type=float, default=float(os.environ.get("VAL_RATIO", "0.2")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "123")))
    p.add_argument("--smoke-n-per-class", type=int, default=int(os.environ.get("SMOKE_N_PER_CLASS", "0")))
    p.add_argument(
        "--materialize-train-aug",
        action="store_true",
        default=os.environ.get("MATERIALIZE_TRAIN_AUG", "0") == "1",
        help="Write train_aug_15mods.pt. This is very large for 15 groups and is disabled by default.",
    )
    return p


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def center_crop_or_pad3d(img, out_shape):
    oh, ow, od = out_shape
    h, w, d = img.shape
    ch = min(h, oh)
    cw = min(w, ow)
    cd = min(d, od)
    sh = max((h - ch) // 2, 0)
    sw = max((w - cw) // 2, 0)
    sd = max((d - cd) // 2, 0)
    cropped = img[sh:sh + ch, sw:sw + cw, sd:sd + cd]

    out = np.zeros((oh, ow, od), dtype=cropped.dtype)
    th = (oh - ch) // 2
    tw = (ow - cw) // 2
    td = (od - cd) // 2
    out[th:th + ch, tw:tw + cw, td:td + cd] = cropped
    return out


def minmax01(x, eps=1e-8):
    vmin = float(x.min())
    vmax = float(x.max())
    if vmax - vmin > eps:
        return ((x - vmin) / (vmax - vmin)).astype(np.float32)
    return np.zeros_like(x, dtype=np.float32)


def find_modality_file(case_dir, mod_key):
    patterns = [
        os.path.join(case_dir, f"*_{mod_key}.nii*"),
        os.path.join(case_dir, f"*-{mod_key}.nii*"),
        os.path.join(case_dir, f"*{mod_key}*.nii*"),
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"[{case_dir}] missing modality file for key: {mod_key}")


def load_case_4mod(case_dir, out_shape):
    vols = []
    for mk in modalities_to_file_keys(MODALITY_ORDER):
        fp = find_modality_file(case_dir, mk)
        x = nib.load(fp).get_fdata().astype(np.float32)
        x = center_crop_or_pad3d(x, out_shape)
        x = minmax01(x)
        vols.append(x)
    return np.stack(vols, axis=0)


def scan_from_label_xlsx(root_dir, label_xlsx, case_col, label_col, smoke_n_per_class=0):
    df = pd.read_excel(label_xlsx)
    if case_col not in df.columns or label_col not in df.columns:
        raise KeyError(f"label file must contain {case_col!r} and {label_col!r}; got {list(df.columns)}")

    df = df[[case_col, label_col]].dropna()
    df[case_col] = df[case_col].astype(str)
    df[label_col] = df[label_col].astype(int)

    if smoke_n_per_class > 0:
        df = df.groupby(label_col, group_keys=False).head(smoke_n_per_class).reset_index(drop=True)

    items = []
    missing = []
    for _, row in df.iterrows():
        case = row[case_col]
        label = int(row[label_col])
        case_dir = os.path.join(root_dir, case)
        if os.path.isdir(case_dir):
            items.append((case_dir, label, case))
        else:
            missing.append(case)

    if missing:
        raise FileNotFoundError(f"{len(missing)} case dirs are missing, e.g. {missing[:5]}")
    return items


def make_omega_for_group(num_samples, group_modalities):
    row = torch.tensor(group_to_omega_row(group_modalities), dtype=torch.float32)
    return row.view(1, len(MODALITY_ORDER), 1, 1, 1).repeat(num_samples, 1, 1, 1, 1)


def apply_missing(x_full, omega):
    n, m, h, w, d = x_full.shape
    return x_full * omega.expand(n, m, h, w, d)


def label_counts(labels):
    u, c = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def dump_train_aug(fold_dir, x_full, y_all, idx_train):
    x_train_full = x_full.index_select(0, torch.tensor(idx_train).long())
    y_train = y_all.index_select(0, torch.tensor(idx_train).long())

    x_list, o_list, y_list, base_list, group_list = [], [], [], [], []
    for base_i in range(len(idx_train)):
        for group_name, present in MODALITY_GROUPS:
            omega = make_omega_for_group(1, present)
            xmiss = apply_missing(x_train_full[base_i:base_i + 1], omega)
            x_list.append(xmiss)
            o_list.append(omega)
            y_list.append(y_train[base_i:base_i + 1])
            base_list.append(base_i)
            group_list.append(group_name)

    pack = {
        "X_miss": torch.cat(x_list, dim=0),
        "Omega": torch.cat(o_list, dim=0),
        "y": torch.cat(y_list, dim=0).long(),
        "base_idx": torch.tensor(base_list).long(),
        "idx_train_global": torch.tensor(idx_train).long(),
        "modality_group": group_list,
        "groups": [{"name": n, "present": p} for n, p in MODALITY_GROUPS],
        "mods": MODALITY_ORDER,
    }
    torch.save(pack, os.path.join(fold_dir, "train_aug_15mods.pt"))
    print("Saved train_aug_15mods.pt", tuple(pack["X_miss"].shape))


def dump_eval_packs(fold_dir, split_name, x_full, y_all, idx_split):
    idx_t = torch.tensor(idx_split).long()
    x_split = x_full.index_select(0, idx_t)
    y_split = y_all.index_select(0, idx_t)
    for group_name, present in MODALITY_GROUPS:
        omega = make_omega_for_group(len(idx_split), present)
        xmiss = apply_missing(x_split, omega)
        out_path = os.path.join(fold_dir, f"{split_name}_{group_name}.pt")
        torch.save(
            {
                "X_miss": xmiss,
                "Omega": omega,
                "y": y_split.long(),
                "idx": idx_t,
                "modality_group": group_name,
                "present_modalities": present,
                "mods": MODALITY_ORDER,
            },
            out_path,
        )
    print(f"Saved {split_name} packs:", len(MODALITY_GROUPS))


def make_folds(labels, n_splits, val_ratio, seed):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_idx = np.arange(len(labels))
    for fold, (trainval_idx, test_idx) in enumerate(skf.split(all_idx, labels)):
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=val_ratio,
            random_state=seed + fold,
            stratify=labels[trainval_idx],
        )
        yield fold, np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def main():
    args = build_argparser().parse_args()
    args.image_size = parse_shape(args.image_size)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Data root:", args.data_root)
    print("Label file:", args.label_xlsx)
    print("Output dir:", args.out_dir)
    print("Image size:", args.image_size)
    print("Modality groups:", len(MODALITY_GROUPS))

    items = scan_from_label_xlsx(
        args.data_root,
        args.label_xlsx,
        args.case_col,
        args.label_col,
        smoke_n_per_class=args.smoke_n_per_class,
    )
    case_dirs = [x[0] for x in items]
    labels = np.array([x[1] for x in items], dtype=np.int64)
    names = [x[2] for x in items]
    print("Cases:", len(items), "label distribution:", label_counts(labels))

    x_list = []
    for i, cdir in enumerate(case_dirs):
        x_list.append(load_case_4mod(cdir, args.image_size))
        if (i + 1) % 25 == 0 or (i + 1) == len(case_dirs):
            print(f"  loaded {i + 1}/{len(case_dirs)}")

    x_full = torch.from_numpy(np.stack(x_list, axis=0)).float()
    y_all = torch.from_numpy(labels).long()
    xfull_path = os.path.join(args.out_dir, "X_full_4mod.pt")
    torch.save(
        {
            "X_full": x_full,
            "y": y_all,
            "names": names,
            "mods": MODALITY_ORDER,
            "out_shape": args.image_size,
            "label_map": {"0": 0, "1": 1},
        },
        xfull_path,
    )
    print("Saved", xfull_path, tuple(x_full.shape))

    config = {
        "data_root": args.data_root,
        "label_xlsx": args.label_xlsx,
        "case_col": args.case_col,
        "label_col": args.label_col,
        "image_size": args.image_size,
        "n_splits": args.n_splits,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "groups": [{"name": n, "present": p} for n, p in MODALITY_GROUPS],
    }
    save_json(os.path.join(args.out_dir, "config.json"), config)

    for fold, idx_train, idx_val, idx_test in make_folds(labels, args.n_splits, args.val_ratio, args.seed):
        fold_dir = os.path.join(args.out_dir, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        print(f"\n[fold{fold}] train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")
        print("  train labels:", label_counts(labels[idx_train]))
        print("  val labels:", label_counts(labels[idx_val]))
        print("  test labels:", label_counts(labels[idx_test]))
        save_json(
            os.path.join(fold_dir, "split.json"),
            {
                "fold": fold,
                "idx_train": idx_train.tolist(),
                "idx_val": idx_val.tolist(),
                "idx_test": idx_test.tolist(),
                "train_labels": label_counts(labels[idx_train]),
                "val_labels": label_counts(labels[idx_val]),
                "test_labels": label_counts(labels[idx_test]),
            },
        )
        if args.materialize_train_aug:
            dump_train_aug(fold_dir, x_full, y_all, idx_train)
        else:
            save_json(
                os.path.join(fold_dir, "train_groups.json"),
                {
                    "note": "train_aug_15mods.pt is intentionally not materialized. GRNNTCtrain.py defaults to patient mode.",
                    "groups": [{"name": n, "present": p} for n, p in MODALITY_GROUPS],
                },
            )
            print("Skipped train_aug_15mods.pt materialization; saved train_groups.json")
        dump_eval_packs(fold_dir, "val", x_full, y_all, idx_val)
        dump_eval_packs(fold_dir, "test", x_full, y_all, idx_test)

    print("\nDone.")


if __name__ == "__main__":
    main()
