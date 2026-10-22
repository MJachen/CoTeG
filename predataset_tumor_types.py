import argparse
import os
import glob

import nibabel as nib
import numpy as np
import torch

from predataset import (
    apply_missing,
    center_crop_or_pad3d,
    dump_eval_packs,
    dump_train_aug,
    label_counts,
    make_folds,
    minmax01,
    save_json,
    set_seed,
)
from modality_groups import MODALITY_GROUPS, MODALITY_ORDER


DEFAULT_ALIASES = {
    "T1": ("t1n", "t1"),
    "T1CE": ("t1c", "t1ce"),
    "T2": ("t2w", "t2"),
    "FLAIR": ("t2f", "flair"),
}


def parse_shape(values):
    if len(values) != 3:
        raise argparse.ArgumentTypeError("image size must contain exactly three integers")
    return tuple(int(v) for v in values)


def build_argparser():
    p = argparse.ArgumentParser(
        description="Build current GRNNTC packs from folder-labeled BraTS2025 tumor-type data."
    )
    p.add_argument("--data-root", default=os.environ.get("BRATS2025_DATA_ROOT", "/home/cjc/brats2025"))
    p.add_argument("--out-dir", default=os.environ.get("BRATS2025_OUT_DIR", "./brats2025_tumor_tasks"))
    p.add_argument("--classes", nargs="+", default=os.environ.get("BRATS2025_CLASSES", "GLI MEN MET").split())
    p.add_argument(
        "--tasks",
        nargs="+",
        default=os.environ.get("BRATS2025_TASKS", "all3 pairwise").split(),
        choices=("all3", "pairwise"),
        help="Build one 3-class task, all pairwise binary tasks, or both.",
    )
    p.add_argument("--image-size", nargs=3, default=os.environ.get("OUT_SHAPE", "128,128,96").split(","))
    p.add_argument("--n-splits", type=int, default=int(os.environ.get("N_SPLITS", "5")))
    p.add_argument("--val-ratio", type=float, default=float(os.environ.get("VAL_RATIO", "0.2")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "123")))
    p.add_argument("--smoke-n-per-class", type=int, default=int(os.environ.get("SMOKE_N_PER_CLASS", "0")))
    p.add_argument(
        "--materialize-train-aug",
        action="store_true",
        default=os.environ.get("MATERIALIZE_TRAIN_AUG", "0") == "1",
        help="Write train_aug_15mods.pt. This is very large and is disabled by default.",
    )
    return p


def find_modality_file(case_dir, modality):
    aliases = DEFAULT_ALIASES[modality]
    patterns = []
    for key in aliases:
        patterns.extend(
            [
                os.path.join(case_dir, f"*-{key}.nii*"),
                os.path.join(case_dir, f"*_{key}.nii*"),
                os.path.join(case_dir, f"*{key}.nii*"),
            ]
        )
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"[{case_dir}] missing modality {modality}; tried aliases {aliases}")


def load_case_4mod(case_dir, out_shape):
    vols = []
    files = {}
    for modality in MODALITY_ORDER:
        fp = find_modality_file(case_dir, modality)
        files[modality] = os.path.basename(fp)
        x = nib.load(fp).get_fdata().astype(np.float32)
        x = center_crop_or_pad3d(x, out_shape)
        x = minmax01(x)
        vols.append(x)
    return np.stack(vols, axis=0), files


def scan_folder_classes(data_root, classes, smoke_n_per_class=0):
    items = []
    missing_class_dirs = []
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            missing_class_dirs.append(class_dir)
            continue
        case_dirs = [
            os.path.join(class_dir, name)
            for name in sorted(os.listdir(class_dir))
            if os.path.isdir(os.path.join(class_dir, name))
        ]
        if smoke_n_per_class > 0:
            case_dirs = case_dirs[:smoke_n_per_class]
        for case_dir in case_dirs:
            items.append(
                {
                    "case_dir": case_dir,
                    "label": label,
                    "name": os.path.basename(case_dir),
                    "class_name": class_name,
                }
            )
    if missing_class_dirs:
        raise FileNotFoundError(f"Class directories not found: {missing_class_dirs}")
    if not items:
        raise RuntimeError(f"No case directories found under {data_root} for classes {classes}")
    return items


def validate_split_feasibility(labels, n_splits):
    counts = label_counts(labels)
    too_small = {k: v for k, v in counts.items() if v < n_splits}
    if too_small:
        raise ValueError(
            f"Each class must have at least n_splits={n_splits} cases for stratified CV; got {too_small}"
        )


def task_specs(classes, tasks):
    specs = []
    if "all3" in tasks:
        specs.append(("all3", list(classes)))
    if "pairwise" in tasks:
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                a, b = classes[i], classes[j]
                specs.append((f"{a}_vs_{b}", [a, b]))
    return specs


def build_one_task(args, task_name, task_classes):
    task_out = os.path.join(args.out_dir, task_name)
    os.makedirs(task_out, exist_ok=True)

    items = scan_folder_classes(args.data_root, task_classes, args.smoke_n_per_class)
    labels = np.array([item["label"] for item in items], dtype=np.int64)
    names = [item["name"] for item in items]
    class_names = [item["class_name"] for item in items]
    label_map = {name: i for i, name in enumerate(task_classes)}
    validate_split_feasibility(labels, args.n_splits)

    print("\n" + "=" * 80)
    print(f"Task: {task_name}")
    print("Classes:", label_map)
    print("Output dir:", task_out)
    print("Cases:", len(items), "label distribution:", label_counts(labels))

    x_list = []
    file_records = []
    for i, item in enumerate(items):
        x_case, files = load_case_4mod(item["case_dir"], args.image_size)
        x_list.append(x_case)
        file_records.append({"case": item["name"], "class": item["class_name"], "files": files})
        if (i + 1) % 25 == 0 or (i + 1) == len(items):
            print(f"  loaded {i + 1}/{len(items)}")

    x_full = torch.from_numpy(np.stack(x_list, axis=0)).float()
    y_all = torch.from_numpy(labels).long()
    xfull_path = os.path.join(task_out, "X_full_4mod.pt")
    torch.save(
        {
            "X_full": x_full,
            "y": y_all,
            "names": names,
            "classes": class_names,
            "mods": MODALITY_ORDER,
            "out_shape": args.image_size,
            "label_map": label_map,
            "modality_aliases": {k: list(v) for k, v in DEFAULT_ALIASES.items()},
        },
        xfull_path,
    )
    print("Saved", xfull_path, tuple(x_full.shape))

    save_json(
        os.path.join(task_out, "config.json"),
        {
            "task_name": task_name,
            "data_root": args.data_root,
            "classes": task_classes,
            "label_map": label_map,
            "image_size": args.image_size,
            "n_splits": args.n_splits,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "modalities": MODALITY_ORDER,
            "modality_aliases": {k: list(v) for k, v in DEFAULT_ALIASES.items()},
            "groups": [{"name": n, "present": p} for n, p in MODALITY_GROUPS],
            "files": file_records,
        },
    )

    for fold, idx_train, idx_val, idx_test in make_folds(labels, args.n_splits, args.val_ratio, args.seed):
        fold_dir = os.path.join(task_out, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        print(f"\n[{task_name} fold{fold}] train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")
        print("  train labels:", label_counts(labels[idx_train]))
        print("  val labels:", label_counts(labels[idx_val]))
        print("  test labels:", label_counts(labels[idx_test]))
        save_json(
            os.path.join(fold_dir, "split.json"),
            {
                "task_name": task_name,
                "fold": fold,
                "label_map": label_map,
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


def main():
    args = build_argparser().parse_args()
    args.image_size = parse_shape(args.image_size)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Data root:", args.data_root)
    print("Output root:", args.out_dir)
    print("Image size:", args.image_size)
    print("Tasks:", args.tasks)
    print("Requested classes:", args.classes)

    for task_name, classes in task_specs(args.classes, args.tasks):
        build_one_task(args, task_name, classes)

    print("\nDone.")


if __name__ == "__main__":
    main()
