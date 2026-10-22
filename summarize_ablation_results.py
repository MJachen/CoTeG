import argparse
import csv
import json
import os


METRIC_KEYS = ["ACC", "BAC", "F1", "F1_macro", "AUC", "PR_AUC", "Sensitivity", "Specificity"]


def parse_args():
    p = argparse.ArgumentParser(description="Summarize 15-modality-group ablation metrics.")
    p.add_argument(
        "--items",
        nargs="+",
        required=True,
        help="Pairs in name=metrics_json_path format, e.g. full=fold0/fusion_out/metrics_modality_groups.json",
    )
    p.add_argument("--section", default="test_per_k_thr", choices=["test_per_k_thr", "test_overall_thr"])
    p.add_argument("--out-csv", default="ablation_summary.csv")
    return p.parse_args()


def load_item(name, path, section):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sec = data[section]
    macro = sec.get("overall_k_macroavg", {})
    concat = sec.get("overall_macro_avg", {})
    by_group = sec.get("by_k", {})
    row = {
        "method": name,
        "metrics_path": os.path.abspath(path),
        "section": section,
        "n": macro.get("n", concat.get("n")),
    }
    for key in METRIC_KEYS:
        value = macro.get(key)
        if value is None and by_group:
            vals = [
                float(metrics[key])
                for metrics in by_group.values()
                if key in metrics and metrics[key] is not None
            ]
            value = sum(vals) / len(vals) if vals else None
        row[f"{key}_concat"] = concat.get(key)
        row[f"{key}_15group_mean"] = value
    return row


def main():
    args = parse_args()
    rows = []
    for item in args.items:
        if "=" not in item:
            raise ValueError(f"Expected name=path item, got {item!r}")
        name, path = item.split("=", 1)
        rows.append(load_item(name, path, args.section))

    fieldnames = ["method", "section", "n"]
    for key in METRIC_KEYS:
        fieldnames.extend([f"{key}_15group_mean", f"{key}_concat"])
    fieldnames.append("metrics_path")

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("saved:", args.out_csv)
    for row in rows:
        print(
            row["method"],
            "ACC=", row.get("ACC_15group_mean"),
            "BAC=", row.get("BAC_15group_mean"),
            "F1=", row.get("F1_15group_mean"),
            "AUC=", row.get("AUC_15group_mean"),
        )


if __name__ == "__main__":
    main()
