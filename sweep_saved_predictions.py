# -*- coding: utf-8 -*-
"""
Threshold diagnostics for saved fusion predictions.

This script does not train a model and does not load image tensors. It reads a
sample-level CSV produced by fusiontraintest_brats.py, sweeps thresholds, and
writes diagnostic tables. If the CSV is a TEST prediction file, use the output
for analysis only; final thresholds should still be selected on VAL.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def threshold_grid(start, end, step):
    return np.arange(float(start), float(end) + 0.5 * float(step), float(step))


def sweep_one(name, y, p, thresholds):
    rows = []
    for thr in thresholds:
        detail, _ = eval_with_threshold(y, p, float(thr))
        row = {
            "scope": name,
            "thr": float(thr),
            "n": int(len(y)),
        }
        row.update(detail)
        rows.append(row)
    return rows


def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def eval_with_threshold(y_true, prob1, thr):
    y_true = np.asarray(y_true).astype(np.int64)
    prob1 = np.asarray(prob1).astype(np.float32)
    pred = (prob1 >= float(thr)).astype(np.int64)
    tp, fp, tn, fn = confusion_counts(y_true, pred)
    eps = 1e-12
    out = {
        "ACC": float((tp + tn) / (tp + tn + fp + fn + eps)),
        "Sensitivity": float(tp / (tp + fn + eps)),
        "Specificity": float(tn / (tn + fp + eps)),
        "Precision": float(tp / (tp + fp + eps)),
        "NPV": float(tn / (tn + fn + eps)),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "thr": float(thr),
    }
    out["BAC"] = float(0.5 * (out["Sensitivity"] + out["Specificity"]))
    out["F1"] = float(2.0 * out["Precision"] * out["Sensitivity"] / (out["Precision"] + out["Sensitivity"] + eps))
    try:
        out["F1_macro"] = float(f1_score(y_true, pred, average="macro"))
    except Exception:
        out["F1_macro"] = float("nan")
    try:
        out["AUC"] = float(roc_auc_score(y_true, prob1))
    except Exception:
        out["AUC"] = float("nan")
    try:
        out["PR_AUC"] = float(average_precision_score(y_true, prob1))
    except Exception:
        out["PR_AUC"] = float("nan")
    return out, pred


def pick_recommendations(sweep_df, spec_targets):
    recs = []
    for scope, sub in sweep_df.groupby("scope", sort=False):
        sub = sub.sort_values("thr")
        best_f1 = sub.sort_values(
            ["F1", "AUC", "ACC", "Specificity", "Sensitivity"],
            ascending=[False, False, False, False, False],
        ).iloc[0]
        recs.append({"scope": scope, "criterion": "best_f1", **best_f1.to_dict()})

        best_youden = sub.assign(youden=sub["Sensitivity"] + sub["Specificity"] - 1.0).sort_values(
            ["youden", "F1", "ACC"],
            ascending=[False, False, False],
        ).iloc[0]
        recs.append({"scope": scope, "criterion": "best_bac", **best_youden.drop(labels=["youden"]).to_dict()})

        for spec_target in spec_targets:
            ok = sub[sub["Specificity"] >= spec_target]
            if len(ok) == 0:
                picked = sub.sort_values(
                    ["Specificity", "Sensitivity", "F1"],
                    ascending=[False, False, False],
                ).iloc[0]
                criterion = f"fallback_max_spec_for_{spec_target:.2f}"
            else:
                picked = ok.sort_values(
                    ["Sensitivity", "F1", "ACC"],
                    ascending=[False, False, False],
                ).iloc[0]
                criterion = f"min_spec_{spec_target:.2f}"
            recs.append({"scope": scope, "criterion": criterion, **picked.to_dict()})
    return pd.DataFrame(recs)


def main():
    parser = argparse.ArgumentParser(description="Sweep thresholds from saved fusion sample predictions.")
    parser.add_argument("--out-dir", required=True, help="Fusion output directory.")
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Prediction CSV. Default: <out-dir>/test_sample_level_results_overall.csv",
    )
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--spec-targets", default="0.50,0.60,0.70")
    args = parser.parse_args()

    input_csv = args.input_csv or os.path.join(args.out_dir, "test_sample_level_results_overall.csv")
    df = pd.read_csv(input_csv)
    required = {"modality_group", "y_true", "prob_HG"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_csv} missing required columns: {sorted(missing)}")

    thresholds = threshold_grid(args.threshold_min, args.threshold_max, args.threshold_step)
    spec_targets = parse_float_list(args.spec_targets)

    rows = []
    rows.extend(
        sweep_one(
            "overall",
            df["y_true"].to_numpy(dtype=np.int64),
            df["prob_HG"].to_numpy(dtype=np.float32),
            thresholds,
        )
    )
    for group, sub in df.groupby("modality_group", sort=True):
        rows.extend(
            sweep_one(
                str(group),
                sub["y_true"].to_numpy(dtype=np.int64),
                sub["prob_HG"].to_numpy(dtype=np.float32),
                thresholds,
            )
        )

    sweep_df = pd.DataFrame(rows)
    rec_df = pick_recommendations(sweep_df, spec_targets)

    os.makedirs(args.out_dir, exist_ok=True)
    sweep_path = os.path.join(args.out_dir, "threshold_sweep_from_saved_predictions.csv")
    rec_path = os.path.join(args.out_dir, "threshold_recommendations_from_saved_predictions.csv")
    sweep_df.to_csv(sweep_path, index=False)
    rec_df.to_csv(rec_path, index=False)

    print("Input:", input_csv)
    print("Saved sweep:", sweep_path)
    print("Saved recommendations:", rec_path)
    print("\nTop overall recommendations:")
    cols = ["criterion", "thr", "ACC", "BAC", "F1", "F1_macro", "AUC", "PR_AUC", "Sensitivity", "Specificity", "TP", "FP", "TN", "FN"]
    print(rec_df[rec_df["scope"] == "overall"][cols].to_string(index=False))
    print("\nNote: if the input CSV is from TEST, these thresholds are diagnostic only.")


if __name__ == "__main__":
    main()
