# -*- coding: utf-8 -*-
"""
Calibrate saved fusion probabilities with validation predictions.

Expected inputs are sample-level CSV files with columns:
  idx, modality_group, y_true, prob_HG

Use fusiontraintest_brats.py outputs:
  val_sample_level_results_raw.csv
  test_sample_level_results_overall.csv

The script fits calibration on VAL only, applies it to TEST, then reports fixed,
best-on-VAL, and VAL specificity-constrained thresholds.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


EPS = 1e-6


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def threshold_grid(start, end, step):
    return np.arange(float(start), float(end) + 0.5 * float(step), float(step))


def require_columns(df, path):
    required = {"idx", "modality_group", "y_true", "prob_HG"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def metric_at_threshold(y_true, prob, thr):
    y_true = np.asarray(y_true).astype(np.int64)
    prob = np.asarray(prob).astype(np.float64)
    pred = (prob >= float(thr)).astype(np.int64)
    tp, fp, tn, fn = confusion_counts(y_true, pred)
    eps = 1e-12
    out = {
        "thr": float(thr),
        "n": int(len(y_true)),
        "ACC": float((tp + tn) / (tp + tn + fp + fn + eps)),
        "Sensitivity": float(tp / (tp + fn + eps)),
        "Specificity": float(tn / (tn + fp + eps)),
        "Precision": float(tp / (tp + fp + eps)),
        "NPV": float(tn / (tn + fn + eps)),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }
    out["BAC"] = float(0.5 * (out["Sensitivity"] + out["Specificity"]))
    out["F1"] = float(2.0 * out["Precision"] * out["Sensitivity"] / (out["Precision"] + out["Sensitivity"] + eps))
    try:
        out["F1_macro"] = float(f1_score(y_true, pred, average="macro"))
    except Exception:
        out["F1_macro"] = float("nan")
    try:
        out["AUC"] = float(roc_auc_score(y_true, prob))
    except Exception:
        out["AUC"] = float("nan")
    try:
        out["PR_AUC"] = float(average_precision_score(y_true, prob))
    except Exception:
        out["PR_AUC"] = float("nan")
    return out, pred


def choose_threshold(y_true, prob, thresholds, mode):
    best = None
    best_key = None
    for thr in thresholds:
        detail, _ = metric_at_threshold(y_true, prob, thr)
        if mode in {"bac", "balanced_accuracy"}:
            key = (
                0.5 * (detail["Sensitivity"] + detail["Specificity"]),
                detail.get("F1", float("nan")),
                detail.get("ACC", float("nan")),
            )
        elif mode in {"f1", "binary_f1", "f1_pos"}:
            key = (
                detail.get("F1", float("nan")),
                detail.get("Specificity", float("nan")),
                detail.get("Sensitivity", float("nan")),
                detail.get("ACC", float("nan")),
            )
        elif mode == "macro_f1":
            key = (
                detail.get("F1_macro", float("nan")),
                detail.get("Specificity", float("nan")),
                detail.get("Sensitivity", float("nan")),
                detail.get("ACC", float("nan")),
            )
        elif mode == "youden":
            key = (
                detail["Sensitivity"] + detail["Specificity"] - 1.0,
                detail.get("F1", float("nan")),
                detail.get("ACC", float("nan")),
            )
        else:
            raise ValueError("threshold mode must be bac, youden, f1, or macro_f1")
        if any(np.isnan(x) for x in key):
            continue
        if best is None or key > best_key:
            best = (float(thr), detail)
            best_key = key
    if best is None:
        return 0.5, metric_at_threshold(y_true, prob, 0.5)[0]
    return best


def choose_threshold_min_specificity(y_true, prob, thresholds, min_spec):
    fallback = None
    fallback_key = None
    best = None
    best_key = None
    for thr in thresholds:
        detail, _ = metric_at_threshold(y_true, prob, thr)
        fb_key = (
            detail.get("Specificity", float("nan")),
            detail.get("Sensitivity", float("nan")),
            detail.get("F1", float("nan")),
        )
        if fallback is None or fb_key > fallback_key:
            fallback = (float(thr), detail)
            fallback_key = fb_key
        if detail.get("Specificity", 0.0) + 1e-12 < min_spec:
            continue
        key = (
            0.5 * (detail.get("Sensitivity", float("nan")) + detail.get("Specificity", float("nan"))),
            detail.get("F1", float("nan")),
            detail.get("ACC", float("nan")),
        )
        if best is None or key > best_key:
            best = (float(thr), detail)
            best_key = key
    return best if best is not None else fallback


class PlattCalibrator:
    def __init__(self, class_weight="balanced"):
        self.model = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000, class_weight=class_weight)
        self.constant = None

    def fit(self, prob, y):
        y = np.asarray(y).astype(np.int64)
        if len(np.unique(y)) < 2:
            self.constant = float(y[0]) if len(y) else 0.5
            return self
        self.model.fit(logit(prob).reshape(-1, 1), y)
        return self

    def predict(self, prob):
        if self.constant is not None:
            return np.full(len(prob), self.constant, dtype=np.float64)
        return self.model.predict_proba(logit(prob).reshape(-1, 1))[:, 1]


def calibrate_probabilities(val_df, test_df, mode):
    val_df = val_df.copy()
    test_df = test_df.copy()
    if mode == "none":
        val_df["prob_HG_cal"] = val_df["prob_HG"].astype(float)
        test_df["prob_HG_cal"] = test_df["prob_HG"].astype(float)
        return val_df, test_df, {"mode": mode}

    if mode == "overall":
        cal = PlattCalibrator().fit(val_df["prob_HG"].to_numpy(), val_df["y_true"].to_numpy())
        val_df["prob_HG_cal"] = cal.predict(val_df["prob_HG"].to_numpy())
        test_df["prob_HG_cal"] = cal.predict(test_df["prob_HG"].to_numpy())
        return val_df, test_df, {"mode": mode}

    if mode == "per_group":
        meta = {"mode": mode, "groups": {}}
        global_cal = PlattCalibrator().fit(val_df["prob_HG"].to_numpy(), val_df["y_true"].to_numpy())
        val_df["prob_HG_cal"] = np.nan
        test_df["prob_HG_cal"] = np.nan
        for group in sorted(test_df["modality_group"].unique()):
            val_mask = val_df["modality_group"] == group
            test_mask = test_df["modality_group"] == group
            v = val_df[val_mask]
            if len(v) >= 8 and v["y_true"].nunique() == 2:
                cal = PlattCalibrator().fit(v["prob_HG"].to_numpy(), v["y_true"].to_numpy())
                meta["groups"][str(group)] = "per_group"
            else:
                cal = global_cal
                meta["groups"][str(group)] = "fallback_overall"
            val_df.loc[val_mask, "prob_HG_cal"] = cal.predict(val_df.loc[val_mask, "prob_HG"].to_numpy())
            test_df.loc[test_mask, "prob_HG_cal"] = cal.predict(test_df.loc[test_mask, "prob_HG"].to_numpy())
        return val_df, test_df, meta

    raise ValueError("calibration mode must be none, overall, or per_group")


def evaluate_threshold_map(df, thresholds_by_group, prob_col):
    rows = []
    sample_rows = []
    y_all = []
    p_all = []
    pred_all = []
    for group, sub in df.groupby("modality_group", sort=True):
        y = sub["y_true"].to_numpy(dtype=np.int64)
        p = sub[prob_col].to_numpy(dtype=np.float64)
        thr = float(thresholds_by_group[str(group)])
        detail, pred = metric_at_threshold(y, p, thr)
        detail["group"] = str(group)
        rows.append(detail)
        tmp = sub.copy()
        tmp["prob_HG_eval"] = p
        tmp["threshold"] = thr
        tmp["y_pred"] = pred
        tmp["correct"] = pred == y
        sample_rows.append(tmp)
        y_all.append(y)
        p_all.append(p)
        pred_all.append(pred)

    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    pred_all = np.concatenate(pred_all)
    tp, fp, tn, fn = confusion_counts(y_all, pred_all)
    overall = {
        "thr": float("nan"),
        "n": int(len(y_all)),
        "ACC": float((tp + tn) / max(len(y_all), 1)),
        "Sensitivity": float(tp / (tp + fn + 1e-12)),
        "Specificity": float(tn / (tn + fp + 1e-12)),
        "Precision": float(tp / (tp + fp + 1e-12)),
        "NPV": float(tn / (tn + fn + 1e-12)),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }
    overall["BAC"] = float(0.5 * (overall["Sensitivity"] + overall["Specificity"]))
    overall["F1"] = float(2.0 * overall["Precision"] * overall["Sensitivity"] / (overall["Precision"] + overall["Sensitivity"] + 1e-12))
    overall["F1_macro"] = float(f1_score(y_all, pred_all, average="macro"))
    try:
        overall["AUC"] = float(roc_auc_score(y_all, p_all))
    except Exception:
        overall["AUC"] = float("nan")
    try:
        overall["PR_AUC"] = float(average_precision_score(y_all, p_all))
    except Exception:
        overall["PR_AUC"] = float("nan")
    return overall, rows, pd.concat(sample_rows, ignore_index=True)


def print_overall(name, detail):
    print(
        f"{name}: ACC={detail['ACC']:.4f} F1={detail['F1']:.4f} F1_macro={detail['F1_macro']:.4f} "
        f"BAC={detail['BAC']:.4f} "
        f"AUC={detail['AUC']:.4f} PR={detail['PR_AUC']:.4f} "
        f"Sens={detail['Sensitivity']:.4f} Spec={detail['Specificity']:.4f} "
        f"TP={detail['TP']} FP={detail['FP']} TN={detail['TN']} FN={detail['FN']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Calibrate saved fusion predictions using VAL predictions.")
    parser.add_argument("--fusion-out-dir", required=True)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--calibration", choices=["none", "overall", "per_group"], default="overall")
    parser.add_argument("--threshold-mode", choices=["bac", "youden", "f1", "macro_f1"], default="bac")
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.995)
    parser.add_argument("--threshold-step", type=float, default=0.0025)
    parser.add_argument("--fixed-thresholds", default="0.5,0.7,0.8,0.9,0.95,0.99")
    parser.add_argument("--spec-targets", default="0.50,0.55,0.60,0.65")
    args = parser.parse_args()

    val_csv = args.val_csv or os.path.join(args.fusion_out_dir, "val_sample_level_results_raw.csv")
    test_csv = args.test_csv or os.path.join(args.fusion_out_dir, "test_sample_level_results_overall.csv")
    out_dir = args.out_dir or os.path.join(args.fusion_out_dir, f"calibrated_{args.calibration}")
    os.makedirs(out_dir, exist_ok=True)

    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    require_columns(val_df, val_csv)
    require_columns(test_df, test_csv)

    val_df, test_df, cal_meta = calibrate_probabilities(val_df, test_df, args.calibration)
    val_out = os.path.join(out_dir, "val_predictions_calibrated.csv")
    test_out = os.path.join(out_dir, "test_predictions_calibrated.csv")
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    thresholds = threshold_grid(args.threshold_min, args.threshold_max, args.threshold_step)
    fixed_thresholds = parse_float_list(args.fixed_thresholds)
    spec_targets = parse_float_list(args.spec_targets)

    yv = val_df["y_true"].to_numpy(dtype=np.int64)
    pv = val_df["prob_HG_cal"].to_numpy(dtype=np.float64)
    yt = test_df["y_true"].to_numpy(dtype=np.int64)
    pt = test_df["prob_HG_cal"].to_numpy(dtype=np.float64)

    reports = {}
    recommendation_rows = []

    raw_test, _ = metric_at_threshold(yt, test_df["prob_HG"].to_numpy(dtype=np.float64), 0.5)
    reports["raw_fixed_0.5"] = {"test_overall": raw_test}

    for thr in fixed_thresholds:
        detail, _ = metric_at_threshold(yt, pt, thr)
        key = f"cal_fixed_{thr:.4f}".rstrip("0").rstrip(".")
        reports[key] = {"test_overall": detail}
        recommendation_rows.append({"criterion": key, **detail})
        print_overall(key, detail)

    thr_best, val_best = choose_threshold(yv, pv, thresholds, args.threshold_mode)
    test_best, _ = metric_at_threshold(yt, pt, thr_best)
    reports[f"val_best_{args.threshold_mode}"] = {
        "val_threshold": float(thr_best),
        "val_detail": val_best,
        "test_overall": test_best,
    }
    recommendation_rows.append({"criterion": f"val_best_{args.threshold_mode}", **test_best})
    print_overall(f"val_best_{args.threshold_mode}", test_best)

    for spec in spec_targets:
        thr_spec, val_spec = choose_threshold_min_specificity(yv, pv, thresholds, spec)
        test_spec, _ = metric_at_threshold(yt, pt, thr_spec)
        key = f"val_min_spec_{spec:.2f}"
        reports[key] = {
            "val_threshold": float(thr_spec),
            "val_detail": val_spec,
            "test_overall": test_spec,
        }
        recommendation_rows.append({"criterion": key, **test_spec})
        print_overall(key, test_spec)

    # Per-group thresholds selected on VAL after calibration.
    per_group_thr = {}
    per_group_val = {}
    for group, sub in val_df.groupby("modality_group", sort=True):
        thr, detail = choose_threshold(
            sub["y_true"].to_numpy(dtype=np.int64),
            sub["prob_HG_cal"].to_numpy(dtype=np.float64),
            thresholds,
            args.threshold_mode,
        )
        per_group_thr[str(group)] = float(thr)
        per_group_val[str(group)] = detail
    per_group_overall, per_group_rows, per_group_samples = evaluate_threshold_map(
        test_df, per_group_thr, "prob_HG_cal"
    )
    reports[f"per_group_val_best_{args.threshold_mode}"] = {
        "thresholds": per_group_thr,
        "val_by_group": per_group_val,
        "test_overall": per_group_overall,
        "test_by_group": per_group_rows,
    }
    recommendation_rows.append({"criterion": f"per_group_val_best_{args.threshold_mode}", **per_group_overall})
    per_group_samples.to_csv(os.path.join(out_dir, "test_predictions_per_group_threshold.csv"), index=False)
    print_overall(f"per_group_val_best_{args.threshold_mode}", per_group_overall)

    rec_df = pd.DataFrame(recommendation_rows)
    rec_path = os.path.join(out_dir, "calibrated_threshold_recommendations.csv")
    rec_df.to_csv(rec_path, index=False)

    metrics = {
        "config": {
            "fusion_out_dir": args.fusion_out_dir,
            "val_csv": val_csv,
            "test_csv": test_csv,
            "calibration": cal_meta,
            "threshold_mode": args.threshold_mode,
            "threshold_grid": [args.threshold_min, args.threshold_max, args.threshold_step],
            "fixed_thresholds": fixed_thresholds,
            "spec_targets": spec_targets,
        },
        "reports": reports,
    }
    metrics_path = os.path.join(out_dir, "calibrated_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:", val_out)
    print("Saved:", test_out)
    print("Saved:", rec_path)
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()
