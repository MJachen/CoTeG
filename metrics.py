
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn

def confusion_counts(y_true, y_pred):
    # y_true/y_pred: 0/1 numpy arrays
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn

def derived_metrics_from_conf(tp, fp, tn, fn):
    eps = 1e-12
    sens = tp / (tp + fn + eps)   # TPR / Recall
    spec = tn / (tn + fp + eps)   # TNR
    prec = tp / (tp + fp + eps)   # PPV
    npv  = tn / (tn + fn + eps)
    acc  = (tp + tn) / (tp + tn + fp + fn + eps)
    f1_pos = 2 * prec * sens / (prec + sens + eps)
    return {
        "ACC": acc,
        "BAC": 0.5 * (sens + spec),
        "Sensitivity": sens,
        "Specificity": spec,
        "Precision": prec,
        "NPV": npv,
        "F1": f1_pos,
        "F1_pos": f1_pos,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }

def compute_auc_pr(y_true, prob1):
    y_true = np.asarray(y_true).astype(np.int64)
    prob1 = np.asarray(prob1).astype(np.float32)
    out = {}
    # ROC-AUC
    try:
        out["AUC"] = float(roc_auc_score(y_true, prob1))
    except Exception:
        out["AUC"] = float("nan")
    # PR-AUC (Average Precision)
    try:
        out["PR_AUC"] = float(average_precision_score(y_true, prob1))
    except Exception:
        out["PR_AUC"] = float("nan")
    return out

def eval_with_threshold(y_true, prob1, thr):
    y_true = np.asarray(y_true).astype(np.int64)
    prob1 = np.asarray(prob1).astype(np.float32)
    pred = (prob1 >= thr).astype(np.int64)

    # confusion-derived
    tp, fp, tn, fn = confusion_counts(y_true, pred)
    d = derived_metrics_from_conf(tp, fp, tn, fn)

    # macro f1 (uses sklearn; safe)
    try:
        d["F1_macro"] = float(f1_score(y_true, pred, average="macro"))
    except Exception:
        d["F1_macro"] = float("nan")

    # AUCs
    d.update(compute_auc_pr(y_true, prob1))
    d["thr"] = float(thr)
    return d, pred

def choose_best_threshold(y_true, prob1, mode="bac", thr_grid=None):
    """
    mode:
      - "bac": maximize balanced accuracy, (Sensitivity + Specificity) / 2
      - "youden": equivalent ranking to BAC, maximize Sensitivity + Specificity - 1
      - "f1": maximize positive-class F1 (label 1 / HGG)
      - "macro_f1": maximize macro-F1
    """
    y_true = np.asarray(y_true).astype(np.int64)
    prob1 = np.asarray(prob1).astype(np.float32)

    if thr_grid is None:
        # dense grid; small val set so OK
        thr_grid = np.linspace(0.05, 0.95, 91)

    best_thr = 0.5
    best_score = -1e9
    best_detail = None

    for thr in thr_grid:
        detail, _ = eval_with_threshold(y_true, prob1, thr)
        if mode in {"bac", "balanced_accuracy"}:
            score = 0.5 * (detail["Sensitivity"] + detail["Specificity"])
        elif mode in {"youden"}:
            score = detail["Sensitivity"] + detail["Specificity"] - 1.0
        elif mode in {"f1", "binary_f1", "f1_pos"}:
            score = detail.get("F1", detail.get("F1_pos", float("nan")))
        elif mode == "macro_f1":
            score = detail.get("F1_macro", float("nan"))
        else:
            raise ValueError("mode must be 'bac', 'youden', 'f1', or 'macro_f1'")

        if np.isnan(score):
            continue

        # tie-break: higher sensitivity, then lower threshold (more conservative for recall)
        if (score > best_score) or (abs(score - best_score) < 1e-12 and detail["Sensitivity"] > best_detail["Sensitivity"]):
            best_score = score
            best_thr = float(thr)
            best_detail = detail

    if best_detail is None:
        best_detail, _ = eval_with_threshold(y_true, prob1, 0.5)
        best_thr, best_score = 0.5, -1e9

    return best_thr, best_score, best_detail

def print_report_table(title, overall, by_k_dict):
    """
    overall: dict from eval_with_threshold (includes thr, ACC, F1, AUC, PR_AUC, Sens, Spec, TP/FP/TN/FN...)
    by_k_dict: mapping k->dict
    """
    print(f"\n=== {title} (thr={overall['thr']:.3f}) ===")
    print(f"Overall: ACC={overall['ACC']:.4f} BAC={overall['BAC']:.4f} F1={overall['F1']:.4f} F1_macro={overall['F1_macro']:.4f} "
          f"AUC={overall['AUC']:.4f} PR_AUC={overall['PR_AUC']:.4f} "
          f"Sens={overall['Sensitivity']:.4f} Spec={overall['Specificity']:.4f} "
          f"TP={overall['TP']} FP={overall['FP']} TN={overall['TN']} FN={overall['FN']}")
    print("By group:")
    for kk in by_k_dict:
        d = by_k_dict[kk]
        print(f"  group={kk} n={d['n']} thr={d['thr']:.3f} ACC={d['ACC']:.4f} BAC={d['BAC']:.4f} F1={d['F1']:.4f} F1_macro={d['F1_macro']:.4f} "
              f"AUC={d['AUC']:.4f} PR_AUC={d['PR_AUC']:.4f} "
              f"Sens={d['Sensitivity']:.4f} Spec={d['Specificity']:.4f} "
              f"TP={d['TP']} FP={d['FP']} TN={d['TN']} FN={d['FN']}")


def fit_platt_calibrator(prob1_val, y_val):
    """
    prob1_val: numpy array, shape (n,)
    y_val: numpy array, shape (n,) with 0/1
    return: sklearn LogisticRegression calibrator
    """
    prob1_val = np.asarray(prob1_val).reshape(-1, 1)
    y_val = np.asarray(y_val).astype(np.int64)

    # class_weight='balanced' 可选：不平衡时更稳
    lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000, class_weight="balanced")
    lr.fit(prob1_val, y_val)
    return lr

def apply_platt_calibrator(calibrator, prob1):
    prob1 = np.asarray(prob1).reshape(-1, 1)
    # predict_proba gives [P(class=0), P(class=1)]
    return calibrator.predict_proba(prob1)[:, 1]


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # tensor shape [2] or None
        self.gamma = gamma

    def forward(self, logits, y):
        logp = torch.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        pt = p[torch.arange(y.size(0), device=y.device), y]
        logpt = logp[torch.arange(y.size(0), device=y.device), y]
        if self.alpha is not None:
            at = self.alpha[y]
            loss = -at * (1 - pt).pow(self.gamma) * logpt
        else:
            loss = -(1 - pt).pow(self.gamma) * logpt
        return loss.mean()
