# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from GRNNTCtrain import build_tv_pairs, tv3d_reconstruction_loss
from modality_prior import apply_modality_prior_chunk, summarize_prior_state
from project_paths import get_pack_dir


# =========================
# 你只需要改这几行路径
# =========================
# 批量输入：会匹配 val_k*.pt 和 test_k*.pt
EVAL_PACK_GLOBS = [
    os.path.join(get_pack_dir(), "val_*.pt"),
    os.path.join(get_pack_dir(), "test_*.pt"),
]
FACTORS_PATH = os.environ.get("FACTORS_PATH", os.path.join(get_pack_dir(), "tucker_factors_train.pt"))
OUT_DIR = get_pack_dir()


# =========================
# 训练/优化超参数（默认可用）
# =========================
LR = float(os.environ.get("GRNNTC_EVAL_LR", "1e-3"))
MAX_ITER = int(os.environ.get("GRNNTC_EVAL_MAX_ITER", "800"))
LOG_EVERY = 100
LAMBDA_SUBJ = float(os.environ.get("GRNNTC_EVAL_LAMBDA_SUBJ", "5e-4"))        # U_N 图正则权重
CHUNK_S = int(os.environ.get("GRNNTC_EVAL_CHUNK_S", os.environ.get("GRNNTC_CHUNK_S", "2048")))
LAMBDA_PRIOR = float(os.environ.get("GRNNTC_EVAL_LAMBDA_PRIOR", os.environ.get("GRNNTC_LAMBDA_PRIOR", "0.0")))
USE_MODALITY_PRIOR = os.environ.get("GRNNTC_USE_MODALITY_PRIOR", "0") == "1"
PRIOR_BLEND = float(os.environ.get("GRNNTC_PRIOR_BLEND", "0.0"))
LAMBDA_TV = float(os.environ.get("GRNNTC_EVAL_LAMBDA_TV", os.environ.get("GRNNTC_LAMBDA_TV", "1e-2")))
TV_MAX_PAIRS = int(os.environ.get("GRNNTC_TV_MAX_PAIRS", "200000"))
TV_PAIR_CHUNK = int(os.environ.get("GRNNTC_TV_PAIR_CHUNK", "32768"))
SEED = 123

# patient graph 参数
SIM_TYPE = "rbf"          # "rbf" / "cosine" / "euclidean"
KNN_K = 10
SIGMA = 1.0


def build_modality_laplacian(X_true_hwdmn):
    """
    输入:
        X_true_hwdmn: torch or numpy, shape [H, W, D, M, N]
    输出:
        L_M: numpy, shape [M, M]
    """
    if isinstance(X_true_hwdmn, torch.Tensor):
        X = X_true_hwdmn.cpu().numpy()
    else:
        X = X_true_hwdmn

    H, W, D, M, N = X.shape
    W_mat = np.zeros((M, M), dtype=np.float32)

    for i in range(M):
        for j in range(i + 1, M):
            v1 = X[..., i, :].reshape(-1)
            v2 = X[..., j, :].reshape(-1)
            corr = np.corrcoef(v1, v2)[0, 1]
            if not np.isfinite(corr):
                corr = 0.0
            corr = max(float(corr), 0.0)
            W_mat[i, j] = corr
            W_mat[j, i] = corr

    D_mat = np.diag(W_mat.sum(axis=1))
    L = D_mat - W_mat
    return L


def build_patient_graph_laplacian(features_nf,
                                  sim_type="rbf",
                                  knn_k=10,
                                  sigma=1.0,
                                  device="cpu"):
    """
    输入:
        features_nf: (N, F) numpy or torch
    输出:
        L_N: (N, N) torch on device
    """
    if isinstance(features_nf, torch.Tensor):
        X = features_nf.cpu().numpy()
    else:
        X = features_nf

    X = StandardScaler().fit_transform(X)
    N = X.shape[0]

    if sim_type == "cosine":
        W = cosine_similarity(X)
        W = (W + 1) / 2
    elif sim_type == "euclidean":
        dist = euclidean_distances(X)
        dist = dist / (dist.max() + 1e-12)
        W = 1 - dist
    elif sim_type == "rbf":
        dist = euclidean_distances(X)
        W = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    else:
        raise ValueError("sim_type must be 'rbf'/'cosine'/'euclidean'")

    np.fill_diagonal(W, 0)

    # KNN 稀疏化 + 对称化
    W_knn = np.zeros_like(W)
    for i in range(N):
        idx = np.argsort(W[i])[::-1][:knn_k]
        W_knn[i, idx] = W[i, idx]
    W = np.maximum(W_knn, W_knn.T)

    D = np.diag(W.sum(axis=1))
    L = D - W
    return torch.tensor(L, dtype=torch.float32, device=device)


def graph_quadratic(U, L):
    """
    tr(U^T L U)
    U: [n, r], L: [n, n]
    """
    return torch.sum(U * (L @ U))


def maybe_pin_memory(x):
    if os.environ.get("GRNNTC_PIN_MEMORY", "0") != "1":
        return x
    try:
        return x.pin_memory()
    except RuntimeError as exc:
        print(f"[Warn] pin_memory failed, continuing without pinned memory: {exc}")
        return x


class Tucker3DModel(nn.Module):
    """
    X_flat ∈ R^{S×M×N} ≈ G ×_1 U_S ×_2 U_M ×_3 U_N
    """
    def __init__(self, S, M, N, rank_S, rank_M, rank_N, device="cpu"):
        super().__init__()
        self.S, self.M, self.N = S, M, N
        self.rS, self.rM, self.rN = rank_S, rank_M, rank_N

        self.core = nn.Parameter(0.1 * torch.randn(rank_S, rank_M, rank_N, device=device))
        self.U_S = nn.Parameter(0.1 * torch.randn(S, rank_S, device=device))
        self.U_M = nn.Parameter(0.1 * torch.randn(M, rank_M, device=device))
        self.U_N = nn.Parameter(0.1 * torch.randn(N, rank_N, device=device))

    def forward_flat(self):
        # (S,M,N)
        X_flat = torch.einsum("sa,abc,mb,nc->smn", self.U_S, self.core, self.U_M, self.U_N)
        return torch.relu(X_flat)


def _adapt_xmiss_to_NMHWD(X_miss: torch.Tensor, Omega_nm111: torch.Tensor):
    """
    兼容两种输入：
    - 原脚本：X_miss (N,M,H,W,D)
    - 你的 pituitary pack：X_miss (N,C,D,H,W)
    返回：
    - X_miss_nm: (N,M,H,W,D)
    """
    if X_miss.ndim != 5:
        raise ValueError("X_miss 必须是 5D 张量。")

    # 目标 M 从 Omega 的第二维推断（你的 pack Omega 是 (N,3,1,1,1)）
    if Omega_nm111.ndim != 5:
        raise ValueError("Omega 必须是 (N,M,1,1,1) 的 5D 张量。")
    M_expected = Omega_nm111.shape[1]
    input_layout = os.environ.get("EVAL_X_LAYOUT", "NMHWD").upper()

    if X_miss.shape[1] != M_expected:
        raise ValueError(f"无法适配 X_miss 形状 {tuple(X_miss.shape)}，Omega M={M_expected}。")

    # 默认项目 pack 是 (N,M,H,W,D)。如果外部数据确实是 (N,M,D,H,W)，显式设置 EVAL_X_LAYOUT=NMDHW。
    if input_layout == "NMHWD":
        return X_miss
    if input_layout in {"NMDHW", "NCDHW"}:
        return X_miss.permute(0, 1, 3, 4, 2).contiguous()

    raise ValueError("EVAL_X_LAYOUT must be NMHWD or NMDHW/NCDHW.")


def run_one(eval_pack_path: str, factors_path: str, out_pack_path: str):
    # ----------------- device -----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ----------------- load eval_pack -----------------
    if not os.path.exists(eval_pack_path):
        raise FileNotFoundError(f"Eval pack not found: {eval_pack_path}")

    pack = torch.load(eval_pack_path, map_location="cpu")
    if "X_miss" not in pack or "Omega" not in pack:
        raise KeyError("eval_pack 必须包含 X_miss 和 Omega 两个 key。")

    X_miss_raw = pack["X_miss"].float()
    Omega_nm111 = pack["Omega"].float()    # (N,M,1,1,1)

    if Omega_nm111.ndim != 5:
        raise ValueError("Omega 维度不正确，期望 (N,M,1,1,1)。")

    # ★ 关键适配：把你的 (N,C,D,H,W) 转为 (N,M,H,W,D)
    X_miss_nm = _adapt_xmiss_to_NMHWD(X_miss_raw, Omega_nm111)

    if X_miss_nm.ndim != 5:
        raise ValueError("X_miss 维度不正确，期望 (N,M,H,W,D)。")

    N, M, H, W, D = X_miss_nm.shape
    S = H * W * D
    print(f"Loaded eval_pack: {os.path.basename(eval_pack_path)}  X_miss={tuple(X_miss_nm.shape)} Omega={tuple(Omega_nm111.shape)} S={S}")

    # 可选真值
    X_full_nm = pack.get("X_full", None)
    if X_full_nm is not None:
        X_full_nm = X_full_nm.float()
        if tuple(X_full_nm.shape) != (N, M, H, W, D):
            print(f"[Warn] X_full shape {tuple(X_full_nm.shape)} != {(N,M,H,W,D)}; skip GT MSE.")
            X_full_nm = None

    # internal: (H,W,D,M,N)
    X_miss_hwdmn = X_miss_nm.permute(2, 3, 4, 1, 0).contiguous()
    Omega_nm = Omega_nm111.expand(N, M, H, W, D)
    Omega_hwdmn = Omega_nm.permute(2, 3, 4, 1, 0).contiguous()

    X_gt_hwdmn = None
    if X_full_nm is not None:
        X_gt_hwdmn = X_full_nm.permute(2, 3, 4, 1, 0).contiguous()

    # flatten: (S,M,N)
    X_true_flat = maybe_pin_memory(X_miss_hwdmn.view(S, M, N))
    Omega_flat = maybe_pin_memory(Omega_hwdmn.view(S, M, N))

    # ----------------- Laplacians -----------------
    L_M_np = build_modality_laplacian(X_miss_hwdmn)
    L_M = torch.tensor(L_M_np, dtype=torch.float32, device=device)

    # patient features: observed mean per modality -> (N,M)
    num_obs_mn = Omega_flat.sum(dim=0).clamp_min(1.0)              # (M,N)
    feat_mn = (X_true_flat * Omega_flat).sum(dim=0) / num_obs_mn   # (M,N)
    patient_features = feat_mn.T                                   # (N,M)

    Xp = patient_features.cpu().numpy()
    Xp = StandardScaler().fit_transform(Xp)
    Xp_pca = PCA(n_components=min(M, 10, Xp.shape[0], Xp.shape[1])).fit_transform(Xp)

    L_N = build_patient_graph_laplacian(
        Xp_pca, sim_type=SIM_TYPE, knn_k=KNN_K, sigma=SIGMA, device=device
    )

    # ----------------- load pretrained factors -----------------
    if not os.path.exists(factors_path):
        raise FileNotFoundError(f"Factors not found: {factors_path}")

    fac = torch.load(factors_path, map_location="cpu")
    for k in ["G", "U_S", "U_M"]:
        if k not in fac:
            raise KeyError(f"Factors 文件缺少 key: {k}（必须至少包含 G/U_S/U_M）")

    G = fac["G"].float()         # (rS,rM,rN)
    U_S_pre = fac["U_S"].float() # (S,rS)
    U_M_pre = fac["U_M"].float() # (M,rM)
    prior_state = fac.get("modality_prior_state", None) if USE_MODALITY_PRIOR else None
    if USE_MODALITY_PRIOR:
        if prior_state is None:
            print("[Warn] GRNNTC_USE_MODALITY_PRIOR=1 but factors do not contain modality_prior_state.")
        else:
            print("Using modality prior:", summarize_prior_state(prior_state))

    rS, rM, rN = G.shape
    if U_S_pre.shape != (S, rS):
        raise ValueError(f"U_S shape mismatch. expected {(S,rS)}, got {tuple(U_S_pre.shape)}")
    if U_M_pre.shape != (M, rM):
        raise ValueError(f"U_M shape mismatch. expected {(M,rM)}, got {tuple(U_M_pre.shape)}")

    print(f"Loaded pretrained factors: G={tuple(G.shape)} U_S={tuple(U_S_pre.shape)} U_M={tuple(U_M_pre.shape)}")
    print(f"Will optimize U_N only: shape {(N, rN)}")

    # ----------------- init model & freeze -----------------
    model = Tucker3DModel(S, M, N, rank_S=rS, rank_M=rM, rank_N=rN, device=device).to(device)
    with torch.no_grad():
        model.core.copy_(G.to(device))
        model.U_S.copy_(U_S_pre.to(device))
        model.U_M.copy_(U_M_pre.to(device))
        # U_N 随机初始化后优化

    model.core.requires_grad_(False)
    model.U_S.requires_grad_(False)
    model.U_M.requires_grad_(False)
    model.U_N.requires_grad_(True)

    optimizer = optim.Adam([model.U_N], lr=LR, weight_decay=1e-5)
    tv_pairs = build_tv_pairs(H, W, D, TV_MAX_PAIRS, seed=123) if LAMBDA_TV > 0 else None

    # ----------------- optimize U_N -----------------
    print("Start optimizing U_N ...")
    for it in range(MAX_ITER):
        optimizer.zero_grad()

        data_num = 0.0
        data_den = 0.0
        prior_num = torch.tensor(0.0, device=device)
        prior_den = torch.tensor(0.0, device=device)

        for s0 in range(0, S, CHUNK_S):
            s1 = min(S, s0 + CHUNK_S)
            X_true_c = X_true_flat[s0:s1].to(device, non_blocking=True)
            Omega_c = Omega_flat[s0:s1].to(device, non_blocking=True)

            U_S_c = model.U_S[s0:s1]
            X_hat_c = torch.einsum("sa,abc,mb,nc->smn", U_S_c, model.core, model.U_M, model.U_N)
            X_hat_c = torch.relu(X_hat_c)

            diff = (X_hat_c - X_true_c) * Omega_c
            data_num = data_num + diff.pow(2).sum()
            data_den = data_den + Omega_c.sum()

            if prior_state is not None and LAMBDA_PRIOR > 0:
                omega_mn = Omega_c[0]
                prior_c, prior_mask_c = apply_modality_prior_chunk(X_true_c * Omega_c, omega_mn, prior_state)
                prior_diff = (X_hat_c - prior_c) * prior_mask_c
                prior_num = prior_num + prior_diff.pow(2).sum()
                prior_den = prior_den + prior_mask_c.sum()

        data_loss = data_num / (data_den + 1e-8)
        prior_loss = prior_num / (prior_den + 1e-8)
        subj_loss = graph_quadratic(model.U_N, L_N)
        tv_loss = tv3d_reconstruction_loss(model, tv_pairs, pair_chunk=TV_PAIR_CHUNK)

        loss = data_loss + LAMBDA_SUBJ * subj_loss + LAMBDA_PRIOR * prior_loss + LAMBDA_TV * tv_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.U_N.data.clamp_(min=0)

        if (it % LOG_EVERY == 0) or (it == MAX_ITER - 1):
            print(
                f"[Iter {it:04d}] loss={loss.item():.4e} "
                f"data={data_loss.item():.4e} subj={subj_loss.item():.4e} "
                f"prior={prior_loss.item():.4e} tv={tv_loss.item():.4e}"
            )

    # ----------------- reconstruct X_hat -----------------
    print("Reconstructing X_hat ...")
    X_hat_cpu = torch.zeros((S, M, N), dtype=torch.float32)

    with torch.no_grad():
        for s0 in range(0, S, CHUNK_S):
            s1 = min(S, s0 + CHUNK_S)
            U_S_c = model.U_S[s0:s1]
            X_hat_c = torch.einsum("sa,abc,mb,nc->smn", U_S_c, model.core, model.U_M, model.U_N)
            X_hat_c = torch.relu(X_hat_c)
            X_hat_cpu[s0:s1] = X_hat_c.cpu()

    X_hat_hwdmn = X_hat_cpu.view(H, W, D, M, N)

    X_prior_hwdmn = None
    Prior_mask_hwdmn = None
    if prior_state is not None:
        print("Building modality-prior image for missing entries ...")
        X_prior_cpu = torch.zeros((S, M, N), dtype=torch.float32)
        Prior_mask_cpu = torch.zeros((S, M, N), dtype=torch.float32)
        omega_mn_full = Omega_flat[0]
        with torch.no_grad():
            for s0 in range(0, S, CHUNK_S):
                s1 = min(S, s0 + CHUNK_S)
                X_true_c = X_true_flat[s0:s1]
                Omega_c = Omega_flat[s0:s1]
                prior_c, prior_mask_c = apply_modality_prior_chunk(X_true_c * Omega_c, omega_mn_full, prior_state)
                X_prior_cpu[s0:s1] = prior_c.cpu()
                Prior_mask_cpu[s0:s1] = prior_mask_c.cpu()
        X_prior_hwdmn = X_prior_cpu.view(H, W, D, M, N)
        Prior_mask_hwdmn = Prior_mask_cpu.view(H, W, D, M, N)
        if PRIOR_BLEND > 0:
            blend_mask = Prior_mask_hwdmn * (1.0 - Omega_hwdmn)
            X_hat_hwdmn = X_hat_hwdmn * (1.0 - PRIOR_BLEND * blend_mask) + X_prior_hwdmn * (PRIOR_BLEND * blend_mask)
            print(f"Blended tensor completion with modality prior on missing entries: alpha={PRIOR_BLEND}")

    # 观测处回填：严格补全（observed 不变）
    X_hat_hwdmn = X_hat_hwdmn * (1.0 - Omega_hwdmn) + X_miss_hwdmn * Omega_hwdmn

    # 可选：缺失处 MSE
    if X_gt_hwdmn is not None:
        miss_mask = (Omega_hwdmn == 0)
        mse_missing = ((X_hat_hwdmn - X_gt_hwdmn)[miss_mask].pow(2).mean()).item()
        print(f"GT MSE on missing entries: {mse_missing:.6e}")

    # ----------------- save completed pack -----------------
    X_hat_nm = X_hat_hwdmn.permute(4, 3, 0, 1, 2).contiguous()  # (N,M,H,W,D)

    out = {
        "idx": pack.get("idx", None),
        "y": pack.get("y", None),
        "k": pack.get("k", None),
        "Omega": Omega_nm111,   # 原格式 (N,M,1,1,1)
        "X_miss": X_miss_nm,    # 原格式 (N,M,H,W,D)
        "X_hat": X_hat_nm,      # 补全结果 (N,M,H,W,D)
    }
    if X_full_nm is not None:
        out["X_full"] = X_full_nm
    if X_prior_hwdmn is not None:
        out["X_prior"] = X_prior_hwdmn.permute(4, 3, 0, 1, 2).contiguous()
        out["Prior_mask"] = Prior_mask_hwdmn.permute(4, 3, 0, 1, 2).contiguous()

    torch.save(out, out_pack_path)
    print(f"Saved completed pack to: {out_pack_path}")

    # ===============================
    # 保存 val / test 的 Tucker 参数
    # ===============================
    tucker_save_path = out_pack_path.replace(".pt", "_tucker_factors.pt")

    save_dict = {
        "G": model.core.detach().cpu(),
        "U_S": model.U_S.detach().cpu(),
        "U_M": model.U_M.detach().cpu(),
        "U_N": model.U_N.detach().cpu(),
        "modality_prior_state": prior_state,
        "lambda_prior": float(LAMBDA_PRIOR),
        "prior_blend": float(PRIOR_BLEND),
        "lambda_tv": float(LAMBDA_TV),
        "tv_max_pairs": int(TV_MAX_PAIRS),
    }

    torch.save(save_dict, tucker_save_path)
    print(f"Saved Tucker factors to: {tucker_save_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 收集待处理文件
    eval_paths = []
    for pat in EVAL_PACK_GLOBS:
        eval_paths.extend(sorted(glob.glob(pat)))

    # 过滤掉已经是 completed 的文件，避免重复跑
    eval_paths = [
        p for p in eval_paths
        if "_completed" not in os.path.basename(p)
        and not p.endswith("_tucker_factors.pt")
    ]

    if not eval_paths:
        raise FileNotFoundError("未找到任何待处理 eval pack。请检查 EVAL_PACK_GLOBS 路径。")

    print("Found eval packs:")
    for p in eval_paths:
        print("  -", p)

    if not os.path.exists(FACTORS_PATH):
        raise FileNotFoundError(f"Factors not found: {FACTORS_PATH}")

    # 批量处理
    for eval_pack_path in eval_paths:
        base = os.path.basename(eval_pack_path)
        stem = os.path.splitext(base)[0]
        out_pack_path = os.path.join(OUT_DIR, stem + "_completed.pt")

        print("\n" + "=" * 80)
        print(f"[RUN] {eval_pack_path} -> {out_pack_path}")
        run_one(eval_pack_path, FACTORS_PATH, out_pack_path)

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
