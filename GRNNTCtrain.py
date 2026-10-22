# -*- coding: utf-8 -*-
"""
Train graph-regularized non-negative Tucker completion.

Default mode is patient mode:
- use one subject per training case from X_full_4mod.pt + fold split.json
- simulate the 15 missing-modality groups during optimization
- save X_hat_train_5d.pt and tucker_factors_train.pt for downstream scripts

Legacy augmented mode is still available with GRNNTC_TRAIN_MODE=augmented if a
train_aug_15mods.pt/train_aug_k1to4.pt file exists.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler

from modality_groups import MODALITY_GROUPS, group_to_omega_row
from modality_prior import apply_modality_prior_chunk, fit_modality_prior, summarize_prior_state
from project_paths import find_train_pack_path, find_xfull_path, get_pack_dir


def build_modality_laplacian(X_true):
    if isinstance(X_true, torch.Tensor):
        X = X_true.cpu().numpy()
    else:
        X = X_true

    _, _, _, M, _ = X.shape
    W_mat = np.zeros((M, M), dtype=np.float32)

    print("Computing modality similarity...")
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
    return D_mat - W_mat


def build_patient_graph_laplacian(features, sim_type="rbf", knn_k=10, sigma=1.0, device="cuda"):
    if isinstance(features, torch.Tensor):
        X = features.cpu().numpy()
    else:
        X = features

    N = X.shape[0]
    if N <= 1:
        return torch.zeros((N, N), dtype=torch.float32, device=device)

    X = StandardScaler().fit_transform(X)

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
        raise ValueError("sim_type must be cosine / euclidean / rbf")

    np.fill_diagonal(W, 0)

    knn_k = min(int(knn_k), max(N - 1, 1))
    W_knn = np.zeros_like(W)
    for i in range(N):
        idx = np.argsort(W[i])[::-1][:knn_k]
        W_knn[i, idx] = W[i, idx]

    W = np.maximum(W_knn, W_knn.T)
    D = np.diag(W.sum(axis=1))
    L = D - W
    return torch.tensor(L, dtype=torch.float32, device=device)


def graph_quadratic(U, L):
    return torch.sum(U * (L @ U))


def spatial_3d_smoothness_loss(U_S, H, W, D):
    _, rS = U_S.shape
    U_vol = U_S.view(H, W, D, rS).permute(3, 0, 1, 2)

    dx = U_vol[:, 1:, :, :] - U_vol[:, :-1, :, :]
    dy = U_vol[:, :, 1:, :] - U_vol[:, :, :-1, :]
    dz = U_vol[:, :, :, 1:] - U_vol[:, :, :, :-1]
    return dx.pow(2).mean() + dy.pow(2).mean() + dz.pow(2).mean()


def build_tv_pairs(H, W, D, max_pairs=200000, seed=123):
    max_pairs = int(max_pairs)
    if max_pairs <= 0:
        return None

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    per_axis = max(1, max_pairs // 3)
    pairs = []

    if H > 1:
        h = torch.randint(0, H - 1, (per_axis,), generator=gen)
        w = torch.randint(0, W, (per_axis,), generator=gen)
        d = torch.randint(0, D, (per_axis,), generator=gen)
        i0 = (h * W + w) * D + d
        i1 = ((h + 1) * W + w) * D + d
        pairs.append((i0, i1))
    if W > 1:
        h = torch.randint(0, H, (per_axis,), generator=gen)
        w = torch.randint(0, W - 1, (per_axis,), generator=gen)
        d = torch.randint(0, D, (per_axis,), generator=gen)
        i0 = (h * W + w) * D + d
        i1 = (h * W + (w + 1)) * D + d
        pairs.append((i0, i1))
    if D > 1:
        h = torch.randint(0, H, (per_axis,), generator=gen)
        w = torch.randint(0, W, (per_axis,), generator=gen)
        d = torch.randint(0, D - 1, (per_axis,), generator=gen)
        i0 = (h * W + w) * D + d
        i1 = (h * W + w) * D + (d + 1)
        pairs.append((i0, i1))

    if not pairs:
        return None
    idx0 = torch.cat([p[0] for p in pairs], dim=0)[:max_pairs].long()
    idx1 = torch.cat([p[1] for p in pairs], dim=0)[:max_pairs].long()
    return idx0, idx1


def tv3d_reconstruction_loss(model, tv_pairs, pair_chunk=32768):
    if tv_pairs is None:
        return torch.tensor(0.0, device=model.core.device)
    idx0_cpu, idx1_cpu = tv_pairs
    total = torch.tensor(0.0, device=model.core.device)
    count = 0
    for p0 in range(0, idx0_cpu.numel(), pair_chunk):
        p1 = min(idx0_cpu.numel(), p0 + pair_chunk)
        idx0 = idx0_cpu[p0:p1].to(model.core.device)
        idx1 = idx1_cpu[p0:p1].to(model.core.device)
        x0 = torch.einsum("sa,abc,mb,nc->smn", model.U_S.index_select(0, idx0), model.core, model.U_M, model.U_N)
        x1 = torch.einsum("sa,abc,mb,nc->smn", model.U_S.index_select(0, idx1), model.core, model.U_M, model.U_N)
        total = total + (torch.relu(x0) - torch.relu(x1)).abs().sum()
        count += x0.numel()
    return total / max(count, 1)


class Tucker3DModel(nn.Module):
    def __init__(self, S, M, N, rank_S, rank_M, rank_N, device="cpu"):
        super().__init__()
        self.S, self.M, self.N = S, M, N
        self.rS, self.rM, self.rN = rank_S, rank_M, rank_N

        self.core = nn.Parameter(0.1 * torch.randn(rank_S, rank_M, rank_N, device=device))
        self.U_S = nn.Parameter(torch.randn(S, rank_S, device=device) * 0.1)
        self.U_M = nn.Parameter(torch.randn(M, rank_M, device=device) * 0.1)
        self.U_N = nn.Parameter(torch.randn(N, rank_N, device=device) * 0.1)

    def chunk_forward(self, s0, s1):
        U_S_c = self.U_S[s0:s1]
        X_hat_c = torch.einsum("sa,abc,mb,nc->smn", U_S_c, self.core, self.U_M, self.U_N)
        return torch.relu(X_hat_c)


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def maybe_pin_memory(x):
    if os.environ.get("GRNNTC_PIN_MEMORY", "0") != "1":
        return x
    try:
        return x.pin_memory()
    except RuntimeError as exc:
        print(f"[Warn] pin_memory failed, continuing without pinned memory: {exc}")
        return x


def _pca_features(patient_features):
    Xpf = patient_features.cpu().numpy()
    Xpf = StandardScaler().fit_transform(Xpf)
    n_components = min(10, Xpf.shape[0], Xpf.shape[1])
    if n_components <= 0:
        return Xpf
    return PCA(n_components=n_components).fit_transform(Xpf)


def _save_outputs(model, save_factors, pack_dir, H, W, D, M, N, rank, chunk_S, metadata):
    print("Reconstructing train X_hat ...")
    S = H * W * D
    X_hat_cpu = torch.zeros((S, M, N), dtype=torch.float32)
    with torch.no_grad():
        for s0 in range(0, S, chunk_S):
            s1 = min(S, s0 + chunk_S)
            X_hat_cpu[s0:s1] = model.chunk_forward(s0, s1).cpu()

    X_hat = X_hat_cpu.view(H, W, D, M, N)

    save_dict = {
        "G": model.core.detach().cpu(),
        "U_S": model.U_S.detach().cpu(),
        "U_M": model.U_M.detach().cpu(),
        "U_N": model.U_N.detach().cpu(),
        "HWD": (H, W, D),
        "rank": rank,
        "M": M,
        "N": N,
        **metadata,
    }
    torch.save(save_dict, save_factors)
    print("Saved Tucker factors to:", save_factors)

    xhat_path = os.path.join(pack_dir, "X_hat_train_5d.pt")
    torch.save(X_hat.detach().cpu(), xhat_path)
    print("Saved X_hat (H,W,D,M,N) to:", xhat_path)
    print("X_hat range:", float(X_hat.min()), float(X_hat.max()))


def run_patient_mode(cfg):
    pack_dir = cfg["pack_dir"]
    device = cfg["device"]
    xfull_path = find_xfull_path(pack_dir)
    split_path = os.path.join(pack_dir, "split.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"patient mode needs split.json in {pack_dir}")

    full = torch.load(xfull_path, map_location="cpu")
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    X_full_all = full["X_full"].float()
    y_all = full.get("y", None)
    idx_train_global = torch.tensor(split["idx_train"], dtype=torch.long)
    X_train_nm = X_full_all.index_select(0, idx_train_global).contiguous()
    y_train = None if y_all is None else y_all.index_select(0, idx_train_global).long()

    N, M, H, W, D = X_train_nm.shape
    S = H * W * D
    print("Train mode: patient")
    print("X_train:", tuple(X_train_nm.shape), "S:", S)

    X_train_hwdmn = X_train_nm.permute(2, 3, 4, 1, 0).contiguous()
    X_true_flat = maybe_pin_memory(X_train_hwdmn.view(S, M, N))

    L_M_np = build_modality_laplacian(X_train_hwdmn)
    L_M = torch.tensor(L_M_np, dtype=torch.float32, device=device)

    prior_state = None
    if cfg["use_modality_prior"]:
        print("Fitting observed-modality linear image prior...")
        prior_state = fit_modality_prior(
            X_train_nm,
            max_samples=cfg["prior_max_samples"],
            ridge=cfg["prior_ridge"],
            seed=123,
        )
        print("Modality prior:", summarize_prior_state(prior_state))

    patient_features = X_true_flat.mean(dim=0).T
    Xp_pca = _pca_features(patient_features)
    L_N = build_patient_graph_laplacian(Xp_pca, sim_type="rbf", device=device)

    model = Tucker3DModel(
        S, M, N,
        rank_S=cfg["rank_S"],
        rank_M=cfg["rank_M"],
        rank_N=cfg["rank_N"],
        device=device,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    tv_pairs = build_tv_pairs(H, W, D, cfg["tv_max_pairs"], seed=123) if cfg["lambda_tv"] > 0 else None

    omega_groups = torch.tensor(
        [group_to_omega_row(present) for _, present in MODALITY_GROUPS],
        dtype=torch.float32,
    )
    group_names = [name for name, _ in MODALITY_GROUPS]
    subject_offsets = torch.arange(N, dtype=torch.long)

    print("Start training...")
    for it in range(cfg["max_iter"]):
        optimizer.zero_grad()

        group_idx = (subject_offsets + it) % len(group_names)
        omega_mn = omega_groups.index_select(0, group_idx).T.contiguous().to(device)
        omega_1mn = omega_mn.unsqueeze(0)

        data_num = 0.0
        data_den = 0.0
        prior_num = torch.tensor(0.0, device=device)
        prior_den = torch.tensor(0.0, device=device)
        for s0 in range(0, S, cfg["chunk_S"]):
            s1 = min(S, s0 + cfg["chunk_S"])
            X_true_c = X_true_flat[s0:s1].to(device, non_blocking=True)
            X_hat_c = model.chunk_forward(s0, s1)

            diff = (X_hat_c - X_true_c) * omega_1mn
            data_num = data_num + diff.pow(2).sum()
            data_den = data_den + omega_mn.sum() * (s1 - s0)

            if prior_state is not None and cfg["lambda_prior"] > 0:
                X_obs_c = X_true_c * omega_1mn
                prior_c, prior_mask_c = apply_modality_prior_chunk(X_obs_c, omega_mn, prior_state)
                prior_diff = (X_hat_c - prior_c) * prior_mask_c
                prior_num = prior_num + prior_diff.pow(2).sum()
                prior_den = prior_den + prior_mask_c.sum()

        data_loss = data_num / (data_den + 1e-8)
        prior_loss = prior_num / (prior_den + 1e-8)
        space_loss = spatial_3d_smoothness_loss(model.U_S, H, W, D)
        tv_loss = tv3d_reconstruction_loss(model, tv_pairs, pair_chunk=cfg["tv_pair_chunk"])
        mod_loss = graph_quadratic(model.U_M, L_M)
        subj_loss = graph_quadratic(model.U_N, L_N)
        loss = (
            data_loss
            + cfg["lambda_space"] * space_loss
            + cfg["lambda_tv"] * tv_loss
            + cfg["lambda_mod"] * mod_loss
            + cfg["lambda_subj"] * subj_loss
            + cfg["lambda_prior"] * prior_loss
        )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.U_S.data.clamp_(min=0)
            model.U_M.data.clamp_(min=0)
            model.U_N.data.clamp_(min=0)
            model.core.data.clamp_(min=0)
            model.U_S.data /= model.U_S.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

        if (it % cfg["log_every"] == 0) or (it == cfg["max_iter"] - 1):
            print(
                f"[Iter {it:04d}] loss={loss.item():.4e} "
                f"data={data_loss.item():.4e} space={space_loss.item():.4e} "
                f"mod={mod_loss.item():.4e} subj={subj_loss.item():.4e} "
                f"prior={prior_loss.item():.4e} tv={tv_loss.item():.4e}"
            )

    metadata = {
        "train_mode": "patient",
        "idx_train_global": idx_train_global.cpu(),
        "groups": [{"name": name, "present": present} for name, present in MODALITY_GROUPS],
        "lambda_tv": float(cfg["lambda_tv"]),
        "tv_max_pairs": int(cfg["tv_max_pairs"]),
    }
    if prior_state is not None:
        metadata["modality_prior_state"] = prior_state
        metadata["lambda_prior"] = float(cfg["lambda_prior"])
    if y_train is not None:
        metadata["y_train"] = y_train.cpu()

    _save_outputs(
        model=model,
        save_factors=cfg["save_factors"],
        pack_dir=pack_dir,
        H=H,
        W=W,
        D=D,
        M=M,
        N=N,
        rank=(cfg["rank_S"], cfg["rank_M"], cfg["rank_N"]),
        chunk_S=cfg["chunk_S"],
        metadata=metadata,
    )

    with torch.no_grad():
        mse_full_num = 0.0
        mse_full_den = 0
        for s0 in range(0, S, cfg["chunk_S"]):
            s1 = min(S, s0 + cfg["chunk_S"])
            X_true_c = X_true_flat[s0:s1].to(device, non_blocking=True)
            mse_full_num += (model.chunk_forward(s0, s1) - X_true_c).pow(2).sum().item()
            mse_full_den += X_true_c.numel()
    print(f"Train full-entry reconstruction MSE: {mse_full_num / max(mse_full_den, 1):.6e}")


def run_augmented_mode(cfg):
    pack_dir = cfg["pack_dir"]
    device = cfg["device"]
    train_aug_path = find_train_pack_path(pack_dir)
    xfull_path = find_xfull_path(pack_dir)
    if not os.path.exists(train_aug_path):
        raise FileNotFoundError(
            f"augmented mode needs train_aug_15mods.pt/train_aug_k1to4.pt: {train_aug_path}"
        )

    pack = torch.load(train_aug_path, map_location="cpu")
    full = torch.load(xfull_path, map_location="cpu")

    X_miss_nm = pack["X_miss"].float()
    Omega_nm111 = pack["Omega"].float()
    base_idx = pack["base_idx"].long()
    idx_train_global = pack["idx_train_global"].long()
    y_train = pack.get("y", None)

    X_full_all = full["X_full"].float()
    N_aug, M, H, W, D = X_miss_nm.shape
    S = H * W * D
    print("Train mode: augmented")
    print("X_miss_nm:", tuple(X_miss_nm.shape), "Omega:", tuple(Omega_nm111.shape))

    global_idx = idx_train_global.index_select(0, base_idx)
    X_gt_nm = X_full_all.index_select(0, global_idx).contiguous()

    X_miss = X_miss_nm.permute(2, 3, 4, 1, 0).contiguous()
    X_gt = X_gt_nm.permute(2, 3, 4, 1, 0).contiguous()
    Omega = Omega_nm111.expand(N_aug, M, H, W, D).permute(2, 3, 4, 1, 0).contiguous()

    X_true_flat = maybe_pin_memory(X_miss.view(S, M, N_aug))
    Omega_flat = maybe_pin_memory(Omega.view(S, M, N_aug))

    L_M_np = build_modality_laplacian(X_miss.cpu())
    L_M = torch.tensor(L_M_np, dtype=torch.float32, device=device)

    prior_state = None
    if cfg["use_modality_prior"]:
        print("Fitting observed-modality linear image prior...")
        train_full_nm = X_full_all.index_select(0, idx_train_global).contiguous()
        prior_state = fit_modality_prior(
            train_full_nm,
            max_samples=cfg["prior_max_samples"],
            ridge=cfg["prior_ridge"],
            seed=123,
        )
        print("Modality prior:", summarize_prior_state(prior_state))

    num_obs_mn = Omega_flat.sum(dim=0).clamp_min(1.0)
    feat_mn = (X_true_flat * Omega_flat).sum(dim=0) / num_obs_mn
    Xp_pca = _pca_features(feat_mn.T)
    L_N = build_patient_graph_laplacian(Xp_pca, sim_type="rbf", device=device)

    model = Tucker3DModel(
        S, M, N_aug,
        rank_S=cfg["rank_S"],
        rank_M=cfg["rank_M"],
        rank_N=cfg["rank_N"],
        device=device,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    tv_pairs = build_tv_pairs(H, W, D, cfg["tv_max_pairs"], seed=123) if cfg["lambda_tv"] > 0 else None

    print("Start training...")
    for it in range(cfg["max_iter"]):
        optimizer.zero_grad()
        data_num = 0.0
        data_den = 0.0
        prior_num = torch.tensor(0.0, device=device)
        prior_den = torch.tensor(0.0, device=device)
        for s0 in range(0, S, cfg["chunk_S"]):
            s1 = min(S, s0 + cfg["chunk_S"])
            X_true_c = X_true_flat[s0:s1].to(device, non_blocking=True)
            Omega_c = Omega_flat[s0:s1].to(device, non_blocking=True)
            X_hat_c = model.chunk_forward(s0, s1)
            diff = (X_hat_c - X_true_c) * Omega_c
            data_num = data_num + diff.pow(2).sum()
            data_den = data_den + Omega_c.sum()

            if prior_state is not None and cfg["lambda_prior"] > 0:
                omega_mn = Omega_c[0]
                prior_c, prior_mask_c = apply_modality_prior_chunk(X_true_c * Omega_c, omega_mn, prior_state)
                prior_diff = (X_hat_c - prior_c) * prior_mask_c
                prior_num = prior_num + prior_diff.pow(2).sum()
                prior_den = prior_den + prior_mask_c.sum()

        data_loss = data_num / (data_den + 1e-8)
        prior_loss = prior_num / (prior_den + 1e-8)
        space_loss = spatial_3d_smoothness_loss(model.U_S, H, W, D)
        tv_loss = tv3d_reconstruction_loss(model, tv_pairs, pair_chunk=cfg["tv_pair_chunk"])
        mod_loss = graph_quadratic(model.U_M, L_M)
        subj_loss = graph_quadratic(model.U_N, L_N)
        loss = (
            data_loss
            + cfg["lambda_space"] * space_loss
            + cfg["lambda_tv"] * tv_loss
            + cfg["lambda_mod"] * mod_loss
            + cfg["lambda_subj"] * subj_loss
            + cfg["lambda_prior"] * prior_loss
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.U_S.data.clamp_(min=0)
            model.U_M.data.clamp_(min=0)
            model.U_N.data.clamp_(min=0)
            model.core.data.clamp_(min=0)
            model.U_S.data /= model.U_S.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

        if (it % cfg["log_every"] == 0) or (it == cfg["max_iter"] - 1):
            print(
                f"[Iter {it:04d}] loss={loss.item():.4e} data={data_loss.item():.4e} "
                f"prior={prior_loss.item():.4e} tv={tv_loss.item():.4e}"
            )

    _save_outputs(
        model=model,
        save_factors=cfg["save_factors"],
        pack_dir=pack_dir,
        H=H,
        W=W,
        D=D,
        M=M,
        N=N_aug,
        rank=(cfg["rank_S"], cfg["rank_M"], cfg["rank_N"]),
        chunk_S=cfg["chunk_S"],
        metadata={
            "train_mode": "augmented",
            "base_idx": base_idx.cpu(),
            "idx_train_global": idx_train_global.cpu(),
            "y_train": None if y_train is None else y_train.long().cpu(),
            "modality_prior_state": prior_state,
            "lambda_prior": float(cfg["lambda_prior"]),
            "lambda_tv": float(cfg["lambda_tv"]),
            "tv_max_pairs": int(cfg["tv_max_pairs"]),
        },
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    pack_dir = get_pack_dir()
    cfg = {
        "device": device,
        "pack_dir": pack_dir,
        "save_factors": os.path.join(pack_dir, "tucker_factors_train.pt"),
        "rank_S": _env_int("GRNNTC_RANK_S", 6),
        "rank_M": _env_int("GRNNTC_RANK_M", 3),
        "rank_N": _env_int("GRNNTC_RANK_N", 32),
        "lambda_space": _env_float("GRNNTC_LAMBDA_SPACE", 1e-3),
        "lambda_mod": _env_float("GRNNTC_LAMBDA_MOD", 5e-4),
        "lambda_subj": _env_float("GRNNTC_LAMBDA_SUBJ", 5e-4),
        "lambda_prior": _env_float("GRNNTC_LAMBDA_PRIOR", 0.0),
        "use_modality_prior": os.environ.get("GRNNTC_USE_MODALITY_PRIOR", "0") == "1",
        "lambda_tv": _env_float("GRNNTC_LAMBDA_TV", 1e-2),
        "tv_max_pairs": _env_int("GRNNTC_TV_MAX_PAIRS", 200000),
        "tv_pair_chunk": _env_int("GRNNTC_TV_PAIR_CHUNK", 32768),
        "prior_max_samples": _env_int("GRNNTC_PRIOR_MAX_SAMPLES", 200000),
        "prior_ridge": _env_float("GRNNTC_PRIOR_RIDGE", 1e-3),
        "lr": _env_float("GRNNTC_LR", 1e-4),
        "max_iter": _env_int("GRNNTC_MAX_ITER", 2000),
        "log_every": _env_int("GRNNTC_LOG_EVERY", 200),
        "chunk_S": _env_int("GRNNTC_CHUNK_S", 2048),
    }

    print(
        "GRNNTC config:",
        {k: v for k, v in cfg.items() if k not in {"device", "pack_dir", "save_factors"}},
    )

    mode = os.environ.get("GRNNTC_TRAIN_MODE", "patient").strip().lower()
    if mode == "patient":
        run_patient_mode(cfg)
    elif mode == "augmented":
        run_augmented_mode(cfg)
    else:
        raise ValueError("GRNNTC_TRAIN_MODE must be patient or augmented")


if __name__ == "__main__":
    main()
