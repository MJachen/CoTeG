"""Completion backends and measurement helpers for experiment T2.

All public tensor inputs use the repository convention ``(N, M, H, W, D)``.
The two population arms share the same implementation and differ only in the
subject-graph coefficient.  The per-subject arm fits one spatial x modality
Tucker model per case and never shares factors between subjects.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler


ARM_PER_SUBJECT = "per_subject"
ARM_POPULATION_NO_GRAPH = "population_no_subject_graph"
ARM_FULL_COTEG = "full_coteg"
ARM_ORDER = [ARM_PER_SUBJECT, ARM_POPULATION_NO_GRAPH, ARM_FULL_COTEG]


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_sha256(x: torch.Tensor) -> str:
    arr = x.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    # memoryview avoids a second full-size byte copy for large MRI packs.
    h.update(memoryview(arr).cast("B"))
    return h.hexdigest()


def parameter_bytes(parameters: Iterable[torch.Tensor]) -> int:
    return int(sum(p.numel() * p.element_size() for p in parameters))


def build_modality_laplacian(x_nmhwd: torch.Tensor) -> torch.Tensor:
    """Build a non-negative correlation graph over modalities."""
    x = x_nmhwd.detach().cpu().float().permute(1, 0, 2, 3, 4).reshape(x_nmhwd.shape[1], -1).numpy()
    m = x.shape[0]
    w = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(i + 1, m):
            if float(np.std(x[i])) <= 1e-12 or float(np.std(x[j])) <= 1e-12:
                corr = 0.0
            else:
                corr = np.corrcoef(x[i], x[j])[0, 1]
            corr = float(corr) if np.isfinite(corr) else 0.0
            corr = max(corr, 0.0)
            w[i, j] = corr
            w[j, i] = corr
    return torch.from_numpy(np.diag(w.sum(axis=1)) - w).float()


def observed_subject_features(x_nmhwd: torch.Tensor, omega_nm: torch.Tensor) -> np.ndarray:
    """Return observed mean intensity per modality, with missingness indicators."""
    n, m = x_nmhwd.shape[:2]
    means = x_nmhwd.reshape(n, m, -1).mean(dim=2)
    obs = omega_nm.float()
    features = torch.cat([means * obs, obs], dim=1).cpu().numpy()
    return features


def build_subject_laplacian(
    features: np.ndarray,
    knn_k: int,
    sigma: float,
) -> torch.Tensor:
    n = int(features.shape[0])
    if n <= 1:
        return torch.zeros((n, n), dtype=torch.float32)
    scaled = StandardScaler().fit_transform(features)
    dist = euclidean_distances(scaled)
    sigma = max(float(sigma), 1e-8)
    w = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(w, 0.0)
    k = min(max(int(knn_k), 1), n - 1)
    sparse = np.zeros_like(w)
    for i in range(n):
        idx = np.argsort(w[i])[::-1][:k]
        sparse[i, idx] = w[i, idx]
    w = np.maximum(sparse, sparse.T).astype(np.float32)
    return torch.from_numpy(np.diag(w.sum(axis=1)) - w).float()


def graph_quadratic(u: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    return torch.sum(u * (laplacian @ u)) / max(int(u.shape[0]), 1)


def spatial_smoothness(u_s: torch.Tensor, hwd: Tuple[int, int, int]) -> torch.Tensor:
    h, w, d = hwd
    r_s = int(u_s.shape[-1])
    leading = u_s.shape[:-2]
    vol = u_s.reshape(*leading, h, w, d, r_s).movedim(-1, -4)
    dx = vol[..., 1:, :, :] - vol[..., :-1, :, :]
    dy = vol[..., :, 1:, :] - vol[..., :, :-1, :]
    dz = vol[..., :, :, 1:] - vol[..., :, :, :-1]
    return dx.square().mean() + dy.square().mean() + dz.square().mean()


def modality_quadratic_shared(u_m: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    return torch.sum(u_m * (laplacian @ u_m)) / max(int(u_m.shape[0]), 1)


def modality_quadratic_per_subject(u_m: torch.Tensor, laplacians: torch.Tensor) -> torch.Tensor:
    # u_m: (N,M,rM), laplacians: (N,M,M)
    return torch.einsum("nmr,nmk,nkr->", u_m, laplacians, u_m) / max(int(u_m.shape[0]), 1)


def build_tv_pairs(hwd: Tuple[int, int, int], max_pairs: int, seed: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    h, w, d = hwd
    if max_pairs <= 0:
        return None
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    per_axis = max(1, int(max_pairs) // 3)
    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    if h > 1:
        ih = torch.randint(0, h - 1, (per_axis,), generator=gen)
        iw = torch.randint(0, w, (per_axis,), generator=gen)
        id_ = torch.randint(0, d, (per_axis,), generator=gen)
        pairs.append(((ih * w + iw) * d + id_, (((ih + 1) * w + iw) * d + id_)))
    if w > 1:
        ih = torch.randint(0, h, (per_axis,), generator=gen)
        iw = torch.randint(0, w - 1, (per_axis,), generator=gen)
        id_ = torch.randint(0, d, (per_axis,), generator=gen)
        pairs.append(((ih * w + iw) * d + id_, ((ih * w + iw + 1) * d + id_)))
    if d > 1:
        ih = torch.randint(0, h, (per_axis,), generator=gen)
        iw = torch.randint(0, w, (per_axis,), generator=gen)
        id_ = torch.randint(0, d - 1, (per_axis,), generator=gen)
        pairs.append(((ih * w + iw) * d + id_, ((ih * w + iw) * d + id_ + 1)))
    if not pairs:
        return None
    idx0 = torch.cat([p[0] for p in pairs])[:max_pairs].long()
    idx1 = torch.cat([p[1] for p in pairs])[:max_pairs].long()
    return idx0, idx1


@dataclass
class CompletionConfig:
    rank_s: int
    rank_m: int
    rank_n: int
    train_iterations: int
    eval_iterations: int
    lr_train: float
    lr_eval: float
    lambda_space: float
    lambda_mod: float
    lambda_subject: float
    lambda_tv: float
    tv_max_pairs: int
    chunk_s: int
    knn_k: int
    graph_sigma: float
    seed: int
    per_subject_batch_size: int = 1
    log_every: int = 0


@dataclass
class StageStats:
    seconds: float
    peak_cpu_rss_bytes: int
    peak_cuda_allocated_bytes: int


class ResourceMeter:
    def __init__(self, device: torch.device):
        self.device = device
        self.start_time = 0.0
        self.peak_rss = 0
        self._process = None

    def __enter__(self):
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
        except ImportError:
            self._process = None
        self.sample()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.start_time = time.perf_counter()
        return self

    def sample(self) -> None:
        if self._process is not None:
            self.peak_rss = max(self.peak_rss, int(self._process.memory_info().rss))

    def __exit__(self, exc_type, exc, tb):
        self.sample()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.seconds = time.perf_counter() - self.start_time
        self.peak_cuda = (
            int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else 0
        )

    def stats(self) -> StageStats:
        return StageStats(self.seconds, self.peak_rss, self.peak_cuda)


class PopulationTucker(nn.Module):
    def __init__(self, s: int, m: int, n: int, rank_s: int, rank_m: int, rank_n: int, device: torch.device):
        super().__init__()
        self.core = nn.Parameter(0.1 * torch.randn(rank_s, rank_m, rank_n, device=device))
        self.u_s = nn.Parameter(0.1 * torch.randn(s, rank_s, device=device))
        self.u_m = nn.Parameter(0.1 * torch.randn(m, rank_m, device=device))
        self.u_n = nn.Parameter(0.1 * torch.randn(n, rank_n, device=device))

    def reconstruct_chunk(self, s0: int, s1: int) -> torch.Tensor:
        return torch.relu(torch.einsum("sa,abc,mb,nc->smn", self.u_s[s0:s1], self.core, self.u_m, self.u_n))

    def reconstruct_indices(self, idx: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.einsum("pa,abc,mb,nc->pmn", self.u_s.index_select(0, idx), self.core, self.u_m, self.u_n))


class PerSubjectTucker(nn.Module):
    """Independent spatial x modality Tucker models stored in one batched module."""

    def __init__(self, s: int, m: int, n: int, rank_s: int, rank_m: int, device: torch.device):
        super().__init__()
        self.core = nn.Parameter(0.1 * torch.randn(n, rank_s, rank_m, device=device))
        self.u_s = nn.Parameter(0.1 * torch.randn(n, s, rank_s, device=device))
        self.u_m = nn.Parameter(0.1 * torch.randn(n, m, rank_m, device=device))

    def reconstruct_chunk(self, s0: int, s1: int) -> torch.Tensor:
        # Return (S_chunk,M,N), matching PopulationTucker.
        x = torch.einsum("nsa,nab,nmb->nsm", self.u_s[:, s0:s1], self.core, self.u_m)
        return torch.relu(x).permute(1, 2, 0).contiguous()

    def reconstruct_indices(self, idx: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("npa,nab,nmb->npm", self.u_s.index_select(1, idx), self.core, self.u_m)
        return torch.relu(x).permute(1, 2, 0).contiguous()


def _reconstruct_all(model: nn.Module, s: int, m: int, n: int, chunk_s: int) -> torch.Tensor:
    out = torch.empty((s, m, n), dtype=torch.float32)
    with torch.no_grad():
        for s0 in range(0, s, chunk_s):
            s1 = min(s, s0 + chunk_s)
            out[s0:s1] = model.reconstruct_chunk(s0, s1).detach().cpu()
    return out


def _tv_loss(model: nn.Module, pairs: Optional[Tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> torch.Tensor:
    if pairs is None:
        first = next(model.parameters())
        return torch.zeros((), device=first.device, dtype=first.dtype)
    i0, i1 = pairs
    i0 = i0.to(device)
    i1 = i1.to(device)
    return (model.reconstruct_indices(i0) - model.reconstruct_indices(i1)).abs().mean()


def _clamp_nonnegative(model: nn.Module) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.clamp_(min=0.0)


def fit_population_training_basis(
    x_train_nmhwd: torch.Tensor,
    omega_groups_gm: torch.Tensor,
    cfg: CompletionConfig,
    device: torch.device,
    use_subject_graph: bool,
) -> Tuple[PopulationTucker, torch.Tensor, StageStats]:
    stage_start = time.perf_counter()
    set_deterministic_seed(cfg.seed)
    n, m, h, w, d = x_train_nmhwd.shape
    s = h * w * d
    x_flat = x_train_nmhwd.permute(2, 3, 4, 1, 0).reshape(s, m, n).contiguous()
    l_m = build_modality_laplacian(x_train_nmhwd).to(device)
    full_omega = torch.ones((n, m), dtype=torch.float32)
    l_n = None
    if use_subject_graph:
        l_n = build_subject_laplacian(
            observed_subject_features(x_train_nmhwd, full_omega), cfg.knn_k, cfg.graph_sigma
        ).to(device)
    tv_pairs = build_tv_pairs((h, w, d), cfg.tv_max_pairs, cfg.seed) if cfg.lambda_tv > 0 else None
    model = PopulationTucker(s, m, n, cfg.rank_s, cfg.rank_m, cfg.rank_n, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_train, weight_decay=1e-5)
    subject_offsets = torch.arange(n, dtype=torch.long)

    with ResourceMeter(device) as meter:
        for iteration in range(cfg.train_iterations):
            optimizer.zero_grad(set_to_none=True)
            group_idx = (subject_offsets + iteration) % int(omega_groups_gm.shape[0])
            omega_mn = omega_groups_gm.index_select(0, group_idx).T.contiguous().to(device)
            numerator = torch.zeros((), device=device)
            denominator = torch.zeros((), device=device)
            for s0 in range(0, s, cfg.chunk_s):
                s1 = min(s, s0 + cfg.chunk_s)
                true = x_flat[s0:s1].to(device, non_blocking=True)
                diff = (model.reconstruct_chunk(s0, s1) - true) * omega_mn.unsqueeze(0)
                numerator = numerator + diff.square().sum()
                denominator = denominator + omega_mn.sum() * (s1 - s0)
            data_loss = numerator / denominator.clamp_min(1.0)
            space_loss = spatial_smoothness(model.u_s, (h, w, d))
            mod_loss = modality_quadratic_shared(model.u_m, l_m)
            subj_loss = graph_quadratic(model.u_n, l_n) if use_subject_graph else torch.zeros((), device=device)
            tv_loss = _tv_loss(model, tv_pairs, device)
            loss = (
                data_loss
                + cfg.lambda_space * space_loss
                + cfg.lambda_mod * mod_loss
                + (cfg.lambda_subject if use_subject_graph else 0.0) * subj_loss
                + cfg.lambda_tv * tv_loss
            )
            loss.backward()
            optimizer.step()
            _clamp_nonnegative(model)
            meter.sample()
            if cfg.log_every and (iteration % cfg.log_every == 0 or iteration + 1 == cfg.train_iterations):
                print(
                    f"  train iter={iteration:04d} loss={loss.item():.4e} data={data_loss.item():.4e} "
                    f"subject={subj_loss.item():.4e}",
                    flush=True,
                )
        x_hat_flat = _reconstruct_all(model, s, m, n, cfg.chunk_s)
        meter.sample()
    x_hat = x_hat_flat.reshape(h, w, d, m, n).permute(4, 3, 0, 1, 2).contiguous()
    model = model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    stats = meter.stats()
    stats.seconds = time.perf_counter() - stage_start
    return model, x_hat, stats


def fit_per_subject_training_completion(
    x_train_nmhwd: torch.Tensor,
    omega_groups_gm: torch.Tensor,
    cfg: CompletionConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, StageStats]:
    outputs = []
    total_seconds = 0.0
    peak_rss = 0
    peak_cuda = 0
    batch_size = max(int(cfg.per_subject_batch_size), 1)
    for subject_index in range(0, int(x_train_nmhwd.shape[0]), batch_size):
        batch = x_train_nmhwd[subject_index : subject_index + batch_size]
        omega_schedule = omega_groups_gm.roll(shifts=-subject_index, dims=0)
        completed, stats = _fit_per_subject_batch(
            batch,
            omega_schedule,
            cfg,
            device,
            iterations=cfg.train_iterations,
            seed=cfg.seed + subject_index,
            cyclic_schedule=True,
            learning_rate=cfg.lr_train,
        )
        outputs.append(completed)
        total_seconds += stats.seconds
        peak_rss = max(peak_rss, stats.peak_cpu_rss_bytes)
        peak_cuda = max(peak_cuda, stats.peak_cuda_allocated_bytes)
    return torch.cat(outputs, dim=0), StageStats(total_seconds, peak_rss, peak_cuda)


def _fit_per_subject_batch(
    x_nmhwd: torch.Tensor,
    omega_gm: torch.Tensor,
    cfg: CompletionConfig,
    device: torch.device,
    iterations: int,
    seed: int,
    cyclic_schedule: bool,
    learning_rate: float,
) -> Tuple[torch.Tensor, StageStats]:
    stage_start = time.perf_counter()
    set_deterministic_seed(seed)
    n, m, h, w, d = x_nmhwd.shape
    s = h * w * d
    x_flat = x_nmhwd.permute(2, 3, 4, 1, 0).reshape(s, m, n).contiguous()
    laplacians = torch.stack([build_modality_laplacian(x_nmhwd[i : i + 1]) for i in range(n)], dim=0).to(device)
    tv_pairs = build_tv_pairs((h, w, d), cfg.tv_max_pairs, seed) if cfg.lambda_tv > 0 else None
    model = PerSubjectTucker(s, m, n, cfg.rank_s, cfg.rank_m, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    offsets = torch.arange(n, dtype=torch.long)

    with ResourceMeter(device) as meter:
        for iteration in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            if cyclic_schedule:
                group_idx = (offsets + iteration) % int(omega_gm.shape[0])
                omega_nm = omega_gm.index_select(0, group_idx).to(device)
            else:
                if omega_gm.shape[0] != n:
                    raise ValueError("Fixed per-subject masks must have shape (N,M).")
                omega_nm = omega_gm.to(device)
            omega_mn = omega_nm.T.contiguous()
            numerator = torch.zeros((), device=device)
            denominator = torch.zeros((), device=device)
            for s0 in range(0, s, cfg.chunk_s):
                s1 = min(s, s0 + cfg.chunk_s)
                true = x_flat[s0:s1].to(device, non_blocking=True)
                diff = (model.reconstruct_chunk(s0, s1) - true) * omega_mn.unsqueeze(0)
                numerator = numerator + diff.square().sum()
                denominator = denominator + omega_mn.sum() * (s1 - s0)
            data_loss = numerator / denominator.clamp_min(1.0)
            loss = (
                data_loss
                + cfg.lambda_space * spatial_smoothness(model.u_s, (h, w, d))
                + cfg.lambda_mod * modality_quadratic_per_subject(model.u_m, laplacians)
                + cfg.lambda_tv * _tv_loss(model, tv_pairs, device)
            )
            loss.backward()
            optimizer.step()
            _clamp_nonnegative(model)
            meter.sample()
        x_hat_flat = _reconstruct_all(model, s, m, n, cfg.chunk_s)
        meter.sample()
    x_hat = x_hat_flat.reshape(h, w, d, m, n).permute(4, 3, 0, 1, 2).contiguous()
    stats = meter.stats()
    stats.seconds = time.perf_counter() - stage_start
    return x_hat, stats


def infer_population_completion(
    basis: PopulationTucker,
    x_full_nmhwd: torch.Tensor,
    omega_m: torch.Tensor,
    cfg: CompletionConfig,
    device: torch.device,
    use_subject_graph: bool,
    seed: int,
) -> Tuple[torch.Tensor, StageStats, int, int]:
    stage_start = time.perf_counter()
    set_deterministic_seed(seed)
    n, m, h, w, d = x_full_nmhwd.shape
    s = h * w * d
    omega_nm = omega_m.view(1, m).expand(n, m).contiguous()
    x_miss = x_full_nmhwd * omega_nm.view(n, m, 1, 1, 1)
    x_flat = x_miss.permute(2, 3, 4, 1, 0).reshape(s, m, n).contiguous()
    rank_n = int(basis.core.shape[2])
    eval_model = PopulationTucker(s, m, n, int(basis.u_s.shape[1]), int(basis.u_m.shape[1]), rank_n, device).to(device)
    with torch.no_grad():
        eval_model.core.copy_(basis.core.detach())
        eval_model.u_s.copy_(basis.u_s.detach())
        eval_model.u_m.copy_(basis.u_m.detach())
    eval_model.core.requires_grad_(False)
    eval_model.u_s.requires_grad_(False)
    eval_model.u_m.requires_grad_(False)
    optimizer = torch.optim.Adam([eval_model.u_n], lr=cfg.lr_eval, weight_decay=1e-5)
    l_n = None
    if use_subject_graph:
        l_n = build_subject_laplacian(
            observed_subject_features(x_miss, omega_nm), cfg.knn_k, cfg.graph_sigma
        ).to(device)
    tv_pairs = build_tv_pairs((h, w, d), cfg.tv_max_pairs, seed) if cfg.lambda_tv > 0 else None

    with ResourceMeter(device) as meter:
        for _ in range(cfg.eval_iterations):
            optimizer.zero_grad(set_to_none=True)
            numerator = torch.zeros((), device=device)
            denominator = torch.zeros((), device=device)
            for s0 in range(0, s, cfg.chunk_s):
                s1 = min(s, s0 + cfg.chunk_s)
                true = x_flat[s0:s1].to(device, non_blocking=True)
                diff = (eval_model.reconstruct_chunk(s0, s1) - true) * omega_nm.T.to(device).unsqueeze(0)
                numerator = numerator + diff.square().sum()
                denominator = denominator + omega_nm.sum().to(device) * (s1 - s0)
            data_loss = numerator / denominator.clamp_min(1.0)
            subj_loss = graph_quadratic(eval_model.u_n, l_n) if use_subject_graph else torch.zeros((), device=device)
            loss = data_loss + (cfg.lambda_subject if use_subject_graph else 0.0) * subj_loss + cfg.lambda_tv * _tv_loss(eval_model, tv_pairs, device)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                eval_model.u_n.clamp_(min=0.0)
            meter.sample()
        pred_flat = _reconstruct_all(eval_model, s, m, n, cfg.chunk_s)
        meter.sample()
    pred = pred_flat.reshape(h, w, d, m, n).permute(4, 3, 0, 1, 2).contiguous()
    completed = x_miss + pred * (1.0 - omega_nm.view(n, m, 1, 1, 1))
    representation_params = sum(p.numel() for p in eval_model.parameters())
    representation_bytes = parameter_bytes(eval_model.parameters())
    stats = meter.stats()
    stats.seconds = time.perf_counter() - stage_start
    return completed, stats, int(representation_params), representation_bytes


def infer_per_subject_completion(
    x_full_nmhwd: torch.Tensor,
    omega_m: torch.Tensor,
    cfg: CompletionConfig,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, StageStats, int, int]:
    n, m = x_full_nmhwd.shape[:2]
    outputs = []
    total_seconds = 0.0
    peak_rss = 0
    peak_cuda = 0
    one_subject_params = 0
    one_subject_bytes = 0
    batch_size = max(int(cfg.per_subject_batch_size), 1)
    for subject_index in range(0, n, batch_size):
        x_true = x_full_nmhwd[subject_index : subject_index + batch_size]
        batch_n = int(x_true.shape[0])
        omega_nm = omega_m.view(1, m).expand(batch_n, m).contiguous()
        x_miss = x_true * omega_m.view(1, m, 1, 1, 1)
        pred, stats = _fit_per_subject_batch(
            x_miss,
            omega_nm,
            cfg,
            device,
            iterations=cfg.eval_iterations,
            seed=seed + subject_index,
            cyclic_schedule=False,
            learning_rate=cfg.lr_eval,
        )
        completed = x_miss + pred * (1.0 - omega_m.view(1, m, 1, 1, 1))
        outputs.append(completed)
        total_seconds += stats.seconds
        peak_rss = max(peak_rss, stats.peak_cpu_rss_bytes)
        peak_cuda = max(peak_cuda, stats.peak_cuda_allocated_bytes)
        if one_subject_params == 0:
            s = math.prod(x_true.shape[2:])
            one_subject_params = cfg.rank_s * cfg.rank_m + s * cfg.rank_s + m * cfg.rank_m
            one_subject_bytes = one_subject_params * torch.tensor([], dtype=torch.float32).element_size()
    return (
        torch.cat(outputs, dim=0),
        StageStats(total_seconds, peak_rss, peak_cuda),
        int(one_subject_params * n),
        int(one_subject_bytes * n),
    )


def training_capacity(
    arm: str,
    shape_nmhwd: Sequence[int],
    cfg: CompletionConfig,
) -> Dict[str, int]:
    n, m, h, w, d = [int(v) for v in shape_nmhwd]
    s = h * w * d
    if arm == ARM_PER_SUBJECT:
        per_subject = cfg.rank_s * cfg.rank_m + s * cfg.rank_s + m * cfg.rank_m
        total = n * per_subject
        return {
            "rank_s": cfg.rank_s,
            "rank_m": cfg.rank_m,
            "rank_n": 1,
            "parameters_per_subject": per_subject,
            "parameters_total": total,
            "factor_storage_bytes": total * 4,
        }
    total = s * cfg.rank_s + m * cfg.rank_m + n * cfg.rank_n + cfg.rank_s * cfg.rank_m * cfg.rank_n
    return {
        "rank_s": cfg.rank_s,
        "rank_m": cfg.rank_m,
        "rank_n": cfg.rank_n,
        "parameters_per_subject": -1,
        "parameters_total": total,
        "factor_storage_bytes": total * 4,
    }
