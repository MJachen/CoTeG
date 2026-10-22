import itertools

import torch


def _mask_key(present):
    return "".join("1" if bool(v) else "0" for v in present)


def _all_nonempty_masks(m):
    for bits in itertools.product([0, 1], repeat=m):
        if any(bits):
            yield bits


def fit_modality_prior(
    x_full_nm,
    max_samples=200000,
    ridge=1e-3,
    seed=123,
):
    """
    Fit global linear image priors for missing modalities.

    x_full_nm: (N,M,H,W,D), values expected in [0,1].

    For each observed-modality mask and each missing target modality, fit:
        x_target ~= intercept + sum_j coeff_j * x_observed_j

    The fitted state is small and can be saved inside tucker_factors_train.pt.
    """
    if x_full_nm.ndim != 5:
        raise ValueError(f"x_full_nm must be (N,M,H,W,D), got {tuple(x_full_nm.shape)}")

    x_full_nm = x_full_nm.float().cpu()
    n, m, h, w, d = x_full_nm.shape
    s = h * w * d
    total = n * s
    k = int(min(max_samples, total))
    if k <= 0:
        raise ValueError("Cannot fit modality prior from an empty tensor.")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    sample_linear = torch.randperm(total, generator=gen)[:k]
    n_idx = sample_linear // s
    rem = sample_linear % s
    h_idx = rem // (w * d)
    rem = rem % (w * d)
    w_idx = rem // d
    d_idx = rem % d
    samples = x_full_nm[n_idx, :, h_idx, w_idx, d_idx].float()  # (K,M)

    entries = []
    eye_cache = {}
    for present_bits in _all_nonempty_masks(m):
        present = [i for i, bit in enumerate(present_bits) if bit]
        x = samples[:, present]
        x_aug = torch.cat([torch.ones((k, 1), dtype=x.dtype), x], dim=1)
        xtx = x_aug.T @ x_aug
        p = x_aug.shape[1]
        if p not in eye_cache:
            eye_cache[p] = torch.eye(p, dtype=x.dtype)
            eye_cache[p][0, 0] = 0.0
        xtx = xtx + float(ridge) * eye_cache[p]

        for target in range(m):
            if present_bits[target]:
                continue
            y = samples[:, target:target + 1]
            xty = x_aug.T @ y
            try:
                beta = torch.linalg.solve(xtx, xty).flatten()
            except RuntimeError:
                beta = torch.linalg.pinv(xtx) @ xty
                beta = beta.flatten()
            entries.append(
                {
                    "key": _mask_key(present_bits),
                    "target": int(target),
                    "present": [int(i) for i in present],
                    "intercept": float(beta[0].item()),
                    "coeff": [float(v) for v in beta[1:].tolist()],
                }
            )

    return {
        "type": "global_linear_modality_prior",
        "M": int(m),
        "max_samples": int(k),
        "ridge": float(ridge),
        "entries": entries,
    }


def summarize_prior_state(prior_state):
    if not prior_state:
        return "none"
    return (
        f"{prior_state.get('type', 'unknown')} "
        f"M={prior_state.get('M')} entries={len(prior_state.get('entries', []))} "
        f"samples={prior_state.get('max_samples')}"
    )


def build_prior_lookup(prior_state):
    lookup = {}
    if not prior_state:
        return lookup
    for entry in prior_state.get("entries", []):
        lookup[(entry["key"], int(entry["target"]))] = entry
    return lookup


def apply_modality_prior_chunk(x_obs_smn, omega_mn, prior_state, clamp=True):
    """
    x_obs_smn: (S_chunk,M,N), observed entries present and missing entries zeroed.
    omega_mn: (M,N), 1 for observed, 0 for missing.

    Returns:
        prior_smn, prior_mask_smn. prior_mask is 1 only for missing entries where
        a prior can be computed from at least one observed modality.
    """
    if not prior_state:
        return torch.zeros_like(x_obs_smn), torch.zeros_like(x_obs_smn)

    lookup = build_prior_lookup(prior_state)
    if not lookup:
        return torch.zeros_like(x_obs_smn), torch.zeros_like(x_obs_smn)

    s_chunk, m, n = x_obs_smn.shape
    prior = torch.zeros_like(x_obs_smn)
    prior_mask = torch.zeros_like(x_obs_smn)
    omega_cpu = omega_mn.detach().cpu()

    for subj in range(n):
        present_bits = [bool(v > 0.5) for v in omega_cpu[:, subj].tolist()]
        key = _mask_key(present_bits)
        if key.count("1") == 0:
            continue
        for target in range(m):
            if present_bits[target]:
                continue
            entry = lookup.get((key, target))
            if entry is None:
                continue
            pred = torch.full(
                (s_chunk,),
                float(entry["intercept"]),
                dtype=x_obs_smn.dtype,
                device=x_obs_smn.device,
            )
            for src, coeff in zip(entry["present"], entry["coeff"]):
                pred = pred + float(coeff) * x_obs_smn[:, int(src), subj]
            if clamp:
                pred = pred.clamp(0.0, 1.0)
            prior[:, target, subj] = pred
            prior_mask[:, target, subj] = 1.0

    return prior, prior_mask
