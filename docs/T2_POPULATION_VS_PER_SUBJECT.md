# T2: Population-level versus per-subject completion

## Question and primary criterion

T2 isolates whether the gain comes from population relationships rather than
only from fitting a larger joint tensor. It compares:

1. `per_subject`: each case has independent spatial and modality Tucker factors;
2. `population_no_subject_graph`: one population tensor with `lambda_subject=0`;
3. `full_coteg`: the same population tensor with subject-graph regularization.

The primary criterion is deliberately strict: `full_coteg` must exceed both
baselines on both 15-combination macro BAC and macro AUC. A smoke run validates
the execution path only and cannot establish this claim.

## Controlled quantities

- The same `split.json` indices are used by every arm.
- The same canonical 15 non-empty modality masks from `modality_groups.py` are
  used by every arm. Their per-split SHA-256 hashes are stored in
  `protocol_manifest.json`.
- Both population arms use identical Tucker ranks, initialization seed,
  optimizer, iterations, spatial/modality/TV terms, and classifier protocol.
  The only difference is whether the subject-graph coefficient is zero.
- Every arm trains the same tensor-only `ConvTRLNet` downstream classifier with
  the same seed and hyperparameters. The graph classification branch is not
  used, because it would confound completion-stage population integration with
  downstream graph propagation.
- One validation threshold is selected by maximizing macro BAC across all 15
  validation combinations, then frozen for all test combinations.

Training uses all 15 masks as deterministic augmentation. At validation and
test time, population methods optimize only the split-specific subject factor
`U_N`; `G`, `U_S`, and `U_M` are learned from the training split and frozen.
Per-subject completion fits each case independently and shares no factor or
graph across cases. Its independent models may be vectorized in small batches
(`per_subject_batch_size`) for GPU efficiency; no parameter or loss term is
shared within such a batch. Validation and test cohorts are never mixed.

## Capacity and storage

The component ranks `rank_s` and `rank_m` are matched. The population arms also
share the same `rank_n`. Exact factor parameter counts and float32 storage are
reported in `capacity.csv` for the training cohort and a test inference cohort.
This is important because equal component ranks do not imply equal total
capacity: per-subject spatial factors are duplicated once per case.

## Metrics and files

- `summary.csv`: 15-combination macro BAC/AUC, four `K=1` combination macro
  BAC/AUC, and missing-group reconstruction quality.
- `per_combination.csv`: BAC, AUC, MAE, and SSIM for every modality combination.
- `completion_quality_detail.csv`: subject/modality-level MAE and SSIM.
- `sample_predictions.csv`: auditable validation/test probabilities.
- `resource_usage.csv`: completion wall time, sampled peak process RSS, and
  CUDA peak allocated memory for each stage/group.
- `classifier_resources.csv`: classifier time, parameters, and CUDA peak.
- `capacity.csv`: actual Tucker parameter and factor-storage differences.
- `decision.json`: automatic strict-criterion result and full-minus-baseline
  deltas.
- `protocol_manifest.json`: input/config/split/mask hashes and environment.

`MAE_brain` is calculated over the union brain support defined by any non-zero
ground-truth modality. `SSIM_brain_bbox` is 3D SSIM over that support's bounding
box, averaged over missing modalities and cases. The full-modality group has no
missing entries and therefore has undefined reconstruction metrics.

## Local smoke run

```powershell
E:\cjj\anaconda3\envs\tensor\python.exe run_t2_completion_ablation.py `
  --config configs\t2_smoke.json
```

The smoke config uses the existing 10-case, `32 x 32 x 32`, fold-0 fixture,
all 15 modality combinations, one completion iteration, and one classifier
epoch. It tests interfaces and output completeness, not scientific performance.

## Full run

First update the two paths in `configs/t2_full.example.json`, copy it to a
run-specific file, and preserve that exact config with the results. Then run:

```powershell
E:\cjj\anaconda3\envs\tensor\python.exe run_t2_completion_ablation.py `
  --config configs\t2_full_seed42.json
```

Run the full protocol for every prespecified seed. Aggregate test metrics at
the seed level (each seed shown, plus mean and sample SD); do not treat the 15
modality combinations as independent replicates.

The current local audit found 15.69 GiB total RAM (4.35 GiB available) and an
8 GiB RTX 4060 Laptop GPU. A 1,251-case float32 `64^3 x 4` full pack is already
about 4.89 GiB before preprocessing/training copies, so the full configuration
is intended for a higher-memory server or a prebuilt, carefully staged pack.
