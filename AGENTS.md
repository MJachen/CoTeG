# AGENTS.md

## Scope

This file guides Codex or other coding agents working in this repository.

The project is a research codebase for multimodal brain MRI missing-modality classification. The main method follows the paper `MICCAI2026_3066(1).pdf`: graph regularized non-negative tensor completion, tensor/GNN dual-branch learning, and interaction-aware MoE fusion.

## Working Rules

- Preserve business logic unless the user explicitly asks for changes.
- Do not rewrite the project into a new framework or package layout without permission.
- Keep edits small and localized.
- Treat existing `.pth` files and generated experiment outputs as artifacts; do not delete or overwrite them casually.
- Assume a dirty worktree may contain user work. Never revert unrelated changes.
- Prefer adding documentation, config, or small compatibility fixes over broad refactors.

## Main Pipeline

The primary workflow is:

1. `predataset.py`
   - Build GLI/MET packs from raw BraTS-style NIfTI files.
   - Output to `./brats_gli_met_packs`.

2. `GRNNTCtrain.py`
   - Train graph regularized Tucker completion on augmented train data.
   - Save train Tucker factors and completed train tensor.

3. `GRNNTCeval.py`
   - Complete validation/test packs using trained factors.
   - Optimize eval-set `U_N` only.

4. `threebranchesv3.py`
   - Pretrain tensor branch and graph branch.
   - Save `tensor_branch.pth` and `gnn_branch.pth`.

5. `fusiontraintest_brats.py`
   - Freeze branches.
   - Train MoE fusion head.
   - Evaluate test performance for `K=1..4`.

See `PROJECT_OVERVIEW.md` for more context.

## Important Files

- `predataset.py`: data pack generation and missing modality simulation.
- `GRNNTCtrain.py`: train-set Tucker completion.
- `GRNNTCeval.py`: val/test Tucker completion.
- `twotrlv2.py`: tensor branch with 3D CNN and TRL.
- `twognn.py`: GraphSAGE branch and earlier fusion helpers.
- `threebranchesv3.py`: branch pretraining.
- `fusiontraintest_brats.py`: final fusion training/evaluation.
- `metrics.py`: binary metrics, threshold selection, focal loss.
- `zerotoolfunc.py`: utility functions and KNN graph construction.

## Shape Conventions

Use these conventions carefully:

- Data packs: `(N, M, H, W, D)`
- GRNNTC internal tensor: `(H, W, D, M, N)`
- GRNNTC flattened tensor: `(S, M, N)`, `S=H*W*D`
- CNN input: `(N, 4, D, H, W)`
- Modality mask: `(N, M, 1, 1, 1)`

Most bugs in this repo are likely to come from silent `H,W,D` vs `D,H,W` permutation mistakes.

## Environment Notes

No dependency manifest currently exists. Imports indicate these external packages:

- `torch`
- `torch_geometric`
- `monai`
- `nibabel`
- `numpy`
- `pandas`
- `scikit-learn`
- `tensorly`

Many scripts contain hard-coded paths from the original experiment environment. Before running end-to-end experiments, check path constants near the top of the relevant script.

## Testing And Verification

There are no automated tests at the moment.

For code changes, prefer the lightest meaningful verification:

- Syntax check edited Python files with `python -m py_compile <file>`.
- For data-flow changes, inspect tensor shapes with a tiny synthetic tensor if possible.
- Avoid launching long GPU training jobs unless the user asks for an experiment run.

## Documentation Style

When updating documentation:

- Keep the pipeline order explicit.
- Record tensor shapes near any description of data flow.
- Mention whether a script is mainline, baseline, or exploratory.
- Keep paper terminology aligned with code terminology:
  - `GRNNTC` means graph regularized non-negative Tucker tensor completion.
  - `U_N` is the subject-mode factor matrix used by the graph branch.
  - `X_hat` is the completed multimodal tensor.
  - `K` is the number of available modalities.

## Cautions

- `zerotoolfunc.knn_graph_pytorch` currently sends indices to `cuda` explicitly. This can break CPU-only runs.
- Some comments are mojibake-encoded Chinese. Avoid mass re-encoding unless requested.
- `fusiontraintest_brats.py` references external checkpoint paths under `/home/cjc/tensor deco/...`; these may need adjustment for another machine.
- The project mixes two-class GLI/MET experiments with older three-class GLI/MEN/MET code. Confirm which task the user wants before changing labels or class counts.
