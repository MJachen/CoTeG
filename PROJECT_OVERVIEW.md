# Project Overview

## What This Project Is

This repository implements the method described in `MICCAI2026_3066(1).pdf`: **Tensor Completion and Dual-Branch Learning on Multimodal Brain MRI with Missing Modality**.

The project studies robust brain tumor classification when some MRI modalities are missing. It focuses on multimodal BraTS-style MRI volumes with four modalities:

- `t1n`
- `t1c`
- `t2w`
- `t2f` / FLAIR-like modality naming

The main experimental target in the current code is **GLI vs MET binary classification** on BraTS 2025-style data. Some older or baseline code also supports **GLI / MEN / MET three-class classification**.

The core idea is:

1. Simulate missing modalities by keeping `K=1..4` available modalities per sample.
2. Complete the missing multimodal MRI tensor using **graph regularized non-negative Tucker tensor completion**.
3. Train two complementary branches:
   - a tensor branch over completed 3D volumes;
   - a graph branch over subject-mode Tucker factors.
4. Fuse both branches with an interaction-aware mixture-of-experts head.
5. Evaluate robustness across missing-modality levels.

## Paper-To-Code Mapping

The paper proposes a tensor-graph dual-branch framework:

- **GRNNTC**: graph regularized non-negative tensor completion.
- **Tensor branch**: 3D CNN plus Tensor Regression Layer.
- **Graph branch**: GraphSAGE over subject embeddings.
- **MoE fusion**: adaptive fusion of tensor and graph embeddings.

The corresponding implementation is:

- `predataset.py`
  - Builds the experiment packs from raw NIfTI files.
  - Uses GLI/MET labels.
  - Crops/pads volumes to `64x64x64`.
  - Creates train/val/test splits with a 6:2:2 stratified ratio.
  - Generates train augmentation for `K=1,2,3,4`.
  - Generates validation and test packs for each fixed missing level.

- `GRNNTCtrain.py`
  - Implements the training-time graph regularized Tucker completion.
  - Converts `(N,M,H,W,D)` packs into internal `(S,M,N)` tensors where `S=H*W*D`.
  - Learns `G`, `U_S`, `U_M`, and `U_N`.
  - Uses observed-entry reconstruction loss, spatial smoothness, modality graph regularization, subject graph regularization, and non-negativity constraints.
  - Saves `tucker_factors_train.pt` and `X_hat_train_5d.pt`.

- `GRNNTCeval.py`
  - Applies trained Tucker factors to validation/test packs.
  - Freezes learned `G`, `U_S`, and `U_M`.
  - Optimizes only the eval-set `U_N`.
  - Saves completed packs and eval-specific Tucker factors.

- `twotrlv2.py`
  - Implements the tensor branch.
  - Uses a 3-layer 3D CNN feature extractor.
  - Uses a Tensor Regression Layer instead of a plain fully connected classifier.
  - Can return both logits and tensor embeddings.

- `twognn.py`
  - Implements the graph branch using `torch_geometric.nn.SAGEConv`.
  - Also contains a simpler gated fusion module used by earlier experiments.

- `threebranchesv3.py`
  - Pretrains the tensor branch and GNN branch.
  - Saves `tensor_branch.pth` and `gnn_branch.pth`.

- `fusiontraintest_brats.py`
  - Implements the final frozen-branch fusion experiment.
  - Loads completed volumes and Tucker factors.
  - Extracts tensor and graph embeddings.
  - Trains `MoEFusion`, an interaction-aware mixture-of-experts head.
  - Selects thresholds on validation data.
  - Evaluates test performance for `K=1..4`.
  - Exports metrics, sample-level results, and ROC data.

## Main Data Flow

The intended pipeline is:

1. Prepare raw data under a BraTS-style directory:

   ```text
   /home/cjc/brats2025/
     GLI/
       case_*/
         *t1n*.nii*
         *t1c*.nii*
         *t2w*.nii*
         *t2f*.nii*
     MET/
       case_*/
         ...
   ```

2. Run preprocessing:

   ```bash
   python predataset.py
   ```

   Expected outputs under `./brats_gli_met_packs/` include:

   - `split.json`
   - `X_full_4mod_gli_met.pt`
   - `train_aug_k1to4.pt`
   - `val_keep1.pt` through `val_keep4.pt`
   - `test_keep1.pt` through `test_keep4.pt`

3. Train GRNNTC on the augmented train pack:

   ```bash
   python GRNNTCtrain.py
   ```

   Expected outputs:

   - `brats_gli_met_packs/tucker_factors_train.pt`
   - `brats_gli_met_packs/X_hat_train_5d.pt`

4. Complete validation and test packs:

   ```bash
   python GRNNTCeval.py
   ```

   Expected outputs:

   - `val_keep{k}_completed.pt`
   - `val_keep{k}_completed_tucker_factors.pt`
   - `test_keep{k}_completed.pt`
   - `test_keep{k}_completed_tucker_factors.pt`

5. Pretrain tensor and graph branches:

   ```bash
   python threebranchesv3.py
   ```

   Expected outputs:

   - `tensor_branch.pth`
   - `gnn_branch.pth`

6. Train and evaluate final fusion:

   ```bash
   python fusiontraintest_brats.py
   ```

   Expected outputs under `./fusion_out/` include:

   - `best_fusion_head.pth`
   - `metrics_k1k4.json`
   - `test_sample_level_results.csv`
   - `roc_export_8models.txt`

## Important Tensor Shapes

Common project shapes:

- Raw/completed pack volume: `(N, M, H, W, D)`
- Internal GRNNTC representation: `(H, W, D, M, N)`
- Flattened GRNNTC representation: `(S, M, N)` where `S = H * W * D`
- CNN/TRL input: `(N, 4, D, H, W)`
- Modality mask `Omega`: `(N, M, 1, 1, 1)`
- Training completed tensor saved by `GRNNTCtrain.py`: `(H, W, D, M, N)`
- Completed eval pack `X_hat`: `(N, M, H, W, D)`

Be careful with `H,W,D` versus `D,H,W` permutations. Most downstream CNN code expects `(N,4,D,H,W)`.

## Baselines And Experimental Scripts

Several baseline or exploratory scripts are present:

- `train.py`
  - Simple 3D CNN baseline for GLI/MEN/MET three-class classification.

- `resnet2025.py`
  - MONAI ResNet baseline with LRTC/HaLRTC-style tensor completion utilities.

- `casemean.py`
  - Case mean imputation baseline.

- `CP.py`
  - CP decomposition completion baseline.

- `KNN.py`
  - KNN imputation baseline.

- `maskchannel.py`
  - Missing-mask-as-channel baseline.

- `moddrop.py`
  - Modality dropout baseline.

These scripts are useful for comparison but are not the main tensor-graph pipeline.

## Dependencies

No dependency manifest is currently present. Imports indicate the project depends on:

- `torch`
- `torch_geometric`
- `monai`
- `nibabel`
- `numpy`
- `pandas`
- `scikit-learn`
- `tensorly`

The repository also contains two model checkpoint files:

- `tensor_branch.pth`
- `gnn_branch.pth`

## Current Repository State

This is an experiment-oriented research codebase rather than a packaged application.

Observed characteristics:

- No `README.md` was present before this overview.
- No `requirements.txt`, `pyproject.toml`, or environment file was present.
- No test directory or automated tests were found.
- Most scripts are flat in the repository root.
- Several file paths are hard-coded to the original experiment machine, such as `/home/cjc/brats2025` and `/home/cjc/tensor deco/...`.
- Some Chinese comments appear mojibake-encoded, likely due to an encoding mismatch.

## High-Level Interpretation

The code is trying to reproduce the paper's claim that missing-modality MRI classification benefits from preserving both:

- **high-order spatial/modality/subject tensor structure**, via graph regularized Tucker completion;
- **population-level relational information**, via GNN learning on subject-mode Tucker factors.

The final classifier combines completed-image tensor features and subject-graph features through adaptive MoE fusion, aiming to remain stable when only one, two, three, or all four modalities are available.
