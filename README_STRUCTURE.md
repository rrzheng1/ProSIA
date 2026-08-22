# mt_esm_prott5_gated_structure

This directory contains the standalone script for the best custom model variant:
`mt_esm_prott5_gated_structure`.

## Architecture

```text
ESM branch:
  ESM2 site window [31, 1280]
  -> MultiScaleCNN kernels [1, 9, 11], channels 200
  -> MultiScaleCNN kernels [1, 9, 11], channels 200
  -> center lysine representation [600]

ProtT5 branch:
  ProtT5 site window [31, 1024]
  -> Linear + ReLU + Dropout
  -> BiGRU
  -> mean pooling [256]
  -> Linear projection [600]

ESM-ProtT5 gated fusion:
  prott5_gate = sigmoid(Linear([ESM_600, ProtT5_256]))
  fused = prott5_gate * ESM_600
        + (1 - prott5_gate) * projected_ProtT5_600

Structure branch:
  structure site window [31, 112]
  -> Linear + ReLU + Dropout
  -> BiGRU
  -> mean pooling [128]
  -> Linear projection [600]

Structure gated fusion:
  structure_gate = sigmoid(Linear([fused_600, Structure_128]))
  final = structure_gate * fused_600
        + (1 - structure_gate) * projected_structure_600

Prediction:
  final [600] -> task-specific binary classifier head
```

## Main Files

- `run_mt_esm_prott5_gated_structure.py`: standalone training/evaluation script.
- `run_gated_structure.sh`: GPU launch wrapper.
- `all_ptm_task_mean_metrics.csv`: five-fold independent-test mean metrics by PTM.
- `all_ptm_independent_test_metrics_5fold.csv`: independent-test metrics for each fold.

## Run

Run the existing configuration:

```bash
bash /data/ranran/my_ptm/多位点/experiment/mt_esm_prott5_custom/mt_esm_prott5_gated_structure/run_gated_structure.sh
```

Rerun and overwrite only this variant's result directory:

```bash
bash /data/ranran/my_ptm/多位点/experiment/mt_esm_prott5_custom/mt_esm_prott5_gated_structure/run_gated_structure.sh --overwrite-variant
```

Useful options are passed through to the Python script, for example:

```bash
bash /data/ranran/my_ptm/多位点/experiment/mt_esm_prott5_custom/mt_esm_prott5_gated_structure/run_gated_structure.sh \
  --epochs 30 \
  --patience 6 \
  --batch-size 16
```
