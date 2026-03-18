# 2D Pipeline Report

## Scope

This report tracks the end-to-end commands executed for the current 2D pipeline:

1. preprocessing QC
2. offline preprocessing dataset generation
3. baseline training on the preprocessed dataset

It also highlights the important outputs and comments on the observed results.

## 1. Preprocessing QC

### Command

```bash
python scripts/preprocess_qc_2d.py
```

### Important output

```text
Sampled slices: 20
Slices with warnings: 0
Processed image mean min/avg/max: 0.0018 0.238221 0.496401
Processed image std min/avg/max: 0.861094 1.017904 1.20182
Processed foreground fraction min/avg/max: 0.0 0.068629 0.152995
Saved report to: /home/carlo/Projects/UT/acdc/artifacts/preprocess_qc_report_2d.json
Saved figures to: /home/carlo/Projects/UT/acdc/artifacts/preprocess_visual_qc_2d
```

### Comment

The QC run did not report any warnings on the sampled slices. This is the first sign that the deterministic preprocessing and the online augmentation pipeline are not obviously breaking shapes, labels, or foreground content.

The foreground fraction varies from `0.0` to about `0.153`, which is expected because some sampled slices are background-only while others intersect the heart more centrally. The processed image standard deviation stays around `1.0`, which is consistent with the intended normalization behavior.

## 2. Offline Preprocessing Dataset

### Command

```bash
python scripts/preprocess_dataset_2d.py
```

### Important output

```text
Saved 1902 preprocessed slices to: /home/carlo/Projects/UT/acdc/artifacts/preprocessed_2d/samples
Saved manifest to: /home/carlo/Projects/UT/acdc/artifacts/preprocessed_2d/manifest.json
```

### Comment

This step materialized the deterministic preprocessing on disk as individual 2D `.npz` slices plus a `manifest.json`.

This is important because it removes the most expensive preprocessing work from the training loop. After this step, training reads already-preprocessed 2D slices and only applies online augmentation. In practice this should make the baseline more stable and easier to benchmark.

## 3. Baseline Training

### Command requested first

```bash
python scripts/train_baseline_2d.py
```

### Important output from the final training log

```text
Epoch 50: train_loss=0.1724 val_loss=0.2222 val_dice=0.8432 lr=0.000125 time=4.04s throughput=376.43 samples/s
Epoch 51: train_loss=0.1723 val_loss=0.2205 val_dice=0.8453 lr=0.000125 time=4.01s throughput=379.18 samples/s
...
Epoch 71: train_loss=0.1683 val_loss=0.2257 val_dice=0.8383 lr=0.000031 time=4.08s throughput=372.96 samples/s
Early stopping at epoch 71 after 20 stale validations.
Saved metrics to: artifacts/baseline_metrics_2d.json
Saved final model to: artifacts/baseline_model_2d.pt
```

### Final summary file

The resulting summary file contains:

```json
{
  "fold": 0,
  "epochs_run": 71,
  "train_slices": 1520,
  "val_slices": 382,
  "best_epoch": 51,
  "best_val_dice": 0.845254,
  "best_val_loss": 0.220488,
  "best_train_loss": 0.172278,
  "best_val_dice_per_class": [0.794308, 0.847539, 0.904794],
  "class_names": ["rv", "myo", "lv"],
  "best_epoch_duration_sec": 4.0086,
  "best_epoch_train_samples_per_sec": 379.1816,
  "best_epoch_max_gpu_memory_mb": 643.58,
  "model_path": "artifacts/baseline_model_2d.pt",
  "preprocessed_root": "/home/carlo/Projects/UT/acdc/artifacts/preprocessed_2d"
}
```

### Comment

The baseline reached a best validation Dice of about `0.8453` at epoch `51` and then plateaued. Early stopping triggered at epoch `71`, so the scheduler and stopping criterion behaved as intended.

The per-class Dice values are:

- RV: `0.7943`
- MYO: `0.8475`
- LV: `0.9048`

This is a strong baseline for the current 2D setup. The LV class is clearly the easiest of the three, while RV remains the hardest.

## Artifacts Produced

- QC JSON: `artifacts/preprocess_qc_report_2d.json`
- QC figures: `artifacts/preprocess_visual_qc_2d/`
- Preprocessed dataset manifest: `artifacts/preprocessed_2d/manifest.json`
- Preprocessed dataset samples: `artifacts/preprocessed_2d/samples/`
- Baseline training log: `artifacts/train_baseline_2d_run.log`
- Baseline summary: `artifacts/baseline_metrics_2d.json`
- Final model: `artifacts/baseline_model_2d.pt`

## Overall Conclusion

The end-to-end pipeline completed successfully:

- the QC stage did not surface obvious preprocessing failures
- the offline preprocessing stage produced the full 2D dataset
- the baseline training converged to a strong validation Dice around `0.845`

At this point the project has a solid benchmark to beat. Future preprocessing or training changes should be compared against:

- `best_val_dice = 0.845254`
- `best_epoch = 51`
- per-class Dice `[0.794308, 0.847539, 0.904794]`
