# November 2026 experiments: from Exp-25 to the presented model

These notebooks take the project from **Exp-25** (U-Net + DAG on SRAD + CLAHE input, 82.76% Dice) to the model we presented: **ENS_R5_aug_x3**, a 3-seed U-Net++ ensemble with an ImageNet ResNet-34 encoder on raw scans, **89.36% Dice** on the same 266 held-out test scans. They also cover the app model trained from the same recipe (**ENS_FINAL_x3**, 89.06%).

All notebooks run on Kaggle (GPU T4 x2, internet on). Each one writes its own copy of the training code (`g17core.py`) in its first code cell, so it runs exactly as it did at the time. The title cell of R3 onwards still says "U-Net + DAG, 200 epochs"; that header is left over from the template. The real configuration is the `PLAN` cell.

## What each round did

| Notebook | Question | Outcome (internal test Dice, flip TTA) |
| --- | --- | --- |
| `00_loss_variations_U-Net_DAG` | Focal Tversky and boundary-aware losses on U-Net + DAG | First loss screen |
| `01_round_R1` | Which loss for U-Net + DAG? (8 runs, 200 epochs) | Focal Tversky α0.5/β0.5 best, 85.90 |
| `02_round_R2` | Is the gain real? (3 seeds per loss) | FT beats BCE+Dice by ~0.6; 9-model ensemble 87.09 |
| `03_round_R3` | Remove stray islands; try a pretrained encoder | **U-Net++ with ImageNet ResNet-34: 88.90** single model (+3.0) |
| `04_round_R4` | Consolidate the new backbone (3 seeds) | FT 0.5/0.5: 88.63 ± 0.16; ensemble 89.27 |
| `05_round_R5` | Close the gap on a new hospital (strong scanner-appearance augmentation) | **89.36 internal, 79.66 on BrEaST** → paper model |
| `06_round_R6` | How stable is the recipe? (5-fold patient-level CV) | 86.38 ± 0.82 CV Dice |
| `07_round_R7` | More resolution or capacity? (384 px, EfficientNet-B4, ResNeXt-50) | No gain; best 88.93 |
| `08_round_FINAL_app_model` | Production weights: R5 recipe on all labelled data incl. BrEaST, 3 seeds | 89.06; deployed in the app |
| `09_export_app_model_to_huggingface` | Export both ensembles, calibrate the Check flag, publish | `aee4/G17-BUS-Segmentation` (`r5_paper/`, `final/`) |

The **paper model** is the average of `R5_aug_s42`, `R5_aug_s1` and `R5_aug_s2` from `05_round_R5`. Each member averages the scan and its mirror image (flip TTA).

## Analysis notebooks (`analysis/`)

| Notebook | What it shows |
| --- | --- |
| `R1_boundary_audit`, `R2_boundary_audit` | Contours on the hardest test scans (boundary audit) |
| `R3_gallery` | Ground truth vs prediction on 30 test scans |
| `BrEaST_external_validation` | Zero-shot test on BrEaST (Poland, 4 scanners, never seen in training) |
| `BrEaST_intensity_diagnosis` | Whether the BrEaST drop comes from brightness/contrast, plus test-time intensity fixes |
| `BrEaST_failure_breakdown` | BrEaST Dice by scanner, pathology, BI-RADS, shape and margin |

## Final recipe (paper model)

- **Input:** raw grayscale scan resized to 256 × 256. No SRAD or CLAHE.
- **Model:** U-Net++ (`segmentation_models_pytorch`), ImageNet ResNet-34 encoder, 1 input channel.
- **Loss:** BCE + Focal Tversky (α 0.5, β 0.5, γ 1.33).
- **Augmentation:** flip, rotation ±15°, scale 0.9–1.1, translation ±10%. Strong scanner-appearance augmentation: brightness/contrast, gamma, speckle, noise, blur and fan-shaped crops.
- **Training:** 100 epochs, Adam lr 1e-3, cosine schedule, batch 4 × 3 accumulation. The checkpoint is chosen on validation Dice.
- **Inference:** flip TTA per model, then the mean of 3 seeds; mask at 0.5.

## How to rerun

1. Import a notebook into Kaggle (File → Import Notebook), set GPU T4 x2 and turn internet on.
2. Attach the inputs it names:
   - R5–FINAL: the **Breast-Lesions-USG** dataset (`rabeasaleh2/breast-lesions-usg`).
   - BrEaST analyses: the R4 output, plus the BrEaST dataset or the `BrEaST_external_validation` output.
   - Export: the FINAL and R5 outputs, plus a write-scoped `HF_TOKEN` Kaggle secret.
3. Save & Run All.

The splits come from `Ahorts/G17-preprocessed-dataset`. The test set is pinned to the same 266 scans as Exp-16 to Exp-25, so every number above compares directly with Exp-25.
