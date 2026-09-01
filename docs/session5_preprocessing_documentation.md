# Session 5 Preprocessing Documentation

## Simple Presentation Script

For preprocessing, our goal was to prepare public breast ultrasound datasets for AMFU-Net training.

The original AMFU-Net paper used fused ultrasound images. Their system produced two images of the same lesion: a reflection image and a transmission image. These two images were fused before segmentation. However, in our earlier sessions, we established that public datasets such as BUSI, BUS-UCLM, and BUS-BRA only provide normal single-modality ultrasound images. They do not provide paired reflection and transmission images, so the original fusion stage cannot be reproduced directly.

Because of that, we made a controlled pivot. Instead of image fusion, we use a single-image enhancement pipeline:

```text
Raw ultrasound image -> SRAD despeckling -> CLAHE contrast enhancement -> AMFU-Net input
```

The aim is not to claim that SRAD and CLAHE are the same as multimodal fusion. The aim is to replace the unavailable fusion stage with a defensible ultrasound preprocessing stage that improves a single image before segmentation.

The preprocessing notebook first loads three public breast ultrasound datasets: BUSI, BUS-UCLM, and BUS-BRA. Since each dataset has a different file structure, the notebook standardizes them into one common table. Each accepted sample has an image path, a mask path, a dataset name, an image ID, a patient or group ID, and a class label.

Next, the notebook checks that every ultrasound image is correctly paired with its lesion mask. This is important because segmentation is pixel-level learning. If the image and mask do not match, the model would learn the wrong lesion location.

After pairing, the notebook performs an audit. It checks that image and mask dimensions match and that the mask actually contains a lesion. It also hashes images to detect exact duplicates. Known patients, derived patient groups, and duplicate images are kept inside one split to reduce data leakage.

The dataset is then split into training, validation, and internal test sets. The training set is used to update the model, the validation set is used to monitor learning, and the internal test set is kept unseen until final evaluation.

For the image enhancement itself, the notebook applies robust intensity scaling first. This clips extreme dark and bright pixels and normalizes the image range. Then SRAD is applied to reduce speckle noise while preserving lesion boundaries. Finally, CLAHE is applied to improve local contrast and make faint lesion boundaries clearer.

The final export saves four versions of each accepted case: the raw image, the SRAD image, the enhanced SRAD plus CLAHE image, and the unchanged mask. Training uses the enhanced image and its matching mask.

In summary, our preprocessing pipeline converts multiple public datasets into a clean, traceable, leakage-aware, training-ready dataset for AMFU-Net.

## Block-by-Block Code Explanation

## 1. Setup and Dependency Installation

```python
!pip install -q "huggingface_hub>=0.30,<2" "kagglehub>=0.3,<1" ...
```

This cell installs the libraries needed inside the Kaggle environment.

The important packages are:

- `huggingface_hub`: used to download and upload datasets.
- `kagglehub`: used to access Kaggle datasets such as BUSI.
- `opencv-python-headless`: used for image processing, especially CLAHE and image writing.
- `scikit-image`, `scikit-learn`: used for processing support and grouped splitting.
- `pandas`, `numpy`: used for metadata tables and numerical image processing.
- `openpyxl`: useful if spreadsheet metadata is encountered.
- `tqdm`: gives progress bars during long dataset operations.

Decision made: install exact version ranges so the notebook is more reproducible and less likely to break because of a future package update.

## 2. Imports and Configuration

```python
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
```

This block imports general Python tools.

- `hashlib` is used to compute image hashes for duplicate detection.
- `io` is used to read image bytes from dataset records.
- `json` is used to save the final preprocessing configuration.
- `random` and `numpy.random` support reproducibility.
- `re` is used for cleaning IDs and matching filenames.
- `zipfile` is used to extract downloaded archives.
- `Path` gives cleaner file path handling.
- `dataclass` is used to define the configuration object.

```python
import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from huggingface_hub import HfApi, create_repo, login, snapshot_download
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient
```

This imports the project-specific libraries.

- `cv2` handles CLAHE and PNG writing.
- `PIL.Image` reads images reliably from files or bytes.
- `requests` downloads BUS-BRA from Zenodo.
- `StratifiedGroupKFold` creates splits that preserve class balance while keeping patient groups separate.
- `UserSecretsClient` retrieves the Hugging Face token securely from Kaggle secrets.

## 3. The Config Object

```python
@dataclass(frozen=True)
class Config:
    SEED: int = 42
    WORK_ROOT: str = "/kaggle/working/G17_multi_dataset"
    OUTPUT_ROOT: str = "/kaggle/working/G17_Multi_Dataset_Output"
```

The configuration object stores the values that control the whole preprocessing run.

Important values:

- `SEED = 42`: makes splits and random choices repeatable.
- `WORK_ROOT`: temporary working area for downloads and canonical files.
- `OUTPUT_ROOT`: final output folder for manifests and exported images.

```python
UCLM_REPO: str = "MedOtter/BUS-UCLM"
BUSI_KAGGLE_HANDLE: str = "aryashah2k/breast-ultrasound-images-dataset"
BUS_BRA_ZENODO_RECORD: int = 8231412
```

These define the three dataset sources.

Decision made: use official or traceable public dataset locations instead of manually copied folders.

```python
EXCLUDE_UCLM_MARKS: bool = True
EXCLUDE_UCLM_DOPPLER: bool = True
EXCLUDE_UCLM_COMBINED: bool = True
LESION_ONLY: bool = True
```

These options clean BUS-UCLM.

- Images with marks are excluded because visible annotations could leak information to the model.
- Doppler and combined images are excluded to keep the input type consistent.
- `LESION_ONLY` keeps benign and malignant lesion cases only, matching the segmentation focus.

```python
SRAD_ITERATIONS: int = 30
SRAD_STEP: float = 0.20
SRAD_DECAY: float = 0.125
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_GRID: tuple[int, int] = (8, 8)
```

These control image enhancement.

- SRAD iterations control how long diffusion runs.
- SRAD step controls update strength.
- SRAD decay gradually reduces the speckle scale estimate.
- CLAHE clip limit prevents over-amplification.
- CLAHE grid controls local tile size.

Decision made: use SRAD first and CLAHE second, based on the earlier pivot session. Despeckling first avoids enhancing speckle noise.

```python
OUTER_FOLDS: int = 5
INNER_FOLDS: int = 5
RUN_FULL_EXPORT: bool = True
HF_REPO_ID: str = "aee4/G17-preprocessed-dataset"
HF_PRIVATE: bool = True
UPLOAD_TO_HF: bool = True
```

These control splitting and export.

- `OUTER_FOLDS` creates the internal test split.
- `INNER_FOLDS` creates train and validation splits from the remaining data.
- `HF_REPO_ID` is the Hugging Face dataset destination.

## 4. Reproducibility and Folder Setup

```python
CFG = Config()
random.seed(CFG.SEED)
np.random.seed(CFG.SEED)
cv2.setRNGSeed(CFG.SEED)
```

This initializes the configuration and seeds the random number generators.

Decision made: use the same seed across Python, NumPy, and OpenCV so repeated runs produce the same splits and previews.

```python
WORK_ROOT = Path(CFG.WORK_ROOT)
CANONICAL_ROOT = WORK_ROOT / "canonical"
OUTPUT_ROOT = Path(CFG.OUTPUT_ROOT)
WORK_ROOT.mkdir(parents=True, exist_ok=True)
CANONICAL_ROOT.mkdir(parents=True, exist_ok=True)
```

This creates the working folders.

- `WORK_ROOT` stores downloaded and intermediate files.
- `CANONICAL_ROOT` stores standardized image-mask pairs.
- `OUTPUT_ROOT` stores final exported results.

## 5. Hugging Face Login

```python
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
login(token=HF_TOKEN, add_to_git_credential=False)
hf_api = HfApi(token=HF_TOKEN)
```

This logs into Hugging Face using a Kaggle secret.

Decision made: keep the token outside the notebook code. This is safer because the notebook can be shared without exposing credentials.

## 6. Shared Helper Functions

### Image Extensions

```python
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
```

This defines the image file types the loaders should accept.

### Reading Binary Payloads

```python
def payload_bytes(value) -> bytes:
    if isinstance(value, dict) and value.get("bytes") is not None:
        return value["bytes"]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError(...)
```

Some datasets store images as file paths, while others store them as binary payloads. This function converts supported binary formats into raw bytes.

Decision made: support both disk files and dataset byte objects so the same reader can work across different sources.

### Reading Grayscale Images

```python
def read_gray(source) -> np.ndarray:
    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)
    with Image.open(io.BytesIO(payload_bytes(source))) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)
```

This function reads any ultrasound image as grayscale.

Why grayscale? Public breast ultrasound datasets are single-channel in practice, and the training model expects one input channel. Converting everything to grayscale standardizes the data.

### Reading Binary Masks

```python
def read_binary_mask(source) -> np.ndarray:
    ...
    if arr.ndim == 2:
        binary = arr > 0
    elif arr.ndim == 3:
        binary = np.any(arr[..., :3] > 0, axis=-1)
    ...
    return binary.astype(np.uint8) * 255
```

This function converts masks into binary images.

- Pixel value `255` means lesion.
- Pixel value `0` means background.

It supports grayscale masks and RGB masks.

Decision made: keep masks binary because segmentation training needs a clear lesion/background target.

### Writing PNGs

```python
def write_png(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(...)
```

This saves images and masks as PNG files.

Decision made: PNG is lossless, so masks and processed images are not damaged by compression artifacts.

### File Hashing

```python
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    ...
    return digest.hexdigest()
```

This computes a SHA-256 hash for each image.

Purpose: detect exact duplicate images even if filenames differ.

### Safe IDs

```python
def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
```

This cleans dataset IDs so they can safely be used as filenames.

### Saving Canonical Pairs

```python
def save_canonical_pair(dataset, image_id, image, mask):
    image = np.asarray(image, dtype=np.uint8)
    mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    image_path = CANONICAL_ROOT / dataset / "images" / f"{safe_id(image_id)}.png"
    mask_path = CANONICAL_ROOT / dataset / "masks" / f"{safe_id(image_id)}.png"
    write_png(image_path, image)
    write_png(mask_path, mask)
    return image_path, mask_path
```

This writes each accepted image-mask pair into a consistent folder structure.

Decision made: convert every dataset into a canonical local format before auditing, splitting, and preprocessing. This reduces dataset-specific complexity later.

## 7. Loading BUS-UCLM

```python
def load_bus_uclm() -> pd.DataFrame:
    root = WORK_ROOT / "downloads" / "BUS_UCLM"
    snapshot_download(repo_id=CFG.UCLM_REPO, repo_type="dataset", local_dir=root)
```

This downloads BUS-UCLM from Hugging Face.

```python
parquet_files = sorted(root.rglob("*.parquet"))
raw = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
```

BUS-UCLM is stored in Parquet files, so the notebook reads and combines them into one table.

```python
keep = np.ones(len(raw), dtype=bool)
if CFG.EXCLUDE_UCLM_MARKS:
    keep &= ~raw["has_marks"].astype(bool).to_numpy()
...
if CFG.LESION_ONLY:
    keep &= raw["class_label"].isin(["benign", "malignant"]).to_numpy()
```

This applies the BUS-UCLM cleaning rules.

Reasoning:

- Marked images are excluded to avoid annotation leakage.
- Doppler and combined images are excluded to keep modality consistent.
- Normal cases are excluded because this project is lesion segmentation, and normal images do not contain lesion masks.

```python
image = read_gray(row["image"])
mask = read_binary_mask(row["mask"])
image_id = f"UCLM_{row['image_id']}"
image_path, mask_path = save_canonical_pair("BUS_UCLM", image_id, image, mask)
```

Each accepted BUS-UCLM row is converted into a canonical grayscale image and binary mask.

```python
"patient_id": f"UCLM_{row['patient_id']}",
"group_quality": "true_patient_id",
```

BUS-UCLM provides true patient IDs, so these can be used for grouped splitting.

## 8. Loading BUSI

```python
def load_busi() -> pd.DataFrame:
    root = Path(kagglehub.dataset_download(CFG.BUSI_KAGGLE_HANDLE))
```

This downloads BUSI from Kaggle.

```python
all_files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
originals = [p for p in all_files if "_mask" not in p.stem.lower()]
```

BUSI stores original images and masks as separate image files. This separates original images from mask files.

```python
label = next((part.lower() for part in image_path.parts
              if part.lower() in {"benign", "malignant", "normal"}), None)
if label not in {"benign", "malignant"}:
    continue
```

The label is inferred from the folder path. Normal cases are skipped because the project focuses on lesion segmentation.

```python
mask_paths = sorted(image_path.parent.glob(f"{image_path.stem}_mask*{image_path.suffix}"))
```

This finds the mask or masks belonging to the image.

```python
combined = np.zeros(image.shape, dtype=np.uint8)
for mask_path in mask_paths:
    mask = read_binary_mask(mask_path)
    if mask.shape != image.shape:
        valid = False
        break
    combined = np.maximum(combined, mask)
```

Some BUSI images can have more than one mask file. The notebook combines them into one lesion mask using `np.maximum`.

Decision made: if multiple lesion regions exist, preserve all positive mask pixels instead of choosing only one.

```python
"patient_id": image_id,
"group_quality": "image_level_only_patient_ids_unavailable",
```

BUSI does not provide reliable patient IDs in this loader, so each image is treated as its own group.

Caveat to mention in presentation: BUSI grouping is weaker than BUS-UCLM and BUS-BRA because patient-level IDs are unavailable.

## 9. Downloading BUS-BRA from Zenodo

```python
def download_zenodo_record(record_id: int, destination: Path) -> Path:
    metadata = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
```

This contacts the Zenodo API and retrieves file metadata for BUS-BRA.

```python
with requests.get(url, stream=True, timeout=120) as response:
    ...
    for chunk in response.iter_content(1024 * 1024):
        stream.write(chunk)
```

Files are downloaded in chunks to avoid loading large archives into memory.

```python
for archive in destination.glob("*.zip"):
    extract_dir = destination / archive.stem
    if not extract_dir.exists():
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(extract_dir)
```

Any ZIP archives are extracted after download.

## 10. BUS-BRA Pairing Helpers

### Detecting Mask Files

```python
def mask_like(path: Path) -> bool:
    if path.parent.name.lower() in {"gt", "gts", "mask", "masks", ...}:
        return True
    text = f"{path.parent.name}_{path.stem}".lower()
    return any(token in text for token in ["mask", "groundtruth", ...])
```

BUS-BRA file structures may vary, so this function identifies likely mask files using folder names and filename tokens.

### Normalized Pair Keys

```python
def normalized_pair_key(path: Path) -> str:
    key = path.stem.lower()
    key = re.sub(r"^(bus|mask)[_-]", "", key)
    ...
    return re.sub(r"[^a-z0-9]+", "", key)
```

This creates a simplified key from each filename so an image and its mask can be matched even if their names are not identical.

Decision made: normalize filenames before matching to handle real-world naming inconsistencies.

### Inferring Labels and Patient IDs

```python
def infer_label(path: Path) -> str | None:
    text = " ".join(part.lower() for part in path.parts)
    if "malignant" in text:
        return "malignant"
    if "benign" in text:
        return "benign"
    return None
```

This infers the class label from folder names if metadata is incomplete.

```python
def infer_patient_id(stem: str, dataset: str) -> str:
    match = re.match(r"([A-Za-z]*[-_]?\d+)", stem)
    value = match.group(1) if match else stem
    return f"{dataset}_{safe_id(value)}"
```

This derives a patient or case group from the filename when metadata does not provide one.

Decision made: use metadata patient IDs where available, otherwise use conservative filename-derived grouping.

## 11. BUS-BRA Metadata Loading

```python
def load_bus_bra_metadata(root: Path) -> dict:
    lookup = {}
    for csv_path in sorted(root.rglob("*.csv"), ...):
        table = pd.read_csv(csv_path)
```

This searches for CSV metadata files inside BUS-BRA.

```python
columns = normalized_columns(table)
filename_column = first_matching_column(columns, ["id", "imagefilename", ...])
class_column = first_matching_column(columns, ["pathology", "classlabel", ...])
patient_column = first_matching_column(columns, ["patientid", "caseid", ...])
```

Because metadata column names may differ, the notebook normalizes column names and searches for likely matches.

Decision made: make metadata parsing flexible so small column naming differences do not break the loader.

## 12. BUS-BRA Loading

```python
files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
masks = [p for p in files if mask_like(p)]
images = [p for p in files if not mask_like(p)]
```

This separates BUS-BRA image files from mask files.

```python
mask_map = {}
for path in masks:
    mask_map.setdefault(normalized_pair_key(path), []).append(path)
```

This builds a lookup table from normalized keys to possible mask files.

```python
candidates = mask_map.get(key, [])
metadata = metadata_lookup.get(key, {})
label = normalize_class(metadata.get("class_value")) or infer_label(image_path)
if len(candidates) != 1 or label is None:
    rejected.append(...)
    continue
```

The notebook only accepts a BUS-BRA image if:

- exactly one matching mask is found;
- the class label can be determined.

Decision made: reject ambiguous cases instead of guessing. This protects dataset quality.

```python
pd.DataFrame(rejected).to_csv(OUTPUT_ROOT / "bus_bra_unpaired_or_unlabelled.csv", index=False)
```

Rejected BUS-BRA cases are saved for review.

## 13. Combining the Development Dataset

```python
development_index = pd.concat([uclm_index, bus_bra_index, busi_index], ignore_index=True)
```

This combines the three cleaned dataset indexes into one development pool.

At this point, every row should have the same structure:

- dataset source;
- image ID;
- patient or group ID;
- class label;
- image path;
- mask path.

## 14. Geometry and Mask Audit

```python
def audit_geometry(frame: pd.DataFrame, report_name: str):
    rows = []
    for index, row in tqdm(frame.iterrows(), ...):
        image = read_gray(row["image_path"])
        mask = read_binary_mask(row["mask_path"])
```

This reads every image-mask pair again for validation.

```python
rows.append({
    "image_height": image.shape[0],
    "image_width": image.shape[1],
    "mask_height": mask.shape[0],
    "mask_width": mask.shape[1],
    "geometry_match": image.shape == mask.shape,
    "mask_has_lesion": bool(np.any(mask)),
})
```

The audit checks:

- image height and width;
- mask height and width;
- whether geometry matches;
- whether the mask contains any lesion pixel.

```python
valid_rows = audit.loc[audit.geometry_match & audit.mask_has_lesion, "row_index"].to_numpy()
return frame.iloc[valid_rows].reset_index(drop=True), audit
```

Only valid rows are kept.

Decision made: remove shape mismatches and empty masks before splitting so invalid samples do not leak into train, validation, or test.

## 15. Duplicate Detection

```python
development_index["image_hash"] = [
    file_sha256(Path(path)) for path in tqdm(development_index.image_path, ...)
]
```

This computes a content hash for every image.

```python
development_index["split_group"] = development_index["patient_id"].astype(str)
duplicate_hashes = development_index.loc[
    development_index.duplicated("image_hash", keep=False), "image_hash"
].unique()
```

The default split group is the patient ID. But if two images have the same hash, they are treated as duplicates.

```python
for image_hash in duplicate_hashes:
    rows = development_index.image_hash.eq(image_hash)
    development_index.loc[rows, "split_group"] = f"DUP_{image_hash[:16]}"
```

Duplicate images are forced into the same split group.

Decision made: prevent the same image from appearing in both training and testing under different filenames.

## 16. Grouped Train/Validation/Test Splitting

```python
def grouped_split(frame: pd.DataFrame, seed: int):
    y = frame.class_label.astype(str).to_numpy()
    groups = frame.split_group.astype(str).to_numpy()
    x = np.zeros((len(frame), 1), dtype=np.uint8)
```

This prepares labels and groups for splitting.

- `y` is the class label, benign or malignant.
- `groups` contains patient or duplicate group IDs.
- `x` is a dummy feature array because the splitter only needs labels and groups.

```python
outer = StratifiedGroupKFold(n_splits=CFG.OUTER_FOLDS, shuffle=True, random_state=seed)
train_val_idx, test_idx = next(outer.split(x, y, groups))
```

The outer split separates the internal test set.

```python
inner = StratifiedGroupKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=seed + 100)
train_idx, val_idx = next(inner.split(x_tv, y_tv, groups_tv))
```

The inner split separates training and validation from the remaining data.

```python
assert group_sets[0].isdisjoint(group_sets[1])
assert group_sets[0].isdisjoint(group_sets[2])
assert group_sets[1].isdisjoint(group_sets[2])
```

These assertions verify that no group appears in more than one split.

Decision made: use stratified grouped splitting to balance class labels while reducing patient and duplicate leakage.

## 17. Per-Dataset Splitting and Combined Manifest

```python
source_splits = {}
for offset, (dataset, part) in enumerate(development_index.groupby("dataset")):
    source_splits[dataset] = grouped_split(part.reset_index(drop=True), CFG.SEED + offset)
```

Each dataset is split separately.

Reasoning: BUSI, BUS-UCLM, and BUS-BRA may have different sizes and class distributions. Splitting per source helps preserve representation from each dataset.

```python
combined_splits = {}
for split_name in ["train", "val", "internal_test"]:
    combined_splits[split_name] = pd.concat(
        [parts[split_name] for parts in source_splits.values()], ignore_index=True
    ).assign(split=split_name)
```

The per-dataset splits are then combined into final train, validation, and internal test sets.

```python
split_manifest = pd.concat(combined_splits.values(), ignore_index=True)
split_manifest.to_csv(OUTPUT_ROOT / "combined_split_manifest.csv", index=False)
```

The split manifest is saved for traceability.

## 18. Robust Intensity Scaling

```python
def robust_unit_scale(image_u8: np.ndarray) -> np.ndarray:
    image = np.asarray(image_u8, dtype=np.float32)
    lo, hi = np.percentile(image, (0.5, 99.5))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)
```

This normalizes ultrasound brightness values.

Instead of using the absolute minimum and maximum, it uses the 0.5th and 99.5th percentiles.

Why this matters:

- Ultrasound images can contain very dark or very bright outlier pixels.
- Outliers can distort normal min-max scaling.
- Percentile scaling makes the image range more stable.

Decision made: robust scaling before SRAD gives the diffusion algorithm a consistent numeric range.

## 19. SRAD Despeckling

```python
def srad(image_u8, iterations=30, step=0.20, decay=0.125, eps=1e-8):
    if not (0 < step <= 0.25):
        raise ValueError("SRAD step must satisfy 0 < step <= 0.25")
```

SRAD stands for Speckle Reducing Anisotropic Diffusion.

The step-size check prevents unstable updates.

```python
image = robust_unit_scale(image_u8)
if iterations == 0 or image.max() == image.min():
    return np.round(image * 255).astype(np.uint8)
```

The image is first robustly scaled. If there is nothing to process, the function returns safely.

```python
central = j[h // 10:max(h // 10 + 1, 9 * h // 10),
            w // 10:max(w // 10 + 1, 9 * w // 10)]
roi = central[central > np.percentile(central, 5)]
```

The algorithm estimates speckle behavior from a central region of the image.

Reasoning: the center of an ultrasound image is more likely to contain useful tissue information than borders or labels.

```python
mean, variance = float(roi.mean()), float(roi.var())
q0_sq = max(variance / (mean * mean + eps), eps) * np.exp(-decay * t)
```

This estimates the speckle coefficient of variation. The decay term reduces the influence over iterations.

```python
p = np.pad(j, 1, mode="edge")
north, south = p[:-2, 1:-1] - j, p[2:, 1:-1] - j
west, east = p[1:-1, :-2] - j, p[1:-1, 2:] - j
```

These lines compute local differences between each pixel and its neighbors.

```python
grad_sq = (north**2 + south**2 + west**2 + east**2) / (j**2 + eps)
lap = (north + south + west + east) / (j + eps)
```

These approximate gradient and Laplacian information, which SRAD uses to distinguish speckle regions from structural edges.

```python
c = np.clip(1 / (1 + denominator), 0, 1)
```

This is the diffusion coefficient.

- Close to 1: allow more smoothing.
- Close to 0: reduce smoothing near likely edges.

```python
j = np.clip(j + 0.25 * step * divergence, eps, 1)
```

This updates the image gradually.

Presentation explanation:

SRAD smooths uniform speckled regions while trying not to blur lesion boundaries. This is why it is better suited to ultrasound than ordinary blur.

## 20. CLAHE Enhancement

```python
def apply_clahe(image_u8, clip_limit=2.0, grid=(8, 8)):
    if image_u8.ndim != 2 or image_u8.dtype != np.uint8:
        raise ValueError("CLAHE expects a 2-D uint8 image")
    return cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=tuple(map(int, grid))
    ).apply(image_u8)
```

CLAHE means Contrast Limited Adaptive Histogram Equalization.

It improves local contrast by dividing the image into small tiles and equalizing each tile. The clip limit prevents extreme contrast amplification.

Decision made: apply CLAHE after SRAD because applying it before SRAD would enhance speckle noise.

## 21. Full Preprocessing Function

```python
def preprocess_image(image_u8):
    despeckled = srad(
        image_u8,
        iterations=CFG.SRAD_ITERATIONS,
        step=CFG.SRAD_STEP,
        decay=CFG.SRAD_DECAY,
    )
    enhanced = apply_clahe(
        despeckled,
        clip_limit=CFG.CLAHE_CLIP_LIMIT,
        grid=CFG.CLAHE_GRID,
    )
    assert image_u8.shape == despeckled.shape == enhanced.shape
    return despeckled, enhanced
```

This function applies the actual preprocessing sequence:

```text
raw image -> SRAD -> CLAHE
```

The assertion confirms that preprocessing does not change image dimensions.

Decision made: use one function for every dataset and every split so the preprocessing rule stays consistent.

## 22. Deterministic Safety Test

```python
rng = np.random.default_rng(CFG.SEED)
test_image = rng.integers(0, 256, (97, 131), dtype=np.uint8)
d1, e1 = preprocess_image(test_image)
d2, e2 = preprocess_image(test_image)
assert np.array_equal(d1, d2) and np.array_equal(e1, e2)
```

This creates a random test image and preprocesses it twice.

The outputs must be identical.

Purpose:

- confirms the preprocessing is deterministic;
- catches unexpected randomness;
- verifies that unusual image sizes still work.

## 23. Preview Function

```python
def show_source_examples(frame, seed=CFG.SEED):
    samples = []
    for dataset, part in frame.groupby("dataset"):
        samples.append(part.sample(1, random_state=seed))
```

This selects one example from each dataset.

```python
for column, (image, title) in enumerate([
    (raw, "Raw"), (despeckled, "SRAD"),
    (enhanced, "SRAD + CLAHE"), (mask, "Mask -- unchanged")
]):
    axes[row_number, column].imshow(image, cmap="gray", vmin=0, vmax=255)
```

It displays:

- raw image;
- SRAD output;
- SRAD plus CLAHE output;
- unchanged mask.

Decision made: include visual inspection as a safety check. Medical image preprocessing should not be trusted from code alone; the outputs must be visually checked.

## 24. Exporting Each Partition

```python
def export_partition(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    records = []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc=f"Exporting {split_name}"):
        raw = read_gray(row.image_path)
        mask = read_binary_mask(row.mask_path)
```

This exports one split at a time: train, validation, or internal test.

```python
if raw.shape != mask.shape:
    raise ValueError(f"Geometry changed after audit: {row.image_id}")
```

This is a second safety check. Even after the audit, the exporter refuses to continue if geometry is wrong.

```python
despeckled, enhanced = preprocess_image(raw)
```

The raw image is processed into SRAD and enhanced versions.

```python
base = OUTPUT_ROOT / "dataset" / split_name / row.dataset
paths = {
    "raw_path": base / "raw" / f"{row.image_id}.png",
    "srad_path": base / "srad" / f"{row.image_id}.png",
    "enhanced_path": base / "enhanced" / f"{row.image_id}.png",
    "mask_path_exported": base / "masks" / f"{row.image_id}.png",
}
```

This defines the output folder structure.

Each split and dataset gets four folders:

- `raw`;
- `srad`;
- `enhanced`;
- `masks`.

```python
write_png(paths["raw_path"], raw)
write_png(paths["srad_path"], despeckled)
write_png(paths["enhanced_path"], enhanced)
write_png(paths["mask_path_exported"], mask)
```

This writes all four versions.

Important point: the mask is saved unchanged. We enhance the input image, not the ground-truth label.

```python
records.append({
    **row.to_dict(),
    **{key: str(value.relative_to(OUTPUT_ROOT)) for key, value in paths.items()},
    "srad_iterations": CFG.SRAD_ITERATIONS,
    "srad_step": CFG.SRAD_STEP,
    "srad_decay": CFG.SRAD_DECAY,
    "clahe_clip_limit": CFG.CLAHE_CLIP_LIMIT,
    "clahe_grid": "x".join(map(str, CFG.CLAHE_GRID)),
})
```

The export manifest records:

- original metadata;
- paths to exported files;
- SRAD settings;
- CLAHE settings.

Decision made: save parameters with each record so the dataset is traceable and reproducible.

## 25. Full Export and Upload

```python
if CFG.RUN_FULL_EXPORT:
    exported = []
    for split_name, part in combined_splits.items():
        exported.append(export_partition(part, split_name))
```

This runs the export for all splits.

```python
export_manifest = pd.concat(exported, ignore_index=True)
export_manifest.to_csv(OUTPUT_ROOT / "preprocessing_manifest.csv", index=False)
```

The full preprocessing manifest is saved.

```python
(OUTPUT_ROOT / "preprocessing_config.json").write_text(
    json.dumps(asdict(CFG), indent=2), encoding="utf-8"
)
```

The full configuration is also saved as JSON.

Decision made: save both the manifest and configuration so the run can be audited later.

```python
create_repo(
    repo_id=CFG.HF_REPO_ID,
    repo_type="dataset",
    private=CFG.HF_PRIVATE,
    exist_ok=True,
    token=HF_TOKEN,
)
hf_api.upload_folder(
    folder_path=str(OUTPUT_ROOT),
    repo_id=CFG.HF_REPO_ID,
    repo_type="dataset",
    commit_message="Upload G17 preprocessed dataset",
)
```

This creates or reuses the Hugging Face dataset repository and uploads the preprocessing output.

Purpose:

- makes the dataset available to the training notebook;
- avoids manually moving thousands of image files;
- preserves the preprocessing output in a versioned location.

## Final Summary for Presentation

The preprocessing pipeline performs three major jobs.

First, it builds a reliable dataset. It downloads BUSI, BUS-UCLM, and BUS-BRA, pairs images with masks, removes invalid samples, detects duplicates, and creates grouped train-validation-test splits.

Second, it enhances each ultrasound image. It uses robust scaling to stabilize brightness, SRAD to reduce speckle noise, and CLAHE to improve local contrast.

Third, it exports a traceable training-ready dataset. It saves raw images, SRAD images, enhanced images, unchanged masks, split manifests, and the preprocessing configuration, then uploads the result to Hugging Face.

This preprocessing stage is the bridge between the original AMFU-Net paper and our practical implementation. Since we cannot reproduce the paper's private two-image fusion stage, we use a documented single-image enhancement pipeline that prepares public ultrasound images for the same segmentation objective.
