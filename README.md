# AMFU-Net Breast Ultrasound Lesion Segmentation Research

This repository tracks our final year project work on:

**AMFU-Net: A Multi-Scale Attention Network for Lesion Segmentation in Fused Breast Ultrasound Imaging**

## Project Summary

The original AMFU-Net paper uses fused breast ultrasound images. It first combines reflection and transmission ultrasound images from a special USCT system, then segments the lesion using AMFU-Net.

Our team cannot reproduce the original fusion stage exactly because the paper's dataset is private and public breast ultrasound datasets do not provide paired reflection and transmission images. Because of this, our project adapts AMFU-Net for public single-modality ultrasound datasets.

Our adapted pipeline is:

```text
Public breast ultrasound image -> SRAD despeckling -> CLAHE enhancement -> AMFU-Net segmentation
```

The segmentation architecture remains based on AMFU-Net, while the unavailable fusion stage is replaced with a documented single-image enhancement pipeline.

## Repository Structure

```text
.
├── papers/
│   ├── AMFU-Net original research paper
│   └── supporting preprocessing/segmentation paper
├── sessions/
│   ├── session-01-understanding-paper/
│   ├── session-02-preprocessing-pivot/
│   ├── session-04-enhancement-pipeline/
│   └── session-05-current-progress/
├── preprocessing/
│   └── amfu-net-processing-2.ipynb
├── training/
│   └── training-pipe.ipynb
├── docs/
│   └── session5_preprocessing_documentation.md
├── tracking/
│   └── note.txt
└── archive/
    └── imported-notebooks/
```

## Main Files

- `papers/AMFU_Net_A_multi_scale_attention_network_for_lesion_segmentation.pdf`
  - The main research paper this project is based on.

- `papers/Development of a Deep-Learning-Based Method for Breast Ultrasound Image Segmentation.pdf`
  - Supporting paper for breast ultrasound preprocessing and segmentation.

- `preprocessing/amfu-net-processing-2.ipynb`
  - Current tracked preprocessing notebook.
  - Builds the public dataset pipeline using BUSI, BUS-UCLM, and BUS-BRA.
  - Performs image-mask pairing, auditing, grouped splitting, SRAD, CLAHE, and export.

- `training/training-pipe.ipynb`
  - Current tracked training notebook.
  - Defines and trains the adapted AMFU-Net model.
  - Tracks Dice, Jaccard/IoU, HD95, and ASD.

- `docs/session5_preprocessing_documentation.md`
  - Detailed explanation of the preprocessing notebook for presentation.

- `tracking/note.txt`
  - Simple session-by-session summary of project progress.

- `archive/imported-notebooks/`
  - Older or imported notebook copies preserved for reference.

## Current Status

Completed so far:

- Original AMFU-Net paper reviewed.
- Dataset limitation identified and documented.
- Decision made to replace unavailable fusion with SRAD + CLAHE preprocessing.
- Preprocessing notebook implemented.
- Training notebook implemented.
- Session 5 presentation materials prepared for preprocessing, training, and demo.

## Current Adaptation

We are not claiming to fully reproduce the original AMFU-Net study because the original fused USCT dataset is unavailable.

Instead, we are adapting the AMFU-Net segmentation idea to public breast ultrasound datasets using a traceable preprocessing stage:

```text
Robust scaling -> SRAD despeckling -> CLAHE enhancement
```

The enhanced images are then used as input to AMFU-Net, with the original lesion masks kept unchanged as ground truth.

## Team Use

Use this repo as the shared research record:

- Add new papers to `papers/`.
- Add meeting or presentation materials under `sessions/`.
- Keep preprocessing implementation under `preprocessing/`.
- Keep training implementation under `training/`.
- Add written explanations under `docs/`.
- Add progress notes and summaries under `tracking/`.
