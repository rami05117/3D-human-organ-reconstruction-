# PFE_Final — 3D Medical Image Segmentation (Liver, Heart, Spleen)

Final year graduation project (PFE) for automatic segmentation of abdominal/thoracic organs from CT scans (NIfTI format), combining classical image preprocessing with deep learning segmentation models.

## Overview

- **Preprocessing** ([pretraitment.py](pretraitment.py)): CT volume normalization, CLAHE contrast enhancement, anisotropic (CED) diffusion filtering, and Frangi vesselness filtering for vessel-aware preprocessing.
- **Segmentation** ([unet3d_predictions.py](unet3d_predictions.py)):
  - A custom **3D U-Net** (PyTorch) with CBAM attention blocks, DropBlock3D regularization, deep supervision, and multi-scale skip connections.
  - **nnU-Net v2** integration for liver segmentation.
  - Dedicated inference pipelines (MONAI transforms) for heart and spleen segmentation.
  - Post-processing: largest connected component extraction, morphological opening, Gaussian smoothing.

## Project structure

```
.
├── pretraitment.py         # Image preprocessing / filtering utilities
├── unet3d_predictions.py   # Models, training-free inference pipelines, pre/post-processing
├── requirements.txt        # Python dependencies
├── Models/                 # Trained model weights (not tracked in git, see below)
├── patients/                # Input CT volumes in NIfTI (.nii.gz) format (not tracked in git)
└── predicted*/               # Inference outputs (not tracked in git)
```

## Setup

```bash
pip install -r requirements.txt
```

> For GPU-accelerated PyTorch, follow the commented instructions in `requirements.txt` to install the CUDA build matching your system.

## Data & model weights

The `patients/`, `Models/`, and `predicted*/` folders are excluded from version control (large binary files / patient data).

The CT volumes were gathered from free, publicly available imaging benchmarks/datasets rather than a single source. Manual annotation (ground-truth segmentation masks) was performed by hand for the gathered cases to build the training/evaluation set used in this project.

To run inference:

1. Place input CT scans (`.nii.gz`) in `patients/`.
2. Place trained model weights in `Models/`:
   - `Models/heart.pth`, `Models/spleen_model.pth` — custom 3D U-Net checkpoints.
   - `Models/checkpoint_best.pth` — nnU-Net liver checkpoint.
   - `Models/Dataset100_LIVER/` — nnU-Net trained model output folder (referenced via the `nnUNet_results` environment variable).

## License

TODO: add a license if you want others to reuse this work (e.g. MIT).
