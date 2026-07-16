# Computer Vision for Extracting Construction Graphics from PDFs

Source code for the Final Year Project: **Computer Vision for Extracting Construction Graphics from PDFs** (Goh Peng Yuan, Multimedia University, Session 2025/2026).

This project implements an end-to-end pipeline for (1) converting architectural floor plan PDFs to raster images, (2) instance segmentation of structural elements (Wall, Door, Window / Openings, Closet, Bathroom, Living Room, Bedroom, Hall, Balcony) using YOLOv11s-seg, and (3) OCR-based text extraction of room labels and dimensions using EasyOCR with custom pre/postprocessing.

## Source Code

Latest source code (GitHub): **https://github.com/Pyuan57/Computer-Vision-For-Extracting-Construction-Graphs-From-PDFs**

---

## 1. Requirements

### 1.1 Software & Tools

| Tool | Version | Download Link |
|---|---|---|
| Python | 3.10 or later | https://www.python.org/downloads/ |
| Google Colab (recommended for training) | — | https://colab.research.google.com/ |
| Roboflow (dataset annotation & augmentation) | — | https://roboflow.com/ |
| Label Studio (OCR ground-truth annotation) | — | https://labelstud.io/ |

Model training was performed on Google Colab using a GPU runtime (Nvidia Tesla T4). For local inference and OCR evaluation, a machine with at least 16 GB RAM and a dedicated GPU (e.g., NVIDIA GTX 1650 Ti or better) is recommended.

### 1.2 Required Python Libraries

Install all dependencies with:

```bash
pip install ultralytics pymupdf opencv-python easyocr jiwer python-Levenshtein pandas numpy matplotlib pillow
```

| Library | Purpose | Reference / Docs |
|---|---|---|
| ultralytics | YOLOv11 instance segmentation (train, inference, evaluation) | https://docs.ultralytics.com/ |
| pymupdf (fitz) | PDF-to-PNG rasterisation (600 DPI) | https://pymupdf.readthedocs.io/ |
| opencv-python | Image preprocessing (grayscale, thresholding, morphology) | https://opencv.org/ |
| easyocr | OCR text detection (CRAFT) and recognition (CRNN + CTC) | https://github.com/JaidedAI/EasyOCR |
| jiwer | OCR evaluation metric (Word Error Rate) | https://github.com/jitsi/jiwer |
| python-Levenshtein | OCR evaluation metric (Character Error Rate) & fuzzy room-label matching | https://pypi.org/project/python-Levenshtein/ |
| numpy | Numerical operations | https://numpy.org/ |
| pandas | Data handling/analysis | https://pandas.pydata.org/ |
| matplotlib | Visualisation of results and metrics | https://matplotlib.org/ |
| pillow | Image input/manipulation | https://python-pillow.org/ |

> Note: exact library version numbers used were not pinned/recorded during development. Install the latest versions compatible with Python 3.10+, or run `pip freeze > requirements.txt` after your first successful setup and commit it for future exact-version reproducibility.

---

## 2. Dataset

Both datasets used in this project, together with their annotations, are hosted on Roboflow. You can either **fork** the project into your own Roboflow account or **sign in and download** the specified version directly.

### 2.1 Self-Collected Dataset (Wall, Door, Window)

50 residential floor plan PDFs manually sourced from [HousePlans.net](https://www.houseplans.net/), rasterised to 600 DPI PNG and annotated in Roboflow.

Roboflow project: https://app.roboflow.com/pygoh-yazcv/self-collected-floor-plan-houseplans-net/14

- **Version 14** — original dataset with annotations (no preprocessing/augmentation).
- **Version 13** — dataset used for the reported training/testing results, including preprocessing and augmentation (resizing/auto-orientation, horizontal/vertical flips, 90°/180°/270° rotations, grayscale conversion).

### 2.2 R3D (Rent3D) Benchmark Dataset (8 room-level classes)

- Original dataset: Rent3D project page, University of Toronto — https://www.cs.toronto.edu/~fidler/projects/rent3D.html
- Dataset and annotations as compiled by Zeng, Z., Li, X., Yu, Y. K., & Fu, C.-W. (2019), *Deep Floor Plan Recognition Using a Multi-Task Network With Room-Boundary-Guided Attention*, ICCV 2019 — GitHub: https://github.com/zlzeng/DeepFloorplan

For ease of access, this dataset was also transferred to Roboflow together with its annotations:

Roboflow project: https://app.roboflow.com/pygoh-yazcv/rent3d-floor-plan-yvyfm/10

- **Version 10** — original dataset with annotations (no preprocessing).
- **Version 8** — dataset used for the reported training/testing results, including preprocessing described above.

### 2.3 Downloading a Dataset Version

1. Sign in / create a free account at https://roboflow.com/.
2. Open the relevant project link above and select the version noted (e.g., Version 13 or Version 8).
3. Click **Download Dataset**, choose the export format expected by Ultralytics (**YOLOv11 / YOLO Segmentation**), and download as a ZIP, or use the Roboflow-generated Python snippet to download directly into your Colab/local environment via the `roboflow` pip package:

```bash
pip install roboflow
```

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("pygoh-yazcv").project("self-collected-floor-plan-houseplans-net")
dataset = project.version(13).download("yolov11")
```

(Replace the workspace/project/version with the R3D project details for the second dataset.)

---

## 3. Execution Instructions

### 3.1 PDF-to-Raster Conversion

Convert source floor plan PDFs to 600 DPI PNG images using PyMuPDF before uploading to Roboflow for annotation (only needed if starting from raw PDFs rather than the Roboflow exports above).

### 3.2 Train the Instance Segmentation Model (YOLOv11s-seg)

```bash
pip install ultralytics
```

```bash
# Self-annotated dataset
yolo segment train model=yolo11s-seg.pt data=<self_annotated_data.yaml> \
  imgsz=1024 batch=8 epochs=300 patience=40 optimizer=AdamW \
  lr0=0.001 lrf=0.01 warmup_epochs=10 momentum=0.9 weight_decay=0.0001 cls=2.0

# R3D dataset (baseline and SMOTE-rebalanced variants)
yolo segment train model=yolo11s-seg.pt data=<r3d_data.yaml> \
  imgsz=1024 batch=8 epochs=200 patience=40 optimizer=AdamW \
  lr0=0.001 lrf=0.01 warmup_epochs=10 momentum=0.9 weight_decay=0.0001 cls=2.0
```

`data.yaml` paths come from the Roboflow dataset export (Section 2.3). For the SMOTE-rebalanced R3D run, apply SMOTE-guided oversampling to the training split only (see Section 4 below) before pointing `data=` to the rebalanced training list.

### 3.3 Evaluate the Segmentation Model

```bash
yolo segment val model=<best.pt> data=<data.yaml>
```

### 3.4 Run OCR Extraction

1. Apply preprocessing to each image: grayscale conversion (OpenCV) → Sauvola adaptive thresholding (25×25 window, k=0.25, R=128) → morphological closing (3×3 cross kernel, 1 iteration).
2. Run EasyOCR (`Reader.readtext()`) in a rotation-aware dual pass — once for horizontal text, once for vertical (`rotation_info=[90, 270]`) — on both the original and preprocessed images.
3. Apply postprocessing: rule-based dimension string correction and fuzzy room-label matching (Levenshtein distance ≤ 2) against a predefined room dictionary.
4. Compute CER/WER using `jiwer` and `python-Levenshtein` against ground-truth text exported from Label Studio.

---

## 4. Model Architecture & Hyperparameters

| Parameter | Value |
|---|---|
| Model | YOLOv11s-seg (pretrained: `yolo11s-seg.pt`) |
| Optimizer | AdamW |
| Initial learning rate (lr0) | 0.001 |
| Final learning rate fraction (lrf) | 0.01 (cosine-decayed) |
| Warm-up epochs | 10 |
| Momentum | 0.9 |
| Weight decay | 0.0001 |
| Classification loss weight (cls) | 2.0 |
| Early stopping patience | 40 epochs |
| Input image size | 1024 × 1024 |
| Batch size | 8 |
| Max epochs (self-annotated dataset) | 300 |
| Max epochs (R3D dataset) | 200 |

OCR model: EasyOCR (CRAFT detector + CRNN/BiLSTM recognizer + CTC decoder), pretrained, used without fine-tuning.

---

## 5. Evaluation

- **Segmentation:** Class Accuracy, Mean Accuracy, Overall Accuracy, Precision, Recall, F1-score, IoU, mIoU, mAP@0.50, mAP@0.50:0.95. R3D results additionally benchmarked against Knechtel et al. (2024) and De Nardin et al. (2025).
- **OCR:** Character Error Rate (CER) and Word Error Rate (WER), computed at three stages (no preprocessing / preprocessed / postprocessed) against Label Studio ground-truth annotations.

---
