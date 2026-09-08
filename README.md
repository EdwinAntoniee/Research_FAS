# Robust Face Anti-Spoofing & Biometric Verification System

> **Two-Stage Presentation Attack Detection (PAD) and Identity Verification Pipeline Under Extreme Illumination Variations**  
> *Research Study & Prototype — School of Computer Science, Bina Nusantara University*  
> **Authors**: Edwin Antonie, David Christian Golden Mahaviro, Hasan, Irene Anindaputri Iswanto, Arya Krisna Putra.

---

## 📌 Overview

Facial recognition systems deployed on smartphones, physical access points, and digital banking are vulnerable to **presentation attacks**—including high-resolution photo printouts, tablet/screen video replays, and adversarial spoofing. These vulnerabilities worsen significantly under unpredictable real-world lighting and severe shadows.

This repository implements a **two-stage real-time biometric verification pipeline**:
1. **Stage 1 — Presentation Attack Detection (Liveness Gate)**: Evaluates whether the presented face is bona fide (live human) or an adversarial spoof attack using a lightweight **MobileNetV2** backbone trained with illumination simulation augmentation.
2. **Stage 2 — Identity Verification (ArcFace)**: If and only if the subject passes the liveness test ($P(\text{live}) \ge 0.50$), high-dimensional 512-d ArcFace facial embeddings are extracted and matched against an enrolled identity gallery using cosine distance.

---

## 🏗️ Architecture

```
[ Video / Webcam Stream ]
           │
           ▼
[ InsightFace Face Detector ]
           │ (Face Bounding Box)
           ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 1: Face Anti-Spoofing (FAS) Liveness Gate        │
│ • Backbone: MobileNetV2 (224x224 RGB input)            │
│ • Output: Liveness Probability P(live) ∈ [0.0, 1.0]    │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
     [ P(live) < 0.50 ]          [ P(live) >= 0.50 ]
             │                           │
             ▼                           ▼
  ⛔ [SPOOF ATTACK]            ✅ [BONA FIDE / LIVE]
  • Bounding Box: RED         • Bounding Box: GREEN / AMBER
  • Identity: ACCESS DENIED   • Passed to Stage 2
  • Presentation Attack Blocked          │
                                         ▼
                 ┌────────────────────────────────────────────────┐
                 │ STAGE 2: ArcFace Identity Verification         │
                 │ • Backbone: ArcFace 512-d Embedding Extractor  │
                 │ • Metric: Cosine Distance vs Enrolled Gallery  │
                 │   - dist < 0.65 ➔ Authenticated (Green)       │
                 │   - dist ≥ 0.65 ➔ Unregistered Guest (Amber)   │
                 └────────────────────────────────────────────────┘
```

---

## 🔬 Benchmark Results (OULU-NPU Protocol 4)

The methodology benchmarked three edge-efficient lightweight backbones on **Protocol 4 of the OULU-NPU mobile benchmark** (evaluating generalization across unseen environmental lighting and diverse camera sensors):

| Architecture | Total Parameters | Trainable Parameters | APCER (%) | BPCER (%) | ACER (%) | Evaluation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ShuffleNetV2** | 1.25 M | 21,082 | 48.0% | 52.0% | 50.0% | Underfitted |
| **MobileNetV2** | 2.22 M | 21,082 | 42.0% | 48.0% | 45.0% | Overfitted |
| **EfficientNet-B0** | **4.01 M** | **21,082** | **3.0%** | **27.0%** | **15.0%** | **Champion Baseline** |

- **APCER (Attack Presentation Classification Error Rate)**: 3.0% (blocks 97% of presentation attacks).
- **ACER (Average Classification Error Rate)**: 15.0% on unseen illumination testing.
- **Explainability**: Spatial activation heatmaps confirm that the network detects material anomalies and screen moiré artifacts rather than memorizing individual facial features.

The complete academic paper is available in [`docs/Robust_Face_Anti_Spoofing_Paper.pdf`](docs/Robust_Face_Anti_Spoofing_Paper.pdf).

---

## 🚀 Getting Started

### 1. Requirements & Dependencies
- Python 3.10
- Dependencies: `tensorflow`, `insightface`, `opencv-python`, `scipy`, `numpy`

```bash
pip install tensorflow insightface opencv-python scipy numpy
```

### 2. Enrolling Faces into Gallery
To register identities, place face images in `dataset_wajah/<person_name>/` and run:

```bash
python code_recognition.py --dataset dataset_wajah --output model_recognition_facenet.pkl
```

### 3. Training the FAS Model
To retrain the MobileNetV2 liveness detection classifier with illumination simulation:

```bash
python code_fas.py --dataset dataset_lighting --epochs 10 --batch_size 32
```

### 4. Running Real-Time Verification
Launch the live pipeline with webcam feed:

```bash
python test_recognition.py --source 0
```

Or evaluate against a video file:

```bash
python test_recognition.py --source "path/to/video.mp4"
```

---

## 📂 Repository Structure

```
.
├── code_fas.py                     # MobileNetV2 FAS training pipeline
├── code_recognition.py             # Face enrollment & ArcFace embedding generator
├── test_recognition.py             # Integrated Two-Stage real-time verification pipeline
├── model_fas_mobilenet.h5          # Trained MobileNetV2 anti-spoofing checkpoint (11.5 MB)
├── model_recognition_facenet.pkl   # Enrolled ArcFace identity embeddings
├── dataset_wajah/                  # Identity enrollment galleries
├── docs/                           # Research paper PDF
└── README.md
```