# 🚗 DrowsyGuard
### Real-Time Driver Fatigue Detection System Using Computer Vision

> **Bina Nusantara University — Computer Vision Project**  
> Nikolas Tjahjadi · Nur Hady · Luis Alexandro Soetherio · Vincent Lee · Michael Owen Muliawan · Alethea Agung Yodha Pratama

---

## Overview

DrowsyGuard is a multi-modal driver fatigue detection system that combines geometric facial feature analysis with deep learning classification. The system extracts Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and 3D head pose via PnP solving from MediaPipe facial landmarks, then fuses these with a fine-tuned MobileNetV2 CNN using a learned Logistic Regression meta-learner.

```
Frame → MediaPipe FaceLandmarker (478-point mesh)
              ↓
    EAR / MAR / PnP Head Pose  +  MobileNetV2 CNN prob
              ↓
     LR Meta-Learner (Learned Fusion Score)
              ↓
     Temporal Smoother  (deployment only)
              ↓
          🚨 Alert
```

---

## Dataset

**NTHU-DDD Multi-Class** (National Tsing Hua University Driver Drowsiness Detection)

| Split | Images | Subjects |
|-------|--------|----------|
| Train | ~46,000 | 001, 002, 005, 006 |
| Val   | ~9,849  | 001, 002, 005, 006 |
| Test  | ~9,849  | 001, 002, 005, 006 |

**Sub-classes:**

| Class | Sub-Class | Count |
|-------|-----------|-------|
| Drowsy | sleepyCombination | ~17,757 |
| Drowsy | slowBlinkWithNodding | ~9,143 |
| Drowsy | yawning | ~8,863 |
| NotDrowsy | — | ~30,942 |

**Split methodology:** Chronological per-subject × per-subclass temporal split (70/15/15) to prevent frame leakage. Yawning oversampled in train set to match `sleepyCombination` count.

---

## Methods

### 1. Face & Landmark Detection
MediaPipe `FaceLandmarker` task API (478-point mesh, IMAGE mode) — not the legacy `solutions` API.

### 2. Eye Aspect Ratio (EAR)
Soukupová & Čech (2016) formulation using 6 landmark points per eye. Average of left and right EAR.

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
```

Threshold: `EAR < 0.25` → eye closure detected.

### 3. Mouth Aspect Ratio (MAR)
Geometric ratio of vertical to horizontal mouth distances. Threshold: `MAR > 0.60` → yawn detected.

### 4. Head Pose Estimation (PnP)
True 3D Euler angles (pitch, yaw, roll) via `cv2.solvePnP` with a 6-point canonical 3D face model. Replaces 2D proxy approximations.

```python
_, rvec, tvec = cv2.solvePnP(FACE_3D_MODEL, img_pts, cam_matrix, dist_coeffs,
                              flags=cv2.SOLVEPNP_ITERATIVE)
angles, _, _, _, _, _ = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])
pitch, yaw, roll = angles
```

### 5. CNN Classifier (MobileNetV2)
Fine-tuned MobileNetV2 (ImageNet pretrained), 128×128 input, binary sigmoid output.

- **Phase A:** Head layers only, 10 epochs, LR=1e-3
- **Phase B:** Top 30 layers unfrozen, 25 epochs, LR=2e-5
- **Callbacks:** EarlyStopping (patience=5), ReduceLROnPlateau, ModelCheckpoint
- **Class weights:** Computed from train distribution to handle sub-class imbalance

**Augmentation:** random flip, brightness, contrast, saturation, random erasing (`tf.cond` — graph-safe, 40% chance)

### 6. Learned Fusion (LR Meta-Learner)
Logistic Regression trained on validation set combining:

| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| CNN prob | +6.50 | ✅ Positive |
| EAR (inverted, normalised) | +4.51 | ✅ Positive |
| MAR (normalised) | -3.52 | ⚠️ Negative (low discriminability on NTHU-DDD) |
| HeadPitch (normalised) | -0.54 | — Near-zero |

Threshold tuned on val set by maximising F1, applied once to test set.

### 7. Temporal Smoothing (Deployment Only)
Sliding-window mean over fusion scores — suppresses false positives from normal blinks.

> **Not evaluated on the static image dataset** — temporal smoothing requires real sequential video frames. Reserved for deployment.

```python
smoother = TemporalSmoother(window_size=5, alert_threshold=best_thresh)
status = smoother.step(fusion_score)   # → {'smoothed': 0.62, 'alert': True}
```

---

## Results

### 4-Model Comparison (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | FPR |
|-------|----------|-----------|--------|----------|-----|
| Rule-Based | 0.5815 | 0.5757 | 0.8986 | 0.7018 | 0.8031 |
| CNN (MobileNetV2) | 0.8569 | 0.9331 | 0.7960 | 0.8591 | **0.0692** |
| Hybrid (OR-fusion) | 0.6271 | 0.5990 | 0.9661 | 0.7395 | 0.7837 |
| **LR Fusion** | **0.8843** | **0.9278** | **0.8553** | **0.8901** | 0.0806 |

### Sub-Class Breakdown (CNN, Test Set)

| Sub-Class | N | Accuracy | F1 | Recall |
|-----------|---|----------|----|--------|
| slowBlinkWithNodding | 1,408 | 0.896 | 0.945 | 0.896 |
| sleepyCombination | 2,660 | 0.799 | 0.888 | 0.799 |
| yawning | 1,328 | 0.684 | 0.813 | 0.684 |
| notdrowsy | 4,453 | 0.931 | — | — |

**Key finding:** Eye-closure signals (EAR-correlated: `slowBlinkWithNodding`) are most reliably detected. Yawning is hardest — MAR has near-zero correlation with label on this dataset (`|r|=0.033`), suggesting mouth-based signals are less discriminative in NTHU-DDD recording conditions.

### Feature Discriminativeness

| Feature | \|Correlation with label\| |
|---------|--------------------------|
| EAR | 0.327 |
| HeadPitch | 0.183 |
| HeadRoll | 0.070 |
| MAR | 0.033 |
| HeadYaw | 0.021 |

---

## Limitations

- **4 subjects only:** The Kaggle version of NTHU-DDD contains recordings from 4 subjects. Subject-independent evaluation (train on subjects A/B, test on C/D) was not feasible at this scale — results reflect image-level classification with temporal frame splits per subject.
- **MAR not discriminative:** MAR correlation with fatigue label is near-zero on this dataset. Geometric mouth features do not reliably separate drowsy/non-drowsy in NTHU-DDD recording conditions.
- **FPR:** LR Fusion FPR (8.06%) does not meet the ≤5% target. CNN alone achieves lower FPR (6.92%) at the cost of lower recall.
- **Temporal evaluation:** PERCLOS and blink rate cannot be properly evaluated on static image datasets. Temporal smoothing is implemented but only deployable with real video input.

---

## Saved Artifacts

| File | Description |
|------|-------------|
| `drowsyguard_mobilenetv2.h5` | MobileNetV2 weights |
| `drowsyguard_best.h5` | Best checkpoint by val_loss |
| `drowsyguard_lr_fusion.pkl` | LR meta-learner |
| `drowsyguard_features.csv` | EAR, MAR, HeadPitch/Yaw/Roll per image |
| `drowsyguard_results.csv` | 4-model evaluation metrics |
| `drowsyguard_subclass_results.csv` | Per sub-class breakdown |
| `drowsyguard_inference_config.json` | Thresholds, scaler params, smoother config |

---

## Requirements

```
tensorflow >= 2.13
mediapipe >= 0.10
opencv-python
scikit-learn
scipy
pandas
matplotlib
seaborn
tqdm
```

Install:
```bash
pip install tensorflow mediapipe opencv-python scikit-learn scipy pandas matplotlib seaborn tqdm
```

---

## Usage

### 1. Feature Extraction
Run cells 1–2 in `DrowsyGuard_v5.ipynb`. Set `DATASET_ROOT` in Cell 1.3:

```python
DATASET_ROOT = r'/path/to/Multi class/train'
```

### 2. Training
Run cells 3–4. Training takes approximately 2–3 hours on an Apple M4 (MPS) or 45–90 minutes on Colab T4.

### 3. Inference (single image)

```python
import json, pickle
import cv2, numpy as np

# Load config
with open('drowsyguard_inference_config.json') as f:
    cfg = json.load(f)

with open('drowsyguard_lr_fusion.pkl', 'rb') as f:
    lr_meta = pickle.load(f)

model = tf.keras.models.load_model('drowsyguard_best.h5')
smoother = TemporalSmoother(
    window_size=cfg['temporal_smoother']['window_size'],
    alert_threshold=cfg['temporal_smoother']['alert_threshold']
)

# Run pipeline
result = predict_full_pipeline(img_path)
# → {'EAR': 0.21, 'MAR': 0.45, 'HeadPitch': 12.3,
#    'CNN_Prob': 0.87, 'FusionScore': 0.74, 'Smoothed': 0.68, 'Alert': '🚨 FATIGUE'}
```

---

## Project Structure

```
DrowsyGuard/
├── DrowsyGuard_v5.ipynb          # Main notebook
├── drowsyguard_mobilenetv2.h5    # CNN model
├── drowsyguard_best.h5           # Best checkpoint
├── drowsyguard_lr_fusion.pkl     # LR meta-learner
├── drowsyguard_inference_config.json
├── drowsyguard_features.csv
├── drowsyguard_results.csv
├── drowsyguard_subclass_results.csv
└── README.md
```

---

## References

1. Wahid et al., "Driver Drowsiness Detection: A Review," JISEBI, 2021.
2. Prasetyo et al., "Real-Time Driver Drowsiness Detection Based on Facial Landmark," TNS, 2023.
3. Fu et al., "Advancements in Intelligent Detection of Driver Fatigue," Applied Sciences, 2024.
4. Villanueva et al., "Eye Tracking Based on Facial Landmarks for Fatigue Detection," ELCVIA, 2017.
5. Shi et al., "A Review on Fatigue Driving Detection," ITM Web of Conferences, 2017.
6. Rahman et al., "Drowsiness Detection Based on Eye and Yawn Analysis," IJAII, 2013.
7. Park et al., "Efficient Head Pose Estimation and Fatigue Detection," J. Phys.: Conf. Ser., 2021.
8. Wijaya & Handoko, "Deteksi Kantuk Pengemudi Menggunakan EAR dan MAR," JPSTTD, 2022.
9. Nsimba et al., "Real-Time Drowsiness Detection Using MediaPipe," ACM MMSys, 2023.
