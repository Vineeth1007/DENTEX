# 🦷 Dental Pathology Classification using Deep–Hybrid Feature Fusion

### 🔬 A Research-Grade Framework for Tooth-Level Disease Classification from Panoramic Dental Radiographs

---

## 📘 Overview

This repository presents a complete AI framework for automated analysis of panoramic dental radiographs.  
The pipeline performs **tooth detection**, **ROI extraction**, and **multi-class disease classification** using both **deep learning** and **classical feature descriptors**.

The proposed **hybrid fusion model** combines pretrained deep embeddings (ResNet50 / EfficientNet-B3) with handcrafted descriptors (HOG, LBP, GLCM, Color Histogram), followed by PCA-based dimensionality reduction and classical machine learning classifiers (SVM, Random Forest).

---

## 🧠 Research Objectives

- Build an end-to-end dental disease classification pipeline.  
- Compare **classical**, **deep**, and **hybrid** approaches.  
- Improve diagnostic reliability using **feature-level fusion**.  
- Evaluate models using **accuracy, precision, recall, and F1-score**.

---


---

## 🧩 Dataset

- **Input:** Panoramic dental radiographs (X-rays)  
- **Classes:**
  - Caries  
  - Deep Caries *(optionally merged with Caries)*  
  - Impacted  
  - Periapical Lesion  
  - Healthy  
- **Split:** 80% Train / 10% Validation / 10% Test  
- **Detection:** YOLOv8 single-class model (`class_id = 0`)  
- **Classification:** Crop-based tooth images generated from YOLO detections

---

## 🧰 Implementation Details

| Component | Configuration |
|:--|:--|
| **Backbones** | ResNet50, EfficientNet-B3 (ImageNet pre-trained) |
| **Classifiers** | SVM (RBF / Linear), Random Forest |
| **Feature Descriptors** | HOG, LBP, GLCM, Color Histogram |
| **Optimizer** | AdamW (LR: 1e-3 → 5e-5 → 1e-5, weight decay: 1e-4) |
| **Loss** | Cross-Entropy with Label Smoothing (α = 0.1) |
| **Batch Size** | 32 |
| **Epochs** | 80 (deep models) |
| **Gradient Clipping** | 1.0 |
| **Early Stopping** | Patience = 15 epochs |
| **LR Scheduler** | ReduceLROnPlateau (factor = 0.5, patience = 5) |
| **PCA Dimension** | 500 |
| **Random Forest** | 200 trees, max_depth = 20 |
| **Augmentation Target Count** | 180 samples per class |

---

## 🧠 Hyperparameter Tuning

- Manual and grid-based tuning were used.  
- Validation F1-score guided the parameter search.  
- Hyperparameters tuned: learning rate, weight decay, dropout, label smoothing, PCA dimensions, and SVM/RF parameters.  
- Final configuration:
  - **Learning rate:** 1e-3 → 5e-5 → 1e-5  
  - **Weight decay:** 1e-4  
  - **Dropout:** 0.4–0.5  
  - **Label smoothing:** 0.1  
  - **PCA dimensions:** 500  
  - **Random Forest:** 200 trees, depth 20  
  - **Batch size:** 32  
  - **Early stopping patience:** 15  
  - **LR reduce factor:** 0.5  

---

## 🧩 Results (Ranked by Accuracy)

| Model                              | Type       | **Accuracy** | Precision | Recall | F1-Score |
|:-----------------------------------|:-----------|--------------:|-----------:|--------:|----------:|
| **Hybrid ResNet50 + SVM**          | Hybrid     | **0.9030** | 0.9241 | 0.8957 | 0.9066 |
| **Hybrid EfficientNet + SVM**      | Hybrid     | 0.8993 | 0.9279 | 0.8931 | 0.9068 |
| Hybrid EfficientNet + SVM (Linear) | Hybrid     | 0.8993 | 0.9138 | 0.8901 | 0.9011 |
| HOG + SVM                          | Classical  | 0.8993 | 0.9222 | 0.8818 | 0.8970 |
| HOG + RF                           | Classical  | 0.8172 | 0.8967 | 0.7802 | 0.8207 |
| GLCM + RF                          | Classical  | 0.7873 | 0.7996 | 0.7819 | 0.7881 |
| LBP + RF                           | Classical  | 0.7799 | 0.8298 | 0.7452 | 0.7728 |
| GLCM + SVM                         | Classical  | 0.7948 | 0.8518 | 0.6861 | 0.7140 |
| LBP + SVM                          | Classical  | 0.7500 | 0.8271 | 0.6252 | 0.6305 |

---

### ✅ Observations
- **Hybrid ResNet50 + SVM** achieved the highest accuracy (0.9030).  
- **Hybrid EfficientNet + SVM** closely followed (0.8993).  
- Hybrid models significantly outperform pure classical approaches by over **10%** in both accuracy and F1-score.  
- Among classical methods, **HOG + SVM** remains the most effective.  

---


---

## 🚀 Getting Started

```bash
# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/dental-pathology-classification.git
cd dental-pathology-classification

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ (Optional) Run detection on panoramic X-rays
python detection/yolo_detect.py --input data/xrays/ --output data/crops/

# 4️⃣ Train classification model
python classification/deep_models/train_resnet.py
# or
python classification/hybrid_fusion/train_hybrid.py

# 5️⃣ Evaluate model
python evaluation/evaluate_metrics.py

