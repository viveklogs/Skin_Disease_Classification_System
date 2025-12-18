#  Skin Disease Classification System  
### CNN–Transformer Ensemble for Malignant vs Benign Lesion Detection

---

##  Project Overview

The **Skin Disease Classification System** is an end-to-end deep learning–based medical AI application designed to assist in the **early screening of skin cancer**. The system classifies dermoscopic skin lesion images into **Malignant** or **Benign** categories.

Unlike traditional single-model approaches, this project implements a **heterogeneous ensemble architecture** that combines:
- **Convolutional Neural Networks (CNNs)** for fine-grained local feature extraction  
- **Vision Transformers (ViT)** for global contextual understanding  

The system is deployed through a **Flask-based web application** and integrates **Explainable AI (XAI)** using **Grad-CAM**, enabling visual interpretation of model predictions.

---

##  Key Features

- Binary classification: Benign / Malignant  
- CNN–Transformer ensemble architecture  
- Transfer learning with fine-tuned pretrained models  
- Explainable AI using Grad-CAM heatmaps  
- REST API–based backend  
- Web-based interface for real-time predictions  

---

##  Tools & Technologies Used

### Programming Languages
- Python
- JavaScript
- HTML
- CSS

### Deep Learning & Machine Learning
- PyTorch
- Torchvision
- timm
- NumPy
- Scikit-learn

### Model Architectures
- EfficientNet-B4
- Vision Transformer (ViT-Base)
- CNN–Transformer Ensemble

### Computer Vision & XAI
- OpenCV
- Pillow (PIL)
- Grad-CAM

### Backend & API
- Flask
- Flask-CORS
- REST API

### Frontend
- HTML5, CSS, JavaScript
- React.js

### Development & Experimentation
- Kaggle
- Git
- GitHub
- VS Code

### Visualization
- Matplotlib

---

## System Architecture

The pipeline consists of two parallel deep learning branches whose predictions are combined using weighted averaging.

###  CNN Branch — EfficientNet-B4
- Captures fine-grained texture, border irregularities, and color variations  
- Uses `tf_efficientnet_b4_ns` (Noisy Student pretrained weights)

###  Transformer Branch — Vision Transformer (ViT-Base)
- Models global context using self-attention  
- Captures symmetry and long-range lesion dependencies

###  Ensemble Strategy
- Validation-based weighted averaging of CNN and ViT predictions

---

## Data Preprocessing & Augmentation

- Image resizing to `224 × 224`
- ImageNet-based normalization  
- Random horizontal & vertical flips  
- Rotation up to ±20°  
- Brightness, contrast, and saturation jittering  

---

## Training Strategy & Optimization

1. Transfer learning with frozen backbones  
2. Stabilized fine-tuning with low learning rates  
3. Ensemble weight optimization using validation performance  

- Loss Function: Cross Entropy Loss  
- Optimizer: AdamW  
- Scheduler: StepLR  

---

## Explainable AI (Grad-CAM)

Grad-CAM visualizations highlight the image regions that contribute most to the malignant or benign prediction, improving transparency and trust in medical decision support.

---

## Performance Metrics

| Metric | EfficientNet-B4 | ViT-Base | **Ensemble (Final)** |
|------|----------------|---------|---------------------|
| Accuracy | 91.2% | 90.5% | **92.88%** |
| Precision | 0.90 | 0.89 | **0.93** |
| Recall | 0.91 | 0.89 | **0.92** |
| F1-Score | 0.90 | 0.89 | **0.92** |

---

## Deployment & Web Interface

- Flask-based REST API  
- Real-time inference and prediction  
- JSON responses with Base64-encoded Grad-CAM heatmaps  

---

## Repository Structure

```text
├── app.py
├── test.py
├── models/
│   ├── EfficientNet_Final.pth
│   └── ViT_Final.pth
├── static/
├── templates/
├── requirements.txt
└── README.md

## Use Cases

- Early skin cancer screening
- AI-assisted dermatological diagnosis
- Medical image analysis research
- Educational demonstrations of Explainable AI
