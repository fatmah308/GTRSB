# 🚦 German Traffic Sign Recognition (GTSRB)

An end-to-end deep learning pipeline for classifying traffic signs from the German Traffic Sign Recognition Benchmark dataset, achieving **95%+ accuracy** across 43 sign classes.

![Traffic Signs](https://img.shields.io/badge/Classes-43-blue) ![Accuracy](https://img.shields.io/badge/Accuracy-95%25+-green) ![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

This project implements a Convolutional Neural Network (CNN) to classify German traffic signs into 43 categories. The system includes:
- Custom preprocessing pipeline with ROI-based cropping
- Deep CNN with batch normalization and dropout
- Real-time prediction via Streamlit web interface
- Comprehensive data augmentation for robust generalization

---

## ✨ Features

- **High Accuracy**: 95%+ test accuracy with a 3-block CNN architecture
- **ROI-Based Preprocessing**: Crops images to the traffic sign region for cleaner inputs
- **Data Augmentation**: Rotation, shifting, zoom, and shear to prevent overfitting
- **Interactive Web App**: Upload and classify traffic signs in real-time using Streamlit
- **Top-3 Predictions**: View confidence scores for the most likely sign classes
- **Training Visualization**: Automatic generation of accuracy/loss curves and confusion matrix

---

## 📊 Dataset

**German Traffic Sign Recognition Benchmark (GTSRB)**

- **Training samples**: 39,209 images
- **Test samples**: 12,630 images
- **Classes**: 43 traffic sign categories
- **Image size**: Variable (resized to 48×48 for model input)

### Dataset Structure
```
├── Train.csv          # Training metadata (Width, Height, ROI coords, ClassId, Path)
├── Test.csv           # Test metadata (same structure)
├── Meta.csv           # Class mapping (ClassId → ShapeId, ColorId, SignId)
├── Train/             # Training images organized by class
└── Test/              # Test images
```

---

## 🏗️ Model Architecture

**3-Block CNN with Batch Normalization**

```
Input (48×48×3)
    ↓
┌─────────────────────┐
│ Conv Block 1        │
│ - Conv2D(32) + BN   │
│ - Conv2D(32) + BN   │
│ - MaxPool(2×2)      │
│ - Dropout(0.25)     │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Conv Block 2        │
│ - Conv2D(64) + BN   │
│ - Conv2D(64) + BN   │
│ - MaxPool(2×2)      │
│ - Dropout(0.25)     │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Conv Block 3        │
│ - Conv2D(128) + BN  │
│ - Conv2D(128) + BN  │
│ - MaxPool(2×2)      │
│ - Dropout(0.25)     │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Classifier          │
│ - Flatten           │
│ - Dense(256) + BN   │
│ - Dropout(0.5)      │
│ - Dense(43, softmax)│
└─────────────────────┘
```

**Key Design Choices:**
- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout**: Prevents overfitting (0.25 after conv blocks, 0.5 before output)
- **No Horizontal Flip**: Traffic signs are not horizontally symmetric
- **ReduceLROnPlateau**: Halves learning rate when validation loss plateaus

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- Streamlit

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gtsrb-traffic-sign-recognition.git
cd gtsrb-traffic-sign-recognition
```

2. **Install dependencies**
```bash
pip install tensorflow opencv-python pandas numpy matplotlib streamlit scikit-learn
```

3. **Download the GTSRB dataset**
   - Place `Train.csv`, `Test.csv`, `Meta.csv` in the project root
   - Ensure image paths in CSVs are correct

4. **Create `class_labels.py`**
```python
# Generate from Meta.csv
import pandas as pd
meta = pd.read_csv("Meta.csv")
CLASS_NAMES = {row['ClassId']: f"Sign_{row['SignId']}" for _, row in meta.iterrows()}
```

---

## 🚀 Usage

### Training the Model

```bash
python gtsrb_fixed.py
```

**What happens:**
1. Loads training/test data with ROI-based preprocessing
2. Applies data augmentation (rotation, shifts, zoom)
3. Trains the model for up to 50 epochs (early stopping enabled)
4. Saves the best model as `model.h5`
5. Generates `training_curves.png` and `confusion_matrix.png`

**Expected output:**
```
Train samples: 39209 | Test samples: 12630 | Classes: 43
X_train shape: (39209, 48, 48, 3) | X_test shape: (12630, 48, 48, 3)
Pixel range check — min: 0.0, max: 255.0

Epoch 1/50
...
✓ Test Accuracy: 0.9512 | Test Loss: 0.2134
Model saved as model.h5
```

---

### Running the Streamlit App

```bash
streamlit run app.py
```

**Features:**
- Upload a traffic sign image (JPG, PNG)
- View the predicted class and confidence score
- See top-3 predictions with confidence bars

**Screenshot:**
```
🚦 Traffic Sign Recognition System
Upload a traffic sign image to classify it.

[Uploaded Image]

✓ Predicted Sign: Sign_12
ℹ️ Confidence: 98.34%

Top 3 Predictions
1. Sign_12    98.34%  ████████████████████
2. Sign_15    1.23%   █
3. Sign_9     0.43%   
```

---

## 📈 Results

### Model Performance

| Metric          | Value     |
|-----------------|-----------|
| Test Accuracy   | **95.1%** |
| Test Loss       | 0.213     |
| Training Time   | ~25 min (GPU) |
| Parameters      | ~1.2M     |

### Training Curves
- **Accuracy**: Converges around epoch 15-20
- **Loss**: Steady decrease with minimal overfitting
- **Validation**: Tracks training closely (good generalization)

### Common Misclassifications
- Similar-shaped signs (e.g., circular speed limits)
- Low-resolution or occluded test images
- Signs with lighting variations

---

## 📁 Project Structure

```
gtsrb-traffic-sign-recognition/
│
├── gtsrb_fixed.py          # Model training script
├── app.py                  # Streamlit web app
├── class_labels.py         # ClassId → Sign name mapping
├── README.md               # This file
│
├── Train.csv               # Training metadata
├── Test.csv                # Test metadata
├── Meta.csv                # Class information
│
├── Train/                  # Training images (by class)
├── Test/                   # Test images
│
├── model.h5                # Saved trained model
├── training_curves.png     # Accuracy/loss plots
└── confusion_matrix.png    # 43×43 confusion matrix
```

---

## 🔮 Future Improvements

- [ ] **Transfer Learning**: Fine-tune ResNet50 or EfficientNet for better accuracy
- [ ] **Real-Time Video**: Extend to video stream processing
- [ ] **Model Compression**: Quantization for mobile deployment
- [ ] **Explainability**: Grad-CAM visualizations to see what the model focuses on
- [ ] **Multi-Language Support**: Add sign name translations in the app
- [ ] **Ensemble Models**: Combine multiple CNNs for robust predictions

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Dataset**: [GTSRB from INI Benchmark](http://benchmark.ini.rub.de/)
- **Framework**: TensorFlow/Keras
- **Inspiration**: Traffic safety and autonomous driving research

---

## 📧 Contact

**Fatima Aleem**  
- Email: fatima.308.99@gmail.com
- GitHub: [@fatmah308](https://github.com/fatmah308)
- LinkedIn: [Fatima Aleem](https://linkedin.com/in/fatima-aleem)

---

**⭐ If you found this project helpful, please give it a star!**
