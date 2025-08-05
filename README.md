# Traffic Sign Recognition with CNN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)

A deep learning model that classifies 43 types of traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset using a custom Convolutional Neural Network (CNN).

## 📌 Key Features
- **Data Preprocessing Pipeline**: ROI cropping, resizing (32x32), and normalization
- **CNN Architecture**: 2 convolutional layers with dropout for regularization
- **Model Evaluation**: 86.7% test accuracy with per-class metrics
- **Class Imbalance Handling**: Stratified sampling across 43 sign classes
- **Tools**: TensorFlow/Keras, OpenCV, scikit-learn

## 📊 Results Summary
| Metric          | Score  |
|-----------------|--------|
| Test Accuracy   | 86.7%  |
| Precision (avg) | 88.0%  |
| Recall (avg)    | 86.7%  |
| F1-Score (avg)  | 86.3%  |

**Top Performing Classes**:
- Speed Limit 50 (Sign_3.29): 96.6% precision
- Stop Sign (Sign_2.2): 100% F1-score

**Challenging Classes**:
- Pedestrians (Sign_nan): 59.4% F1-score
- Road Work (Sign_1.32): 66.0% F1-score

## 🛠️ Installation
```bash
git clone https://github.com/yourusername/traffic-sign-recognition.git
cd traffic-sign-recognition
pip install -r requirements.txt

##🚀 Usage
Download GTSRB dataset
Place Train.csv, Test.csv, and Meta.csv in /data
Run the pipeline:
python train.py

##🧠 Model Architecture
Model: "sequential"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 30, 30, 32)     │           896 │
│ max_pooling2d (MaxPooling2D)    │ (None, 15, 15, 32)     │             0 │
│ conv2d_1 (Conv2D)               │ (None, 13, 13, 64)     │        18,496 │
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 6, 6, 64)       │             0 │
│ flatten (Flatten)               │ (None, 2304)           │             0 │
│ dense (Dense)                   │ (None, 128)            │       295,040 │
│ dropout (Dropout)               │ (None, 128)            │             0 │
│ dense_1 (Dense)                 │ (None, 43)             │         5,547 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
Total params: 319,979 (1.22 MB)

##📂 Dataset Structure
39,209 training images across 43 classes
12,630 test images

##📝 Key Code Features
# Balanced dataset loading
def load_balanced_dataset(df):
    grouped = df.groupby('ClassId')
    sampled_df = grouped.apply(lambda x: x.sample(min(len(x), samples_per_class)))
    ...

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1
)

##📈 Future Improvements
Implement ResNet50 for comparison
Add Grad-CAM visualization for model interpretability
Deploy as a web app using Flask

##🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first.

##📜 License
MIT
