# Environment Setup
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings('ignore')

# Imports
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Data Loading
def load_datasets():
    """Load all datasets with validation"""
    try:
        train = pd.read_csv('Train.csv')
        test = pd.read_csv('Test.csv')
        meta = pd.read_csv('Meta.csv')
        print("Datasets loaded successfully")
        print(f"Train: {len(train)} samples | Test: {len(test)} samples | Classes: {len(meta)}")
        return train, test, meta
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

train, test, meta = load_datasets()
if train is None:
    exit()

# After loading train/test datasets but BEFORE preprocessing
# ======================
# CLASS DISTRIBUTION CHECK
# ======================
from collections import Counter

print("Full dataset class distribution:")
print(Counter(train['ClassId']))

missing_classes = set(range(43)) - set(train['ClassId'].unique())
if missing_classes:
    print(f"⚠️ Warning: Missing classes in original data: {missing_classes}")
else:
    print("✅ All classes present in raw data")

# Data Preprocessing
def preprocess_image(img_path, roi_coords, target_size=(32, 32)):
    """Robust image preprocessing with error handling"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = map(int, roi_coords)
        
        # Validate ROI coordinates
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        cropped = img[y1:y2, x1:x2]
        resized = cv2.resize(cropped, target_size)
        return resized.astype(np.float32) / 255.0
    except:
        return None

# ======================
# MODIFIED DATA LOADER
# ======================
def load_balanced_dataset(df, target_size=(32, 32), sample_fraction=1.0):
    """Ensures all classes are represented"""
    # Group by class and sample evenly
    grouped = df.groupby('ClassId')
    samples_per_class = int(len(df)/len(grouped) * sample_fraction)
    sampled_df = grouped.apply(lambda x: x.sample(min(len(x), samples_per_class)))
    
    # Load and preprocess
    images, labels = [], []
    for _, row in sampled_df.iterrows():
        img = preprocess_image(row['Path'], row[['Roi.X1','Roi.Y1','Roi.X2','Roi.Y2']].values, target_size)
        if img is not None:
            images.append(img)
            labels.append(row['ClassId'])
    
    # Verify all classes loaded
    loaded_classes = set(np.unique(labels))
    print(f"Loaded {len(images)} images ({len(loaded_classes)} classes)")
    missing = set(range(43)) - loaded_classes
    if missing:
        print(f"Still missing classes: {missing}")
    else:
        print("All classes represented in final dataset")
    
    return np.array(images), np.array(labels)

# Usage (replace your current load calls):
X_train, y_train = load_balanced_dataset(train, sample_fraction=0.3)
X_test, y_test = load_balanced_dataset(test, sample_fraction=0.3)


# Metadata Handling
# Create class name mapping
class_names = {row['ClassId']: f"Sign_{row['SignId']}" 
              for _, row in meta.iterrows()} if 'SignId' in meta.columns else None

# Verify all classes are represented
train_classes = set(np.unique(y_train))
meta_classes = set(meta['ClassId'])
missing_classes = meta_classes - train_classes
if missing_classes:
    print(f" Warning: Missing training data for classes: {missing_classes}")

num_classes = len(meta)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Data Visualization
def plot_samples(images, labels, class_names=None, n=5):
    plt.figure(figsize=(15, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i])
        title = class_names[np.argmax(labels[i])] if class_names else f"Class {np.argmax(labels[i])}"
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("\n Sample training images:")
plot_samples(X_train, y_train, class_names)

# Model Building
def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()
    return model

print("\n Building model...")
model = build_model(X_train.shape[1:], num_classes)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='constant',
    cval=0  
)

# Model Training
print("\nTraining model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=5,
    validation_data=(X_test, y_test),
    verbose=1
)


# Model Evaluation
def evaluate_model(model, X_test, y_test, class_names):
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n Test Accuracy: {test_acc:.4f}")
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    target_names = [class_names[i] for i in sorted(class_names.keys())] if class_names else None
    print(classification_report(y_true, y_pred, target_names=target_names, digits=3))

print("\n Evaluating model...")
evaluate_model(model, X_test, y_test, class_names)

# BONUS: MobileNetV2
def build_mobilenet():
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D
    
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

print("\n Bonus: MobileNetV2 Model")
X_train_mobile = np.array([cv2.resize(img, (96, 96)) for img in X_train])
X_test_mobile = np.array([cv2.resize(img, (96, 96)) for img in X_test])

mobilenet = build_mobilenet()
mobilenet.fit(
    datagen.flow(X_train_mobile, y_train, batch_size=32),
    epochs=5,
    validation_data=(X_test_mobile, y_test)
)
