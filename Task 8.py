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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

# CLASS DISTRIBUTION CHECK
from collections import Counter

print("Full dataset class distribution:")
print(Counter(train['ClassId']))

missing_classes = set(range(43)) - set(train['ClassId'].unique())
if missing_classes:
    print(f"Warning: Missing classes in original data: {missing_classes}")
else:
    print("All classes present in raw data")

#Class Distribution Plot (Before Training)
def plot_class_distribution(y_train, class_names):
    plt.figure(figsize=(15, 6))
    class_counts = np.bincount(np.argmax(y_train, axis=1))
    plt.bar(range(len(class_counts)), class_counts, color='skyblue')
    plt.xticks(range(len(class_names)), list(class_names.values()), rotation=90)
    plt.title('Training Data Class Distribution')
    plt.xlabel('Traffic Sign Class')
    plt.ylabel('Number of Samples')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# 2. Sample Predictions with Ground Truth (After Evaluation)
def plot_sample_predictions(X_test, y_true, y_pred, class_names, n=5):
    errors = np.where(y_pred != y_true)[0]
    correct = np.where(y_pred == y_true)[0]
    
    plt.figure(figsize=(15, 8))
    
    # Correct predictions
    plt.suptitle('Model Predictions (Green=Correct, Red=Wrong)', y=1.02)
    for i in range(min(n, len(correct))):
        plt.subplot(2, n, i+1)
        plt.imshow(X_test[correct[i]])
        plt.title(f"True: {class_names[y_true[correct[i]]]}\nPred: {class_names[y_pred[correct[i]]]}")
        plt.axis('off')
        plt.gca().set_facecolor('lightgreen')
    
    # Incorrect predictions
    for i in range(min(n, len(errors))):
        plt.subplot(2, n, n+i+1)
        plt.imshow(X_test[errors[i]])
        plt.title(f"True: {class_names[y_true[errors[i]]]}\nPred: {class_names[y_pred[errors[i]]]}")
        plt.axis('off')
        plt.gca().set_facecolor('lightcoral')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

# 3. Enhanced Confusion Matrix (Top 20 Classes)
def plot_confusion_matrix(y_true, y_pred, class_names):
    plt.figure(figsize=(15, 12))
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm[:20, :20],
                                display_labels=list(class_names.values())[:20])
    disp.plot(cmap='Blues', xticks_rotation=90, values_format='.2f')
    plt.title('Normalized Confusion Matrix (Top 20 Classes)', pad=20)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

# 4. Training History Plots
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# 5. Per-Class Metrics Bar Chart
def plot_metrics_by_class(y_true, y_pred, class_names):
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    metrics_df = pd.DataFrame({
        'Class': [class_names[i] for i in range(len(class_names))],
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }).sort_values('F1-Score', ascending=False)
    
    plt.figure(figsize=(12, 8))
    metrics_df.head(20).plot(x='Class', y=['Precision', 'Recall', 'F1-Score'], 
                            kind='bar', colormap='coolwarm')
    plt.title('Top 20 Classes by F1-Score')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('class_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

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

# MODIFIED DATA LOADER
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

def test_single_image(image_path, model, class_names):
    """
    Test a single traffic sign image with the trained model
    
    Args:
        image_path (str): Path to the test image
        model (keras.Model): Your trained CNN model
        class_names (dict): Dictionary mapping class IDs to sign names
    """
    # 1. Manually define ROI coordinates or use default full-image
    # For real use, you'd want to detect ROI automatically or use known coordinates
    roi_coords = [5, 5, 28, 28]  # [x1, y1, x2, y2] - adjust based on your image
    
    # 2. Preprocess the image exactly like training data
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Convert and crop
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = roi_coords
    cropped = img[y1:y2, x1:x2]
    
    # Resize and normalize
    resized = cv2.resize(cropped, (32, 32))
    normalized = resized.astype(np.float32) / 255.0
    
    # 3. Add batch dimension (model expects [batch, height, width, channels])
    input_tensor = np.expand_dims(normalized, axis=0)
    
    # 4. Make prediction
    predictions = model.predict(input_tensor)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # 5. Display results
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(resized)
    plt.title("Processed Input")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.barh(list(class_names.values()), predictions[0])
    plt.title(f"Prediction: {class_names[predicted_class]}\nConfidence: {confidence:.2%}")
    plt.xlabel("Probability")
    plt.tight_layout()
    
    plt.show()

    return predicted_class, confidence

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

# 1. Class distribution
plot_class_distribution(y_train, class_names)

# 2. Training curves
plot_training_history(history)

# 3. After evaluation:
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

plot_sample_predictions(X_test, y_true, y_pred, class_names)
plot_confusion_matrix(y_true, y_pred, class_names)
plot_metrics_by_class(y_true, y_pred, class_names)

# Example usage:
test_image_path = "your_test_image.jpg"  # Replace with your image path
predicted_class, confidence = test_single_image(test_image_path, model, class_names)

print(f"Predicted: {class_names[predicted_class]} (Class {predicted_class})")
print(f"Confidence: {confidence:.2%}")
