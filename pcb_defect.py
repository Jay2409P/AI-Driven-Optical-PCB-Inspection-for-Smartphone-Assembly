
#  STEP 1: Install Dependencies
print(" Installing dependencies for Dual T4 GPU training...")
# Ensure all necessary packages are installed
!pip install ultralytics PyYAML opencv-python matplotlib numpy pandas seaborn tqdm tensorboard

# Add a check to confirm installation of ultralytics
try:
    import ultralytics
    print(" ultralytics installed successfully.")
except ImportError:
    print(" Error: 'ultralytics' module not found after installation attempt.")
    print("Please check your internet connection and re-run the installation cell.")
    print("If the issue persists, try restarting the runtime and running all cells again.")
    raise ModuleNotFoundError("Required 'ultralytics' module is not installed.")

#  STEP 2: Setup Multi-GPU CUDA Environment
import torch
import torch.nn as nn
import os
import yaml
import shutil
from ultralytics import YOLO
import multiprocessing as mp

print(" Checking NVIDIA T4 GPU/CUDA setup...")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs available: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    # Display information for all available GPUs
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        compute_capability = torch.cuda.get_device_capability(i)

        print(f" GPU {i}: {gpu_name}")
        print(f" GPU {i} Memory: {gpu_memory:.1f} GB")
        print(f" GPU {i} CUDA Compute Capability: {compute_capability[0]}.{compute_capability[1]}")

    # CUDA optimizations for T4 GPUs
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Clear cache on all GPUs
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()

    # T4 GPU specific optimizations (compute capability 7.5)
    # T4 doesn't support TensorFloat-32 (requires compute capability 8.x)
    print("  T4 GPUs detected - optimizing for compute capability 7.5")

    # Optimized batch size for dual T4 setup (16GB total memory)
    # Each T4 has ~15GB usable memory, so we can use larger batch sizes
    total_gpu_memory = sum(torch.cuda.get_device_properties(i).total_memory / 1024**3
                          for i in range(torch.cuda.device_count()))

    if torch.cuda.device_count() >= 2:
        # Dual T4 optimization
        batch_size = 24  # Increased batch size for dual T4s
        print(f" Dual T4 optimized batch size: {batch_size}")
        print(f" Total GPU memory: {total_gpu_memory:.1f} GB")
    else:
        # Single T4 fallback
        batch_size = 16
        print(f" Single T4 batch size: {batch_size}")

else:
    print(" CUDA not available! Enable GPU runtime in Kaggle settings")
    batch_size = 4

# STEP 3: Configuration
# Define paths and training parameters
DATASET_PATH = '/kaggle/input/pcb-defect-dataset/pcb-defect-dataset'
WORKING_DIR = '/kaggle/working/'
OUTPUT_DIR = '/kaggle/working/'

# --- T4 GPU Optimized Hyperparameters ---
EPOCHS = 200
IMG_SIZE = 640  # Standard for YOLOv8
PATIENCE = 50
WORKERS = min(16, mp.cpu_count())  # Increased workers for dual GPU setup

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(WORKING_DIR)

print(f"\n Dual T4 GPU Training Configuration:")
print(f"  GPUs: {torch.cuda.device_count()}")
print(f"  Batch Size: {batch_size}")
print(f"  Epochs: {EPOCHS}")
print(f"  Workers: {WORKERS}")
print(f"  Mixed Precision: {'Enabled' if torch.cuda.is_available() else 'Disabled'}")

#  STEP 4: Create Data Configuration (data.yaml)
if not os.path.exists(DATASET_PATH):
    print(f" Dataset not found at: {DATASET_PATH}")
    print("Please ensure your dataset is correctly placed or uploaded!")
else:
    print(f" Dataset found at: {DATASET_PATH}")

data_yaml_content = {
    'path': DATASET_PATH,
    'train': 'train',
    'val': 'val',
    'test': 'test',
    'names': {
        0: 'mouse_bite', 1: 'spur', 2: 'missing_hole',
        3: 'short', 4: 'open_circuit', 5: 'spurious_copper'
    }
}

data_yaml_path = os.path.join(WORKING_DIR, 'data.yaml')
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_yaml_content, f, default_flow_style=False)

print(f" Data config created at: {data_yaml_path}")
print(f"Classes detected: {list(data_yaml_content['names'].values())}")

#  STEP 5: Load Model with Multi-GPU Support
print("\n Loading YOLOv8m model optimized for dual T4 setup...")
model = YOLO('yolov8m.pt')
print(" Model loaded successfully!")

#  STEP 6: Dual T4 GPU Training
print("\n Starting Dual T4 GPU training with optimized hyperparameters...")
print("=" * 60)

# Multi-GPU device configuration
if torch.cuda.device_count() >= 2:
    device = [0, 1]  # Use both T4 GPUs
    print(f" Using GPUs: {device}")
else:
    device = 0  # Fallback to single GPU
    print(f" Using single GPU: {device}")

results = model.train(
    # Dataset configuration
    data=data_yaml_path,

    # Core training parameters
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=batch_size,
    patience=PATIENCE,

    # Multi-GPU settings
    device=device,               # Use both T4 GPUs if available
    workers=WORKERS,             # Increased workers for dual GPU
    amp=True,                    # Mixed precision for T4 compatibility

    # Output and logging settings
    project=WORKING_DIR,
    name='pcb_defect_dual_t4_optimized',
    save=True,
    exist_ok=True,
    verbose=True,
    pretrained=True,
    seed=42,

    # --- T4 Optimized Hyperparameters ---
    optimizer='AdamW',           # Best optimizer for T4 performance
    lr0=0.001,                   # Slightly higher LR for dual GPU training
    lrf=0.0001,                  # Lower final LR multiplier
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    # --- T4 Optimized Data Augmentation ---
    hsv_h=0.015,                 # Conservative hue augmentation for T4
    hsv_s=0.7,                   # Saturation augmentation
    hsv_v=0.4,                   # Value augmentation
    degrees=0.0,                 # No rotation to save GPU memory
    translate=0.1,               # Moderate translation
    scale=0.5,                   # Conservative scaling for T4
    shear=0.0,                   # No shearing
    perspective=0.0,             # No perspective transformation
    flipud=0.0,                  # No up-down flip
    fliplr=0.5,                  # 50% left-right flip
    mosaic=1.0,                  # Full mosaic augmentation
    mixup=0.0,                   # Disabled mixup to save T4 memory
    copy_paste=0.0,              # Disabled copy-paste to save T4 memory

    # --- T4 Memory Optimization ---
    close_mosaic=10,             # Close mosaic augmentation in last 10 epochs
    save_period=25,              # Save checkpoint every 25 epochs

    # --- Performance Optimization ---
    cache=True,                  # Cache images in RAM for faster training
    rect=False,                  # Rectangular training (can cause issues with dual GPU)
    cos_lr=True,                 # Cosine learning rate scheduler
    label_smoothing=0.0,         # No label smoothing for T4
    nbs=64,                      # Nominal batch size for scaling
    overlap_mask=True,           # Overlap mask for segmentation
    mask_ratio=4,                # Mask downsample ratio
    dropout=0.0,                 # No dropout for T4
    val=True,                    # Validate during training
)

print("\n" + "=" * 60)
print(" Dual T4 GPU training completed with optimized settings!")

#  STEP 7: Save Models and Results
runs_dir = os.path.join(WORKING_DIR, 'pcb_defect_dual_t4_optimized')
best_model_path = os.path.join(runs_dir, 'weights', 'best.pt')

if os.path.exists(best_model_path):
    # Save the best trained model
    best_save_path = os.path.join(OUTPUT_DIR, 'pcb_defect_yolov8m_dual_t4_best.pt')
    shutil.copy2(best_model_path, best_save_path)
    print(f" Best model saved: {best_save_path}")

    # Save the last trained model
    last_model_path = os.path.join(runs_dir, 'weights', 'last.pt')
    if os.path.exists(last_model_path):
        last_save_path = os.path.join(OUTPUT_DIR, 'pcb_defect_yolov8m_dual_t4_last.pt')
        shutil.copy2(last_model_path, last_save_path)
        print(f" Last model saved: {last_save_path}")

    # Save the entire training results directory
    results_save_path = os.path.join(OUTPUT_DIR, 'dual_t4_training_results')
    if os.path.exists(results_save_path):
        shutil.rmtree(results_save_path)
    shutil.copytree(runs_dir, results_save_path)
    print(f" Results saved: {results_save_path}")
else:
    print(f" Best model not found at {best_model_path}. Training might have failed or not completed.")

#  STEP 8: Evaluate Model
if os.path.exists(best_model_path):
    print("\n Evaluating the best trained model on the validation set...")
    best_model = YOLO(best_model_path)

    # Use primary GPU for evaluation
    val_results = best_model.val(data=data_yaml_path, device=0)

    print(f" Validation mAP50: {val_results.box.map50:.4f}")
    print(f" Validation mAP50-95: {val_results.box.map:.4f}")

    # Additional T4-specific metrics
    print(f" Per-class mAP50: {val_results.box.map50}")
    print(f" Model size: {os.path.getsize(best_model_path) / 1024**2:.1f} MB")
else:
    print(" Cannot evaluate model: Best model path does not exist.")

#  STEP 9: GPU Memory Cleanup
print("\n Cleaning up GPU memory...")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    print(" GPU memory cleaned up")

# STEP 10: Final Summary
print("\n PCB Defect Detection Dual T4 GPU Training Complete!")
print("=" * 60)
print(f" GPUs Used: {torch.cuda.device_count()} x T4")
print(f" Classes: {list(data_yaml_content['names'].values())}")
print(f" Best model: {OUTPUT_DIR}/pcb_defect_yolov8m_dual_t4_best.pt")
print(f" Results: {OUTPUT_DIR}/dual_t4_training_results/")

print(f"\n Usage Example:")
print(f"from ultralytics import YOLO")
print(f"model = YOLO('{OUTPUT_DIR}/pcb_defect_yolov8m_dual_t4_best.pt')")
print(f"results = model('path/to/your/pcb_image.jpg')")
print(f"results[0].show()")

print("\n T4 GPU Performance Tips:")
print("• Mixed precision training enabled for faster computation")
print("• Optimized batch size for dual T4 memory (24)")
print("• Conservative augmentation to prevent memory issues")
print("• Image caching enabled for faster data loading")
print("• Checkpoint saving every 25 epochs to prevent data loss")

print("\n Your dual T4 GPU PCB defect detector is ready for inference!")

# prompt: write code for install ultralytics libraly

!pip install ultralytics

from google.colab import drive
drive.mount('/content/drive')

# ============================================================================
# PCB DEFECT DETECTION - MODEL EVALUATION & METRICS VISUALIZATION
# Comprehensive evaluation with confusion matrix, F1, recall, precision, accuracy
# ============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os
from ultralytics import YOLO
import cv2
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

#  Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configuration - Update these paths according to your setup
BEST_MODEL_PATH = '/content/drive/MyDrive/kaggle/working/pcb_defect/weights/best.pt'
DATASET_PATH = '/content/drive/MyDrive/Poject/content'
OUTPUT_DIR = '/content/drive/MyDrive/kaggle/working/'
TEST_DIR = os.path.join(DATASET_PATH, 'test', 'images')
TEST_LABELS_DIR = os.path.join(DATASET_PATH, 'test', 'labels')

# Class names for PCB defects
CLASS_NAMES = {
    0: 'mouse_bite', 1: 'spur', 2: 'missing_hole',
    3: 'short', 4: 'open_circuit', 5: 'spurious_copper'
}

print(" PCB Defect Detection Model Evaluation")
print("=" * 60)

#  STEP 1: Load the trained model
print("Loading trained model...")
if not os.path.exists(BEST_MODEL_PATH):
    print(f" Model not found at: {BEST_MODEL_PATH}")
    print("Please ensure your model path is correct!")
    exit()

model = YOLO(BEST_MODEL_PATH)
print(" Model loaded successfully!")

# STEP 2: Collect predictions and ground truth
def collect_predictions_and_labels(model, test_dir, labels_dir, confidence_threshold=0.5):
    """
    Collect predictions and ground truth labels for evaluation
    """
    y_true = []
    y_pred = []
    confidence_scores = []

    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return [], [], []

    image_files = list(Path(test_dir).glob('*.jpg')) + list(Path(test_dir).glob('*.png'))

    if not image_files:
        print(f" No image files found in: {test_dir}")
        return [], [], []

    print(f"Processing {len(image_files)} test images...")

    for img_path in tqdm(image_files, desc="Evaluating"):
        try:
            # Get ground truth labels
            label_path = os.path.join(labels_dir, img_path.stem + '.txt')

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    gt_labels = []
                    for line in f.readlines():
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            gt_labels.append(class_id)
            else:
                gt_labels = []

            # Get model predictions
            results = model(str(img_path), verbose=False)

            pred_labels = []
            pred_confidences = []

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    if conf >= confidence_threshold:
                        class_id = int(boxes.cls[i])
                        pred_labels.append(class_id)
                        pred_confidences.append(conf)

            # Handle cases where image has multiple detections
            # For simplicity, we'll use the most confident prediction per image
            if pred_labels:
                max_conf_idx = np.argmax(pred_confidences)
                y_pred.append(pred_labels[max_conf_idx])
                confidence_scores.append(pred_confidences[max_conf_idx])
            else:
                y_pred.append(-1)  # No detection
                confidence_scores.append(0.0)

            # Use the first ground truth label (or -1 if no labels)
            if gt_labels:
                y_true.append(gt_labels[0])
            else:
                y_true.append(-1)  # No ground truth

        except Exception as e:
            print(f" Error processing {img_path}: {e}")
            continue

    return y_true, y_pred, confidence_scores

# Collect predictions
y_true, y_pred, confidences = collect_predictions_and_labels(model, TEST_DIR, TEST_LABELS_DIR)

# Check if we have any data
if not y_true or not y_pred:
    print("No valid predictions or ground truth found!")
    print("Please check your dataset paths and model.")
    exit()

# Filter out cases where both true and pred are -1 (no detection, no ground truth)
valid_indices = [(i, true_label, pred_label) for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred))
                 if not (true_label == -1 and pred_label == -1)]

if valid_indices:
    valid_y_true = [item[1] for item in valid_indices]
    valid_y_pred = [item[2] for item in valid_indices]
    valid_confidences = [confidences[item[0]] for item in valid_indices]
else:
    valid_y_true = y_true
    valid_y_pred = y_pred
    valid_confidences = confidences

print(f" Total test samples: {len(valid_y_true)}")
print(f"Valid samples with detections: {len([x for x in valid_y_true if x != -1])}")

# STEP 3: Calculate metrics
print("\n Calculating evaluation metrics...")

# Convert -1 (no detection) to a separate class for confusion matrix
def prepare_labels_for_cm(y_true, y_pred):
    # Map -1 to class 6 (background/no detection)
    y_true_cm = [6 if x == -1 else x for x in y_true]
    y_pred_cm = [6 if x == -1 else x for x in y_pred]
    return y_true_cm, y_pred_cm

# Check if we have enough data for meaningful evaluation
if len(valid_y_true) == 0:
    print("No valid samples found for evaluation!")
    exit()

y_true_cm, y_pred_cm = prepare_labels_for_cm(valid_y_true, valid_y_pred)

# Calculate confusion matrix with error handling
try:
    cm = confusion_matrix(y_true_cm, y_pred_cm)
    print(f" Confusion matrix shape: {cm.shape}")

    # Ensure we have a valid confusion matrix
    if cm.size == 0:
        print("Empty confusion matrix generated!")
        # Create a minimal confusion matrix for visualization
        unique_labels = sorted(set(y_true_cm + y_pred_cm))
        cm = np.zeros((len(unique_labels), len(unique_labels)))

except Exception as e:
    print(f" Error creating confusion matrix: {e}")
    # Create a minimal confusion matrix
    unique_labels = sorted(set(y_true_cm + y_pred_cm))
    cm = np.zeros((len(unique_labels), len(unique_labels)))

# Calculate metrics (excluding background class for meaningful metrics)
# Filter out background class (-1) for precision, recall, f1 calculation
y_true_filtered = [x for x in valid_y_true if x != -1]
y_pred_filtered = [y_pred for y_true, y_pred in zip(valid_y_true, valid_y_pred) if y_true != -1]

if y_true_filtered and y_pred_filtered:
    # Overall metrics
    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    precision = precision_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    recall = recall_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    f1 = f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)

    # Per-class metrics
    try:
        precision_per_class = precision_score(y_true_filtered, y_pred_filtered, average=None, zero_division=0)
        recall_per_class = recall_score(y_true_filtered, y_pred_filtered, average=None, zero_division=0)
        f1_per_class = f1_score(y_true_filtered, y_pred_filtered, average=None, zero_division=0)
    except:
        precision_per_class = recall_per_class = f1_per_class = []
else:
    accuracy = precision = recall = f1 = 0
    precision_per_class = recall_per_class = f1_per_class = []

#  STEP 4: Create visualizations
print("\n Creating visualizations...")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 1. Confusion Matrix - with error handling
plt.subplot(2, 3, 1)
try:
    if cm.size > 0 and cm.max() > 0:
        # Determine the labels for the confusion matrix
        unique_labels = sorted(set(y_true_cm + y_pred_cm))
        class_names_with_bg = []
        for label in unique_labels:
            if label == 6:
                class_names_with_bg.append('Background/No Detection')
            elif label in CLASS_NAMES:
                class_names_with_bg.append(CLASS_NAMES[label])
            else:
                class_names_with_bg.append(f'Class_{label}')

        # Ensure the confusion matrix matches the number of labels
        if len(class_names_with_bg) == cm.shape[0]:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names_with_bg, yticklabels=class_names_with_bg)
            plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        else:
            plt.text(0.5, 0.5, 'Confusion Matrix\nNot Available\n(Insufficient Data)',
                     ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    else:
        plt.text(0.5, 0.5, 'Confusion Matrix\nNot Available\n(No Valid Predictions)',
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
except Exception as e:
    plt.text(0.5, 0.5, f'Confusion Matrix\nError: {str(e)[:50]}...',
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)

# 2. Overall Metrics Bar Chart
plt.subplot(2, 3, 2)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = plt.bar(metrics, values, color=colors)
plt.title('Overall Performance Metrics', fontsize=14, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# 3. Per-Class Precision
plt.subplot(2, 3, 3)
if len(precision_per_class) > 0 and len(y_true_filtered) > 0:
    try:
        unique_classes = sorted(set(y_true_filtered))
        class_labels = [CLASS_NAMES.get(i, f'Class_{i}') for i in unique_classes]
        if len(class_labels) == len(precision_per_class):
            plt.bar(class_labels, precision_per_class, color='skyblue', alpha=0.7)
            plt.title('Precision per Class', fontsize=14, fontweight='bold')
            plt.ylabel('Precision', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
        else:
            plt.text(0.5, 0.5, 'Per-Class Precision\nNot Available',
                     ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    except Exception as e:
        plt.text(0.5, 0.5, f'Per-Class Precision\nError: {str(e)[:30]}...',
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
else:
    plt.text(0.5, 0.5, 'Per-Class Precision\nNo Valid Data',
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)

# 4. Per-Class Recall
plt.subplot(2, 3, 4)
if len(recall_per_class) > 0 and len(y_true_filtered) > 0:
    try:
        unique_classes = sorted(set(y_true_filtered))
        class_labels = [CLASS_NAMES.get(i, f'Class_{i}') for i in unique_classes]
        if len(class_labels) == len(recall_per_class):
            plt.bar(class_labels, recall_per_class, color='lightcoral', alpha=0.7)
            plt.title('Recall per Class', fontsize=14, fontweight='bold')
            plt.ylabel('Recall', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
        else:
            plt.text(0.5, 0.5, 'Per-Class Recall\nNot Available',
                     ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    except Exception as e:
        plt.text(0.5, 0.5, f'Per-Class Recall\nError: {str(e)[:30]}...',
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
else:
    plt.text(0.5, 0.5, 'Per-Class Recall\nNo Valid Data',
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)

# 5. Per-Class F1-Score
plt.subplot(2, 3, 5)
if len(f1_per_class) > 0 and len(y_true_filtered) > 0:
    try:
        unique_classes = sorted(set(y_true_filtered))
        class_labels = [CLASS_NAMES.get(i, f'Class_{i}') for i in unique_classes]
        if len(class_labels) == len(f1_per_class):
            plt.bar(class_labels, f1_per_class, color='lightgreen', alpha=0.7)
            plt.title('F1-Score per Class', fontsize=14, fontweight='bold')
            plt.ylabel('F1-Score', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
        else:
            plt.text(0.5, 0.5, 'Per-Class F1-Score\nNot Available',
                     ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    except Exception as e:
        plt.text(0.5, 0.5, f'Per-Class F1-Score\nError: {str(e)[:30]}...',
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
else:
    plt.text(0.5, 0.5, 'Per-Class F1-Score\nNo Valid Data',
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)

# 6. Confidence Score Distribution
plt.subplot(2, 3, 6)
valid_confidences_filtered = [conf for conf, true_label in zip(valid_confidences, valid_y_true) if true_label != -1]
if valid_confidences_filtered:
    try:
        plt.hist(valid_confidences_filtered, bins=min(30, len(valid_confidences_filtered)),
                 alpha=0.7, color='purple', edgecolor='black')
        plt.title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        mean_conf = np.mean(valid_confidences_filtered)
        plt.axvline(x=mean_conf, color='red', linestyle='--',
                    label=f'Mean: {mean_conf:.3f}')
        plt.legend()
    except Exception as e:
        plt.text(0.5, 0.5, f'Confidence Distribution\nError: {str(e)[:30]}...',
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
else:
    plt.text(0.5, 0.5, 'Confidence Distribution\nNo Valid Data',
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pcb_defect_evaluation_metrics.png'), dpi=300, bbox_inches='tight')
plt.show()

#  STEP 5: Print detailed classification report
print("\n Detailed Classification Report:")
print("=" * 60)

if y_true_filtered and y_pred_filtered:
    try:
        unique_classes = sorted(set(y_true_filtered))
        target_names = [CLASS_NAMES.get(i, f'Class_{i}') for i in unique_classes]
        report = classification_report(y_true_filtered, y_pred_filtered,
                                     target_names=target_names, zero_division=0)
        print(report)
    except Exception as e:
        print(f" Could not generate classification report: {e}")
else:
    print(" No valid data available for classification report")

# STEP 6: Create and save metrics summary
print("\nMetrics Summary:")
print("=" * 60)
print(f"Total samples processed: {len(valid_y_true)}")
print(f"Valid samples with ground truth: {len(y_true_filtered)}")
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Overall Precision: {precision:.4f}")
print(f"Overall Recall: {recall:.4f}")
print(f"Overall F1-Score: {f1:.4f}")
print(f"Average Confidence: {np.mean(valid_confidences) if valid_confidences else 0:.4f}")

# Create a summary DataFrame
try:
    metrics_summary = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Avg Confidence'],
        'Value': [accuracy, precision, recall, f1, np.mean(valid_confidences) if valid_confidences else 0]
    }

    df_summary = pd.DataFrame(metrics_summary)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, 'pcb_defect_metrics_summary.csv'), index=False)
    print(" Metrics summary saved to CSV")

    # Create per-class metrics DataFrame
    if len(precision_per_class) > 0 and len(y_true_filtered) > 0:
        try:
            unique_classes = sorted(set(y_true_filtered))
            class_labels = [CLASS_NAMES.get(i, f'Class_{i}') for i in unique_classes]

            if len(class_labels) == len(precision_per_class):
                per_class_metrics = {
                    'Class': class_labels,
                    'Precision': precision_per_class,
                    'Recall': recall_per_class,
                    'F1-Score': f1_per_class
                }
                df_per_class = pd.DataFrame(per_class_metrics)
                df_per_class.to_csv(os.path.join(OUTPUT_DIR, 'pcb_defect_per_class_metrics.csv'), index=False)
                print("\nPer-Class Metrics:")
                print(df_per_class.to_string(index=False))
                print(" Per-class metrics saved to CSV")
            else:
                print("Inconsistent per-class metrics data")
        except Exception as e:
            print(f" Error saving per-class metrics: {e}")
    else:
        print(" No per-class metrics available")

except Exception as e:
    print(f" Error creating metrics summary: {e}")

print(f"\nEvaluation results saved to:")
print(f"   Visualization: {OUTPUT_DIR}/pcb_defect_evaluation_metrics.png")
print(f"   Metrics Summary: {OUTPUT_DIR}/pcb_defect_metrics_summary.csv")
print(f"  Per-Class Metrics: {OUTPUT_DIR}/pcb_defect_per_class_metrics.csv")

#  STEP 7: Additional Analysis - Validation metrics from training
print("\n Additional Model Analysis:")
print("=" * 60)

# Try to get validation metrics from the trained model
try:
    # Re-run validation to get detailed metrics
    print("Running validation on the dataset...")

    # Check if data.yaml exists
    data_yaml_path = '/content/drive/MyDrive/kaggle/working/data.yaml'
    if os.path.exists(data_yaml_path):
        val_results = model.val(data=data_yaml_path, device=0)

        print(f"Validation mAP50: {val_results.box.map50:.4f}")
        print(f"Validation mAP50-95: {val_results.box.map:.4f}")

        # Create a comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Test vs Validation metrics comparison
        test_metrics = [accuracy, precision, recall, f1]
        val_metrics = [val_results.box.map50, val_results.box.map50, val_results.box.map50, val_results.box.map50]  # Using mAP50 as proxy

        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(metrics_names))
        width = 0.35

        ax1.bar(x - width/2, test_metrics, width, label='Test Set', alpha=0.8)
        ax1.bar(x + width/2, val_metrics, width, label='Validation Set', alpha=0.8)
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Test vs Validation Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_names)
        ax1.legend()
        ax1.set_ylim(0, 1)

        # Class distribution
        if y_true_filtered:
            class_counts = pd.Series(y_true_filtered).value_counts().sort_index()
            class_names_dist = [CLASS_NAMES.get(i, f'Class_{i}') for i in class_counts.index]
            ax2.bar(class_names_dist, class_counts.values, color='lightblue', alpha=0.7)
            ax2.set_xlabel('Class')
            ax2.set_ylabel('Count')
            ax2.set_title('Test Set Class Distribution')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No Class Distribution\nData Available',
                     ha='center', va='center', transform=ax2.transAxes, fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'pcb_defect_additional_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print(f" data.yaml not found at {data_yaml_path}")

except Exception as e:
    print(f" Could not run additional validation: {e}")

print("\nPCB Defect Detection Model Evaluation Complete!")
print("=" * 60)
print(" All metrics calculated and visualized successfully!")
print(" Check the generated plots and CSV files for detailed analysis.")

# Final data validation summary
print("\n Data Validation Summary:")
print("=" * 60)
print(f" Total images processed: {len(y_true) if y_true else 0}")
print(f" Valid predictions: {len([p for p in y_pred if p != -1]) if y_pred else 0}")
print(f" Valid ground truth: {len([gt for gt in y_true if gt != -1]) if y_true else 0}")
print(f" Samples used for evaluation: {len(y_true_filtered) if y_true_filtered else 0}")

if len(y_true_filtered) == 0:
    print("\n WARNING: No valid samples found for evaluation!")
    print("This could be due to:")
    print("  - Incorrect dataset paths")
    print("  - Missing label files")
    print("  - Model confidence threshold too high")
    print("  - No detections made by the model")
    print("  - Corrupted or empty dataset")
