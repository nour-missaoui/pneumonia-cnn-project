# ==========================================
# Pneumonia Detection Using CNN
# Generates model, metrics, reports, and plots
# ==========================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# ==========================================
# Configuration
# ==========================================

img_size = 150
batch_size = 32
epochs = 5

train_dir = "chest_xray/train"
val_dir = "chest_xray/val"
test_dir = "chest_xray/test"

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# Data preprocessing
# ==========================================

# Training images are normalized and augmented.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    zoom_range=0.1,
    horizontal_flip=True,
)

# Validation and test images are only normalized.
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
)

val_data = test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False,
)

class_names = list(test_data.class_indices.keys())

print("Class indices:", test_data.class_indices)

# ==========================================
# Dataset class distribution plot
# ==========================================

train_normal = np.sum(train_data.classes == 0)
train_pneumonia = np.sum(train_data.classes == 1)

val_normal = np.sum(val_data.classes == 0)
val_pneumonia = np.sum(val_data.classes == 1)

test_normal = np.sum(test_data.classes == 0)
test_pneumonia = np.sum(test_data.classes == 1)

plt.figure(figsize=(9, 5))
plt.bar(
    [
        "Train Normal",
        "Train Pneumonia",
        "Val Normal",
        "Val Pneumonia",
        "Test Normal",
        "Test Pneumonia",
    ],
    [
        train_normal,
        train_pneumonia,
        val_normal,
        val_pneumonia,
        test_normal,
        test_pneumonia,
    ],
)
plt.title("Dataset Class Distribution")
plt.xlabel("Dataset Split and Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "dataset_class_distribution.png"), dpi=300)
plt.close()

# ==========================================
# CNN Model
# ==========================================

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ==========================================
# Train model
# ==========================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
)

# ==========================================
# Evaluate model
# ==========================================

loss, acc = model.evaluate(test_data)

predictions = model.predict(test_data)
y_scores = predictions.ravel()
y_pred = (y_scores > 0.5).astype(int)
y_true = test_data.classes

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
)

cm = confusion_matrix(y_true, y_pred)

print("\nCNN Pneumonia Detection Report")
print("=" * 40)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(cm)

# ==========================================
# Save text report
# ==========================================

report_path = os.path.join(output_dir, "cnn_report.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("CNN Pneumonia Detection Report\n")
    f.write("=" * 40)
    f.write("\n\n")

    f.write(f"Test Loss: {loss:.4f}\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n\n")

    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n")

    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write("Interpretation:\n")
    f.write(
        "The model shows high recall for pneumonia detection, meaning it detects most pneumonia cases. "
        "However, if the NORMAL recall is low, the model produces many false positives by classifying "
        "normal X-rays as pneumonia.\n"
    )

# ==========================================
# Plot 1: Training and validation accuracy
# ==========================================

plt.figure(figsize=(8, 6))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cnn_accuracy_plot.png"), dpi=300)
plt.close()

# ==========================================
# Plot 2: Training and validation loss
# ==========================================

plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cnn_loss_plot.png"), dpi=300)
plt.close()

# ==========================================
# Plot 3: Confusion matrix
# ==========================================

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], class_names)
plt.yticks([0, 1], class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cnn_confusion_matrix.png"), dpi=300)
plt.close()

# ==========================================
# Plot 4: ROC curve
# ==========================================

fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.title("CNN ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cnn_roc_curve.png"), dpi=300)
plt.close()

# ==========================================
# Plot 5: Precision-Recall curve
# ==========================================

precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, y_scores)
avg_precision = average_precision_score(y_true, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label=f"Average Precision = {avg_precision:.2f}")
plt.title("CNN Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cnn_precision_recall_curve.png"), dpi=300)
plt.close()

# ==========================================
# Plot 6: Prediction confidence histogram
# ==========================================

plt.figure(figsize=(8, 6))
plt.hist(y_scores, bins=20)
plt.title("CNN Prediction Confidence Distribution")
plt.xlabel("Predicted Pneumonia Probability")
plt.ylabel("Number of Images")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cnn_prediction_confidence.png"), dpi=300)
plt.close()

# ==========================================
# Save trained model
# ==========================================

model.save("cnn_pneumonia_model.h5")

print("\nFiles saved:")
print("- cnn_pneumonia_model.h5")
print(f"- {os.path.join(output_dir, 'cnn_report.txt')}")
print(f"- {os.path.join(output_dir, 'dataset_class_distribution.png')}")
print(f"- {os.path.join(output_dir, 'cnn_accuracy_plot.png')}")
print(f"- {os.path.join(output_dir, 'cnn_loss_plot.png')}")
print(f"- {os.path.join(output_dir, 'cnn_confusion_matrix.png')}")
print(f"- {os.path.join(output_dir, 'cnn_roc_curve.png')}")
print(f"- {os.path.join(output_dir, 'cnn_precision_recall_curve.png')}")
print(f"- {os.path.join(output_dir, 'cnn_prediction_confidence.png')}")