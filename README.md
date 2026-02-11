# Emotion Detection System
Real-time facial emotion detection using Deep Learning and OpenCV.

## Project Overview

This project detects 7 emotions from grayscale images based on facial expressions in real-time using a Convolutional Neural Network (CNN) trained on the FER2013 dataset.

**Emotions Detected:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

## Dataset
    Source: FER2013

    Training images: 28,709

    Test images: 7,178

    Image size: 48×48 grayscale

## Features

- Real-time emotion detection from webcam feed
- Face detection using Haar Cascades
- Custom CNN architecture with BatchNormalization
- Data augmentation to prevent overfitting
- Model checkpointing and early stopping
- Callbacks
- Color-coded emotion display

## 📊 Results

- **Test Accuracy:** 47.69%
- **Model Parameters:** 1,146,247

*Note: FER2013 is a challenging dataset with noisy labels. Human agreement is typically 65-70%, and state-of-the-art models achieve ~70% accuracy.*
# 📁 Project Structure

```
emotion-detection/
├── data/
│   ├── train/                  # Training images (7 emotion folders)
│   └── test/                   # Test images (7 emotion folders)
├── models/
│   ├── best_model.keras        # Trained model
│   └── training_history.png    # Training visualization
├── src/
│   ├── train_model.py          # Model training script
│   ├── load_and_test.py        # Model evaluation script
│   ├── inspect_model.py        # Deep analysis and recommendations
│   ├── real_time_emotion.py    # Real-time emotion detection
│   ├── face_detection.py       # Face detection testing
│   └── test_cama.py            # Webcam testing
├── notebooks/
│   └── explore_data.ipynb      # Data exploration notebook
└── README.md
```

---

# 🚀 How to Run

## 📦 Prerequisites

Install required dependencies:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

---

## ▶️ 1. Train the Model (Optional)

```bash
python src/train_model.py
```

---

## ▶️ 2. Run Real-Time Emotion Detection

```bash
python src/real_time_emotion.py
```

### 🎮 Controls
- Press `q` → Quit  
- Press `s` → Save screenshot  

---

# 🧠 Model Details

---

## 1️⃣ Data Preprocessing

- Rescaling pixel values (1/255)
- Data Augmentation:
  - Rotation
  - Width/Height Shifting
  - Horizontal Flipping
  - Zoom

---

## 2️⃣ Class Imbalance Handling

- Severe imbalance (largest/smallest ≈ 16.5:1)
- Used **class weights** during training to reduce bias toward majority classes
- Helped improve minority emotion performance

---

## 3️⃣ Model Architecture

```
Input (48x48x1 grayscale image)
    ↓
Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten
    ↓
Dense(512) → Dropout(0.5)
    ↓
Dense(7, softmax) → Output probabilities
```

---

## 4️⃣ Training Strategy

- Optimizer: **Adam**
- Loss Function: **Categorical Crossentropy**
- Batch Size: 32
- Epochs: 50 (EarlyStopping enabled)
- Class Weights: Enabled
- Callbacks Used:
  - ModelCheckpoint
  - EarlyStopping
  - ReduceLROnPlateau

---

## 5️⃣ Evaluation Metrics

### 📊 Primary Metric
- Test Accuracy: **47.69%**
- Test Loss: **1.3779**

### 📌 Planned Metrics
- Confusion Matrix
- Per-class Precision
- Recall
- F1-score
- Full Classification Report

---

# 🔍 Face Detection (Haar Cascade)

This project uses **Haar Cascade Classifier** for face detection.

## 📖 What is Haar Cascade?

- Machine learning-based object detection method
- Developed by Paul Viola and Michael Jones
- Uses Haar-like rectangular features
- Cascade of classifiers for fast detection
- Pre-trained model from OpenCV:
  ```
  haarcascade_frontalface_default.xml
  ```

## ✅ Why Haar Cascade?

- Fast enough for real-time detection on CPU
- No GPU required
- Easy to implement
- Works well for frontal faces

## ⚠️ Limitations

- Best with frontal faces
- Can produce false positives
- Less accurate than modern deep learning detectors (MTCNN, RetinaFace)

---

## 🔧 Detection Parameters Used

```python
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)
```

---

# 📊 Results

## 🧪 Test Performance

- Test Accuracy: **47.69%**
- Test Loss: **1.3779**
- Average prediction confidence: ~35–40%

## 📌 Observations

- Confusion between similar emotions (angry ↔ sad)
- Happy and Surprise perform best
- Fear and Disgust are most challenging
- Model correctly predicts ~3 out of 5 samples in test batches

## 📉 Overfitting / Underfitting Analysis

- Moderate generalization
- EarlyStopping prevented overfitting
- Balanced approach overall
- Class weighting improved minority class performance

---

# ⚠️ Limitations

- Dataset imbalance
- Low-resolution images (48x48)
- Emotion overlap (similar facial expressions)
- Challenging dataset

---

# 🚀 Future Improvements

- [ ] Apply Transfer Learning (VGG16, ResNet)
- [ ] Build Ensemble Models
- [ ] Add Emotion History Tracking (Prediction smoothing)
- [ ] Deploy as Web Application
- [ ] Fine-tune on Custom Dataset
- [ ] Add Audio-based Emotion Detection
- [ ] Implement Confusion Matrix Visualization
- [ ] Improve per-class F1 score

---

# 👩‍💻 Author

**Harshitha Rayudu**

---

# ⭐ If you found this project helpful, please consider giving it a star!
