# Emotion Detection System
Real-time facial emotion detection using Deep Learning and OpenCV.

1. Project Overview

This project detects 7 emotions from grayscale images based on facial expressions in real-time using a Convolutional Neural Network (CNN) trained on the FER2013 dataset.

**Emotions Detected:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

2. Dataset

    Source: FER2013

    Training images: 28,709

    Test images: 7,178

    Image size: 48×48 grayscale

3. Features

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

## 📁 Project Structure
```
emotion-detection/
├── data/
│   ├── train/          # Training images (7 emotion folders)
│   └── test/           # Test images (7 emotion folders)
├── models/
│   ├── best_model.keras      # Trained model
│   └── training_history.png  # Training visualization (if available)
├── src/
│   ├── train_model.py        # Model training script
│   ├── load_and_test.py      # Test model and quick evaluation
│   ├── inspect_model.py      # Deep dive analysis with context and recommendations
│   ├── real_time_emotion.py  # Real-time detection
│   ├── face_detection.py     # Face detection testing
│   └── test_cama.py        # Webcam testing
├── notebooks/
│   └── explore_data.ipynb    # Data exploration
└── README.md

## 🚀 How to Run

## Prerequisites

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn

```markdown
### 1. Train the Model (Optional)

```bash
python src/train_model.py

```markdown

### 2. Run Real-Time Detection
```bash
python src/real_time_emotion.py

**Controls:**
- Press `q` to quit
- Press `s` to save screenshot


3. Data Preprocessing

    Rescaling (1/255)

    Data augmentation:

        rotation
        shifting
        flipping
        zoom

4. Class Imbalance Handling

    Severe imbalance (largest/smallest ≈ 16.5:1) 
    We notice a huge difference between the input images of largest emotion and smallest which causes huge data imbalance

    Used class weights to reduce bias

5. Model Architecture

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

6. Training Strategy

    Optimizer: Adam
    Loss: categorical crossentropy
    Batch size: 32
    Epochs: 50 (EarlyStopping used)
    Class weights: Balanced to handle class imbalance
    Callbacks:
        ModelCheckpoint
        EarlyStopping
        ReduceLROnPlateau

7. Evaluation Metrics

    **Primary Metric:**
    - Accuracy: 47.69%

    **Future Metrics to Implement:**
    - Confusion Matrix
    - Per-class Precision, Recall, F1-score
    - Classification Report

## 🔍 Face Detection: Haar Cascade

This project uses **Haar Cascade Classifier** for face detection.

**What is Haar Cascade?**
- A machine learning-based object detection method developed by Paul Viola and Michael Jones
- Uses "Haar-like features" (rectangular patterns) to detect faces
- Employs a cascade of classifiers for fast detection
- Pre-trained model provided by OpenCV: `haarcascade_frontalface_default.xml`

**Why Haar Cascade?**
- Fast enough for real-time detection on CPU
- No GPU required
- Simple to implement
- Good accuracy for frontal faces

**Limitations:**
- Works best with frontal faces (not side profiles)
- Can produce false positives in complex backgrounds
- Less accurate than modern deep learning methods (MTCNN, RetinaFace)

**Detection Parameters Used:**
python
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,    # Image pyramid scaling
    minNeighbors=5,     # Minimum neighbors for valid detection
    minSize=(30, 30)    # Minimum face size
)

8. Results

    **Test Performance:**
    - Test Accuracy: 47.69%
    - Test Loss: 1.3779

    **Observations:**
    - Model shows reasonable performance for a challenging dataset
    - Confusion observed between similar emotions (angry ↔ sad)
    - Average prediction confidence: ~35-40%
    - Model correctly predicts 3 out of 5 samples in test batch

    **Overfitting/Underfitting Analysis:**
    - Model achieved moderate generalization
    - Training stopped early (likely before 50 epochs due to EarlyStopping)
    - Balanced approach - neither severe overfitting nor underfitting
    - Class imbalance handling with weights helped improve minority class performance

    **Per-Emotion Performance (Estimated based on testing):**
    - Best: Happy, Surprise (clear expressions)
    - Moderate: Angry, Neutral, Sad
    - Challenging: Fear, Disgust (fewer training samples, similar to other emotions)

9. Limitations

    Dataset imbalance
    Confusion between similar emotions
    Low resolution images
    Difficult dataset to implement

10. Future Improvements

- [ ] Try transfer learning (VGG16, ResNet)
- [ ] Implement ensemble models
- [ ] Add emotion history tracking (smooth predictions)
- [ ] Deploy as web application
- [ ] Fine-tune on custom dataset
- [ ] Add audio-based emotion detection

```markdown
---

## Author
Harshitha Rayudu
