import cv2
import numpy as np
import tensorflow as tf
import os

# =============================
# Configuration
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")

# Emotion labels (must match training order)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Colors for visualization (BGR format)
COLORS = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 100, 0),     # Dark Green
    'fear': (128, 0, 128),      # Purple
    'happy': (0, 255, 0),       # Green
    'neutral': (255, 255, 255), # White
    'sad': (255, 0, 0),         # Blue
    'surprise': (0, 255, 255)   # Yellow
}

# =============================
# Load Model
# =============================
print("🔄 Loading emotion detection model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# =============================
# Load Face Detector
# =============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# =============================
# Helper function to preprocess face
# =============================
def preprocess_face(face_img):
    """
    Prepare face image for model prediction
    Input: face_img (grayscale, any size)
    Output: preprocessed array ready for model
    """
    # Resize to 48x48 (model input size)
    face_resized = cv2.resize(face_img, (48, 48))
    
    # Normalize pixel values (0-255 → 0-1)
    face_normalized = face_resized / 255.0
    
    # Reshape: (48, 48) → (1, 48, 48, 1)
    # Add batch dimension and channel dimension
    face_input = face_normalized.reshape(1, 48, 48, 1)
    
    return face_input

# =============================
# Start Webcam
# =============================
print("📹 Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam!")
    exit()

print("✅ Webcam started!")
print("\n" + "="*50)
print("CONTROLS:")
print("  Press 'q' to quit")
print("  Press 's' to save screenshot")
print("="*50 + "\n")

frame_count = 0

# =============================
# Main Loop
# =============================
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Failed to grab frame")
        break
    
    frame_count += 1
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Preprocess for model
        face_input = preprocess_face(face_roi)
        
        # Predict emotion
        prediction = model.predict(face_input, verbose=0)
        emotion_idx = np.argmax(prediction)
        emotion = EMOTIONS[emotion_idx]
        confidence = prediction[0][emotion_idx] * 100
        
        # Get color for this emotion
        color = COLORS[emotion]
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Prepare text
        text = f"{emotion}: {confidence:.1f}%"
        
        # Draw background for text (for better readability)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.rectangle(
            frame,
            (x, y - 35),
            (x + text_size[0] + 10, y),
            color,
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (x + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),  # Black text
            2
        )
        
        # Optional: Show all predictions (top 3)
        # Uncomment below if you want to see other emotion probabilities
        """
        top_3_idx = np.argsort(prediction[0])[-3:][::-1]
        y_offset = y + h + 25
        for idx in top_3_idx:
            prob_text = f"{EMOTIONS[idx]}: {prediction[0][idx]*100:.0f}%"
            cv2.putText(frame, prob_text, (x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        """
    
    # Add info text
    info_text = f"Faces detected: {len(faces)} | Press 'q' to quit | 's' to save"
    cv2.putText(
        frame,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    # Display frame
    cv2.imshow('Emotion Detection - Your First AI Project!', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n👋 Quitting...")
        break
    elif key == ord('s'):
        # Save screenshot
        screenshot_path = os.path.join(BASE_DIR, f"screenshot_{frame_count}.png")
        cv2.imwrite(screenshot_path, frame)
        print(f"📸 Screenshot saved: {screenshot_path}")

# =============================
# Cleanup
# =============================
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("✅ Webcam closed successfully!")
print("🎉 Great job completing your first AI/ML project!")
print("="*50)