import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =============================
# Setup paths
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# =============================
# Load the best model
# =============================
model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'best_model.keras'))
print("✅ Model loaded successfully!")

# =============================
# Create test data generator
# =============================
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False  # Important for evaluation!
)

# =============================
# Evaluate the model
# =============================
print("\n" + "="*50)
print("Evaluating model on test set...")
print("="*50)

loss, accuracy = model.evaluate(test_generator, verbose=1)

print("\n" + "="*50)
print(f"📊 Test Loss: {loss:.4f}")
print(f"🎯 Test Accuracy: {accuracy*100:.2f}%")
print("="*50)

# =============================
# Show predictions on a few samples
# =============================
print("\nGetting predictions on test samples...")

# Get one batch
x_batch, y_batch = next(test_generator)

# Predict
predictions = model.predict(x_batch)

# Get class names
class_names = list(test_generator.class_indices.keys())

print("\n" + "="*50)
print("Sample Predictions (first 5 images):")
print("="*50)

for i in range(min(5, len(predictions))):
    true_label = class_names[np.argmax(y_batch[i])]
    pred_label = class_names[np.argmax(predictions[i])]
    confidence = np.max(predictions[i]) * 100
    
    status = "✅" if true_label == pred_label else "❌"
    print(f"{status} True: {true_label:8s} | Predicted: {pred_label:8s} | Confidence: {confidence:.1f}%")

print("="*50)
