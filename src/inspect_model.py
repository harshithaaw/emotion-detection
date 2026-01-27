import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")

print("="*60)
print("INSPECTING YOUR SAVED MODEL")
print("="*60)

# Load model
print("\n🔍 Loading best_model.keras...")
model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "best_model.keras"))
print("✅ Model loaded successfully!")

# Model info
print(f"\n📊 Model Information:")
print(f"   Total parameters: {model.count_params():,}")
print(f"   Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# Create test generator
print(f"\n📁 Loading test data...")
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

# Evaluate
print(f"\n🧪 Evaluating on test set...")
print("   (This will take a minute...)")
loss, accuracy = model.evaluate(test_generator, verbose=0)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"📉 Test Loss: {loss:.4f}")
print(f"🎯 Test Accuracy: {accuracy*100:.2f}%")
print("="*60)

# Performance interpretation
print(f"\n💡 What does this mean?")
if accuracy > 0.60:
    print("   ✅ EXCELLENT! This is very good for FER2013 dataset!")
elif accuracy > 0.50:
    print("   ✅ GOOD! Above average for this difficult dataset.")
elif accuracy > 0.40:
    print("   ⚠️  MODERATE. Usable but has room for improvement.")
else:
    print("   ❌ LOW. Model needs significant improvements.")

print(f"\n   For context:")
print(f"   - Random guessing: ~14% (7 classes)")
print(f"   - Your model: {accuracy*100:.2f}%")
print(f"   - State-of-the-art: ~70%")

# Get some predictions to see behavior
print(f"\n🔮 Testing prediction on sample batch...")
x_batch, y_batch = next(test_generator)
predictions = model.predict(x_batch, verbose=0)

class_names = list(test_generator.class_indices.keys())
confidences = np.max(predictions, axis=1)
avg_confidence = np.mean(confidences) * 100

print(f"   Average prediction confidence: {avg_confidence:.1f}%")

if avg_confidence > 70:
    print("   → Model is very confident (might be overfitting)")
elif avg_confidence > 40:
    print("   → Reasonable confidence levels")
else:
    print("   → Low confidence (model is uncertain)")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

if accuracy >= 0.40:
    print("✅ This model is usable for your webcam demo!")
    print("   Proceed to real-time emotion detection.")
    print("   You can always improve it later.")
else:
    print("⚠️  Consider retraining with improvements:")
    print("   - More epochs")
    print("   - Different architecture")
    print("   - Better data augmentation")

print("="*60)