import os
import numpy as np
from sklearn.utils import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# =============================
# Resolve dataset paths safely
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

print("Train dir:", TRAIN_DIR)
print("Test dir :", TEST_DIR)

# =============================
# Data generators
# =============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'
)

test_generator = test_datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'
)

print("\nClass indices:", train_generator.class_indices)
print(f"Found {train_generator.samples} training images")
print(f"Found {test_generator.samples} testing images")

# =============================
# Model architecture
# =============================
model = models.Sequential([

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =============================
# Compute class weights
# =============================
classes = train_generator.classes
class_labels = np.unique(classes)

weights = compute_class_weight(
    class_weight='balanced',
    classes=class_labels,
    y=classes
)

class_weights = dict(zip(class_labels, weights))

print("\nClass weights:")
for k, v in class_weights.items():
    print(f"Class {k}: {v:.3f}")

# =============================
# Callbacks
# =============================
callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, "best_model.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),

    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )
]


# =============================
# Training
# =============================
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    callbacks=callbacks,
    class_weight=class_weights
)

# =============================
# Save final model
# =============================
model.save(os.path.join(MODELS_DIR, "final_model.keras"))
print(f"\n{'='*50}")
print(f"✅ Final model saved to: {os.path.join(MODELS_DIR, 'final_model.keras')}")
print(f"✅ Best model saved to: {os.path.join(MODELS_DIR, 'best_model.keras')}")
print(f"{'='*50}")

# =============================
# Save history
# =============================
np.save(os.path.join(MODELS_DIR, "history.npy"), history.history)

# =============================
# Plot training history
# =============================
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "training_history.png"))
plt.show()

print("\nTraining complete.")
print("Best model saved to:", os.path.join(MODELS_DIR, "best_model.keras"))
print("Training history saved to:", os.path.join(MODELS_DIR, "history.npy"))
print("Training plot saved to:", os.path.join(MODELS_DIR, "training_history.png"))