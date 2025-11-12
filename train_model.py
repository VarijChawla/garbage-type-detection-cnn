# ==========================================
# Waste Classification using MobileNetV2 (Optimized for CPU)
# ==========================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ==========================================
# STEP 1: CONFIGURATION
# ==========================================
DATA_DIR = "D:/Projects/waste-segregation-system/data/raw"          # Path to your main dataset folder
IMG_SIZE = 128             # Smaller size for faster processing
BATCH_SIZE = 32
EPOCHS = 10                # Reduced for faster testing
LEARNING_RATE = 1e-4

# ==========================================
# STEP 2: DATA LOADING (with train/val split)
# ==========================================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

num_classes = len(train_gen.class_indices)
print("Classes:", train_gen.class_indices)

# ==========================================
# STEP 3: MODEL SETUP (MobileNetV2)
# ==========================================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# ==========================================
# STEP 4: CALLBACKS (Early Stop + Checkpoint)
# ==========================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_mobilenet_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# ==========================================
# STEP 5: TRAINING
# ==========================================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# ==========================================
# STEP 6: SAVE FINAL MODEL
# ==========================================
model.save("final_mobilenet_model.keras")
print("✅ Model training complete and saved as final_mobilenet_model.keras")

# ==========================================
# STEP 7: EVALUATION
# ==========================================
loss, acc = model.evaluate(val_gen)
print(f"\nValidation Accuracy: {acc*100:.2f}%")
