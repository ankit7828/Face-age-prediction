import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

# =========================
# Paths
# =========================
train_dir = "dataset/train"
val_dir = "dataset/val"

IMG_SIZE = 224
BATCH_SIZE = 32

# =========================
# Data Generators (FIXED)
# =========================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Class mapping:", train_data.class_indices)

# =========================
# Model
# =========================
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze ALL layers first
for layer in base_model.layers:
    layer.trainable = False

# Custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

output = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# =========================
# Compile (LOW smoothing)
# =========================
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss=loss_fn,
    metrics=['accuracy']
)

# =========================
# Callbacks
# =========================
os.makedirs("models", exist_ok=True)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1),
    ModelCheckpoint("models/best_model.keras", monitor='val_loss', save_best_only=True, verbose=1)
]

# =========================
# Phase 1 (IMPORTANT)
# =========================
print("\n🚀 Phase 1 Training...\n")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=callbacks
)

# =========================
# Fine-Tuning (SAFE)
# =========================
print("\n🚀 Fine-Tuning...\n")

# Unfreeze ONLY last 20 layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Freeze BatchNorm (VERY IMPORTANT)
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),  # very low LR
    loss=loss_fn,
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks
)

# =========================
# Save
# =========================
model.save("models/final_model.keras")

print("\n✅ FINAL MODEL READY")