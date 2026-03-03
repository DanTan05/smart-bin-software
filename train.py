# train.py
# Smart Recycling Bin — Training Script
# Model: MobileNetV2
# Chosen over MobileNetV3Small because:
#   - Controlled black-background bin environment suits V2's larger feature extractor
#   - Higher confidence on in-distribution images (0.85-0.95 vs 0.65-0.75)
#   - 50ms speed difference on Pi 5 is irrelevant for bin use case
#   - Reliability matters more than inference speed here
#
# USAGE:
#   python train.py
#
# OUTPUTS:
#   waste_classifier.keras   — full model (used by evaluate.py)
#   waste_classifier.tflite  — quantized model (copy to inference/ folder)
#   processed_general/       — auto-created, delete before rerunning if data changed

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
USER_DATASET_DIR  = r'AI model data\dataset'
CATEGORIES        = ['organic', 'paper', 'plastic', 'cans', 'battery']
IMG_SIZE          = 224
BATCH_SIZE        = 16
EPOCHS            = 20
FINE_TUNE_EPOCHS  = 10
OVERSAMPLE_FACTOR = 1

LABEL_SMOOTHING           = 0.1
DROPOUT_RATE              = 0.4
IMBALANCE_THRESHOLD       = 2.5
OVERFITTING_GAP_THRESHOLD = 0.10
# ============================================================


def prepare_dataset():
    processed_dir = './processed_general/'
    for split in ['train', 'val']:
        for cat in CATEGORIES:
            os.makedirs(os.path.join(processed_dir, split, cat), exist_ok=True)

    for split in ['train', 'val']:
        for cat in CATEGORIES:
            src_dir = os.path.join(USER_DATASET_DIR, split, cat)
            if not os.path.exists(src_dir):
                print(f"  Warning: {src_dir} not found — skipping.")
                continue
            images = [f for f in os.listdir(src_dir)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not images:
                continue
            dest_dir = os.path.join(processed_dir, split, cat)
            for i, fname in enumerate(images):
                src = os.path.join(src_dir, fname)
                for dup in range(OVERSAMPLE_FACTOR):
                    shutil.copy(src, os.path.join(dest_dir, f"{i}_dup{dup}_{fname}"))
    print("  Dataset prepared.\n")


def check_class_imbalance():
    counts = {}
    for cat in CATEGORIES:
        d = os.path.join(USER_DATASET_DIR, 'train', cat)
        counts[cat] = len([f for f in os.listdir(d)
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))]) \
                      if os.path.exists(d) else 0

    print("=" * 50)
    print("CLASS IMBALANCE CHECK (train split)")
    print("=" * 50)
    total = sum(counts.values())
    for cat, n in counts.items():
        pct = (n / total * 100) if total > 0 else 0
        print(f"  {cat:<12} {n:>5} images  ({pct:5.1f}%)  {'█' * int(pct / 2)}")

    valid = [v for v in counts.values() if v > 0]
    if not valid:
        return None

    ratio = max(valid) / max(min(valid), 1)
    if ratio > IMBALANCE_THRESHOLD:
        print(f"\n⚠️  Imbalance detected — ratio={ratio:.1f}x. Applying weighted loss.")
        class_weights = {
            i: total / (len(CATEGORIES) * max(counts[cat], 1))
            for i, cat in enumerate(CATEGORIES)
        }
        print("  Weights:", {CATEGORIES[k]: f"{v:.2f}" for k, v in class_weights.items()})
        return class_weights
    else:
        print(f"\n✅ Balanced (ratio={ratio:.1f}x). No weighted loss needed.")
        return None


class OverfitDetector(Callback):
    def __init__(self):
        super().__init__()
        self.history_log = []

    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy', 0)
        val_acc   = logs.get('val_accuracy', 0)
        gap       = train_acc - val_acc
        self.history_log.append(gap)
        if gap > OVERFITTING_GAP_THRESHOLD:
            print(f"\n  ⚠️  Overfit gap at epoch {epoch+1}: "
                  f"train={train_acc:.3f} val={val_acc:.3f} gap={gap:.3f}")


class Top3Callback(Callback):
    def __init__(self, val_gen):
        super().__init__()
        self.val_gen = val_gen

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 != 0:
            return
        batch_x, batch_y = next(iter(self.val_gen))
        preds = self.model.predict(batch_x[:4], verbose=0)
        print(f"\n  --- Top-3 probs (val sample, epoch {epoch+1}) ---")
        for i, pred in enumerate(preds):
            top3 = np.argsort(pred)[::-1][:3]
            true = CATEGORIES[np.argmax(batch_y[i])]
            out  = "  ".join(f"{CATEGORIES[j]}={pred[j]:.2f}" for j in top3)
            print(f"    [{i+1}] True={true:<10} {out}")


def build_model(num_classes):
    """
    MobileNetV2 — 3.4M params, ~3MB TFLite quantized.
    Requires manual rescale=1./255 in ImageDataGenerator (no built-in preprocessing).
    Dense head: 512 units.
    Phase 2 unfreezes last 30 layers.
    """
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    if DROPOUT_RATE > 0:
        x = Dropout(DROPOUT_RATE)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=out)
    return model, base


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in [
        (axes[0], 'accuracy',  'Accuracy'),
        (axes[1], 'loss',      'Loss')
    ]:
        ax.plot(history[key],         label='Train', linewidth=2)
        ax.plot(history[f'val_{key}'], label='Val',  linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  📊 training_curves.png saved")


def save_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open("waste_classifier.tflite", "wb") as f:
        f.write(tflite_model)
    print(f"  ✅ waste_classifier.tflite saved ({len(tflite_model)/1024:.1f} KB)")
    print(f"     → Copy this file to your inference/ folder")


def train():
    print("\n" + "=" * 60)
    print("SMART BIN CLASSIFIER — TRAINING (MobileNetV2)")
    print("=" * 60)

    print("\n📁 Dataset overview:")
    for split in ['train', 'val', 'test']:
        for cat in CATEGORIES:
            d = os.path.join(USER_DATASET_DIR, split, cat)
            if os.path.exists(d):
                n = len([f for f in os.listdir(d)
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                print(f"  {split}/{cat}: {n} images")

    print()
    class_weights = check_class_imbalance()

    print("\n📂 Preparing dataset...")
    if os.path.exists('./processed_general/'):
        print("  processed_general/ exists — using it.")
        print("  ⚠️  Added new images? Delete processed_general/ first.")
    else:
        prepare_dataset()

    # MobileNetV2 requires rescale=1./255 — do not remove
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        './processed_general/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    val_gen = val_datagen.flow_from_directory(
        './processed_general/val',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    print(f"\n  Class indices: {train_gen.class_indices}")
    print(f"  ⚠️  Verify these match model_interface.py self.classes list order")

    if train_gen.samples == 0 or val_gen.samples == 0:
        print("ERROR: No images found. Check USER_DATASET_DIR.")
        return

    num_classes = len(CATEGORIES)
    model, base = build_model(num_classes)
    print(f"\n  Model: MobileNetV2 | Parameters: {model.count_params():,}")

    loss_fn      = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    overfit_cb   = OverfitDetector()
    callbacks    = [
        overfit_cb,
        Top3Callback(val_gen),
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, verbose=1, min_lr=1e-6)
    ]

    print("\n" + "=" * 50)
    print("PHASE 1 — Training head (base frozen)")
    print("=" * 50)
    model.compile(optimizer=Adam(0.0005), loss=loss_fn, metrics=['accuracy'])
    h1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=Adam(0.0001), loss=loss_fn, metrics=['accuracy'])

    print("\n" + "=" * 50)
    print("PHASE 2 — Fine-tuning last 30 layers")
    print("=" * 50)
    h2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    gaps = overfit_cb.history_log
    if gaps:
        print(f"\n  Peak gap:          {max(gaps):.3f}")
        print(f"  Avg gap (last 5):  {np.mean(gaps[-5:]):.3f}")
        if np.mean(gaps[-5:]) > OVERFITTING_GAP_THRESHOLD:
            print("  ⚠️  Overfitting — consider more data or higher dropout")
        else:
            print("  ✅ Gap within acceptable range")

    combined = {
        'accuracy':     h1.history['accuracy']     + h2.history['accuracy'],
        'val_accuracy': h1.history['val_accuracy'] + h2.history['val_accuracy'],
        'loss':         h1.history['loss']         + h2.history['loss'],
        'val_loss':     h1.history['val_loss']     + h2.history['val_loss'],
    }
    plot_training_curves(combined)

    print("\n💾 Saving models...")
    model.save("waste_classifier.keras")
    print("  ✅ waste_classifier.keras saved")
    save_tflite(model)

    print("\n✅ Training complete.")
    print("   → Run: python evaluate.py")
    print("   → Copy waste_classifier.tflite to inference/ folder")


if __name__ == "__main__":
    train()