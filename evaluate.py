# evaluate.py
# Smart Recycling Bin — Evaluation Script
# Run after train.py to get full metrics on the test set.
#
# USAGE:
#   python evaluate.py
#   python evaluate.py --image path/to/image.jpg
#
# REQUIRES: waste_classifier.keras (produced by train.py)

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# CONFIG — must match train.py exactly
# ============================================================
USER_DATASET_DIR = r'AI model data\dataset'
CATEGORIES       = ['organic', 'paper', 'plastic', 'cans', 'battery']
IMG_SIZE         = 224
BATCH_SIZE       = 16
MODEL_PATH       = "waste_classifier.keras"

# Temperature scaling
# MobileNetV2 produces well-calibrated confidence on black-background images.
# Start at 1.0 (off). If avg correct confidence exceeds 0.95, raise to 1.1 or 1.2.
TEMPERATURE = 1.2
# ============================================================


def apply_temperature(probs, temperature):
    if temperature == 1.0:
        return probs
    eps = 1e-7
    log_p  = np.log(np.clip(probs, eps, 1.0))
    scaled = log_p / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_s  = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def evaluate_test_set(model):
    test_dir = os.path.join(USER_DATASET_DIR, 'test')
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return

    # MobileNetV2 requires rescale=1./255
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    if test_gen.samples == 0:
        print("No images found in test set.")
        return

    print(f"\n  Found {test_gen.samples} test images")
    print(f"  Class order: {test_gen.class_indices}")
    print(f"\n  Running predictions...")

    raw_preds = model.predict(test_gen, verbose=1)
    preds     = apply_temperature(raw_preds, TEMPERATURE)

    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    idx_to_name  = {v: k for k, v in test_gen.class_indices.items()}
    target_names = [idx_to_name.get(i, f"class_{i}") for i in range(len(test_gen.class_indices))]

    # ── Classification report ────────────────────────────────
    print("\n" + "=" * 55)
    print("CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=target_names))

    # ── Per-class accuracy ───────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    print("=" * 55)
    print("PER-CLASS ACCURACY")
    print("=" * 55)
    for i, name in enumerate(target_names):
        bar  = '█' * int(per_class_acc[i] * 20)
        flag = "  ⚠️" if per_class_acc[i] < 0.90 else ""
        print(f"  {name:<12} {per_class_acc[i]*100:5.1f}%  {bar}{flag}")

    # ── Confusion matrix text ────────────────────────────────
    print("\n" + "=" * 55)
    print("CONFUSION MATRIX")
    print("=" * 55)
    header = "            " + "  ".join(f"{n[:6]:>7}" for n in target_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {target_names[i]:<12} " + "  ".join(f"{v:>7}" for v in row))

    # ── Key confusion pairs ──────────────────────────────────
    print()
    name_to_idx = {name: i for i, name in enumerate(target_names)}

    def confusion_pair(a, b):
        ai, bi = name_to_idx.get(a), name_to_idx.get(b)
        if ai is None or bi is None:
            return
        ab, ba = cm[ai, bi], cm[bi, ai]
        ta, tb = cm[ai].sum(), cm[bi].sum()
        print(f"  🔍 {a}→{b}: {ab} ({ab/ta*100:.1f}%)   {b}→{a}: {ba} ({ba/tb*100:.1f}%)")

    confusion_pair('plastic', 'paper')
    confusion_pair('battery', 'cans')
    confusion_pair('organic', 'paper')

    # ── Battery recall ───────────────────────────────────────
    bat_idx = name_to_idx.get('battery')
    if bat_idx is not None:
        total   = cm[bat_idx].sum()
        correct = cm[bat_idx, bat_idx]
        missed  = total - correct
        can_idx = name_to_idx.get('cans')
        as_can  = cm[bat_idx, can_idx] if can_idx is not None else 0
        print(f"\n  🔋 BATTERY DETECTION (safety critical)")
        print(f"     Detected correctly: {correct}/{total}")
        print(f"     Missed:             {missed}")
        print(f"     Confused with cans: {as_can}")
        if total > 0:
            recall = correct / total * 100
            flag   = "  ✅" if recall >= 90 else "  ⚠️  Collect more battery images"
            print(f"     Recall:             {recall:.1f}%{flag}")

    # ── Confidence analysis ──────────────────────────────────
    print("\n" + "=" * 55)
    print("CONFIDENCE ANALYSIS")
    print("=" * 55)
    max_conf   = np.max(preds, axis=1)
    wrong_mask = y_pred != y_true
    hc_wrong   = np.sum(wrong_mask & (max_conf > 0.8))

    print(f"  Total:                      {len(y_true)}")
    print(f"  Correct:                    {(~wrong_mask).sum()}")
    print(f"  Wrong:                      {wrong_mask.sum()}")
    print(f"  Wrong with confidence >80%: {hc_wrong}  ← most dangerous")
    if (~wrong_mask).any():
        avg_c = np.mean(max_conf[~wrong_mask])
        print(f"  Avg confidence (correct):   {avg_c:.3f}")
        if avg_c > 0.95:
            print(f"  ℹ️  Very high confidence — consider setting TEMPERATURE to 1.1")
    if wrong_mask.any():
        avg_w = np.mean(max_conf[wrong_mask])
        print(f"  Avg confidence (wrong):     {avg_w:.3f}")
        if avg_w > 0.70:
            print("  ⚠️  High confidence on errors — raise TEMPERATURE")
        else:
            print("  ✅ Wrong predictions have low confidence — threshold will catch them")

    print(f"\n  ℹ️  Temperature: T={TEMPERATURE} | Threshold in result_handler.py: 0.60")

    plot_confusion_matrix(cm, target_names)


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=11)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  📊 confusion_matrix.png saved")


def test_single_image(model, image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read: {image_path}")
        return
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # MobileNetV2: rescale to 0-1
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    raw  = model.predict(image, verbose=0)[0]
    pred = apply_temperature(raw[np.newaxis, :], TEMPERATURE)[0]

    print(f"\n  Image: {os.path.basename(image_path)}")
    print(f"  {'─'*40}")
    for rank, idx in enumerate(np.argsort(pred)[::-1], 1):
        bar    = '█' * int(pred[idx] * 30)
        name   = CATEGORIES[idx] if idx < len(CATEGORIES) else f"class_{idx}"
        marker = " ← predicted" if rank == 1 else ""
        print(f"  {rank}. {name:<12} {pred[idx]*100:5.1f}%  {bar}{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--model', type=str, default=MODEL_PATH)
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("SMART BIN CLASSIFIER — EVALUATION (MobileNetV2)")
    print("=" * 55)

    if not os.path.exists(args.model):
        print(f"\nERROR: Model not found at '{args.model}'")
        print("  Run train.py first.")
        sys.exit(1)

    print(f"\n  Loading: {args.model}")
    model = tf.keras.models.load_model(args.model)
    print("  ✅ Model loaded")

    if args.image:
        test_single_image(model, args.image)
    else:
        evaluate_test_set(model)


if __name__ == "__main__":
    main()