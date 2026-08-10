# Phase 1 — Simulator & Synthetic Dataset pipeline
# Run in the Kaggle notebook with the CIFAR-10-C dataset added as input.

import os
import numpy as np
import torchvision

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
CIFAR10C_DIR = "/kaggle/input/datasets/harshadakhatu/cifar-10-c/CIFAR-10-C"
OUTPUT_DIR = "/kaggle/working/flowpost_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    "speckle_noise", "gaussian_blur", "spatter", "saturate",
]

N_CALIBRATION_PER_CORRUPTION = 10   # keep calibration set small (per corruption, per severity)
N_SEVERITIES = 5
IMAGES_PER_SEVERITY = 10000         # CIFAR-10-C convention: 50000 = 5 x 10000

RNG_SEED = 42


# ---------------------------------------------------------------------
# 1. Simulator domain: clean CIFAR-10 (abundant, used for PostNet training)
# ---------------------------------------------------------------------
def load_clean_cifar10():
    print("Downloading/loading clean CIFAR-10 (simulator domain)...")
    train_set = torchvision.datasets.CIFAR10(root="/kaggle/working/cifar10_clean",
                                              train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root="/kaggle/working/cifar10_clean",
                                             train=False, download=True)

    x_train = np.array(train_set.data)          # (50000, 32, 32, 3)
    y_train = np.array(train_set.targets)       # (50000,)
    x_test = np.array(test_set.data)             # (10000, 32, 32, 3)
    y_test = np.array(test_set.targets)          # (10000,)

    print(f"  clean train: {x_train.shape}, clean test: {x_test.shape}")
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------
# 2. Real domain: CIFAR-10-C -> stratified calibration set + held-out test set
#    theta = (corruption_type, severity)
# ---------------------------------------------------------------------
def build_real_domain_splits():
    print("\nBuilding CIFAR-10-C calibration + held-out test splits...")
    labels = np.load(os.path.join(CIFAR10C_DIR, "labels.npy"))  # (50000,), repeats every 10000

    calib_images, calib_labels, calib_theta = [], [], []
    test_images, test_labels, test_theta = [], [], []

    rng = np.random.default_rng(RNG_SEED)

    N_CLASSES = 10
    # With N_CALIBRATION_PER_CORRUPTION=10 and 10 classes, this samples exactly
    # 1 image per class per (corruption, severity) combo -> perfectly class-balanced
    # calibration set (95 images per class overall, since 19 corruptions x 5 severities = 95 combos).
    assert N_CALIBRATION_PER_CORRUPTION % N_CLASSES == 0, \
        "N_CALIBRATION_PER_CORRUPTION should be a multiple of 10 for even per-class stratification"
    per_class_per_combo = N_CALIBRATION_PER_CORRUPTION // N_CLASSES

    for corruption in CORRUPTIONS:
        arr = np.load(os.path.join(CIFAR10C_DIR, f"{corruption}.npy"), mmap_mode="r")
        for severity in range(1, N_SEVERITIES + 1):
            start = (severity - 1) * IMAGES_PER_SEVERITY
            end = severity * IMAGES_PER_SEVERITY
            sev_images = arr[start:end]
            sev_labels = labels[start:end]

            # Stratified sampling: pick `per_class_per_combo` images per class
            mask = np.zeros(IMAGES_PER_SEVERITY, dtype=bool)
            for cls in range(N_CLASSES):
                cls_idx = np.where(sev_labels == cls)[0]
                chosen = rng.choice(cls_idx, size=per_class_per_combo, replace=False)
                mask[chosen] = True

            calib_images.append(np.array(sev_images[mask]))
            calib_labels.append(sev_labels[mask])
            calib_theta.extend([(corruption, severity)] * N_CALIBRATION_PER_CORRUPTION)

            # Everything else goes to held-out test
            test_images.append(np.array(sev_images[~mask]))
            test_labels.append(sev_labels[~mask])
            test_theta.extend([(corruption, severity)] * (IMAGES_PER_SEVERITY - N_CALIBRATION_PER_CORRUPTION))

        print(f"  {corruption}: done")

    calib_images = np.concatenate(calib_images, axis=0)
    calib_labels = np.concatenate(calib_labels, axis=0)
    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    print(f"\n  Calibration set: {calib_images.shape} "
          f"({len(CORRUPTIONS)} corruptions x {N_SEVERITIES} severities x {N_CALIBRATION_PER_CORRUPTION})")
    print(f"  Held-out test set: {test_images.shape}")

    return (calib_images, calib_labels, calib_theta), (test_images, test_labels, test_theta)


# ---------------------------------------------------------------------
# 3. Save everything
# ---------------------------------------------------------------------
def main():
    x_train, y_train, x_test_clean, y_test_clean = load_clean_cifar10()
    (calib_x, calib_y, calib_theta), (real_test_x, real_test_y, real_test_theta) = build_real_domain_splits()

    np.savez(os.path.join(OUTPUT_DIR, "simulator_domain.npz"),
              x_train=x_train, y_train=y_train,
              x_test=x_test_clean, y_test=y_test_clean)

    np.savez(os.path.join(OUTPUT_DIR, "calibration_set.npz"),
              x=calib_x, y=calib_y, theta=np.array(calib_theta, dtype=object))

    np.savez(os.path.join(OUTPUT_DIR, "real_held_out_test.npz"),
              x=real_test_x, y=real_test_y, theta=np.array(real_test_theta, dtype=object))

    print(f"\nAll saved to {OUTPUT_DIR}:")
    for f in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, f)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {f}  ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
