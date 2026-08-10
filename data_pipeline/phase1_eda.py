# Phase 1 — EDA
# Run in the same Kaggle notebook where flowpost_data/ was generated
# (or with the flowpost-phase1-data dataset added as input).

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

DATA_DIR = "/kaggle/working/flowpost_data"   # change to /kaggle/input/flowpost-phase1-data if loading from the published dataset

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


# ---------------------------------------------------------------------
# 1. Load everything
# ---------------------------------------------------------------------
sim = np.load(os.path.join(DATA_DIR, "simulator_domain.npz"), allow_pickle=True)
calib = np.load(os.path.join(DATA_DIR, "calibration_set.npz"), allow_pickle=True)
test = np.load(os.path.join(DATA_DIR, "real_held_out_test.npz"), allow_pickle=True)

x_train, y_train = sim["x_train"], sim["y_train"]
x_test_clean, y_test_clean = sim["x_test"], sim["y_test"]

calib_x, calib_y, calib_theta = calib["x"], calib["y"], calib["theta"]
test_x, test_y, test_theta = test["x"], test["y"], test["theta"]

print("=== Shapes ===")
print(f"Clean train:      {x_train.shape}")
print(f"Clean test:        {x_test_clean.shape}")
print(f"Calibration set:   {calib_x.shape}")
print(f"Held-out test set: {test_x.shape}\n")


# ---------------------------------------------------------------------
# 2. Class balance check
# ---------------------------------------------------------------------
def print_class_balance(name, y):
    counts = Counter(y.tolist())
    print(f"--- {name} ---")
    for c in range(10):
        print(f"  {CLASS_NAMES[c]:12s}: {counts.get(c, 0)}")
    print()

print("=== Class Balance ===")
print_class_balance("Clean train (simulator)", y_train)
print_class_balance("Clean test", y_test_clean)
print_class_balance("Calibration set (real domain)", calib_y)
print_class_balance("Held-out test set (real domain)", test_y)


# ---------------------------------------------------------------------
# 3. Theta distribution check (calibration set)
# ---------------------------------------------------------------------
print("=== Theta (corruption_type, severity) distribution — calibration set ===")
theta_counts = Counter([tuple(t) for t in calib_theta])
corruptions = sorted(set(t[0] for t in theta_counts))
severities = sorted(set(t[1] for t in theta_counts))

print(f"{'Corruption':20s} | " + " | ".join(f"Sev {s}" for s in severities))
for c in corruptions:
    row = [str(theta_counts.get((c, s), 0)) for s in severities]
    print(f"{c:20s} | " + " | ".join(f"{r:5s}" for r in row))
print()


# ---------------------------------------------------------------------
# 4. Visual sanity check — sample images
# ---------------------------------------------------------------------
def show_samples(images, labels, title, n=8, thetas=None):
    fig, axes = plt.subplots(1, n, figsize=(n * 1.6, 2))
    idx = np.random.choice(len(images), n, replace=False)
    for ax, i in zip(axes, idx):
        ax.imshow(images[i])
        label_str = CLASS_NAMES[labels[i]]
        if thetas is not None:
            c, s = thetas[i]
            label_str += f"\n{c}\nsev {s}"
        ax.set_title(label_str, fontsize=7)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(f"/kaggle/working/{title.replace(' ', '_').lower()}.png", dpi=120)
    plt.show()

np.random.seed(0)
show_samples(x_train, y_train, "Clean CIFAR-10 (simulator domain)")
show_samples(calib_x, calib_y, "Calibration set (real domain, corrupted)", thetas=calib_theta)
show_samples(test_x, test_y, "Held-out test set (real domain, corrupted)", thetas=test_theta)

# Severity progression for a single corruption type, side by side
def show_severity_progression(corruption_name="gaussian_noise"):
    matches = [i for i, t in enumerate(calib_theta) if t[0] == corruption_name]
    matches_by_sev = {}
    for i in matches:
        sev = calib_theta[i][1]
        matches_by_sev.setdefault(sev, i)  # just grab one example per severity

    fig, axes = plt.subplots(1, len(matches_by_sev), figsize=(len(matches_by_sev) * 2, 2.2))
    for ax, sev in zip(axes, sorted(matches_by_sev)):
        i = matches_by_sev[sev]
        ax.imshow(calib_x[i])
        ax.set_title(f"severity {sev}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Severity progression: {corruption_name}", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"/kaggle/working/severity_progression_{corruption_name}.png", dpi=120)
    plt.show()

show_severity_progression("gaussian_noise")
show_severity_progression("fog")

print("\nEDA complete. Sample plots saved to /kaggle/working/ as PNGs.")
