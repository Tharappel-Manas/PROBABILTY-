"""
sample_run.py

Creates a tiny deterministic CIFAR-10 corruption dataset, trains PostNet briefly,
and evaluates on a held-out corrupted split.
"""

import argparse
import io
import os

import numpy as np
import torch
import torchvision
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader

from postnet.phase2_postnet import CIFARDataset, DEVICE, PostNet, uce_loss

CORRUPTIONS = ("brightness", "contrast", "gaussian_blur", "motion_blur", "jpeg")
SEVERITIES = (1, 2, 3)

BRIGHTNESS_FACTORS = {1: 0.8, 2: 0.65, 3: 0.5}
CONTRAST_FACTORS = {1: 0.75, 2: 0.55, 3: 0.4}
GAUSSIAN_RADII = {1: 0.6, 2: 1.1, 3: 1.6}
MOTION_KERNELS = {1: 3, 2: 5, 3: 7}
JPEG_QUALITIES = {1: 60, 2: 40, 3: 25}


def apply_corruption(image_array, corruption, severity):
    img = Image.fromarray(image_array)

    if corruption == "brightness":
        return np.array(ImageEnhance.Brightness(img).enhance(BRIGHTNESS_FACTORS[severity]), dtype=np.uint8)

    if corruption == "contrast":
        return np.array(ImageEnhance.Contrast(img).enhance(CONTRAST_FACTORS[severity]), dtype=np.uint8)

    if corruption == "gaussian_blur":
        return np.array(img.filter(ImageFilter.GaussianBlur(radius=GAUSSIAN_RADII[severity])), dtype=np.uint8)

    if corruption == "motion_blur":
        k = MOTION_KERNELS[severity]
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0
        kernel = (kernel / kernel.sum()).reshape(-1).tolist()
        return np.array(img.filter(ImageFilter.Kernel((k, k), kernel, scale=1.0)), dtype=np.uint8)

    if corruption == "jpeg":
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=JPEG_QUALITIES[severity])
        buffer.seek(0)
        with Image.open(buffer) as jpeg_img:
            return np.array(jpeg_img.convert("RGB"), dtype=np.uint8)

    raise ValueError(f"Unsupported corruption type: {corruption}")


def make_corrupted_split(images, labels):
    combos = [(c, s) for c in CORRUPTIONS for s in SEVERITIES]
    x_out = np.empty_like(images, dtype=np.uint8)
    y_out = labels.copy()
    theta = []

    for i in range(len(images)):
        corruption, severity = combos[i % len(combos)]
        x_out[i] = apply_corruption(images[i], corruption, severity)
        theta.append((corruption, severity))

    return x_out, y_out, theta


def build_data(args):
    os.makedirs(args.sample_dir, exist_ok=True)

    train_set = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True)

    x_train = np.array(train_set.data[: args.n_train], dtype=np.uint8)
    y_train = np.array(train_set.targets[: args.n_train], dtype=np.int64)

    x_val_clean = np.array(test_set.data[: args.n_val], dtype=np.uint8)
    y_val_clean = np.array(test_set.targets[: args.n_val], dtype=np.int64)

    calib_start = args.n_val
    calib_end = calib_start + args.n_calibration
    held_start = calib_end
    held_end = held_start + args.n_heldout

    calib_images = np.array(test_set.data[calib_start:calib_end], dtype=np.uint8)
    calib_labels = np.array(test_set.targets[calib_start:calib_end], dtype=np.int64)
    held_images = np.array(test_set.data[held_start:held_end], dtype=np.uint8)
    held_labels = np.array(test_set.targets[held_start:held_end], dtype=np.int64)

    calib_x, calib_y, calib_theta = make_corrupted_split(calib_images, calib_labels)
    held_x, held_y, held_theta = make_corrupted_split(held_images, held_labels)

    np.savez(
        os.path.join(args.sample_dir, "simulator_domain.npz"),
        x_train=x_train,
        y_train=y_train,
        x_test=x_val_clean,
        y_test=y_val_clean,
    )
    np.savez(
        os.path.join(args.sample_dir, "calibration_set.npz"),
        x=calib_x,
        y=calib_y,
        theta=np.array(calib_theta, dtype=object),
    )
    np.savez(
        os.path.join(args.sample_dir, "real_held_out_test.npz"),
        x=held_x,
        y=held_y,
        theta=np.array(held_theta, dtype=object),
    )


def train_and_evaluate(args):
    sim = np.load(os.path.join(args.sample_dir, "simulator_domain.npz"))
    held = np.load(os.path.join(args.sample_dir, "real_held_out_test.npz"), allow_pickle=True)

    train_ds = CIFARDataset(sim["x_train"], sim["y_train"])
    val_ds = CIFARDataset(sim["x_test"], sim["y_test"])
    held_ds = CIFARDataset(held["x"], held["y"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    held_loader = DataLoader(held_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PostNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = uce_loss(model(x)["alpha"], y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            seen += x.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model.predict(x)
                correct += (pred["pred_class"] == y).sum().item()
                total += y.size(0)

        train_loss = total_loss / max(1, seen)
        print(f"Epoch {epoch}/{args.epochs} | train UCE loss: {train_loss:.4f} | val accuracy: {correct / max(1, total):.4f}")

    checkpoint_path = os.path.join(args.sample_dir, "postnet_sample.pt")
    torch.save(model.state_dict(), checkpoint_path)

    all_correct, all_epi, all_alea = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in held_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model.predict(x)
            all_correct.append((pred["pred_class"] == y).cpu())
            all_epi.append(pred["epistemic"].cpu())
            all_alea.append(pred["aleatoric"].cpu())

    correct = torch.cat(all_correct)
    epistemic = torch.cat(all_epi)
    aleatoric = torch.cat(all_alea)

    print(f"\nSaved checkpoint: {checkpoint_path}")
    print(f"Held-out corrupted accuracy: {correct.float().mean().item():.4f}")
    print(
        f"Epistemic: mean {epistemic.mean().item():.4f}, std {epistemic.std().item():.4f}, "
        f"min {epistemic.min().item():.4f}, max {epistemic.max().item():.4f}"
    )
    print(
        f"Aleatoric: mean {aleatoric.mean().item():.4f}, std {aleatoric.std().item():.4f}, "
        f"min {aleatoric.min().item():.4f}, max {aleatoric.max().item():.4f}"
    )

    mask = correct.bool()
    if mask.any() and (~mask).any():
        print(
            "Split by correctness | "
            f"epi(correct)={epistemic[mask].mean().item():.4f}, epi(wrong)={epistemic[~mask].mean().item():.4f} | "
            f"alea(correct)={aleatoric[mask].mean().item():.4f}, alea(wrong)={aleatoric[~mask].mean().item():.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run tiny PostNet training with deterministic CIFAR-10 corruptions.")
    parser.add_argument("--sample-dir", default="sample_data_corruptions", help="Output directory for sample .npz files.")
    parser.add_argument("--data-root", default="./data", help="Where CIFAR-10 is downloaded/cached.")
    parser.add_argument("--n-train", type=int, default=2000, help="Clean simulator training size.")
    parser.add_argument("--n-val", type=int, default=500, help="Clean simulator validation size.")
    parser.add_argument("--n-calibration", type=int, default=200, help="Corrupted calibration size.")
    parser.add_argument("--n-heldout", type=int, default=500, help="Corrupted held-out test size.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training/eval batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_data(args)
    train_and_evaluate(args)
