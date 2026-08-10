# Phase 2 — Evaluation: accuracy + aleatoric/epistemic uncertainty on simulator val set
# Run in the same session where PostNet/Encoder/etc. classes are already defined,
# after loading a trained checkpoint.

import numpy as np
import torch

CHECKPOINT_PATH = "/kaggle/working/postnet_final.pt"   # or postnet_epoch20.pt for the peak-accuracy checkpoint

def evaluate_postnet(checkpoint_path=CHECKPOINT_PATH, data_dir="/kaggle/working/flowpost_data"):
    sim = np.load(data_dir + "/simulator_domain.npz")
    x_val, y_val = sim["x_test"], sim["y_test"]

    val_ds = CIFARDataset(x_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    model = PostNet().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    all_correct = []
    all_epistemic = []
    all_aleatoric = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model.predict(x)
            all_correct.append((pred["pred_class"] == y).cpu())
            all_epistemic.append(pred["epistemic"].cpu())
            all_aleatoric.append(pred["aleatoric"].cpu())

    correct = torch.cat(all_correct)
    epistemic = torch.cat(all_epistemic)
    aleatoric = torch.cat(all_aleatoric)

    accuracy = correct.float().mean().item()

    print(f"=== PostNet Evaluation ({checkpoint_path}) ===")
    print(f"Accuracy: {accuracy:.4f}\n")

    print("--- Epistemic uncertainty (unfamiliarity with input) ---")
    print(f"  mean: {epistemic.mean():.4f}  std: {epistemic.std():.4f}")
    print(f"  min:  {epistemic.min():.4f}   max: {epistemic.max():.4f}")

    print("\n--- Aleatoric uncertainty (data ambiguity) ---")
    print(f"  mean: {aleatoric.mean():.4f}  std: {aleatoric.std():.4f}")
    print(f"  min:  {aleatoric.min():.4f}   max: {aleatoric.max():.4f}")

    # Key sanity check: uncertainty should be higher on WRONG predictions than
    # correct ones -- this is the core evidence PostNet's uncertainty is meaningful
    correct_mask = correct.bool()
    print("\n--- Uncertainty split by correctness (sanity check) ---")
    print(f"  Epistemic | correct: {epistemic[correct_mask].mean():.4f}  "
          f"wrong: {epistemic[~correct_mask].mean():.4f}")
    print(f"  Aleatoric | correct: {aleatoric[correct_mask].mean():.4f}  "
          f"wrong: {aleatoric[~correct_mask].mean():.4f}")

    if epistemic[~correct_mask].mean() > epistemic[correct_mask].mean():
        print("\n  Epistemic uncertainty is higher on wrong predictions -- looks correct.")
    else:
        print("\n  WARNING: epistemic uncertainty is NOT higher on wrong predictions -- worth investigating.")

    return {
        "accuracy": accuracy,
        "epistemic": epistemic,
        "aleatoric": aleatoric,
        "correct": correct,
    }

if __name__ == "__main__":
    results = evaluate_postnet()
