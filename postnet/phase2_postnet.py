# Phase 2 — PostNet: Encoder + Class-Conditional Normalizing Flow + Dirichlet/UCE
# Trains on simulator_domain.npz (clean CIFAR-10) ONLY.
#
# pip install normflows  (already in requirements.txt)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import normflows as nf
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

N_CLASSES = 10
LATENT_DIM = 6           # kept intentionally small -- flow-based density estimation
                          # degrades badly in high dimensions (curse of dimensionality);
                          # this matches the original PostNet paper's design choice
BATCH_SIZE = 128
EPOCHS = 30
LR = 3e-4


# ---------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------
class CIFARDataset(Dataset):
    def __init__(self, x, y):
        # x: (N, 32, 32, 3) uint8 -> normalize to [0, 1], channels-first
        self.x = torch.from_numpy(x).float().permute(0, 3, 1, 2) / 255.0
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ---------------------------------------------------------------------
# 2. Encoder — simple CNN mapping image -> latent vector z
# ---------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                        # 32 -> 16

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                        # 16 -> 8

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                                # -> (128, 1, 1)
        )
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        z = self.fc(h)
        return z


# ---------------------------------------------------------------------
# 3. Class-conditional Normalizing Flow
#    One flow per class, each modeling p(z | c) in latent space.
#    Using a stack of Radial flows around a standard Gaussian base.
# ---------------------------------------------------------------------
def build_class_flow(latent_dim=LATENT_DIM, n_flows=6):
    base = nf.distributions.base.DiagGaussian(latent_dim)

    flows = []
    for i in range(n_flows):
        # Alternate which half of dimensions gets transformed each layer
        b = torch.zeros(latent_dim)
        b[: latent_dim // 2] = 1 if i % 2 == 0 else 0
        b[latent_dim // 2:] = 0 if i % 2 == 0 else 1

        s = nf.nets.MLP([latent_dim, 2 * latent_dim, latent_dim], init_zeros=True)
        t = nf.nets.MLP([latent_dim, 2 * latent_dim, latent_dim], init_zeros=True)
        flows.append(nf.flows.MaskedAffineFlow(b, t, s))
        # Note: ActNorm removed -- its data-dependent init can produce NaN if a
        # latent dimension has near-zero variance in the first batch (log(0)).
        # BatchNorm in the encoder + gradient clipping + LR already provide
        # enough stabilization without it.

    return nf.NormalizingFlow(base, flows)


class ClassConditionalFlows(nn.Module):
    def __init__(self, n_classes=N_CLASSES, latent_dim=LATENT_DIM, n_flows=6):
        super().__init__()
        self.flows = nn.ModuleList([
            build_class_flow(latent_dim, n_flows) for _ in range(n_classes)
        ])

    def log_prob(self, z, class_idx):
        """log p(z | c) for a specific class index (scalar class, applied to whole batch)."""
        return self.flows[class_idx].log_prob(z)

    def log_prob_all_classes(self, z):
        """Returns (batch_size, n_classes) matrix of log p(z | c) for every class."""
        return torch.stack([self.flows[c].log_prob(z) for c in range(len(self.flows))], dim=1)


# ---------------------------------------------------------------------
# 4. Full PostNet model: Encoder -> per-class density -> Dirichlet pseudo-counts
# ---------------------------------------------------------------------
class PostNet(nn.Module):
    def __init__(self, n_classes=N_CLASSES, latent_dim=LATENT_DIM, n_flows=6):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.class_flows = ClassConditionalFlows(n_classes, latent_dim, n_flows)
        self.n_classes = n_classes
        # N_c: prior pseudo-count budget per class (uniform prior; could be set to
        # training-set class frequencies instead)
        self.register_buffer("N_c", torch.ones(n_classes) * 1.0)

    def forward(self, x):
        z = self.encoder(x)                                  # (B, latent_dim)
        log_p_z_given_c = self.class_flows.log_prob_all_classes(z)  # (B, n_classes)

        # High-dimensional log-densities can be very large; clamp before exponentiating
        # to avoid inf -> NaN propagation. 20 is a safe ceiling (exp(20) ~ 4.8e8).
        log_p_z_given_c = torch.clamp(log_p_z_given_c, min=-30.0, max=20.0)
        p_z_given_c = torch.exp(log_p_z_given_c)

        # Evidence / pseudo-counts: beta_c = N_c * p(z|c)
        beta = self.N_c.unsqueeze(0) * p_z_given_c            # (B, n_classes)

        # Dirichlet concentration parameters: alpha_c = beta_c + 1
        alpha = beta + 1.0

        # Safety net: sanitize any residual NaN/Inf (e.g. from flow numerical
        # instability) so a single bad value doesn't crash the whole batch.
        alpha = torch.nan_to_num(alpha, nan=1.0, posinf=1e6, neginf=1.0)
        alpha = alpha.clamp(min=1e-3)

        return {
            "z": z,
            "log_p_z_given_c": log_p_z_given_c,
            "alpha": alpha,
        }

    def predict(self, x):
        out = self.forward(x)
        alpha = out["alpha"]
        alpha_sum = alpha.sum(dim=1, keepdim=True)

        probs = alpha / alpha_sum                              # expected class probabilities
        # Aleatoric uncertainty: expected entropy of the categorical (data ambiguity)
        # Epistemic uncertainty: inverse of total evidence (unfamiliarity with input)
        epistemic = self.n_classes / alpha_sum.squeeze(1)       # higher = more unfamiliar
        aleatoric = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=1)  # predictive entropy

        return {
            "probs": probs,
            "pred_class": probs.argmax(dim=1),
            "alpha": alpha,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
        }


# ---------------------------------------------------------------------
# 5. UCE loss (Bayesian loss from the PostNet paper)
#    Uncertain Cross Entropy: E_{Dir(alpha)}[-log p_c] for the true class c
#    = digamma(alpha_sum) - digamma(alpha_c)
# ---------------------------------------------------------------------
def uce_loss(alpha, y, entropy_reg=1e-4):
    alpha_sum = alpha.sum(dim=1)
    alpha_true = alpha.gather(1, y.unsqueeze(1)).squeeze(1)

    uce = torch.digamma(alpha_sum) - torch.digamma(alpha_true)

    # Entropy regularizer on the Dirichlet (encourages higher uncertainty where
    # evidence is weak, prevents overconfident collapse) -- standard PostNet addition
    dirichlet = torch.distributions.Dirichlet(alpha)
    entropy = dirichlet.entropy()

    loss = uce.mean() - entropy_reg * entropy.mean()
    return loss


# ---------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------
def train_postnet(data_dir="/kaggle/working/flowpost_data"):
    sim = np.load(os.path.join(data_dir, "simulator_domain.npz"))
    x_train, y_train = sim["x_train"], sim["y_train"]
    x_val, y_val = sim["x_test"], sim["y_test"]     # clean test set used as val here

    train_ds = CIFARDataset(x_train, y_train)
    val_ds = CIFARDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = PostNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = uce_loss(out["alpha"], y)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [warning] NaN/Inf loss encountered, skipping this batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # flow training can be unstable
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_ds)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model.predict(x)
                correct += (pred["pred_class"] == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        print(f"Epoch {epoch:3d}/{EPOCHS} | train UCE loss: {train_loss:.4f} | val accuracy: {val_acc:.4f}")

        # Checkpoint every 5 epochs in case of Kaggle session timeout
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"/kaggle/working/postnet_epoch{epoch}.pt")

    torch.save(model.state_dict(), "/kaggle/working/postnet_final.pt")
    print("\nTraining complete. Model saved to /kaggle/working/postnet_final.pt")
    return model


if __name__ == "__main__":
    train_postnet()
