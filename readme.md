# FlowPost
### Calibrated Uncertainty under Simulation-to-Reality Shift

A hybrid probabilistic deep learning pipeline that combines **Flow Matching**, **Posterior Networks (PostNet)**, and a **Bayesian Network (pgmpy)** decision layer to make trustworthy, uncertainty-aware predictions when a model trained on simulated data is deployed on real-world data.

---

## Problem

Deep learning models are often trained on simulator-generated data because collecting large volumes of real-world data is expensive or impractical. But simulators never match the real world exactly — this is called **model misspecification**. A model trained purely on simulated data can be confidently wrong when deployed on real observations, which is dangerous in domains where decisions have real consequences.

## Domain & Application

This project uses an **epidemic (SIR) simulator** as its case study.

- A clean SIR simulator generates idealized infection curves and is used to train PostNet.
- Real-world epidemic reporting is noisy: cases are **under-reported** and **delayed**. A small calibration set of "real-like" noisy/delayed curves is used to train the Flow Matching correction network.
- The task: given a 30-day infection curve, classify the outbreak severity and produce a calibrated, trustworthy decision — not just a raw label.

## Architecture

```
                     SIMULATION STAGE
   SIR Simulator (β, γ) → Clean Infection Curves → Train PostNet


                     DEPLOYMENT STAGE
   Real-world Observation (noisy, under-reported, delayed)
                       │
                       ▼
     Small Calibration Dataset → Observation Flow (TX)
              (Flow Matching: real → simulator-aligned)
                       │
                       ▼
              Simulation-aligned Observation
                       │
                       ▼
                  Encoder (PostNet)
                       │
                       ▼
            Latent Representation (z)
                       │
                       ▼
        Normalizing Flow → P(z | class)
                       │
                       ▼
        Density-based Pseudo-Counts (β)
                       │
                       ▼
        Dirichlet Posterior Distribution
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Prediction   Aleatoric UQ    Epistemic UQ
        │              │              │
        └──────────────┴──────────────┘
                       │
                       ▼
        pgmpy Bayesian Network (Decision Layer)
                       │
                       ▼
     Final Decision: Trust / Flag for Review / Reject
```

### Components

| Component | Type | Role |
|---|---|---|
| SIR Simulator | — | Generates synthetic (θ, x) training pairs |
| Flow Matching (TX) | Deep Learning | Aligns real, noisy observations to the simulator's distribution |
| Encoder | Deep Learning | Extracts features from the (aligned) observation |
| Normalizing Flow | Deep Learning | Estimates class-conditional density P(z\|class) |
| Dirichlet Posterior | Probabilistic Reasoning | Converts density into calibrated prediction + uncertainty |
| pgmpy Bayesian Network | Probabilistic Reasoning | Fuses uncertainty + context into an interpretable decision |

## Parameters

- **θ (simulator parameters):** infection rate `β ~ U(0.1, 0.9)`, recovery rate `γ ~ U(0.05, 0.5)`; `R0 = β / γ`
- **Class labels:** `R0 < 1` → Controlled | `1 ≤ R0 < 2` → Moderate spread | `R0 ≥ 2` → High spread
- **x (observation):** 30-day daily new-infection count vector
  - Simulator version: clean SIR curve
  - Calibration/"real" version: under-reported (60–80% capture) + delayed/smoothed curve

## Novelty

Standard FMCPE uses two correction flows (input correction + posterior correction). This project keeps only the input-correction flow (TX) and replaces the posterior-correction flow with PostNet's density-based evidential uncertainty — testing whether PostNet's built-in confidence estimation makes the second flow unnecessary. A pgmpy decision layer is added on top to turn raw uncertainty numbers into an explicit, interpretable action.

## Repository Structure

```
flowpost/
├── data/           # Simulated + calibration datasets
├── src/            # Source code (simulator, TX, PostNet, pgmpy network)
├── notebooks/       # Exploratory / training notebooks
├── results/         # Metrics, plots, saved checkpoints
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/flowpost.git
cd flowpost
conda create -n flowpost python=3.10
conda activate flowpost
pip install -r requirements.txt
```

Confirm GPU availability:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

## Team

| Member | Role |
|---|---|
| Manas | Simulator, PostNet training (GPU-heavy), baseline evaluation |
| Zahwa | Flow Matching (TX), pgmpy decision layer, integration support |
| Both | Integration, evaluation, report, presentation |

## References

1. Barzilai, D., Elhadad, T., et al. "Flow Matching Calibration for Simulation-Based Inference under Model Misspecification." *ICML 2026.*
2. Charpentier, B., Zügner, D., Günnemann, S. "Posterior Network: Uncertainty Estimation without OOD Samples via Density-Based Pseudo-Counts." *NeurIPS 2020.*
