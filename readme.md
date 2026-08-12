# Flow Posterior for Uncertainty Estimation

A hybrid uncertainty-estimation pipeline combining a **Posterior Network
(PostNet)** for density-based epistemic and aleatoric uncertainty, a
lightweight **Flow Matching calibration step (FMCPE-lite)** for correcting
simulation-to-reality domain shift, and a **pgmpy Bayesian decision network**
that turns uncertainty into an explainable Trust / Flag / Reject decision.

---

## Problem Statement

Standard deep learning classifiers output a single confidence score and are
frequently overconfident on inputs that differ from their training
distribution — they have no principled way to say "I don't know." Most
existing approaches to fixing this require exposing the model to
out-of-distribution (OOD) examples during training, which is often
impractical: you rarely know in advance what your model will fail on.

This project asks two questions:

1. Can a model learn to recognize unfamiliar inputs **without ever training
   on OOD data**, purely by modeling the density of "normal" data in its own
   learned representation?
2. When that model is deployed on data that has drifted from its training
   distribution (a common industrial problem — simulators and clean
   benchmarks rarely match messy real-world inputs), can a small,
   inexpensive calibration step correct for that gap, and can the resulting
   uncertainty be turned into an auditable decision rather than a raw
   confidence number?

---

## Our Contribution

PostNet and FMCPE are each existing, individually published methods. This
project's contribution is combining them, and adding a decision layer on
top, in a way not found together in the literature:

- **PostNet** provides OOD-free epistemic uncertainty via density estimation
  in latent space, rather than needing adversarial OOD sampling or ensemble
  methods.
- **FMCPE-lite** adds a lightweight, input-space correction for
  simulation-to-reality domain shift — instead of the full two-transport-map
  machinery of the original FMCPE paper, only the observation-space map is
  learned, using a small real-world calibration set.
- **pgmpy decision layer** converts PostNet's raw uncertainty into a
  probabilistic Bayesian Network over discretized evidence, then applies an
  expected-utility decision rule (`argmax_action EU(action)`) to produce an
  explainable Trust / Flag / Reject output — moving from "the model is
  uncertain" to "here is the recommended action and why."

This combination targets a genuinely practical gap: evidential deep learning
models are usually evaluated assuming training and deployment data are
perfectly aligned, which rarely holds in practice.

---

## Domain

- **Simulator (clean) domain:** CIFAR-10 — abundant, well-behaved training
  data, standing in for a "clean" or simulated data source.
- **Real/shifted domain:** CIFAR-10-C — corrupted CIFAR-10 variants (noise,
  blur, fog, brightness, etc. at varying severity), standing in for
  real-world, distribution-shifted deployment data.
- **theta:** corruption type + severity (1-5)
- **x:** the 32x32x3 corrupted image itself

CIFAR-10/CIFAR-10-C was chosen because it is classification-native (matches
PostNet directly), has well-documented, reproducible corruption benchmarks,
and keeps data-engineering overhead low so effort stays focused on the
uncertainty method itself.

---

## Architecture

```
              clean CIFAR-10 (simulator domain)
                        |
                        v
          --------------------------------
          |   PostNet training           |
          |   Encoder -> class-cond.     |
          |   Normalizing Flow -> P(z|c) |
          |   -> Dirichlet pseudo-counts |
          --------------------------------
                        |
        CIFAR-10-C (real/shifted, held out)
                        |
                        v
          --------------------------------
          |   TX: Flow Matching (lite)   |
          |   corrupted -> clean-style   |
          --------------------------------
                        |
                        v
          --------------------------------
          |   PostNet (frozen, trained   |
          |   on clean data only)        |
          --------------------------------
                        |
        prediction + aleatoric/epistemic uncertainty
                        |
                        v
          --------------------------------
          |   pgmpy Bayesian Network     |
          |   uncertainty + context ->    |
          |   P(prediction correct)       |
          --------------------------------
                        |
                        v
          --------------------------------
          |   Utility / Decision layer   |
          |   EU(action) = sum_state      |
          |   P(state|evidence) x         |
          |   utility(state,action)       |
          --------------------------------
                        |
                        v
          Decision: argmax_action EU(action)
                -> Trust / Flag / Reject
```

**How PostNet works, briefly:** an encoder maps each image to a low-dimensional
latent vector `z`. A class-conditional normalizing flow estimates `P(z|c)`
for every class via the change-of-variables formula. This density is
converted into Dirichlet pseudo-counts, `beta_c = N_c * P(z|c)`, giving
Dirichlet parameters `alpha_c = beta_c + 1`. From the resulting
`Dir(alpha)` distribution, PostNet reads off two separate uncertainty
signals: **aleatoric** (entropy of the expected class probabilities — data
is inherently ambiguous) and **epistemic** (inverse of total evidence — the
model hasn't seen enough like this). The model is trained with the UCE
(uncertain cross-entropy) loss, the closed-form expected cross-entropy under
the Dirichlet posterior.

---


---

## Repository Structure

```
data_pipeline/
  phase1_data_pipeline.py   - builds simulator/calibration/held-out splits
  phase1_eda.py              - exploratory data analysis on the splits
postnet/
  phase2_postnet.py          - Encoder, class-conditional flows, Dirichlet
                                pseudo-counts, UCE loss, PostNet model
  phase2_evaluate.py         - correct-vs-wrong uncertainty sanity check
  phase2_shift_eval.py       - clean-vs-CIFAR-10-C shift experiment
  phase2_flow_audit.py       - flow invertibility + discriminative density checks
results/
  headline_result.npz        - raw uncertainty/correctness arrays
  plot1_epistemic_histogram.png
  plot2_accuracy_vs_uncertainty.png
sample_run.py                 - quick end-to-end smoke test
requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Run the data pipeline, then PostNet training/evaluation scripts in
`postnet/`, in order. `sample_run.py` provides a quick smoke test of the
full pipeline on a small subset of data.

---

## Related Work & Scope

PostNet and FMCPE build on existing published methods (density-based
evidential deep learning and flow-based posterior correction,
respectively). Two related directions were considered and deliberately
excluded from this project:

- **floZ (Bayesian evidence estimation)** — requires a fully validated
  posterior estimator as a prerequisite, plus a physics domain (e.g.
  gravitational wave data) outside this project's scope.
- **Manifold-valued flows** — only relevant for inherently directional or
  spherical data, which image classification is not; would add Riemannian
  geometry machinery with no benefit here.
- **Multiplicative Normalizing Flows (MNF)** — a competing architecture to
  PostNet for epistemic uncertainty, solving the same problem rather than
  complementing it.

---

## Team

- **Manas** — PostNet training, flow implementation, Dirichlet/UCE loss,
  verification and evaluation
- **Zahwa** — data pipeline, calibration set construction, EDA
## References

1. Barzilai, D., Elhadad, T., et al. "Flow Matching Calibration for Simulation-Based Inference under Model Misspecification." *ICML 2026.*
2. Charpentier, B., Zügner, D., Günnemann, S. "Posterior Network: Uncertainty Estimation without OOD Samples via Density-Based Pseudo-Counts." *NeurIPS 2020.*
