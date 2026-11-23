# sdp-certified-phase-retrieval
Certified robustness of ReLU phase-retrieval neural networks via Semidefinite Programming (SDP) and S-procedure duality. Benchmarks SDP upper bounds against PGD attacks, IBP/Lipschitz bounds, and PhaseLift.
# Certified Robustness of Deep Neural Network Phase Retrieval via Semidefinite Programming

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![CVXPY](https://img.shields.io/badge/Solver-CVXPY%20%7C%20MOSEK%20%7C%20Clarabel-green.svg)](https://www.cvxpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Course Project for **ECE 285 (2026)** at the Department of Electrical and Computer Engineering, **University of California, San Diego (UC San Diego)**.

📄 **[Project Proposal (PDF)](docs/ECE285_Team_Proposal_Chengen.pdf)**

---

## Overview

Real-valued phase retrieval aims to recover an unknown signal $x \in \mathbb{R}^n$ from phaseless quadratic measurements $y_k = \langle a_k, x \rangle^2$. While convex relaxations such as **PhaseLift** offer strong stability guarantees, solving semidefinite programs (SDPs) lifted to $X = xx^\top$ scales poorly in high dimensions. Conversely, **feed-forward neural networks** provide fast inference but can suffer from severe instabilities under small $\ell_2$-bounded measurement perturbations $\Vert{}\delta\Vert{}_2 \le \epsilon$.

This project bridges the gap by formulating a **dual SDP robustness certificate** for ReLU phase-retrieval networks. We compute a certified upper bound on the worst-case sign-invariant reconstruction error:

$$\mathcal{E}(\delta) = \min_{\sigma \in \{\pm 1\}} \Vert{}f_\theta(y + \delta) - \sigma x\Vert{}_2^2, \quad \text{s.t. } \Vert{}\delta\Vert{}_2 \le \epsilon$$

and systematically evaluate bound tightness and runtime against adversarial lower bounds and classical baselines.

---

## Key Methodology

* **ReLU Quadratic Constraints & S-Procedure Duality**: Encodes layer-wise pre-activation bounds $l \le \hat{z} \le u$ (via IBP/CROWN) and exact/relaxed ReLU quadratic constraints ($z'(z' - \hat{z}) = 0$, $(\hat{z}-l)(\hat{z}-u) \le 0$) into a lifted SDP, solving its Lagrangian dual via the S-procedure.
* **Input SVD Dimensionality Reduction**: Exploits the fact that measurement noise $\delta \in \mathbb{R}^m$ only enters through the first-layer weight matrix $W_0 \in \mathbb{R}^{h \times m}$. Using thin SVD ($W_0 = U\Sigma V^\top$), the perturbation set reduces to $\{U\Sigma w : \Vert{}w\Vert{}_2 \le \epsilon\}$, making the SDP dimension $1 + h + \sum_{l \ge 1} n_l$ independent of the measurement count $m$.
* **Strict Dual Feasibility & Sign-Flip Verification**: Applies slack-matrix shifting ($+tI$) to guarantee rigorous numerical feasibility despite interior-point solver tolerances, and checks both $\pm \sigma_0$ branches to account for potential sign flips inside the $\epsilon$-ball.
* **Comprehensive Benchmarking**: Evaluates certified SDP upper bounds alongside:
  * **PGD Attacks** (empirical worst-case lower bound)
  * **IBP / Lipschitz (LipSDP) Bounds** (fast outer bounds)
  * **PhaseLift SDP** (convex recovery baseline via CVXPY)

---

## Experimental Setup

* **Dataset & Measurements**: $8 \times 8$ downsampled MNIST ($n = 64$, normalized $\Vert{}x\Vert{}_2 = 1$) with fixed Gaussian sensing vectors $a_k \sim \mathcal{N}(0, I_n)$ and $m = 4n = 256$ measurements.
* **Network Architecture**: Sign-invariant ReLU MLP trained to minimize $\min_{\sigma \in \{\pm 1\}} \Vert{}f_\theta(y) - \sigma x\Vert{}_2^2$.
* **Perturbation Grid**: Evaluated across relative noise levels $\epsilon_{\text{rel}} = \epsilon / \Vert{}y\Vert{}_2$ to identify tightness regimes and where the SDP relaxation becomes vacuous.

---

## Repository Structure

```text
├── docs/
│   └── ECE285_Team_Proposal_Chengen.pdf   # Project proposal
├── src/
│   ├── models/                            # Sign-invariant ReLU MLP & training scripts
│   ├── certification/                     # Dual SDP formulation, SVD reduction & tI post-processing
│   ├── baselines/                         # PhaseLift (CVXPY), PGD attack, IBP / LipSDP
│   └── utils/                             # Data loading (8x8 MNIST) & sensing matrix generation
├── scripts/
│   └── run_experiments.py                 # Main benchmarking script across epsilon grids
├── requirements.txt
└── README.md
