# $\ell_2$ Adversarial Robustness Certification for Phase Retrieval Neural Networks (SDP + RLT)

This repository evaluates the $\ell_2$ adversarial robustness certification tightness and computational runtime of a 3-layer Multilayer Perceptron (MLP) trained for Phase Retrieval on the $8 \times 8$ MNIST dataset. By incorporating pairwise product constraints via the Reformulation-Linearization Technique (RLT) into standard Semidefinite Programming (SDP), **SDP+RLT** substantially tightens the certified upper bounds and expands the non-vacuous certification regime.

## Key Highlights

1. **Significant Bound Tightening via SDP+RLT**: Adding pairwise neuron product constraints (RLT) to the plain SDP relaxation tightens the certified worst-case upper bound by up to **87.1%** (at $\epsilon_{\text{rel}} = 0.13$). Compared to empirical PGD attack lower bounds, the median ratio of the SDP+RLT upper bound is virtually exact at $\epsilon_{\text{rel}} = 0.05$ ($1.00\times$) and remains controlled at $2.60\times$ even at $\epsilon_{\text{rel}} = 0.13$.
2. **Substantially Expanded Non-Vacuous Regime**: Using the mean squared error of the trivial zero estimator $f = 0$ (threshold `1.0`) as the vacuousness boundary, IBP and Lipschitz-based methods become completely vacuous by $\epsilon_{\text{rel}} = 0.05$, and plain SDP becomes vacuous at $\epsilon_{\text{rel}} = 0.10$. In contrast, **SDP+RLT extends the non-vacuous certification radius to $\epsilon_{\text{rel}} = 0.13$**, maintaining non-vacuous guarantees on 19 out of 20 test samples.
3. **Ultra-Fast Inference vs. ReLU-Dependent Certification Runtime**: Neural network forward inference takes only **0.05 ms** per sample—over $60,000\times$ faster than classical convex optimization via PhaseLift ($\approx 3.0\text{--}3.4\text{ s}$). During certification, however, as larger perturbations induce more unstable ReLU neurons in the hidden layer, the solve time of SDP+RLT scales up to **93.82 s/sample**.

## Experimental Setup & Baseline Parameters

* **Architecture & Dataset**:
  * Model: 3-layer MLP (`256` input measurements $\to$ `128` hidden ReLU units $\to$ `64` output pixels)
  * Dataset: $8 \times 8$ MNIST phase retrieval (20,000 training samples; $N = 20$ test samples for full Phase 4 certification)
  * Clean Performance: Clean test Mean Squared Error (MSE) is `0.0398`
* **Adversarial Perturbations ($\ell_2$ Norm)**:
  * Mean signal norm: $\mathbb{E}\lVert y \rVert_2 = 29.84$
  * Relative perturbation levels: $\epsilon_{\text{rel}} \in \{0.01, 0.05, 0.10, 0.13\}$
  * Corresponding absolute $\ell_2$ radii: $\epsilon \in \{0.30, 1.49, 2.98, 3.88\}$
* **Solvers & Verification**:
  * Primary solver: **MOSEK**
  * Multi-solver cross-validation (Phase 2 / Phase 3): **CLARABEL** and **SCS**, with duality gaps consistently verified on the order of $10^{-11}$ to $10^{-6}$.

## Core Experimental Results

### 1. Certification Tightness (`tightness.jpg`)

The table below summarizes the **median** worst-case squared error upper bounds (and median ratios relative to empirical PGD lower bounds) across $N = 20$ test samples. The dashed threshold `1.0` denotes the error of the trivial zero estimator ($f = 0$); upper bounds $\ge 1.0$ are considered vacuous:

| Relative Perturbation $\epsilon_{\text{rel}}$ ($\epsilon$) | PGD Lower Bound (Fixed / Inv) | SDP+RLT (Certified Upper) | SDP Plain (Certified Upper) | LipSDP (Lipschitz) | IBP (Interval Bound) | Spectral Lipschitz | Median Ratio (SDP+RLT / PGD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.01** (`0.30`) | 0.0289 / 0.0289 | **0.0289** | 0.0289 | 0.2014 | 0.2419 | 0.3522 | $1.00\times$ |
| **0.05** (`1.49`) | 0.0618 / 0.0618 | **0.0637** | 0.1709 | 2.5434 *(vacuous)* | 3.9794 *(vacuous)* | 5.3732 *(vacuous)* | $1.00\times$ |
| **0.10** (`2.98`) | 0.1436 / 0.1436 | **0.2676** | 1.5224 *(vacuous)* | 9.1646 *(vacuous)* | 24.4160 *(vacuous)* | 20.0144 *(vacuous)* | $1.64\times$ |
| **0.13** (`3.88`) | 0.2169 / 0.2157 | **0.6563** | 5.0869 *(vacuous)* | 15.1073 *(vacuous)* | 83.7629 *(vacuous)* | 33.2603 *(vacuous)* | $2.60\times$ |

**Key Takeaways:**
* **Exactness with Zero Unstable Neurons**: At $\epsilon_{\text{rel}} = 0.01$, there are on average only `0.2` unstable neurons (`0` for most samples), making the network locally affine within the perturbation ball. Both SDP and SDP+RLT recover the exact global optimum matching the PGD lower bound (within $< 10^{-9}$).
* **Decisive Role of RLT Constraints**: For $\epsilon_{\text{rel}} \ge 0.05$ as unstable neurons proliferate, plain SDP loosens rapidly (reaching $23.5\times$ the PGD bound at $\epsilon_{\text{rel}} = 0.13$). SDP+RLT suppresses the upper bound to `0.6563` ($2.60\times$ PGD), making it the only method capable of providing non-vacuous guarantees at $\epsilon_{\text{rel}} = 0.13$.
* **Comparison with PhaseLift**: PhaseLift achieves near-machine-precision recovery in the noiseless setting ($10^{-28}$ to $10^{-15}$ error) and incurs errors of `0.0006` to `0.046` under perturbed inputs. Note that the PhaseLift curves in `tightness.jpg` evaluate empirical reconstruction error under **PGD attacks generated against the neural network** (`y + delta_PGD`) and **random Gaussian noise** (`y + delta_rand`), rather than PhaseLift's own worst-case adversarial error or a certified upper bound.

### 2. Computational Runtime & Scalability (`timing.png`)

Per-sample median runtimes across perturbation levels are summarized below:

| Relative Perturbation $\epsilon_{\text{rel}}$ | Avg. Unstable Neurons (out of 128) | NN Forward Inference (Median) | PhaseLift (Noisy, Median) | SDP Plain (MOSEK, Median) | SDP+RLT (MOSEK, Median) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.01** | 0.2 | 0.050 ms | 3.13 s | 0.087 s | **0.073 s** |
| **0.05** | 2.9 | 0.050 ms | 3.00 s | 0.129 s | **0.130 s** |
| **0.10** | 24.9 | 0.050 ms | 3.41 s | 2.38 s | **2.95 s** |
| **0.13** | 53.4 | 0.050 ms | 3.25 s | 56.07 s | **93.82 s** |

**Key Takeaways:**
* **Inference Speed Advantage**: Neural network forward inference remains constant at `0.050 ms` ($50\ \mu\text{s}$), achieving a $> 60,000\times$ speedup over PhaseLift ($\approx 3.1\text{ s}$). For small-to-moderate perturbations ($\epsilon_{\text{rel}} \le 0.10$), the combined inference and SDP+RLT certification time is still faster than a single PhaseLift solve.
* **Scalability Bottleneck**: PhaseLift's semidefinite matrix dimension depends solely on the signal dimension, yielding nearly constant runtime across $\epsilon$. In contrast, neural network SDP variable counts scale with the number of unstable neurons (`n_unstable`, e.g., matrix dimension $N$ growing from 20 to 141 in Phase 3). Under larger perturbations tested in Phase 1/2 ($\epsilon_{\text{rel}} = 0.20, 0.50$), unstable neuron counts reach 105--128, causing MOSEK solver timeouts (e.g., `1992 s` for plain SDP on `test#0` at $\epsilon_{\text{rel}} = 0.20$). Consequently, Phase 4 focuses full certification evaluation on $\epsilon_{\text{rel}} \le 0.13$.

## Consistency Checks & Future Directions

* **Theoretical Ordering & Solver Consistency**: All automated consistency checks (`checks`) in Phase 2 and Phase 3 passed 100%, strictly satisfying the theoretical hierarchy:

  $$\text{Clean} \le \text{PGD}_{\text{inv}} \le \text{PGD}_{\text{fixed}} \le \text{SDP+RLT} \le \text{SDP} \le \text{IBP}$$

  with close agreement across MOSEK, CLARABEL, and SCS.
* **Future Directions**:
  1. **Sparse RLT Constraint Selection**: Apply pairwise RLT product constraints selectively to the top-$k$ most critical unstable neurons ranked by pre-activation interval width (`width_mean`), preserving tightness while reducing SDP matrix size.
  2. **Certification-Aware Training**: Incorporate $\ell_2$ weight regularization or adversarial training to reduce the number of unstable ReLU activations under large input perturbations.

## Repository Structure

```text
.
├── README.md          # Project documentation and experimental summary
├── tightness.jpg      # Certification tightness comparison across perturbation levels
├── timing.png         # Per-sample solver runtime comparison across perturbation levels
└── results/           # Phase 1 - Phase 4 detailed logs, solver cross-checks, and bound statistics
```
