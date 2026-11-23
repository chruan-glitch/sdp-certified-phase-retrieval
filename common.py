"""Shared setup: fixed Gaussian measurement matrix, measurements, data, eps conventions.

Conventions (must match the proposal):
  * x in R^n with ||x||_2 = 1, y_k = <a_k, x>^2, A in R^{m x n} Gaussian, m = 4n, drawn ONCE.
  * Perturbation budget is ABSOLUTE: ||delta||_2 <= eps, one common eps grid for all samples.
    Plots show eps_rel = eps / mean(||y||_2) over the test set.
  * delta ranges over the full l2 ball (y + delta may be negative: conservative modelling choice).
  * All reported errors are squared l2 errors.
"""
import numpy as np


def make_A(n, m_factor=4, seed=0):
    """Fixed Gaussian measurement matrix A (m x n), float64."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m_factor * n, n))


def measure(A, X):
    """y_k = <a_k, x>^2 for a batch X of shape (N, n) -> (N, m). float64."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return (X @ A.T) ** 2


def normalize_rows(X):
    X = np.asarray(X, dtype=np.float64)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _downsample(imgs, side):
    """imgs: (N, 28, 28) float array -> (N, side, side) by block averaging (28 must be divisible
    after cropping; 28 -> 8 is not an integer factor, so use adaptive-style area averaging)."""
    N, H, W = imgs.shape
    out = np.zeros((N, side, side))
    ys = np.linspace(0, H, side + 1)
    xs = np.linspace(0, W, side + 1)
    for i in range(side):
        for j in range(side):
            y0, y1 = int(np.floor(ys[i])), int(np.ceil(ys[i + 1]))
            x0, x1 = int(np.floor(xs[j])), int(np.ceil(xs[j + 1]))
            out[:, i, j] = imgs[:, y0:y1, x0:x1].mean(axis=(1, 2))
    return out


def load_signals(kind="mnist", side=8, n_train=10000, n_test=500, seed=0, root="./data"):
    """Return (Xtrain, Xtest), each (N, side*side), unit l2 norm, float64.

    kind="mnist": torchvision MNIST (downloaded to `root`), block-averaged to side x side.
    kind="synthetic": random smooth non-negative blobs, for testing without downloads.
    """
    n = side * side
    rng = np.random.default_rng(seed)
    if kind == "mnist":
        from torchvision import datasets  # imported lazily so numpy-only tools still work
        tr = datasets.MNIST(root, train=True, download=True)
        te = datasets.MNIST(root, train=False, download=True)
        xtr = tr.data.numpy().astype(np.float64)[:n_train] / 255.0
        xte = te.data.numpy().astype(np.float64)[:n_test] / 255.0
        Xtr = _downsample(xtr, side).reshape(-1, n)
        Xte = _downsample(xte, side).reshape(-1, n)
    elif kind == "synthetic":
        def blobs(N):
            g = np.linspace(-1, 1, side)
            gx, gy = np.meshgrid(g, g)
            out = np.zeros((N, side, side))
            for i in range(N):
                for _ in range(rng.integers(1, 4)):
                    cx, cy = rng.uniform(-0.7, 0.7, 2)
                    s = rng.uniform(0.15, 0.5)
                    out[i] += np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * s ** 2))
            return out.reshape(N, n)
        Xtr, Xte = blobs(n_train), blobs(n_test)
    else:
        raise ValueError(kind)
    # drop (near-)empty images, then normalise to unit norm
    Xtr = Xtr[np.linalg.norm(Xtr, axis=1) > 1e-6]
    Xte = Xte[np.linalg.norm(Xte, axis=1) > 1e-6]
    return normalize_rows(Xtr), normalize_rows(Xte)


def y_norm_stats(Y):
    """Fluctuation of ||y||_2 across samples (record this in the report)."""
    r = np.linalg.norm(Y, axis=1)
    return dict(mean=r.mean(), std=r.std(), min=r.min(), max=r.max(),
                rel_spread=(r.max() - r.min()) / r.mean())


def eps_grid_abs(Y_test, rel_levels=(0.005, 0.01, 0.02, 0.05, 0.1)):
    """Common ABSOLUTE eps grid built from relative levels and the MEAN ||y||_2 of the test set.
    Returns (eps_abs, eps_rel_for_plot)."""
    mean_norm = np.linalg.norm(Y_test, axis=1).mean()
    rel = np.asarray(rel_levels, dtype=np.float64)
    return rel * mean_norm, rel
