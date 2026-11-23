"""ReLU MLP phase-retrieval network + sign-invariant loss + training loop (PyTorch).

Do NOT add input normalisation / batch-norm layers: the certified object is exactly the stack of
affine layers + ReLUs below, whose weights are read out by bounds.get_weights().
"""
import numpy as np
import torch
import torch.nn as nn


class PhaseNet(nn.Module):
    """f(y) = W_{L-1} relu(... relu(W_0 y + b_0) ...) + b_{L-1}   (last layer affine)."""

    def __init__(self, m, n, hidden=(64,)):
        super().__init__()
        self.dims = [m, *hidden, n]
        self.linears = nn.ModuleList(nn.Linear(a, b) for a, b in zip(self.dims[:-1], self.dims[1:]))

    def forward(self, y):
        z = y
        for i, lin in enumerate(self.linears):
            z = lin(z)
            if i < len(self.linears) - 1:
                z = torch.relu(z)
        return z


def sign_invariant_sq_error(pred, x):
    """Per-sample min_{sigma=+-1} ||pred - sigma x||_2^2 (differentiable almost everywhere)."""
    return torch.minimum(((pred - x) ** 2).sum(1), ((pred + x) ** 2).sum(1))


def train(model, Ytr, Xtr, Yva, Xva, epochs=60, lr=1e-3, batch=128, device="cpu", seed=0,
          verbose=True):
    """Adam + cosine schedule, sign-invariant loss. Inputs are numpy arrays (float64 ok)."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model.to(device)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    Yt, Xt, Yv, Xv = t(Ytr), t(Xtr), t(Yva), t(Xva)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist = []
    N = Yt.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N, generator=g).to(device)
        tot = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            loss = sign_invariant_sq_error(model(Yt[idx]), Xt[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            va = sign_invariant_sq_error(model(Yv), Xv).mean().item()
        hist.append((tot / N, va))
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"epoch {ep:3d}  train {tot / N:.5f}  val {va:.5f}")
    return hist


def save_checkpoint(path, model, A, meta=None):
    torch.save(dict(state=model.state_dict(), dims=model.dims, A=np.asarray(A), meta=meta or {}), path)


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    dims = ck["dims"]
    model = PhaseNet(dims[0], dims[-1], tuple(dims[1:-1]))
    model.load_state_dict(ck["state"])
    return model.eval(), ck["A"], ck["meta"]
