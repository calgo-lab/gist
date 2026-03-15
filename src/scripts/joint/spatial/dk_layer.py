"""dk_layer.py — Deep Kriging spatial interpolation layer.

Replaces the GP kernel with an MLP that learns spatial basis functions.
Predictions are made via closed-form ridge regression on the learned basis:

    Φ(x) = MLP(x)                          # (n, K) basis matrix
    β     = (ΦᵀΦ + λI)⁻¹ Φᵀ y_train       # (K,) coefficients
    ŷ     = Φ(X_test) @ β                  # (n_test,) predictions

Advantages over GP:
  - O(nK + K³) instead of O(n³)  → faster for large n_train
  - Non-stationary: MLP can learn location-dependent patterns
  - Fully differentiable: gradient flows through y_train → β → ŷ → GRU

Interface matches SVGPLayer (forward / forward_with_std) so it is a
drop-in replacement in gru_dk_train.py / gru_dk_eval.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DeepKrigingLayer(nn.Module):
    def __init__(
        self,
        n_spatial_dims: int = 2,
        n_basis: int = 64,
        hidden_sizes: list[int] | None = None,
        lambda_reg: float = 1e-3,
    ):
        """
        Args:
            n_spatial_dims: input coordinate dimensionality (2 for x, y)
            n_basis:        number of basis functions (output size of MLP)
            hidden_sizes:   hidden layer widths; defaults to [64, 64]
            lambda_reg:     ridge regularization for coefficient fitting
        """
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        layers: list[nn.Module] = []
        in_size = n_spatial_dims
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, n_basis))
        self.mlp = nn.Sequential(*layers)

        self.n_basis = n_basis
        self.register_buffer("lambda_reg", torch.tensor(float(lambda_reg)))

    # ------------------------------------------------------------------
    # Public interface (matches SVGPLayer)
    # ------------------------------------------------------------------

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict at X_test given observations (X_train, y_train).

        Gradient flows through y_train → β → output, so the spatial loss
        can update GRU parameters via backprop.

        Returns float32 tensor of shape (n_test,).
        """
        Phi_tr = self.mlp(X_train)   # (n_train, K)
        Phi_te = self.mlp(X_test)    # (n_test,  K)
        beta = self._solve_beta(Phi_tr, y_train)
        return Phi_te @ beta

    def forward_with_std(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and std at X_test.

        Posterior std uses residual variance σ² and the ridge posterior:
            var(x*) = σ² · φ(x*)ᵀ (ΦᵀΦ + λI)⁻¹ φ(x*)

        Returns (y_pred, y_std), both float32 shape (n_test,).
        """
        Phi_tr = self.mlp(X_train)
        Phi_te = self.mlp(X_test)

        A = self._gram(Phi_tr)
        L = torch.linalg.cholesky(A)
        beta = torch.cholesky_solve(
            (Phi_tr.T @ y_train).unsqueeze(-1), L
        ).squeeze(-1)

        y_pred = Phi_te @ beta

        # Residual variance estimate
        residuals = y_train - Phi_tr @ beta
        sigma2 = (residuals ** 2).mean().clamp(min=1e-8)

        # Posterior variance: σ² diag(Φ_te A⁻¹ Φ_te^T)
        V = torch.cholesky_solve(Phi_te.T, L)  # (K, n_test)
        var = sigma2 * (Phi_te * V.T).sum(dim=1)
        y_std = var.clamp(min=0.0).sqrt()

        return y_pred, y_std

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gram(self, Phi: torch.Tensor) -> torch.Tensor:
        """ΦᵀΦ + λI, symmetrised for numerical safety."""
        K = Phi.T @ Phi
        K = 0.5 * (K + K.T)
        lam = self.lambda_reg.to(dtype=K.dtype, device=K.device)
        return K + lam * torch.eye(self.n_basis, dtype=K.dtype, device=K.device)

    def _solve_beta(self, Phi_tr: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        """β = (ΦᵀΦ + λI)⁻¹ Φᵀ y  — differentiable w.r.t. y_train."""
        A = self._gram(Phi_tr)
        b = Phi_tr.T @ y_train
        return torch.linalg.solve(A, b)
