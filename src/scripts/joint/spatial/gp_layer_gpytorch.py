"""
gp_layer_gpytorch.py — gpytorch-backed GP layer for joint GRU+GP training.

SVGPLayer provides two operating modes:

  forward(X_train, y_train, X_test)
      Exact GP predictive mean.  Gradients flow through y_train.
      Used in the joint GRU+GP training loop (Phase 2 spatial step).

  forward_with_std(X_train, y_train, X_test)
      Same as forward, but also returns predictive std.
      Used in evaluation / standalone GP eval.

  elbo_loss(X_train, y_train, n_data)
      SVGP variational ELBO (negative, to be minimised).
      Used in kernel pretraining (gp_eval.py --backend gpytorch).

Kernel: Matern-3/2 with per-dimension ARD lengthscales via gpytorch.
Inducing points are learnable; initialise from data via initialize_inducing().
Float64 computation is optional and only affects the Cholesky path.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


# ---------------------------------------------------------------------------
# Inner gpytorch SVGP model (used for ELBO pretraining only)
# ---------------------------------------------------------------------------

class _SVGPModel(ApproximateGP):
    """Thin gpytorch SVGP.  Only used to compute ELBO in elbo_loss()."""

    def __init__(self, inducing_points: torch.Tensor, ard_num_dims: int = 2, nu: float = 1.5):
        vd = CholeskyVariationalDistribution(inducing_points.size(0))
        vs = VariationalStrategy(
            self, inducing_points, vd, learn_inducing_locations=True
        )
        super().__init__(vs)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=nu,
                ard_num_dims=ard_num_dims,
                lengthscale_constraint=Interval(0.05, 20.0),
            ),
            outputscale_constraint=Interval(0.05, 20.0),
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


# ---------------------------------------------------------------------------
# Public SVGPLayer
# ---------------------------------------------------------------------------

class SVGPLayer(nn.Module):
    """
    gpytorch Matern-3/2 ARD GP layer.

    The kernel (plus noise, inducing points, variational parameters) are
    standard nn.Parameters; the whole module can be added to any optimizer.

    Exact-GP forward path (forward / forward_with_std):
        y_pred = K(X_test, X_train) @ (K(X_train,X_train) + noise·I)^{-1} @ y_train
        Gradient flows from y_pred back through y_train to the GRU.

    SVGP ELBO path (elbo_loss):
        Uses gpytorch's VariationalELBO with mini-batch support.
        Useful for pretraining the kernel on large datasets.
    """

    def __init__(
        self,
        n_spatial_dims: int = 2,
        n_inducing: int = 64,
        nu: float = 1.5,
        jitter: float = 1e-4,
        use_float64: bool = True,
        init_noise: float = 0.1,
        noise_min: float = 1e-4,
        noise_max: float = 2.0,
    ):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.n_inducing = n_inducing
        self.jitter = float(jitter)
        self.use_float64 = bool(use_float64)
        self._work_dtype = torch.float64 if use_float64 else torch.float32

        # Random initial inducing points in [-1, 1]^d (will be overwritten by
        # initialize_inducing() before any serious training).
        inducing_pts = torch.randn(n_inducing, n_spatial_dims)

        self.svgp = _SVGPModel(inducing_pts, ard_num_dims=n_spatial_dims, nu=nu)
        self.likelihood = GaussianLikelihood(
            noise_constraint=Interval(noise_min ** 2, noise_max ** 2)
        )

        # Set initial noise variance (constraint is on variance, not std).
        with torch.no_grad():
            raw = self.likelihood.noise_covar.raw_noise
            target = torch.tensor(init_noise ** 2, dtype=raw.dtype)
            # clamp to constraint range before setting raw
            lo = noise_min ** 2 + 1e-8
            hi = noise_max ** 2 - 1e-8
            target = target.clamp(lo, hi)
            self.likelihood.noise = target

        
        with torch.no_grad():
            self.svgp.covar_module.base_kernel.lengthscale = torch.ones(1, n_spatial_dims)
            self.svgp.covar_module.outputscale = torch.tensor(1.0)

        if use_float64:
            self.svgp = self.svgp.double()
            self.likelihood = self.likelihood.double()

    # ------------------------------------------------------------------
    # Initialize inducing points from data (call before training)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def initialize_inducing(self, X: torch.Tensor) -> None:
        """Place inducing points at a random subset of X."""
        n = min(self.n_inducing, X.size(0))
        idx = torch.randperm(X.size(0))[:n]
        pts = X[idx].to(self.svgp.variational_strategy.inducing_points.dtype)
        self.svgp.variational_strategy.inducing_points.data[:n].copy_(pts)

    # ------------------------------------------------------------------
    # Exact GP forward (joint training — gradients through y_train)
    # ------------------------------------------------------------------

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """
        Exact GP predictive mean.

        Gradients w.r.t. y_train are preserved so that the spatial loss
        can update GRU parameters via backprop.

        Returns float32 tensor of shape (n_test,).
        """
        dt = self._work_dtype
        X_tr = X_train.to(dt)
        y_tr = y_train.to(dt)
        X_te = X_test.to(dt)

        y_pred, _ = self._exact_gp(X_tr, y_tr, X_te, want_var=False)
        return y_pred.to(torch.float32)

    def forward_with_std(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Exact GP predictive mean + std.

        Returns (y_pred, y_std), both float32 shape (n_test,).
        """
        dt = self._work_dtype
        X_tr = X_train.to(dt)
        y_tr = y_train.to(dt)
        X_te = X_test.to(dt)

        y_pred, y_std = self._exact_gp(X_tr, y_tr, X_te, want_var=True)
        return y_pred.to(torch.float32), y_std.to(torch.float32)

    # ------------------------------------------------------------------
    # SVGP ELBO (kernel pretraining)
    # ------------------------------------------------------------------

    def elbo_loss(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        n_data: int,
    ) -> torch.Tensor:
        """
        Negative SVGP ELBO (scalar, to be minimised).

        Call with model in train() mode.  n_data is the total dataset size
        (used to scale the KL term correctly when using mini-batches).
        """
        dt = self._work_dtype
        mll = VariationalELBO(self.likelihood, self.svgp, num_data=n_data)
        output = self.svgp(X_train.to(dt))
        return -mll(output, y_train.to(dt))

    # ------------------------------------------------------------------
    # Compatibility shims matching old GPLayer interface
    # ------------------------------------------------------------------

    def length_scale(self) -> torch.Tensor:
        return self.svgp.covar_module.base_kernel.lengthscale.detach().squeeze()

    def output_scale(self) -> torch.Tensor:
        return self.svgp.covar_module.outputscale.detach().squeeze()

    def noise(self) -> torch.Tensor:
        """Return noise *std* (same convention as old GPLayer)."""
        return self.likelihood.noise.detach().sqrt().squeeze()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _exact_gp(
        self,
        X_tr: torch.Tensor,
        y_tr: torch.Tensor,
        X_te: torch.Tensor,
        want_var: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Core exact-GP computation in the working dtype.
        Gradient w.r.t. y_tr is preserved.
        """
        dt = X_tr.dtype
        n = y_tr.size(0)

        # Cast kernel parameters to working dtype if needed.
        with gpytorch.settings.debug(False):
            K_tt = self.svgp.covar_module(X_tr).evaluate().to(dt)
            K_st = self.svgp.covar_module(X_te, X_tr).evaluate().to(dt)
            if want_var:
                K_ss_diag = self.svgp.covar_module(X_te).evaluate().to(dt).diag()

        noise_var = self.likelihood.noise.to(dt)
        reg = (noise_var + self.jitter) * torch.eye(n, dtype=dt, device=K_tt.device)
        K_reg = 0.5 * (K_tt + K_tt.T) + reg

        L = self._chol_safe(K_reg)
        # alpha = K_reg^{-1} y_tr  — gradient flows through y_tr here
        alpha = torch.cholesky_solve(y_tr.unsqueeze(-1), L).squeeze(-1)
        y_pred = K_st @ alpha

        if not want_var:
            return y_pred, None

        v = torch.linalg.solve_triangular(L, K_st.T, upper=False)
        var = K_ss_diag - (v * v).sum(0)
        y_std = var.clamp(min=0.0).sqrt()
        return y_pred, y_std

    def _chol_safe(self, K: torch.Tensor) -> torch.Tensor:
        """Cholesky with adaptive jitter retry."""
        jitter = self.jitter
        for _ in range(12):
            try:
                return torch.linalg.cholesky(K)
            except RuntimeError:
                K = K + jitter * torch.eye(K.size(0), dtype=K.dtype, device=K.device)
                jitter *= 10.0
        raise RuntimeError(
            f"SVGPLayer: Cholesky failed after 12 jitter retries "
            f"(final jitter={jitter:.2e}, K range=[{K.min():.3g},{K.max():.3g}])"
        )
