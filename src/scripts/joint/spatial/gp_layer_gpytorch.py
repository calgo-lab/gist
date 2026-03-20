from __future__ import annotations

import torch
import torch.nn as nn

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


class _SVGPModel(ApproximateGP):
    def __init__(self, inducing_points, ard_num_dims=2, nu=1.5):
        vd = CholeskyVariationalDistribution(inducing_points.size(0))
        vs = VariationalStrategy(self, inducing_points, vd, learn_inducing_locations=True)
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

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class SVGPLayer(nn.Module):
    def __init__(self, n_spatial_dims=2, n_inducing=64, nu=1.5, jitter=1e-4,
                 use_float64=True, init_noise=0.1, noise_min=1e-4, noise_max=2.0):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.n_inducing = n_inducing
        self.jitter = float(jitter)
        self.use_float64 = bool(use_float64)
        self._work_dtype = torch.float64 if use_float64 else torch.float32

        inducing_pts = torch.randn(n_inducing, n_spatial_dims)
        self.svgp = _SVGPModel(inducing_pts, ard_num_dims=n_spatial_dims, nu=nu)
        self.likelihood = GaussianLikelihood(noise_constraint=Interval(noise_min ** 2, noise_max ** 2))

        with torch.no_grad():
            lo = noise_min ** 2 + 1e-8
            hi = noise_max ** 2 - 1e-8
            self.likelihood.noise = torch.tensor(init_noise ** 2).clamp(lo, hi)

        with torch.no_grad():
            self.svgp.covar_module.base_kernel.lengthscale = torch.ones(1, n_spatial_dims)
            self.svgp.covar_module.outputscale = torch.tensor(1.0)

        if use_float64:
            self.svgp = self.svgp.double()
            self.likelihood = self.likelihood.double()

    @torch.no_grad()
    def initialize_inducing(self, X):
        n = min(self.n_inducing, X.size(0))
        idx = torch.randperm(X.size(0))[:n]
        pts = X[idx].to(self.svgp.variational_strategy.inducing_points.dtype)
        self.svgp.variational_strategy.inducing_points.data[:n].copy_(pts)

    def forward(self, X_train, y_train, X_test):
        dt = self._work_dtype
        y_pred, _ = self._exact_gp(X_train.to(dt), y_train.to(dt), X_test.to(dt), want_var=False)
        return y_pred.to(torch.float32)

    def forward_with_std(self, X_train, y_train, X_test):
        dt = self._work_dtype
        y_pred, y_std = self._exact_gp(X_train.to(dt), y_train.to(dt), X_test.to(dt), want_var=True)
        return y_pred.to(torch.float32), y_std.to(torch.float32)

    def marginal_log_likelihood(self, X, y):
        import math as _math
        dt = self._work_dtype
        X = X.to(dt)
        y = y.to(dt)
        n = y.size(0)
        with gpytorch.settings.debug(False):
            K = self.svgp.covar_module(X).evaluate().to(dt)
        noise_var = self.likelihood.noise.to(dt)
        reg = (noise_var + self.jitter) * torch.eye(n, dtype=dt, device=K.device)
        K_reg = 0.5 * (K + K.T) + reg
        L = self._chol_safe(K_reg)
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
        return 0.5 * (y * alpha).sum() + L.diagonal().log().sum() + 0.5 * n * _math.log(2 * _math.pi)

    def length_scale(self):
        return self.svgp.covar_module.base_kernel.lengthscale.detach().squeeze()

    def output_scale(self):
        return self.svgp.covar_module.outputscale.detach().squeeze()

    def noise(self):
        return self.likelihood.noise.detach().sqrt().squeeze()

    def _exact_gp(self, X_tr, y_tr, X_te, want_var):
        dt = X_tr.dtype
        n = y_tr.size(0)
        with gpytorch.settings.debug(False):
            K_tt = self.svgp.covar_module(X_tr).evaluate().to(dt)
            K_st = self.svgp.covar_module(X_te, X_tr).evaluate().to(dt)
            if want_var:
                K_ss_diag = self.svgp.covar_module(X_te).evaluate().to(dt).diag()
        noise_var = self.likelihood.noise.to(dt)
        reg = (noise_var + self.jitter) * torch.eye(n, dtype=dt, device=K_tt.device)
        K_reg = 0.5 * (K_tt + K_tt.T) + reg
        L = self._chol_safe(K_reg)
        alpha = torch.cholesky_solve(y_tr.unsqueeze(-1), L).squeeze(-1)
        y_pred = K_st @ alpha
        if not want_var:
            return y_pred, None
        v = torch.linalg.solve_triangular(L, K_st.T, upper=False)
        return y_pred, (K_ss_diag - (v * v).sum(0)).clamp(min=0.0).sqrt()

    def _chol_safe(self, K):
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
