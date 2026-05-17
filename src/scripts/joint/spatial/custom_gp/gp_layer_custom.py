from __future__ import annotations

import math
import torch
import torch.nn as nn


def matern32_kernel(X1, X2, length_scale, output_scale):
    diff = X1.unsqueeze(1) - X2.unsqueeze(0)
    r2 = (diff / length_scale).pow(2).sum(-1)
    r = (r2 + 1e-8).sqrt()  # epsilon avoids undefined gradient at r=0
    sqrt3r = math.sqrt(3) * r
    return output_scale.pow(2) * (1.0 + sqrt3r) * torch.exp(-sqrt3r)


def rbf_kernel(X1, X2, length_scale, output_scale):
    diff = X1.unsqueeze(1) - X2.unsqueeze(0)
    r2 = (diff / length_scale).pow(2).sum(-1)
    return output_scale.pow(2) * torch.exp(-0.5 * r2)


class GPLayer(nn.Module):
    def __init__(self, n_spatial_dims=2, init_length_scale=1.0, init_output_scale=1.0,
                 init_noise=0.1, kernel_type="matern32", isotropic=True, jitter=1e-5,
                 mll_diag_eps=1e-4):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.kernel_type = str(kernel_type).lower()
        self.isotropic = bool(isotropic)
        self.jitter = jitter
        self.mll_diag_eps = float(mll_diag_eps)

        if self.isotropic:
            init_ls = torch.tensor(float(init_length_scale), dtype=torch.float32)
        else:
            init_ls = torch.full((n_spatial_dims,), float(init_length_scale), dtype=torch.float32)

        self.ls_min, self.ls_max = 0.05, 20.0
        self.os_min, self.os_max = 0.05, 20.0
        self.noise_min, self.noise_max = 1e-4, 2.0

        self.raw_length_scale = nn.Parameter(self._raw_from_init(init_ls, self.ls_min, self.ls_max))
        self.raw_output_scale = nn.Parameter(
            self._raw_from_init(torch.tensor(float(init_output_scale), dtype=torch.float32), self.os_min, self.os_max)
        )
        self.raw_noise = nn.Parameter(
            self._raw_from_init(torch.tensor(float(init_noise), dtype=torch.float32), self.noise_min, self.noise_max)
        )

    @staticmethod
    def _raw_from_init(v, lo, hi):
        eps = 1e-6
        v = v.clamp(min=lo + eps, max=hi - eps)
        z = (v - lo) / (hi - lo)
        return torch.logit(z.clamp(min=eps, max=1 - eps))

    @staticmethod
    def _bounded(raw, lo, hi):
        return lo + (hi - lo) * torch.sigmoid(raw)

    def length_scale(self):
        return self._bounded(self.raw_length_scale, self.ls_min, self.ls_max)

    def output_scale(self):
        return self._bounded(self.raw_output_scale, self.os_min, self.os_max)

    def noise(self):
        return self._bounded(self.raw_noise, self.noise_min, self.noise_max)

    def set_hyperparameters(self, length_scale, output_scale, noise):
        ls_t = torch.as_tensor(length_scale, dtype=self.raw_length_scale.dtype, device=self.raw_length_scale.device)
        os_t = torch.as_tensor(float(output_scale), dtype=self.raw_output_scale.dtype, device=self.raw_output_scale.device)
        nz_t = torch.as_tensor(float(noise), dtype=self.raw_noise.dtype, device=self.raw_noise.device)
        with torch.no_grad():
            self.raw_length_scale.copy_(self._raw_from_init(ls_t, self.ls_min, self.ls_max))
            self.raw_output_scale.copy_(self._raw_from_init(os_t, self.os_min, self.os_max))
            self.raw_noise.copy_(self._raw_from_init(nz_t, self.noise_min, self.noise_max))

    def _kernel(self, X1, X2):
        ls = self.length_scale()
        os_ = self.output_scale()
        if self.kernel_type == "rbf":
            return rbf_kernel(X1, X2, ls, os_)
        return matern32_kernel(X1, X2, ls, os_)

    def _cholesky_with_jitter(self, K, base_diag):
        K = 0.5 * (K + K.T)
        K = torch.nan_to_num(K, nan=0.0, posinf=1e6, neginf=-1e6)
        I = torch.eye(K.size(0), device=K.device, dtype=K.dtype)
        jitter = float(self.jitter)
        last_err = None
        for _ in range(12):
            try:
                return torch.linalg.cholesky(K + (base_diag + jitter) * I)
            except RuntimeError as e:
                last_err = e
                jitter *= 10.0
        raise last_err

    def forward(self, X_train, y_train, X_test):
        noise = self.noise()
        K_tt = self._kernel(X_train, X_train)
        K_st = self._kernel(X_test, X_train)
        L = self._cholesky_with_jitter(K_tt, noise.pow(2))
        alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L).squeeze(-1)
        return K_st @ alpha

    def forward_with_std(self, X_train, y_train, X_test):
        noise = self.noise()
        K_tt = self._kernel(X_train, X_train)
        K_ss = self._kernel(X_test, X_test)
        K_st = self._kernel(X_test, X_train)
        L = self._cholesky_with_jitter(K_tt, noise.pow(2))
        alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L).squeeze(-1)
        y_pred = K_st @ alpha
        v = torch.linalg.solve_triangular(L, K_st.T, upper=False)
        var = K_ss.diag() - (v * v).sum(0)
        return y_pred, var.clamp(min=0).sqrt()

    def marginal_log_likelihood(self, X, y):
        X = X.to(dtype=torch.float64)
        y = y.to(dtype=torch.float64)
        noise = self.noise().to(dtype=torch.float64)
        N = y.size(0)
        K = self._kernel(X, X).to(dtype=torch.float64)
        L = self._cholesky_with_jitter(K, noise.pow(2) + self.mll_diag_eps)
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
        data_fit = 0.5 * (y * alpha).sum()
        complexity = L.diagonal().log().sum()
        constant = 0.5 * N * math.log(2 * math.pi)
        return data_fit + complexity + constant
