from __future__ import annotations

import math
import torch
import torch.nn as nn


def matern32_kernel(X1, X2, length_scale, output_scale):
    diff = X1.unsqueeze(1) - X2.unsqueeze(0)
    r2 = (diff / length_scale).pow(2).sum(-1)
    r = r2.clamp(min=0).sqrt()
    sqrt3r = math.sqrt(3) * r
    return output_scale.pow(2) * (1.0 + sqrt3r) * torch.exp(-sqrt3r)


def rbf_kernel(X1, X2, length_scale, output_scale):
    diff = X1.unsqueeze(1) - X2.unsqueeze(0)
    r2 = (diff / length_scale).pow(2).sum(-1)
    return output_scale.pow(2) * torch.exp(-0.5 * r2)


class GPLayer(nn.Module):
    def __init__(
        self,
        n_spatial_dims=2,
        init_length_scale=1.0,
        init_output_scale=1.0,
        init_noise=0.1,
        kernel_type="matern32",
        isotropic=True,
        jitter=1e-5,
    ):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.kernel_type = str(kernel_type).lower()
        self.isotropic = bool(isotropic)
        self.jitter = jitter

        if self.isotropic:
            init_ls = torch.tensor(math.log(init_length_scale), dtype=torch.float32)
        else:
            init_ls = torch.full((n_spatial_dims,), math.log(init_length_scale), dtype=torch.float32)
        self.log_length_scale = nn.Parameter(init_ls)
        self.log_output_scale = nn.Parameter(torch.tensor(math.log(init_output_scale)))
        self.log_noise         = nn.Parameter(torch.tensor(math.log(init_noise)))

    @staticmethod
    def _softplus(x):
        return nn.functional.softplus(x)

    def length_scale(self):
        return self._softplus(self.log_length_scale)

    def output_scale(self):
        return self._softplus(self.log_output_scale)

    def noise(self):
        return self._softplus(self.log_noise)

    def _kernel(self, X1, X2):
        ls = self.length_scale()
        os_ = self.output_scale()
        if self.kernel_type == "rbf":
            return rbf_kernel(X1, X2, ls, os_)
        return matern32_kernel(X1, X2, ls, os_)

    def forward(
        self,
        X_train,
        y_train,
        X_test,
    ):
        noise = self.noise()

        K_tt = self._kernel(X_train, X_train)
        K_tt = K_tt + (noise.pow(2) + self.jitter) * torch.eye(K_tt.size(0), device=K_tt.device, dtype=K_tt.dtype)

        K_st = self._kernel(X_test, X_train)

        L = torch.linalg.cholesky(K_tt)
        alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L).squeeze(-1)
        y_pred = K_st @ alpha
        return y_pred

    def forward_with_std(
        self,
        X_train,
        y_train,
        X_test,
    ):
        noise = self.noise()

        K_tt = self._kernel(X_train, X_train)
        K_tt = K_tt + (noise.pow(2) + self.jitter) * torch.eye(K_tt.size(0), device=K_tt.device, dtype=K_tt.dtype)
        K_ss = self._kernel(X_test, X_test)
        K_st = self._kernel(X_test, X_train)

        L = torch.linalg.cholesky(K_tt)
        alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L).squeeze(-1)
        y_pred = K_st @ alpha

        v = torch.linalg.solve_triangular(L, K_st.T, upper=False)
        var = K_ss.diag() - (v * v).sum(0)
        y_std = var.clamp(min=0).sqrt()
        return y_pred, y_std

    def marginal_log_likelihood(
        self,
        X,
        y,
    ):
        noise = self.noise()
        N = y.size(0)

        K = self._kernel(X, X)
        K = K + (noise.pow(2) + self.jitter) * torch.eye(N, device=K.device, dtype=K.dtype)

        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)

        data_fit   = 0.5 * (y * alpha).sum()
        complexity = L.diagonal().log().sum()
        constant   = 0.5 * N * math.log(2 * math.pi)

        return data_fit + complexity + constant
