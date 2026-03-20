from __future__ import annotations


def make_gp_layer(backend="gpytorch", **kwargs):
    backend = str(backend).strip().lower()
    if backend == "gpytorch":
        from gp_layer_gpytorch import SVGPLayer
        accepted = {
            "n_spatial_dims", "n_inducing", "nu", "jitter",
            "use_float64", "init_noise", "noise_min", "noise_max",
        }
        return SVGPLayer(**{k: v for k, v in kwargs.items() if k in accepted})
    raise ValueError(f"Unknown GP backend: {backend!r}. Only 'gpytorch' is supported.")
