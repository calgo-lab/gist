import torch
from torch import nn


class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def normalize(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=self.eps)
        return (x - mean) / std, mean, std

    def denormalize(self, x, mean, std):
        return x * std + mean


class GRUSeq2Seq(nn.Module):
    def __init__(self, past_input_size, future_input_size, hidden_size,
                 num_layers, dropout, out_len, static_input_size=0, use_revin=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(
            input_size=past_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.GRU(
            input_size=future_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)
        self.out_len = out_len
        self.revin = RevIN() if use_revin else None
        if static_input_size > 0:
            self.static_proj = nn.Linear(static_input_size, hidden_size * num_layers)
        else:
            self.static_proj = None

    def forward(self, x_past, x_future, x_static=None):
        revin_mean = revin_std = None
        if self.revin is not None:
            gws_norm, revin_mean, revin_std = self.revin.normalize(x_past[:, :, 0])
            x_past = torch.cat([gws_norm.unsqueeze(-1), x_past[:, :, 1:]], dim=-1)

        if self.static_proj is not None:
            h0 = self.static_proj(x_static)
            h0 = h0.view(-1, self.encoder.num_layers, self.hidden_size)
            h0 = h0.permute(1, 0, 2).contiguous()
            _, h = self.encoder(x_past, h0)
        else:
            _, h = self.encoder(x_past)
        dec_out, _ = self.decoder(x_future, h)
        out = self.head(dec_out).squeeze(-1)

        if self.revin is not None:
            out = self.revin.denormalize(out, revin_mean, revin_std)
        return out
