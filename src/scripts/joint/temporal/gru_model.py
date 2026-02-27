import torch
from torch import nn


class GRUSeq2Seq(nn.Module):
    def __init__(self, past_input_size, future_input_size, hidden_size,
                 num_layers, dropout, out_len, static_input_size=0):
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
        if static_input_size > 0:
            self.static_proj = nn.Linear(static_input_size, hidden_size * num_layers)
        else:
            self.static_proj = None

    def forward(self, x_past, x_future, x_static=None):
        if self.static_proj is not None and x_static is not None:
            h0 = self.static_proj(x_static)
            h0 = h0.view(-1, self.encoder.num_layers, self.hidden_size)
            h0 = h0.permute(1, 0, 2).contiguous()
            _, h = self.encoder(x_past, h0)
        else:
            _, h = self.encoder(x_past)
        dec_out, _ = self.decoder(x_future, h)
        return self.head(dec_out).squeeze(-1)
