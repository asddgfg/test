import torch
import torch.nn as nn


# =========================================================
# 1. TCN
# =========================================================
class TCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_channels = input_dim

        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation

            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            in_channels = hidden_dim

        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, T, F]
        x = x.transpose(1, 2)  # -> [B, F, T]
        x = self.net(x)

        # Padding makes the sequence longer; use the last effective timestep.
        x = x[:, :, -1]  # -> [B, hidden_dim]
        return self.head(x)


# =========================================================
# 2. N-BEATS-like
# =========================================================
class NBeats(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim * seq_len

        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, T, F]
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        x = self.net(x)
        return self.head(x)


# =========================================================
# 3. Kimi-style Transformer
# NOTE:
# This is a lightweight causal Transformer-style baseline,
# not an official reproduction of any proprietary Kimi model.
# =========================================================
def build_causal_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # True means "blocked" for nn.MultiheadAttention attn_mask.
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class KimiBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float, ffn_dim: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        attn_mask = build_causal_attention_mask(seq_len=seq_len, device=x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class KimiModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        ffn_dim: int = 128,
    ):
        super().__init__()

        self.proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            KimiBlock(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                ffn_dim=ffn_dim,
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, T, F]
        x = self.proj(x)

        for layer in self.layers:
            x = layer(x)

        return self.head(x[:, -1, :])


# =========================================================
# Model configs (small / medium / large)
# =========================================================
def get_deep_models():
    return {
        # -------------------------------------------------
        # TCN
        # -------------------------------------------------
        "tcn_small": (
            "tcn",
            {
                "hidden_dim": 32,
                "num_layers": 2,
                "kernel_size": 3,
                "dropout": 0.1,
            },
        ),
        "tcn_medium": (
            "tcn",
            {
                "hidden_dim": 64,
                "num_layers": 3,
                "kernel_size": 3,
                "dropout": 0.1,
            },
        ),
        "tcn_large": (
            "tcn",
            {
                "hidden_dim": 128,
                "num_layers": 4,
                "kernel_size": 5,
                "dropout": 0.2,
            },
        ),

        # -------------------------------------------------
        # N-BEATS-like
        # -------------------------------------------------
        "nbeats_small": (
            "nbeats",
            {
                "hidden_dim": 128,
                "n_layers": 2,
                "dropout": 0.1,
            },
        ),
        "nbeats_medium": (
            "nbeats",
            {
                "hidden_dim": 256,
                "n_layers": 3,
                "dropout": 0.1,
            },
        ),
        "nbeats_large": (
            "nbeats",
            {
                "hidden_dim": 512,
                "n_layers": 4,
                "dropout": 0.2,
            },
        ),

        # -------------------------------------------------
        # Kimi-style
        # -------------------------------------------------
        "kimi_small": (
            "kimi",
            {
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.1,
                "ffn_dim": 128,
            },
        ),
        "kimi_medium": (
            "kimi",
            {
                "d_model": 128,
                "nhead": 4,
                "num_layers": 3,
                "dropout": 0.1,
                "ffn_dim": 256,
            },
        ),
        "kimi_large": (
            "kimi",
            {
                "d_model": 256,
                "nhead": 8,
                "num_layers": 4,
                "dropout": 0.2,
                "ffn_dim": 512,
            },
        ),
    }


def build_model(model_type, input_dim, seq_len, params):
    if model_type == "tcn":
        return TCN(input_dim=input_dim, **params)

    if model_type == "nbeats":
        return NBeats(input_dim=input_dim, seq_len=seq_len, **params)

    if model_type == "kimi":
        return KimiModel(input_dim=input_dim, **params)

    raise ValueError(f"Unknown model_type: {model_type}")
