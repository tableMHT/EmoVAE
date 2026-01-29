# Net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Conv1d, Conv2d, MaxPool2d
from torch_geometric.utils import to_dense_batch


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = float(alpha)

    def set_alpha(self, alpha: float):
        self.alpha = float(alpha)

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)


class VAEModule(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc_mu = Linear(int(latent_dim), int(latent_dim))
        self.fc_logvar = Linear(int(latent_dim), int(latent_dim))

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward(self, tokens: torch.Tensor):
        # tokens: [B,T,D]
        mu = self.fc_mu(tokens)
        logvar = self.fc_logvar(tokens)
        z = self.reparameterize(mu, logvar)

        # KL(q(z|x)||p(z)), sum over (T,D), mean over B
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - torch.exp(logvar), dim=(1, 2)).mean()
        return z, kl, None


class DomainClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            Dropout(float(drop)),
            Linear(int(hidden_dim), 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class _AttnBlock(nn.Module):
    """cross-attention block: MHA + residual + FFN."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(nhead),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(int(d_model))
        self.ln2 = nn.LayerNorm(int(d_model))
        self.ffn = nn.Sequential(
            Linear(int(d_model), int(dim_ff)),
            nn.GELU(),
            Dropout(float(dropout)),
            Linear(int(dim_ff), int(d_model)),
            Dropout(float(dropout)),
        )

    def forward(self, q, kv, attn_mask=None, key_padding_mask=None, need_weights: bool = False):
        qn = self.ln1(q)
        kvn = self.ln1(kv)
        attn_out, attn_w = self.attn(
            query=qn,
            key=kvn,
            value=kvn,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False if need_weights else True,
        )
        x = q + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_w


class PCDA(nn.Module):
    def __init__(
        self,
        classes: int = 3,
        channels: int = 62,
        d_model: int = 256,
        nhead: int = 8,
        num_enc_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        grl_alpha: float = 0.0,
        q_tokens: int = 8,
        time_downsample: int = 1,
        time_L: int = 20,
        time_L_ratio: float = 0.1,
    ):
        super().__init__()
        self.classes = int(classes)
        self.channels = int(channels)
        self.d_model = int(d_model)
        self.q_tokens = int(q_tokens)

        # Fixed SEED DE layout: 1325 = 5 * 265
        self.bands = 5
        self.time_len = 265

        self.time_downsample = max(1, int(time_downsample))
        self.time_L = int(time_L)
        self.time_L_ratio = float(time_L_ratio)

        # Expose last cross-attn weights for optional visualization.
        self.last_cross_attn_weights = None

        # CNN tokens are used only for reconstruction loss (keeps baseline training interface).
        self.conv1 = Conv2d(1, 32, (5, 5))
        self.drop1 = Dropout(0.1)
        self.pool1 = MaxPool2d((1, 4))

        self.conv2 = Conv2d(32, 64, (1, 5))
        self.drop2 = Dropout(0.1)
        self.pool2 = MaxPool2d((1, 4))

        self.conv3 = Conv2d(64, 128, (1, 5))
        self.drop3 = Dropout(0.1)
        self.pool3 = MaxPool2d((1, 4))

        self.token_dim = 2080 + 960 + 256
        self.token_proj = Linear(self.token_dim, self.d_model)

        # Temporal tokens: [B,62,5,T] -> [B,62,T',D]
        self.temporal_conv = nn.Sequential(
            Conv1d(self.bands, self.d_model, kernel_size=5, padding=2, stride=self.time_downsample),
            nn.GELU(),
            Dropout(float(dropout)),
            Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            Dropout(float(dropout)),
        )

        # Channel self-attention.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_enc_layers))

        # Time-selective cross-attention.
        self.learnable_q = nn.Parameter(torch.randn(1, self.q_tokens, self.d_model) * 0.02)
        self.cross_agg = _AttnBlock(self.d_model, int(nhead), int(dim_feedforward), float(dropout))

        # VAE over aggregated tokens.
        self.vae = VAEModule(latent_dim=self.d_model)

        # Cross-attn back to channels.
        self.cross_back = _AttnBlock(self.d_model, int(nhead), int(dim_feedforward), float(dropout))

        self.recon_head = Linear(self.d_model, self.token_dim)

        feat_dim = self.channels * self.d_model
        self.drop_feat = Dropout(float(dropout))
        self.classifier = Linear(feat_dim, self.classes)

        self.grl = GRL(alpha=float(grl_alpha))
        self.domain_classifier = DomainClassifier(feat_dim)

    def set_grl_alpha(self, alpha: float):
        self.grl.set_alpha(alpha)

    @staticmethod
    def _neg_inf(dtype: torch.dtype) -> float:
        return -1e4 if dtype in (torch.float16, torch.bfloat16) else float("-inf")

    def _build_time_selective_attn_mask(self, T: int, device: torch.device, dtype: torch.dtype):
        K = self.q_tokens
        if self.time_L > 0:
            L = int(self.time_L)
        else:
            L = max(1, int(round(self.time_L_ratio * float(T))))
        L = max(1, min(L, T))

        anchors = torch.zeros(1, device=device, dtype=torch.long) if K == 1 else \
            torch.linspace(0, T - 1, steps=K, device=device).round().to(torch.long)

        idx = torch.arange(T, device=device, dtype=torch.long).view(1, T)
        anchors2 = anchors.view(K, 1)
        allowed = (idx < L) | (idx >= anchors2)

        attn_mask = torch.zeros((K, T), device=device, dtype=dtype)
        attn_mask = attn_mask.masked_fill(~allowed, self._neg_inf(dtype))
        return attn_mask, L, anchors

    def _cnn_to_tokens(self, x_dense: torch.Tensor):
        """x_dense:[B,62,1325] -> cnn_tokens:[B,62,token_dim]."""
        B, C, _ = x_dense.shape
        x = x_dense.reshape(-1, 1, 5, 265)

        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = self.pool1(x)
        t1 = x.flatten(1)

        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.pool2(x)
        t2 = x.flatten(1)

        x = F.relu(self.conv3(x))
        x = self.drop3(x)
        x = self.pool3(x)
        t3 = x.flatten(1)

        tokens = torch.cat([t1, t2, t3], dim=1)
        return tokens.view(B, C, -1)

    def forward(self, x, batch):
        # x: node features, batch: PyG batch vector
        x_dense, _ = to_dense_batch(x, batch)          # [B,62,1325]
        B = x_dense.size(0)

        cnn_tokens = self._cnn_to_tokens(x_dense)      # [B,62,token_dim]

        # Restore [B,62,5,T] for temporal tokenization.
        x_de = x_dense.view(B, self.channels, self.bands, self.time_len)
        xt = x_de.reshape(B * self.channels, self.bands, self.time_len)  # [B*62,5,T]

        time_feat = self.temporal_conv(xt)             # [B*62,D,T']
        T_prime = int(time_feat.size(-1))
        time_tokens = time_feat.transpose(1, 2).contiguous().view(
            B, self.channels, T_prime, self.d_model
        )                                              # [B,62,T',D]

        # Channel self-attention at each time step.
        ch_in = time_tokens.permute(0, 2, 1, 3).contiguous().view(
            B * T_prime, self.channels, self.d_model
        )                                              # [B*T',62,D]
        ch_mem = self.encoder(ch_in).view(
            B, T_prime, self.channels, self.d_model
        )                                              # [B,T',62,D]

        # Time KV sequence.
        time_kv = ch_mem.mean(dim=2)                   # [B,T',D]

        # Time-selective cross-attn with learnable queries.
        q = self.learnable_q.expand(B, -1, -1).contiguous()  # [B,K,D]
        attn_mask, L_used, anchors = self._build_time_selective_attn_mask(T_prime, q.device, q.dtype)
        agg_tokens, agg_attn_w = self.cross_agg(q=q, kv=time_kv, attn_mask=attn_mask, need_weights=True)
        self.last_cross_attn_weights = agg_attn_w

        # VAE over aggregated tokens.
        z_lat, kl_loss, _ = self.vae(agg_tokens)

        # Cross-attn back to channels.
        memory_ch = ch_mem.mean(dim=1)                 # [B,62,D]
        quant_tokens, _ = self.cross_back(q=memory_ch, kv=z_lat, need_weights=False)  # [B,62,D]

        recon_tokens = self.recon_head(quant_tokens)   # [B,62,token_dim]
        quant_flat = quant_tokens.reshape(B, -1)       # [B,62*D]
        feat = self.drop_feat(quant_flat)

        dom_feat = self.grl(feat)
        domain_out = self.domain_classifier(dom_feat).squeeze(1)

        class_logits = self.classifier(feat)
        pred = F.softmax(class_logits, dim=1)

        aux = {
            "vq_loss": kl_loss,
            "vq_indices": None,
            "cnn_tokens": cnn_tokens,
            "recon_tokens": recon_tokens,
            "quant_tokens": quant_tokens,
            "quant_flat": quant_flat,
            "agg_tokens": agg_tokens,
            "latent_tokens": z_lat,
            "time_kv": time_kv,
            "time_T": T_prime,
            "time_L": int(L_used),
            "time_anchors": anchors,
        }
        return class_logits, pred, domain_out, aux
