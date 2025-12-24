import math
import torch
from torch import nn
import timm


def causal_mask(T, device):
    return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()


# -------------------------------------------------
# Temporal Convolution (local refinement)
# -------------------------------------------------
class TemporalConvBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


# -------------------------------------------------
# Image Encoder
# -------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, model_name, pretrained=True, freeze=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.out_dim = self.backbone.num_features

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


# -------------------------------------------------
# Improved TS Encoder (deeper, still light)
# -------------------------------------------------
class TS_Encoder(nn.Module):
    def __init__(self, in_dim, embed_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------
# Positional Encoding
# -------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# -------------------------------------------------
# Bidirectional Cross-Modal Fusion
# -------------------------------------------------
class BiCrossModalFusion(nn.Module):
    def __init__(self, img_dim, ts_dim, out_dim, dropout=0.1):
        super().__init__()

        self.img_proj = nn.Linear(img_dim, ts_dim)

        self.ts_to_img = nn.MultiheadAttention(
            ts_dim, num_heads=4, batch_first=True
        )
        self.img_to_ts = nn.MultiheadAttention(
            ts_dim, num_heads=4, batch_first=True
        )

        self.fusion_ffn = nn.Sequential(
            nn.Linear(ts_dim * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, ts, img):
        img = self.img_proj(img)

        ts_attn, _ = self.ts_to_img(ts, img, img)
        img_attn, _ = self.img_to_ts(img, ts, ts)

        fused = torch.cat([ts_attn, img_attn], dim=-1)
        fused = self.fusion_ffn(fused)

        return self.norm(fused)


# -------------------------------------------------
# Temporal Transformer
# -------------------------------------------------
class TemporalTransformer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)

    def forward(self, x):
        mask = causal_mask(x.size(1), x.device)
        return self.encoder(x, mask)


# -------------------------------------------------
# Multimodal Forecaster (Improved)
# -------------------------------------------------
class MultimodalForecaster(nn.Module):
    def __init__(
        self,
        sky_encoder,
        ts_feat_dim,
        ts_embed_dim=64,
        fused_dim=128,
        horizon=25,
        target_dim=1,
        dropout=0.1,
    ):
        super().__init__()

        self.sky_encoder = sky_encoder
        self.ts_encoder = TS_Encoder(ts_feat_dim, ts_embed_dim, dropout)

        self.cross_fusion = BiCrossModalFusion(
            img_dim=sky_encoder.out_dim,
            ts_dim=ts_embed_dim,
            out_dim=fused_dim,
            dropout=dropout,
        )

        self.temp_conv = TemporalConvBlock(fused_dim, dropout)
        self.pos_enc = PositionalEncoding(fused_dim)
        self.temporal_tf = TemporalTransformer(fused_dim, dropout)

        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, horizon * target_dim),
        )

        self.horizon = horizon
        self.target_dim = target_dim

    def forward(self, sky, ts):
        B, T_img = sky.shape[:2]

        B, T, C, H, W = sky.shape
        sky = self.sky_encoder(
            sky.view(B * T, C, H, W)
        ).view(B, T, -1)

        ts = self.ts_encoder(ts)
        ts_img = ts[:, -T_img:]

        fused = self.cross_fusion(ts_img, sky)
        fused = fused + self.temp_conv(fused)

        fused = self.pos_enc(fused)
        fused = self.temporal_tf(fused)

        context = fused[:, -1]
        out = self.head(context)

        return out.view(B, self.horizon, self.target_dim)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sky_enc = ImageEncoder("resnet18", pretrained=True)

    model = MultimodalForecaster(
        sky_encoder=sky_enc,
        ts_feat_dim=3,
    ).to(device)

    B, T_img, T_ts = 2, 5, 30
    sky = torch.randn(B, T_img, 3, 224, 224).to(device)
    ts = torch.randn(B, T_ts, 3).to(device)

    y = model(sky, ts)
    print("Output shape:", y.shape)
