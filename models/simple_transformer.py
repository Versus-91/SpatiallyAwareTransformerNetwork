import torch
import torch.nn as nn


class SimpleTransformer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4, num_layers=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.cls_token_param = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        B = x.size(0)
        cls_tokens = self.cls_token_param.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        cls_output = self.norm(x[:, 0])
        return self.classifier(cls_output).squeeze(-1)
