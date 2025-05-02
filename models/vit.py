import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(t.size(0), t.size(
            1), self.heads, -1).permute(0, 2, 1, 3), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(out.size(0), out.size(1), -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SelfAttention(dim, heads=heads, dim_head=dim_head),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]))

    def forward(self, x):
        for attn, ffn in self.layers:
            x = attn(x) + x
            x = ffn(x) + x
        return self.norm(x)


class SimpleViT(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, mlp_dim):
        super().__init__()

        self.transformer = Transformer(
            dim, depth, heads, dim_head=dim, mlp_dim=mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, patch_embeddings):
        # Pass patch embeddings directly through the transformer
        x = self.transformer(patch_embeddings)

        # Pooling (mean pooling over patch tokens)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)
