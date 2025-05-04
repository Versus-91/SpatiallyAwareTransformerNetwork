import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention_scaled_dot_product(q, k, v, dist_embedding):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    attn_logits[:, :, 1:, 1:] += dist_embedding
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class Attention(nn.Module):
    """
    Attributes
    ----------
    distances_bin_width (int):
        Size of bins used for grouping distances.
    distance_embedding (nn.Embedding):
        Embedding table for distance bins, each represented by a scalar.
    head_dim (int):
        Dimensionality of each attention head.
    num_distance_bins (int):
        The total number of distinct distance bins for which embeddings will be learned.

    """

    def __init__(self, dim, num_heads, **kwargs):
        assert dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        super().__init__()
        self.embedding_dim = dim
        self.num_heads = num_heads
        self.num_distance_bins = kwargs.get("embeddings_num", 256)
        self.distances_bin_width = kwargs.get("distances_bin_width", 1024)
        self.head_dim = self.embedding_dim // num_heads
        self.to_qkv = nn.Linear(self.embedding_dim, 3 * self.embedding_dim)
        self.out = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.distance_embedding = torch.nn.Embedding(
            num_embeddings=self.num_distance_bins, embedding_dim=1)

    def forward(self, x, coordinates: None):
        batch_size, seq_length, _ = x.size()
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size, seq_length,
                          self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # a = coordinates.half().unsqueeze(2)  # [B, N, 1, 2]
        # b = coordinates.half().unsqueeze(1)  # [B, 1, N, 2]
        # distances = torch.linalg.norm(a - b, dim=-1)  # [B, N, N]
        distances = torch.cdist(coordinates.float(), coordinates.float(), p=2)
        distance_bins = (distances / self.distances_bin_width).floor().long()
        values, _ = attention_scaled_dot_product(
            q, k, v, self.distance_embedding(distance_bins).permute(0, 3, 1, 2))
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embedding_dim)
        return self.out(values)


class Transformer(nn.Module):
    """    
    Attributes
    ----------
    norm (nn.LayerNorm): 
        Layer normalization applied to the final output after all Transformer blocks.
    layers (nn.ModuleList): 
        A list of Transformer blocks.

    """

    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, **kwargs),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]))

    def forward(self, x, coordinates: None):
        for attn, ffn in self.layers:
            x = attn(x, coordinates) + x
            x = ffn(x) + x
        return self.norm(x)


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation.

    Parameters
    ----------
    dim (int): 
        Dimensionality of the input embeddings.
    depth (int): 
        Number of Transformer blocks.
    heads (int): 
        Number of attention heads in the multi-head self-attention mechanism.
    mlp_dim (int): 
        Hidden layer size of the feedforward network inside the Transformer.
    num_classes (int): 
        Number of output classes for classification.
    dropout (float, optional): 
        Dropout rate applied to both attention and MLP layers. Default is 0.1.
    **kwargs: Additional arguments.

    Attributes
    ----------
    transformer (nn.Module):
        Transformer encoder consisting of `depth` number of Transformer blocks.
    cls_tokens (shape: [1, 1, dim])):
        Learnable token that is prepended to the input sequence.
    ffn_head (Type: nn.Linear, output: [batch_size, num_classes]):
        Final classification head that maps the cls token embedding to the output class scores.
    """

    def __init__(self, dim, depth, heads, mlp_dim, num_classes, dropout=0.1, **kwargs):
        super().__init__()
        self.transformer = Transformer(
            dim, depth, heads, mlp_dim, dropout, **kwargs)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.ffn_head = nn.Linear(dim, num_classes)

    def forward(self, x, coordinates):
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x, coordinates)
        x = x[:, 0]
        return self.ffn_head(x)


if __name__ == '__main__':
    batch_size = 16
    seq_length = 16
    dim = 64  
    num_classes = 2
    x = torch.randn(batch_size, seq_length, dim)
    coords = torch.randn(batch_size, seq_length, 2)
    model = VisionTransformer(
        dim=dim,
        depth=2,
        heads=4,
        mlp_dim=128,
        num_classes=num_classes
    )
    output = model(x, coords)
    assert output.shape == (
        batch_size, num_classes), f"Expected shape ({batch_size}, {num_classes}), got {output.shape}"
    print("Output shape:", output.shape)
