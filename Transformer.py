import math
import torch
import torch.nn as nn
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        w = torch.softmax(scores, dim=-1)
        y = w @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ : int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ ),
            nn.GELU(),
            nn.Linear(d_ , d_model),
        )
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ff(self.ln2(x))
        return x