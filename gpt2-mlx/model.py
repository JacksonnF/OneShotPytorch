from dataclasses import dataclass
from typing import Any, Optional
import math

import mlx
import mlx.core as mx
import mlx.nn as nn

@dataclass
class ModelArgs:
    n_layers: int = 6
    n_heads: int = 8
    embed_size: int = 512
    feed_fwd_size: int = 2048
    vocab_size: int = 10000
    dropout: float = 0.0
    bias: bool = True

class FeedForward(nn.Module):
    """Simple Feed Forward MLP"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        d = args.embed_size
        ff_d = args.feed_fwd_size
        self.fd_fwd = nn.Sequential(nn.Linear(d, ff_d), nn.GELU(), nn.Linear(ff_d, d))
        self.dropout = nn.Dropout(args.dropout)

    def __call__(self, x: mx.array):
        return self.dropout(self.fd_fwd(x))

class Attention(nn.Module):
    """Multi-Head Attention Module using Scaled Dot-Product Attention"""
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        self.d = args.embed_size
        self.head_dim = self.d // self.n_heads
        self.scale_f = math.sqrt(self.head_dim)

        self.ql = nn.Linear(
            self.d, self.d
        )  # Note: Choosing to keep dim of matrices same as latent size
        self.kl = nn.Linear(self.d, self.d)
        self.vl = nn.Linear(self.d, self.d)
        self.prj = nn.Linear(self.d, self.d)
        self.dpout1 = nn.Dropout(args.dropout)
        self.dpout2 = nn.Dropout(args.dropout)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        mask: Optional[mx.array] = None,
    ):
        # Take input sequence shape: (b_sz, sq_l, embed_dim)
        # Do the Q, K^T matrix multiplication where:
        # Q,K,V: all vectors in sequence stacked row-wise
        # divide by sqrt embed dim and softmax then multiply by V
        b_sz, sq_l, _ = q.shape
        _, sq_l_kv, _ = k.shape

        # projection
        Q, K, V = self.ql(q), self.kl(k), self.vl(v)  # shape: (b_sz, sq_l, d)

        # reshape into multi-headed view (view doesn't copy data)
        Q = Q.reshape(b_sz, sq_l, self.n_heads, self.head_dim)
        K = K.reshape(b_sz, sq_l_kv, self.n_heads, self.head_dim)
        V = V.reshape(b_sz, sq_l_kv, self.n_heads, self.head_dim)

        # Get Q, K, V stacked row-wise (b_sz, n_heads, sq_l, h_d)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        qk = (Q @ K.transpose(0, 1, 3, 2)) / self.scale_f
        if mask != None:
            qk += mask
        atn = mx.softmax(qk, axis=-1) @ V  # shape: (b_sz, n_heads, sq_l, head_d)
        atn = self.dpout1(atn)
        # concat heads and project
        atn = atn.transpose(0, 2, 1, 3)
        atn = atn.reshape(b_sz, sq_l, self.d)
        return self.dpout2(self.prj(atn))
    
class TransfomerBlock(nn.Module):
    """Decoder Style Transformer Block with Feed Forward, Attention and Residual Connections"""
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(args.embed_size, affine=args.bias)
        self.attn_1 = Attention(args)
        self.norm_2 = nn.LayerNorm(args.embed_size, affine=args.bias)
        self.fd_fwd = FeedForward(args)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None,):
        n1 = self.norm_1(x)
        at_x = self.attn_1(q=n1, k=n1, v=n1, mask=mask)
        x = x + at_x

        n2 = self.norm_2(x)
        ax_f = self.fd_fwd(n2)
        x = ax_f + x
        return x
    
class Embedding(nn.Module):
    """Simple Embeddings"""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.embed_size)

    def forward(self, x: mx.array):
        return self.embeddings(x)

def sinusoidal_pos_encoding(d_model, seq_len):
    pos_enc = mx.zeros(seq_len, d_model)
    posns = mx.arange(0, seq_len).unsqueeze(1)
    den = mx.pow(10000, mx.arange(0, d_model, 2).float() / d_model)

    pos_enc[:, 0::2] = mx.sin(posns / den)
    pos_enc[:, 1::2] = mx.cos(posns / den)

    return pos_enc

def create_mask(size):
    mask = mx.triu(mx.ones(size, size), diagonal=1)
    return mask
    
class GPT2(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d = args.embed_size
        self.embedding = Embedding(args)
        self.dropout = nn.Dropout(args.dropout)
        self.blocks = [TransfomerBlock(args) for _ in range(args.n_layers)]
        self.l_norm = nn.LayerNorm(args.embed_size, affine=args.bias)

    def __call__(self, input) -> Any:
        super().__init__()
        for b in self.blocks:
            input = b(input)
        input = self.l_norm(input)
        return input

if __name__ == "__main__":

    args = ModelArgs()
    model = GPT2(args)

    input = mx.random.randint(low=-1, high=1, shape=(64, 5, args.embed_size))
    print("input: ", input, )

    pred = model(input)
    print("\nPrediction: ", pred)