from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class ModelArgs:
    n_layers: int = 6
    n_heads: int = 8
    embed_size: int = 512
    feed_fwd_size: int = 2048
    vocab_size: int = 10000


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.d = args.embed_size
        self.embedding = Embedding(args)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(args) for _ in range(args.n_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(args) for _ in range(args.n_layers)]
        )
        self.linear = nn.Linear(args.embed_size, args.vocab_size)
        self.softmax = nn.Softmax(-1)

    def forward(self, input, output):
        b_sz, seq_l_in, _ = input.shape
        _, seq_l_out, _ = output.shape

        # input, output = self.embedding(input), self.embedding(output)
        input = input + sinusoidal_pos_encoding(self.d, seq_l_in)
        output = output + sinusoidal_pos_encoding(self.d, seq_l_out)

        for enc in self.encoder_layers:
            input = enc(input)

        mask = create_mask(seq_l_out) * -1e9
        for dec in self.decoder_layers:
            output = dec(output, input, mask=mask)

        output = self.linear(output)
        return self.softmax(output)


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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
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
        Q = Q.view(b_sz, sq_l, self.n_heads, self.head_dim)
        K = K.view(b_sz, sq_l_kv, self.n_heads, self.head_dim)
        V = V.view(b_sz, sq_l_kv, self.n_heads, self.head_dim)

        # Get Q, K, V stacked row-wise (b_sz, n_heads, sq_l, h_d)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        qk = torch.matmul(Q, K.transpose(2, 3)) / self.scale_f
        if mask != None:
            qk += mask
        atn = torch.matmul(
            F.softmax(qk, dim=-1), V
        )  # shape: (b_sz, n_heads, sq_l, head_d)

        # concat heads and project
        atn = atn.permute(0, 2, 1, 3).contiguous()
        atn = atn.view(b_sz, sq_l, self.d)
        return self.prj(atn)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        d = args.embed_size
        ff_d = args.feed_fwd_size

        self.fd_fwd = nn.Sequential(nn.Linear(d, ff_d), nn.ReLU(), nn.Linear(ff_d, d))

    def forward(self, x: torch.Tensor):
        return self.fd_fwd(x)


class EncoderLayer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.attn = Attention(args)
        self.fd_fwd = FeedForward(args)
        self.atn_norm = nn.LayerNorm(args.embed_size)
        self.fwd_norm = nn.LayerNorm(args.embed_size)

    def forward(self, x):
        at_x = self.attn(q=x, k=x, v=x)
        at_x = self.atn_norm(x + at_x)

        f_x = self.fd_fwd(at_x)
        ret = self.fwd_norm(f_x + at_x)
        return ret


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.attn_1 = Attention(args)
        self.norm_1 = nn.LayerNorm(args.embed_size)
        self.attn_2 = Attention(args)
        self.norm_2 = nn.LayerNorm(args.embed_size)
        self.fd_fwd = FeedForward(args)
        self.norm_3 = nn.LayerNorm(args.embed_size)

    def forward(self, x: torch.Tensor, enc: torch.Tensor, mask: torch.Tensor):
        at_x = self.attn_1(q=x, k=x, v=x, mask=mask)
        at_x = self.norm_1(at_x + x)

        att_x = self.attn_2(q=at_x, k=enc, v=enc)
        ax = self.norm_2(at_x + att_x)

        ax_f = self.fd_fwd(ax)
        ax = self.norm_3(ax + ax_f)
        return ax


class Embedding(nn.Module):
    """Simple Embeddings"""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.embed_size)

    def forward(self, x: torch.Tensor):
        return self.embeddings(x)

def sinusoidal_pos_encoding(d_model, seq_len):
    pos_enc = torch.zeros(seq_len, d_model)
    posns = torch.arange(0, seq_len).unsqueeze(1)
    den = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)

    pos_enc[:, 0::2] = torch.sin(posns / den)
    pos_enc[:, 1::2] = torch.cos(posns / den)

    return pos_enc

def create_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask

if __name__ == "__main__":
    torch.random.manual_seed(42)

    args = ModelArgs()
    model = Transformer(args)

    input = torch.randint(-1, 1, size=(64, 5, args.embed_size))
    output = torch.randint(-1, 1, size=(64, 1, args.embed_size))
    print("input: ", input, " Output: ", output)

    pred = model(input, output)
    print("\nPrediction: ", pred, "\n\nPrediction Size: ", pred.size())

