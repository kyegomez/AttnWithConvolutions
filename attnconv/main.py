import torch
from einops import rearrange
from torch import einsum, nn


# helpers


def exists(val):
    return val is not None


# normalization
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class ConvolutionLanguageBlock(nn.Module):
    """
    Convolutional block for language modeling.



    Args:
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        kernel_size: int
            Kernel size for the convolutional layer
        padding: int
            Padding for the convolutional layer
        activation: str
            Activation function to use


    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        activation="gelu",
        batchnorm: bool = False,
    ):
        super(ConvolutionLanguageBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.batchnorm = batchnorm

        # Select the activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("Activation function must be either relu ")

        # Define the convolutional layer
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
        )

        # Add BatchNorm
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor):
        # Convolution layer
        x = self.conv1d(x)
        
        # BatchNorm
        if self.batchnorm:
            x = self.batchnorm(x)

        # Activation
        x = self.activation(x)

        return x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# all we need
class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = RMSNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        # self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims))

        # self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim)

        # Swap out the linear layers here
        # self.ff_out = nn.Sequential(nn.GELU(), nn.Linear(ff_inner_dim, dim, bias=False))
        self.ff_out = nn.Sequential(nn.GELU(), nn.Linear(ff_inner_dim, dim))

        # for caching causal mask and rotary embeddings
        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask

        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)


# Transformer


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth,
        heads,
        dim_head: int,
        ff_mult=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        kernel_size = 3
        padding = 1

        for _ in range(depth):
            self.layers.append(
                ConvolutionLanguageBlock(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            self.layers.append(
                ParallelTransformerBlock(dim, dim_head, heads, ff_mult),
            )

    def forward(self, x):
        for block in self.layers:
            x = block(x) + x
        return x


# classes


class ATCTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_tokens,
        dim_head=64,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head, ff_mult=ff_mult
        )

        self.to_logits = nn.Sequential(RMSNorm(dim), nn.Linear(dim, num_tokens))

    def forward(self, x):
        x = self.emb(x)
        x = self.transformer(x)
        return self.to_logits(x)
