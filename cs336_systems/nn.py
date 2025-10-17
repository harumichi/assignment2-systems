import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

from einops import einsum, rearrange


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    e_x = torch.exp(x - x.amax(dim=dim, keepdim=True))
    return e_x / e_x.sum(dim=dim, keepdim=True)


@nvtx.range("scaled_dot_product_attention")
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = query.shape[-1]
    scores = einsum(query, key, '... n d_k, ... m d_k -> ... n m') / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attn = softmax(scores, dim=-1)
    return einsum(attn, value, '... n m, ... m d_v -> ... n d_v')


@nvtx.range("cross_entropy")
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, reduction='mean') -> torch.Tensor:
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
    gathered = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    if reduction == 'mean':
        return -gathered.mean()
    elif reduction == 'sum':
        return -gathered.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(
            weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std,
        )
        self.weight = nn.Parameter(weight)

    @nvtx.range("Linear")
    def forward(self, x):
        return einsum(self.weight, x, 'd_out d_in, ... d_in -> ... d_out')

    def flops_count(self, context_length: int):
        return 2 * self.in_features * self.out_features * context_length


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(
            weight, mean=0.0, std=1.0, a=-3.0, b=3.0,
        )
        self.weights = nn.Parameter(weight)

    @nvtx.range("Embedding")
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]

    def flops_count(self, context_length: int):
        return 0


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    @nvtx.range("RMSNorm")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(
            einsum(x, x, '... d_model, ... d_model -> ...') / self.d_model + self.eps
        )
        return x / rearrange(rms, '... -> ... 1') * self.weight

    def flops_count(self, context_length: int):
        return 0  # Approximation

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    @nvtx.range("SiLU")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.silu = SiLU()
        # w1 and w3 project from d_model -> d_ff; w2 projects back d_ff -> d_model
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    @nvtx.range("SwiGLU")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

    def flops_count(self, context_length: int):
        # without silu
        return (
            self.w1.flops_count(context_length) +
            self.w2.flops_count(context_length) +
            self.w3.flops_count(context_length)
        )

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        exponent = torch.arange(0, d_k // 2, device=device, dtype=torch.float32) * (2.0 / d_k)
        inv_freq = theta ** (-exponent)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = einsum(positions, inv_freq, 'i, d -> i d')
        self.register_buffer('cos', torch.cos(angles), persistent=False)
        self.register_buffer('sin', torch.sin(angles), persistent=False)

    @nvtx.range("RotaryPositionalEmbedding")
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        assert x.shape[-1] == self.d_k
        x1, x2 = rearrange(x, '... (h p) -> p ... h', p=2)
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos
        return rearrange([rot_x1, rot_x2], 'p ... h -> ... (h p)')

    def flops_count(self, context_length: int):
        return 0  # Approximation

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)
        if max_seq_len is None:
            self.rope = nn.Identity()
        else:
            assert theta is not None
            self.rope = RotaryPositionalEmbedding(
                theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device,
            )

    @nvtx.range("MultiHeadSelfAttention")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        Q = rearrange(self.w_q(x), '... n (h d_k) -> ... h n d_k', h=self.num_heads)
        K = rearrange(self.w_k(x), '... n (h d_k) -> ... h n d_k', h=self.num_heads)
        V = rearrange(self.w_v(x), '... n (h d_k) -> ... h n d_k', h=self.num_heads)
        Q = self.rope(Q)
        K = self.rope(K)
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        out = scaled_dot_product_attention(Q, K, V, mask)
        out = rearrange(out, '... h n d_k -> ... n (h d_k)')
        return self.w_o(out)

    def flops_count(self, context_length: int):
        # without softmax
        return (
            2 * self.num_heads * self.d_k * context_length * context_length +  # QK^T
            2 * self.num_heads * self.d_k * context_length * context_length +  # Attention V
            self.w_q.flops_count(context_length) +
            self.w_k.flops_count(context_length) +
            self.w_v.flops_count(context_length) +
            self.w_o.flops_count(context_length)
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ff = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    @nvtx.range("TransformerBlock")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

    def flops_count(self, context_length: int):
        return dict(
            attn=self.attn.flops_count(context_length),
            ff=self.ff.flops_count(context_length),
            norm1=self.norm1.flops_count(context_length),
            norm2=self.norm2.flops_count(context_length),
        )


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        d_ff: int,
        num_layers: int,
        num_heads: int,
        rope_theta: float | None = None,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    max_seq_len=context_length,
                    theta=rope_theta,
                )
            )
        self.norm = RMSNorm(d_model)
        self.output = Linear(d_model, vocab_size)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.embedding(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output(x)
        return x

    def flops_count(self, context_length: int):
        return dict(
            embedding=self.embedding.flops_count(context_length),
            layers=[layer.flops_count(context_length) for layer in self.layers],
            norm=self.norm.flops_count(context_length),
            output=self.output.flops_count(context_length),
        )
