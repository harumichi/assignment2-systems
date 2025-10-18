import torch
import timeit
import numpy as np
from contextlib import nullcontext
from cs336_systems.util import get_device

from cs336_basics.nn import Transformer, cross_entropy


def benchmark(
    *,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    rope_theta: float = 10000.0,
    batch_size: int = 4,
    vocab_size: int = 10000,
    context_length: int = 256,
    steps: int = 10,
    warmup_steps: int = 5,
    with_backward: bool = True,
    mixed_precision_dtype: torch.dtype | None = None,
):
    device = get_device()

    model = Transformer(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        rope_theta=rope_theta,
    ).to(device)
    data = torch.randint(
        0, vocab_size, (batch_size, context_length), dtype=torch.long,
    ).to(device)

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    def run():
        logits = model(data)
        loss = cross_entropy(logits, data)
        if with_backward:
            loss.backward()
        synchronize()

    if mixed_precision_dtype is not None:
        assert isinstance(mixed_precision_dtype, torch.dtype)
        context = torch.autocast(device, dtype=mixed_precision_dtype)
    else:
        context = nullcontext()

    with context:
        timeit.timeit(run, number=warmup_steps)
        return timeit.repeat(run, repeat=steps, number=1)
