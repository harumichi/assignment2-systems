import torch
import time
import torch.cuda.nvtx as nvtx
from torch.cuda import cudart  # type: ignore
import timeit
from contextlib import contextmanager

from cs336_systems.nn import Transformer, cross_entropy
from cs336_basics.optim import AdamW


@contextmanager
def cuda_profile():
    rt = cudart()
    rt.cudaProfilerStart()
    yield
    rt.cudaProfilerStop()


def profiling(
    *,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    rope_theta: float = 10000.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    betas: tuple = (0.9, 0.999),
    batch_size: int = 4,
    vocab_size: int = 10000,
    context_length: int = 256,
    steps: int = 5,
    warmup_steps: int = 5,
):
    assert torch.cuda.is_available()
    device = "cuda"

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
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    def run():
        optimizer.zero_grad()
        with nvtx.range("forward"):
            logits = model(data)
            loss = cross_entropy(logits, data)
        with nvtx.range("backward"):
            loss.backward()
        with nvtx.range("optimizer step"):
            optimizer.step()
        synchronize()
        return loss

    # warmup
    timeit.timeit(run, number=warmup_steps)

    with cuda_profile():
        nvtx.mark("profiling start")
        timeit.timeit(run, number=steps)
        nvtx.mark("profiling end")
