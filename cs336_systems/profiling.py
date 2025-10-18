import torch
import timeit
import logging
from contextlib import contextmanager, nullcontext
import time

import cs336_systems
from cs336_basics.optim import AdamW
from cs336_systems.util import get_device


logger = logging.getLogger(__name__)


if torch.cuda.is_available():
    from torch.cuda import nvtx
    from cs336_systems.nn import Transformer, MultiHeadSelfAttention, cross_entropy
else:
    from cs336_basics.nn import Transformer, MultiHeadSelfAttention, cross_entropy

    class MockNvtx:
        def range(self, *args, **kwargs):
            return nullcontext()

        def mark(self, *args, **kwargs):
            pass

    nvtx = MockNvtx()


@contextmanager
def memory_profile(snapshot_path: str = "memory_snapshot.pickle"):
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    yield
    torch.cuda.memory._dump_snapshot(snapshot_path)
    torch.cuda.memory._record_memory_history(enabled=None)


@contextmanager
def compute_profile():
    from torch.cuda import cudart

    rt = cudart()
    rt.cudaProfilerStart()
    yield
    rt.cudaProfilerStop()


class Timer:
    def __init__(self):
        self.total_time = 0.0

    @contextmanager
    def measure(self, enable: bool = True):
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        if enable:
            self.total_time += end_time - start_time

    def reset(self):
        self.total_time = 0.0

    def get_total_time(self):
        return self.total_time


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
    mixed_precision_dtype: torch.dtype | None = None,
    enable_compute: bool = False,
    enable_memory: bool = False,
    training: bool = True,
    compile: bool = False,
):
    device = get_device()
    logger.info("Using device: %s", device)

    model = Transformer(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        rope_theta=rope_theta,
    ).to(device)
    if compile:
        model = torch.compile(model)

    data = torch.randint(
        0, vocab_size, (batch_size, context_length), dtype=torch.long,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )

    timer = Timer()

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    def run(warmup: bool = False):
        with timer.measure(enable=not warmup):
            if training:
                optimizer.zero_grad()
            with nvtx.range("forward"):
                logits = model(data)
                loss = cross_entropy(logits, data)
            if training:
                with nvtx.range("backward"):
                    loss.backward()
                with nvtx.range("optimizer step"):
                    optimizer.step()
            synchronize()
            return loss

    if device == "cuda" and enable_compute:
        compute_context = compute_profile()
    else:
        compute_context = nullcontext()
    if device == "cuda" and enable_memory:
        memory_context = memory_profile()
    else:
        memory_context = nullcontext()
    if mixed_precision_dtype is not None:
        assert isinstance(mixed_precision_dtype, torch.dtype)
        autocast_context = torch.autocast(device, dtype=mixed_precision_dtype)
    else:
        autocast_context = nullcontext()

    with autocast_context:
        # warmup
        timeit.timeit(
            lambda: run(warmup=True), number=warmup_steps,
        )
        with compute_context, memory_context:
            timeit.timeit(run, number=steps)

    logger.info("Total: %.3f msec / step", timer.get_total_time() / steps * 1000)


def profiling_attention(
    *,
    d_model: int,
    batch_size: int = 8,
    context_length: int = 256,
    steps: int = 100,
    warmup_steps: int = 5,
    mixed_precision_dtype: torch.dtype | None = None,
    enable_compute: bool = False,
    enable_memory: bool = False,
    compile: bool = False,
):
    device = get_device()
    logger.info("Using device: %s", device)

    attention = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=1,
        max_seq_len=context_length,
        theta=10000.0,
        device=device,
        dtype=torch.float32,
    ).to(device)
    if compile:
        attention = torch.compile(attention)

    data = torch.randn(
        batch_size, context_length, d_model,
        dtype=torch.float32,
    ).to(device)

    forward_timer = Timer()
    backward_timer = Timer()

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    def run(warmup: bool = False):
        with forward_timer.measure(enable=not warmup):
            out = attention(data)
            y = out.mean()
        synchronize()
        with backward_timer.measure(enable=not warmup):
            y.backward()
        synchronize()

    if device == "cuda" and enable_compute:
        compute_context = compute_profile()
    else:
        compute_context = nullcontext()
    if device == "cuda" and enable_memory:
        memory_context = memory_profile()
    else:
        memory_context = nullcontext()
    if mixed_precision_dtype is not None:
        assert isinstance(mixed_precision_dtype, torch.dtype)
        autocast_context = torch.autocast(device, dtype=mixed_precision_dtype)
    else:
        autocast_context = nullcontext()

    with autocast_context:
        # warmup
        timeit.timeit(
            lambda: run(warmup=True), number=warmup_steps,
        )
        with compute_context, memory_context:
            timeit.timeit(run, number=steps)

    logger.info("Forward: %.3f msec / step", forward_timer.get_total_time() / steps * 1000)
    logger.info("Backward: %.3f msec / step", backward_timer.get_total_time() / steps * 1000)
