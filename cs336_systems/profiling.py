import torch
import time
import torch.cuda.nvtx as nvtx
from torch.cuda import cudart  # type: ignore
import timeit
from contextlib import contextmanager

from cs336_systems.nn import Transformer, cross_entropy
from cs336_basics.optim import AdamW


def add_nvtx_module_hooks(model):
    for name, module in model.named_modules():
        # Instrument all modules (including non-leaf) to get hierarchical NVTX ranges

        def fwd_pre(m, inp):
            torch.cuda.nvtx.range_push(f"F:{name}:{m.__class__.__name__}")

        def fwd_post(m, inp, out):
            torch.cuda.nvtx.range_pop()

        start_key = f"_bwd_start_{id(module)}"

        def tensor_grad_pre_hook(grad):
            torch.cuda.nvtx.range_push(f"B:{name}:{module.__class__.__name__}")
            module.__dict__[start_key] = time.time()
            return grad

        def bwd_full(m, grad_input, grad_output):
            torch.cuda.nvtx.range_pop()
            st = m.__dict__.get(start_key)
            if st is not None:
                m.__dict__[start_key] = None

        module.register_forward_pre_hook(fwd_pre)
        module.register_forward_hook(fwd_post)
        if hasattr(module, 'register_full_backward_hook'):
            module.register_full_backward_hook(bwd_full)

        def attach_tensor_hook(m, inp, out):
            tensors = []
            if torch.is_tensor(out):
                tensors = [out]
            elif isinstance(out, (tuple, list)):
                tensors = [t for t in out if torch.is_tensor(t)]
            for t in tensors:
                t.register_hook(tensor_grad_pre_hook)

        module.register_forward_hook(attach_tensor_hook)


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
    add_nvtx_module_hooks(model)

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
