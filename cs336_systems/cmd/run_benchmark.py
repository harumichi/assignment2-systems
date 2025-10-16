import numpy as np
import logging
from typing import NamedTuple

from cs336_systems.benchmark import benchmark

logger = logging.getLogger(__name__)


class ModelSize(NamedTuple):
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int

model_sizes = {
    "small": ModelSize(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelSize(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelSize(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelSize(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": ModelSize(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

for key, args in model_sizes.items():
    args = args._asdict()
    for with_backward in [True, False]:
        args["with_backward"] = with_backward
        logger.info("Model size: %s", key)
        logger.info("Benchmarking with args: %s", args)
        ts = benchmark(**args)
        logger.info("mean: %.3f ms, std: %.3f ms", np.mean(ts) * 1000, np.std(ts) * 1000)
