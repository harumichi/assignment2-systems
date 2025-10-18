import logging
import torch
import numpy as np
import pandas as pd
from typing import NamedTuple

from cs336_systems.benchmark import benchmark


logger = logging.getLogger(__name__)

class ModelSize(NamedTuple):
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


model_sizes = {
    "small": ModelSize(768, 3072, 12, 12),
    "medium": ModelSize(1024, 4096, 24, 16),
    "large": ModelSize(1280, 5120, 36, 20),
    "xl": ModelSize(1600, 6400, 48, 25),
    "2.7B": ModelSize(2560, 10240, 32, 32),
}


f_means = {}
t_means = {}

for precision in ["full", "mixed"]:
    logger.info("precision: %s", precision)
    for name, spec in model_sizes.items():
        logger.info("model_size: %s", name)
        base = spec._asdict()
        if precision == "mixed":
            base["mixed_precision_dtype"] = torch.bfloat16
        f_means[name] = float(np.mean(benchmark(**{**base, "with_backward": False})) * 1000)
        t_means[name] = float(np.mean(benchmark(**{**base, "with_backward": True})) * 1000)
    rows = []
    for name in model_sizes.keys():
        fwd = f_means[name]
        total = t_means[name]
        bwd = max(total - fwd, 0.0)
        rows.append({
            "precision": precision,
            "model size": name,
            "forward [msec]": round(fwd, 3),
            "backward [msec]": round(bwd, 3),
        })
    df = pd.DataFrame(rows)
    try:
        print(df.to_markdown(index=False))
    except Exception:
        logger.exception("Failed to print as markdown")
        print(df)
