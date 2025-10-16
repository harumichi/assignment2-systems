import numpy as np
import pandas as pd
from typing import NamedTuple

from cs336_systems.benchmark import benchmark


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


def main():
    f_means = {}
    t_means = {}
    for name, spec in model_sizes.items():
        base = spec._asdict()
        f_means[name] = float(np.mean(benchmark(**{**base, "with_backward": False})) * 1000)
        t_means[name] = float(np.mean(benchmark(**{**base, "with_backward": True})) * 1000)
    rows = []
    for name in model_sizes.keys():
        fwd = f_means[name]
        total = t_means[name]
        bwd = max(total - fwd, 0.0)
        rows.append({
            "model_size": name,
            "forward_ms": round(fwd, 3),
            "backward_ms": round(bwd, 3),
        })
    df = pd.DataFrame(rows)
    try:
        print(df.to_markdown(index=False))
    except Exception:
        print(df)


if __name__ == "__main__":
    main()
