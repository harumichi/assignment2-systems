import argparse
import logging
from typing import NamedTuple
import torch

from cs336_systems.profiling import profiling

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--context-length', type=int)
    parser.add_argument('--enable-compute', action='store_true', default=None)
    parser.add_argument('--enable-memory', action='store_true', default=None)
    parser.add_argument('--mixed-precision-dtype', type=str, choices=['bfloat16', 'float16'])
    return parser.parse_args()

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

args = model_sizes["small"]._asdict()
args.update(
    {k:v for k,v in vars(parse_args()).items() if v is not None}
)
if args.get("mixed_precision_dtype"):
    args["mixed_precision_dtype"] = getattr(torch, args["mixed_precision_dtype"])
profiling(**args)
