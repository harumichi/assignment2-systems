import argparse
import logging
from itertools import product
import torch

from cs336_systems.profiling import profiling_attention

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-compute', action='store_true', default=None)
    parser.add_argument('--enable-memory', action='store_true', default=None)
    return parser.parse_args()


d_model_list = [16, 32, 64, 128]
context_length_list = [256, 1024, 4096, 8192, 16384]

for d_model, context_length in product(d_model_list, context_length_list):
    args = {
        "d_model": d_model,
        "context_length": context_length,
    }
    args.update(
        {k:v for k,v in vars(parse_args()).items() if v is not None}
    )
    logger.info("Profiling with args: %s", args)
    profiling_attention(**args)
