import torch
import torch.nn as nn
import logging

import cs336_systems
from cs336_systems.util import get_device

logger = logging.getLogger(__name__)


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        logger.info("output linear 1: %s", x.dtype)
        x = self.ln(x)
        logger.info("output layernorm: %s", x.dtype)
        x = self.fc2(x)
        logger.info("output linear 2: %s", x.dtype)
        return x


device = get_device()
logger.info("Using device: %s", device)

model = ToyModel(in_features=128, out_features=5).to(device)
cast_dtype = torch.bfloat16

with torch.autocast(device, dtype=cast_dtype):
    target = torch.randint(0, 5, (32,)).to(device)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(32, 128).to(device)
    y = model(x)
    loss = criterion(y, target)
    loss.backward()

    logger.info("logits: %s", y.dtype)
    logger.info("loss: %s", loss.dtype)

    for name, param in model.named_parameters():
        logger.info("param %s: %s", name, param.dtype)
        if param.grad is not None:
            logger.info("grad %s: %s", name, param.grad.dtype)
