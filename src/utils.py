from __future__ import annotations
import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    # Embeddings
    embed_model: str = "esm2"   # esm2 | protbert
    pool: str = "mean"          # mean | cls
    max_len: int = 1024
    batch_size: int = 8

    # Pair representation
    pair_mode: str = "concat"   # concat | absdiff | hadamard | concat_absdiff_hadamard

    # Train
    split_strategy: str = "protein"  # protein | pair
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    hidden_dim: int = 512
    dropout: float = 0.2
    early_stop_patience: int = 3

    # Class imbalance
    pos_weight: Optional[float] = None  # if None, computed from train split

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return Config(**d)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
