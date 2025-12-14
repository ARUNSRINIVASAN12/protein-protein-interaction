from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import torch


def build_pair_features(ea: np.ndarray, eb: np.ndarray, mode: str) -> np.ndarray:
    """ea, eb: [D]"""
    if mode == "concat":
        return np.concatenate([ea, eb], axis=-1)
    if mode == "absdiff":
        return np.abs(ea - eb)
    if mode == "hadamard":
        return ea * eb
    if mode == "concat_absdiff_hadamard":
        return np.concatenate([ea, eb, np.abs(ea-eb), ea*eb], axis=-1)
    raise ValueError(f"Unknown pair_mode: {mode}")


def infer_in_dim(embed_dim: int, mode: str) -> int:
    if mode == "concat":
        return 2*embed_dim
    if mode == "absdiff":
        return embed_dim
    if mode == "hadamard":
        return embed_dim
    if mode == "concat_absdiff_hadamard":
        return 4*embed_dim
    raise ValueError(f"Unknown pair_mode: {mode}")
