from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd
from Bio import SeqIO


@dataclass(frozen=True)
class PairRow:
    protein_a: str
    protein_b: str
    label: int
    seq_a: str
    seq_b: str


def read_fasta_map(fasta_path: str) -> Dict[str, str]:
    if fasta_path is None:
        return {}
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    m: Dict[str, str] = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        pid = str(rec.id)
        m[pid] = str(rec.seq)
    return m


def load_pairs_csv(data_csv: str, fasta_path: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"CSV not found: {data_csv}")
    df = pd.read_csv(data_csv)

    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    # Required: protein_a, protein_b, label
    for req in ["protein_a", "protein_b", "label"]:
        if req not in cols:
            raise ValueError(f"Missing required column '{req}'. Found columns: {list(df.columns)}")

    # If sequences exist in CSV, prefer them.
    has_seq = ("seq_a" in cols) and ("seq_b" in cols)
    if has_seq:
        df = df.rename(columns={cols["protein_a"]: "protein_a", cols["protein_b"]: "protein_b", cols["label"]: "label",
                                cols["seq_a"]: "seq_a", cols["seq_b"]: "seq_b"})
        df["seq_a"] = df["seq_a"].astype(str)
        df["seq_b"] = df["seq_b"].astype(str)
    else:
        # Load from FASTA mapping
        if fasta_path is None:
            raise ValueError("CSV does not contain seq_a/seq_b; please provide --fasta_path.")
        fmap = read_fasta_map(fasta_path)
        df = df.rename(columns={cols["protein_a"]: "protein_a", cols["protein_b"]: "protein_b", cols["label"]: "label"})
        def get_seq(pid: str) -> str:
            if pid not in fmap:
                raise KeyError(f"Protein '{pid}' not found in FASTA map.")
            return fmap[pid]
        df["seq_a"] = df["protein_a"].map(get_seq)
        df["seq_b"] = df["protein_b"].map(get_seq)

    df["label"] = df["label"].astype(int)
    # Basic clean
    df = df.dropna(subset=["protein_a","protein_b","seq_a","seq_b","label"])
    return df


def protein_level_split(df: pd.DataFrame, test_size: float, val_size: float, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split so that proteins don't overlap across splits (best-effort greedy).
    This is stricter than pair-level split and reduces leakage.
    """
    import random
    rng = random.Random(seed)
    proteins = sorted(set(df["protein_a"]).union(set(df["protein_b"])))
    rng.shuffle(proteins)

    n = len(proteins)
    n_test = int(n * test_size)
    n_val = int(n * val_size)

    test_prots = set(proteins[:n_test])
    val_prots  = set(proteins[n_test:n_test+n_val])
    train_prots= set(proteins[n_test+n_val:])

    def mask_for(prot_set: Set[str]) -> pd.Series:
        return df["protein_a"].isin(prot_set) | df["protein_b"].isin(prot_set)

    # Assign pairs to a split only if BOTH proteins in that split; otherwise drop from that split.
    train_df = df[df["protein_a"].isin(train_prots) & df["protein_b"].isin(train_prots)].copy()
    val_df   = df[df["protein_a"].isin(val_prots) & df["protein_b"].isin(val_prots)].copy()
    test_df  = df[df["protein_a"].isin(test_prots) & df["protein_b"].isin(test_prots)].copy()

    return train_df, val_df, test_df


def pair_level_split(df: pd.DataFrame, test_size: float, val_size: float, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    train_df, val_df  = train_test_split(train_df, test_size=val_size/(1.0-test_size), random_state=seed, stratify=train_df["label"])
    return train_df.copy(), val_df.copy(), test_df.copy()
