#!/usr/bin/env python3
import os, gzip, random, itertools
from pathlib import Path

import pandas as pd
from Bio import SeqIO
import requests

# --------- CONFIG (tune these) ----------
SPECIES = "9606"                 # 9606 = Homo sapiens
VERSION = "v12.0"
SCORE_THRESHOLD = 700            # 0-1000; 700 = high-confidence
N_POS = 100_000                  # how many positive pairs to keep
NEG_RATIO = 1                    # 1 => 1:1 pos:neg, 3 => 1:3
SEED = 42

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LINKS_URL = f"https://stringdb-downloads.org/download/protein.links.{VERSION}/{SPECIES}.protein.links.{VERSION}.txt.gz"
SEQS_URL  = f"https://stringdb-downloads.org/download/protein.sequences.{VERSION}/{SPECIES}.protein.sequences.{VERSION}.fa.gz"

LINKS_GZ = OUT_DIR / f"{SPECIES}.protein.links.{VERSION}.txt.gz"
SEQS_GZ  = OUT_DIR / f"{SPECIES}.protein.sequences.{VERSION}.fa.gz"
OUT_CSV  = OUT_DIR / "pairs.csv"
# --------------------------------------


def download(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[OK] Found existing: {dest}")
        return
    print(f"[DL] {url}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print(f"[OK] Saved: {dest} ({dest.stat().st_size/1e6:.1f} MB)")


def load_seq_map(fasta_gz: Path) -> dict:
    """Map STRING protein IDs -> sequence"""
    seq_map = {}
    with gzip.open(fasta_gz, "rt") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            seq_map[str(rec.id)] = str(rec.seq)
    print(f"[OK] Loaded sequences: {len(seq_map):,}")
    return seq_map


def iter_links_filtered(links_gz: Path, score_thr: int):
    """
    STRING links format (gz): protein1 protein2 combined_score
    Returns rows where score >= threshold
    """
    with gzip.open(links_gz, "rt") as f:
        header = next(f).strip().split()
        # Expect: protein1 protein2 combined_score
        p1_col = header[0]
        p2_col = header[1]
        score_col = header[2]
        for line in f:
            a, b, s = line.strip().split()
            s = int(s)
            if s >= score_thr:
                yield a, b, s


def sample_positives(links_gz: Path, score_thr: int, n_pos: int, seed: int):
    """
    Reservoir sample positives from the filtered stream so you don't load everything in RAM.
    """
    rng = random.Random(seed)
    sample = []
    for i, row in enumerate(iter_links_filtered(links_gz, score_thr), start=1):
        if len(sample) < n_pos:
            sample.append(row)
        else:
            j = rng.randint(1, i)
            if j <= n_pos:
                sample[j - 1] = row
    print(f"[OK] Sampled positives: {len(sample):,} (thr={score_thr})")
    return sample


def make_negatives(proteins, pos_set, n_neg: int, seed: int):
    """
    Random negatives: pick random protein pairs not in pos_set.
    (Ensures (a,b) and (b,a) treated the same by storing sorted tuples.)
    """
    rng = random.Random(seed)
    proteins = list(proteins)
    neg = set()
    tries = 0
    max_tries = n_neg * 50  # safety
    while len(neg) < n_neg and tries < max_tries:
        a = rng.choice(proteins)
        b = rng.choice(proteins)
        if a == b:
            tries += 1
            continue
        key = tuple(sorted((a, b)))
        if key in pos_set or key in neg:
            tries += 1
            continue
        neg.add(key)
        tries += 1
    if len(neg) < n_neg:
        print(f"[WARN] Only generated {len(neg):,}/{n_neg:,} negatives (increase max_tries or adjust n_pos)")
    return list(neg)


def main():
    random.seed(SEED)

    # 1) Download
    download(LINKS_URL, LINKS_GZ)
    download(SEQS_URL, SEQS_GZ)

    # 2) Sequences
    seq_map = load_seq_map(SEQS_GZ)
    proteins = set(seq_map.keys())

    # 3) Positives (sampled)
    pos_rows = sample_positives(LINKS_GZ, SCORE_THRESHOLD, N_POS, SEED)
    # pos_set for fast exclusion (unordered)
    pos_set = set(tuple(sorted((a, b))) for a, b, _ in pos_rows)

    # 4) Negatives
    n_neg = len(pos_rows) * NEG_RATIO
    neg_pairs = make_negatives(proteins, pos_set, n_neg, SEED)

    # 5) Build dataframe
    out = []

    # positives
    for a, b, s in pos_rows:
        if a in seq_map and b in seq_map:
            out.append([a, b, seq_map[a], seq_map[b], 1])

    # negatives
    for a, b in neg_pairs:
        if a in seq_map and b in seq_map:
            out.append([a, b, seq_map[a], seq_map[b], 0])

    df = pd.DataFrame(out, columns=["protein_a", "protein_b", "seq_a", "seq_b", "label"])
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # shuffle

    df.to_csv(OUT_CSV, index=False)
    print(f"[DONE] Wrote {len(df):,} rows to {OUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()

