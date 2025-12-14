from __future__ import annotations
import os
import argparse
from typing import Dict, List
import pandas as pd
import torch
from tqdm import tqdm

from .utils import set_seed, get_device, ensure_dir
from .data import load_pairs_csv
from .embedding_models import load_embedder, embed_batch


def unique_proteins(df: pd.DataFrame) -> pd.DataFrame:
    # Build a protein table with unique ID -> sequence
    prots = {}
    for _, r in df.iterrows():
        prots[str(r["protein_a"])] = str(r["seq_a"])
        prots[str(r["protein_b"])] = str(r["seq_b"])
    return pd.DataFrame({"protein_id": list(prots.keys()), "sequence": list(prots.values())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--fasta_path", default=None, help="Only needed if sequences are not in CSV.")
    ap.add_argument("--model", default="esm2", choices=["esm2","protbert"])
    ap.add_argument("--pool", default="mean", choices=["mean","cls"])
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_dir", default="artifacts/embeddings")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.out_dir)

    df = load_pairs_csv(args.data_csv, args.fasta_path)
    prot_df = unique_proteins(df)

    tokenizer, model, hf_id = load_embedder(args.model, device)
    print(f"Using embedding model: {args.model} ({hf_id}) on device={device}")

    out_path = os.path.join(args.out_dir, f"{args.model}_{args.pool}_len{args.max_len}.pt")
    meta_path = os.path.join(args.out_dir, f"{args.model}_{args.pool}_len{args.max_len}.meta.csv")

    # If already exists, skip
    if os.path.exists(out_path) and os.path.exists(meta_path):
        print(f"Found cached embeddings at {out_path}. Nothing to do.")
        return

    embeddings = []
    ids = prot_df["protein_id"].tolist()
    seqs = prot_df["sequence"].tolist()

    for i in tqdm(range(0, len(seqs), args.batch_size), desc="Embedding proteins"):
        batch_seqs = seqs[i:i+args.batch_size]
        emb = embed_batch(batch_seqs, tokenizer, model, device, pool=args.pool, max_len=args.max_len)
        embeddings.append(emb.cpu())

    emb_all = torch.cat(embeddings, dim=0)  # [N, D]
    torch.save({"protein_ids": ids, "embeddings": emb_all}, out_path)
    prot_df.to_csv(meta_path, index=False)

    print(f"Saved embeddings: {out_path}")
    print(f"Saved metadata : {meta_path}")


if __name__ == "__main__":
    main()
