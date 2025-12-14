from __future__ import annotations
import os
import argparse
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .utils import set_seed, get_device, ensure_dir
from .data import load_pairs_csv, protein_level_split, pair_level_split
from .pairs import build_pair_features, infer_in_dim
from .model import MLPClassifier
from .metrics import compute_metrics


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, emb_map: Dict[str, np.ndarray], pair_mode: str):
        self.df = df.reset_index(drop=True)
        self.emb_map = emb_map
        self.pair_mode = pair_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        a = str(r["protein_a"]); b = str(r["protein_b"])
        ea = self.emb_map[a]; eb = self.emb_map[b]
        x = build_pair_features(ea, eb, self.pair_mode).astype(np.float32)
        y = np.int64(r["label"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def load_embedding_map(emb_dir: str) -> Tuple[Dict[str, np.ndarray], int, str]:
    # Find the first .pt in emb_dir
    if not os.path.exists(emb_dir):
        raise FileNotFoundError(f"Embedding directory not found: {emb_dir}")
    pt_files = [f for f in os.listdir(emb_dir) if f.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(f"No .pt embeddings found in {emb_dir}. Run src.embed first.")
    pt_files.sort()
    path = os.path.join(emb_dir, pt_files[0])
    blob = torch.load(path, map_location="cpu")
    ids = blob["protein_ids"]
    emb = blob["embeddings"]  # [N, D]
    emb_dim = int(emb.shape[1])
    emb_map = {pid: emb[i].numpy() for i, pid in enumerate(ids)}
    return emb_map, emb_dim, path


def run_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def predict_proba(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    ps = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        ps.append(prob)
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--fasta_path", default=None)
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--pair_mode", default="concat", choices=["concat","absdiff","hadamard","concat_absdiff_hadamard"])
    ap.add_argument("--split_strategy", default="protein", choices=["protein","pair"])
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--early_stop_patience", type=int, default=3)
    ap.add_argument("--out_dir", default="artifacts/run")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.out_dir)

    df = load_pairs_csv(args.data_csv, args.fasta_path)
    emb_map, emb_dim, emb_path = load_embedding_map(args.emb_dir)
    print(f"Loaded embeddings from: {emb_path} (dim={emb_dim}, n={len(emb_map)})")

    # Filter pairs where both proteins have embeddings (should be all, but be safe)
    df = df[df["protein_a"].astype(str).isin(emb_map.keys()) & df["protein_b"].astype(str).isin(emb_map.keys())].copy()
    df["protein_a"] = df["protein_a"].astype(str)
    df["protein_b"] = df["protein_b"].astype(str)

    if args.split_strategy == "protein":
        train_df, val_df, test_df = protein_level_split(df, args.test_size, args.val_size, seed=args.seed)
        print(f"Protein-split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            print("WARNING: Protein-level split produced empty split(s). Falling back to pair-level split.")
            train_df, val_df, test_df = pair_level_split(df, args.test_size, args.val_size, seed=args.seed)
    else:
        train_df, val_df, test_df = pair_level_split(df, args.test_size, args.val_size, seed=args.seed)

    # Compute pos_weight from train split for BCEWithLogitsLoss
    n_pos = int(train_df["label"].sum())
    n_neg = int((train_df["label"] == 0).sum())
    pos_weight = (n_neg / max(1, n_pos))
    print(f"Train class balance: pos={n_pos} neg={n_neg} -> pos_weight={pos_weight:.3f}")

    train_ds = PairDataset(train_df, emb_map, args.pair_mode)
    val_ds   = PairDataset(val_df, emb_map, args.pair_mode)
    test_ds  = PairDataset(test_df, emb_map, args.pair_mode)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    in_dim = infer_in_dim(emb_dim, args.pair_mode)
    model = MLPClassifier(in_dim=in_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    best_val = -1.0
    best_path = os.path.join(args.out_dir, "best_model.pt")
    patience = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_loader, optimizer, device, loss_fn)

        yv, pv = predict_proba(model, val_loader, device)
        val_metrics = compute_metrics(yv, pv)

        line = {"epoch": epoch, "train_loss": tr_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(line)

        print(f"[Epoch {epoch:02d}] loss={tr_loss:.4f} val_roc={val_metrics['roc_auc']:.4f} val_pr={val_metrics['pr_auc']:.4f} val_f1={val_metrics['f1']:.4f}")

        # Early stopping on PR-AUC (often more meaningful for imbalance)
        score = val_metrics["pr_auc"]
        if np.isnan(score):
            score = -1.0

        if score > best_val:
            best_val = score
            patience = 0
            torch.save({"model_state": model.state_dict(),
                        "in_dim": in_dim,
                        "emb_dim": emb_dim,
                        "pair_mode": args.pair_mode}, best_path)
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"Early stopping (patience={args.early_stop_patience}). Best val PR-AUC={best_val:.4f}")
                break

    # Load best and evaluate on test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    yt, pt = predict_proba(model, test_loader, device)

    np.save("artifacts/run/y_test.npy", yt)
    np.save("artifacts/run/p_test.npy", pt)

    test_metrics = compute_metrics(yt, pt)

    print("TEST METRICS:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save history + test metrics
    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "history.csv"), index=False)
    pd.DataFrame([test_metrics]).to_csv(os.path.join(args.out_dir, "test_metrics.csv"), index=False)
    print(f"Saved artifacts to: {args.out_dir}")


if __name__ == "__main__":
    main()
