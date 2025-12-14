from __future__ import annotations
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModel


MODEL_MAP = {
    # You can swap to larger models if you have GPU RAM:
    "esm2": "facebook/esm2_t6_8M_UR50D",
    "protbert": "Rostlab/prot_bert",
}

def load_embedder(name: str, device: torch.device):
    if name not in MODEL_MAP:
        raise ValueError(f"Unknown embed model '{name}'. Choose from {list(MODEL_MAP.keys())}")
    hf_id = MODEL_MAP[name]
    tokenizer = AutoTokenizer.from_pretrained(hf_id, do_lower_case=False)
    model = AutoModel.from_pretrained(hf_id)
    model.to(device)
    model.eval()
    return tokenizer, model, hf_id


@torch.no_grad()
def embed_batch(
    sequences: List[str],
    tokenizer,
    model,
    device: torch.device,
    pool: str = "mean",
    max_len: int = 1024,
) -> torch.Tensor:
    """Return [B, D] embeddings."""
    # ProtBERT expects spaces between amino acids; ESM tokenizers generally handle raw sequence.
    # We'll handle ProtBERT formatting here.
    if hasattr(tokenizer, "name_or_path") and "prot_bert" in tokenizer.name_or_path:
        sequences = [" ".join(list(s.replace(" ", ""))) for s in sequences]

    enc = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    ).to(device)

    out = model(**enc)
    # Typically: last_hidden_state [B, L, D]
    hidden = out.last_hidden_state

    if pool == "cls":
        emb = hidden[:, 0, :]
    elif pool == "mean":
        # Mask padding tokens
        attn = enc.get("attention_mask", None)
        if attn is None:
            emb = hidden.mean(dim=1)
        else:
            attn = attn.unsqueeze(-1)  # [B, L, 1]
            summed = (hidden * attn).sum(dim=1)
            denom = attn.sum(dim=1).clamp(min=1)
            emb = summed / denom
    else:
        raise ValueError("pool must be 'mean' or 'cls'")

    return emb
