# Protein–Protein Interaction Prediction Using Protein Language Models

This repository contains the complete codebase and documentation for the final course project on **protein–protein interaction (PPI) prediction** using pretrained protein language models.

The objective of this project is to predict whether two proteins interact using **only amino-acid sequence information**.  
This repository accompanies the **final IEEE journal–style (1-column) project report** and is structured to ensure **full reproducibility**.

---

## 1. Project Overview

Protein–protein interactions (PPIs) are fundamental to cellular processes such as signaling, regulation, transport, and complex formation.  
Experimental discovery of PPIs is expensive and incomplete, motivating computational approaches that can prioritize candidate interactions for experimental validation.

In this project:
- Protein sequences are embedded using **ESM-2**, a pretrained transformer-based protein language model.
- Protein pairs are represented using **embedding concatenation**.
- A **multilayer perceptron (MLP)** classifier predicts interaction probabilities.
- Evaluation is performed using a **protein-disjoint split** to prevent information leakage.

---

## 2. Dataset Description

### 2.1 Data Source
- **Database:** STRING  
- **Organism:** *Homo sapiens*  
- **STRING Organism ID:** 9606  
- **Interactions:** High-confidence protein–protein interactions  
- **Sequences:** Corresponding protein sequences in FASTA format  

### 2.2 Label Construction
- **Positive pairs:** Known protein–protein interactions from STRING  
- **Negative pairs:** Randomly sampled protein pairs excluding:
  - Self-interactions (A, A)
  - Any known positive interaction

The training dataset is constructed to be approximately balanced between positive and negative pairs.

### 2.3 Train / Validation / Test Split
A **protein-disjoint split** is used:
- Proteins are split into train, validation, and test sets.
- Only pairs whose two proteins lie within the same split are retained.
- No protein appears in both training and test sets.

This ensures a stricter and more realistic evaluation of generalization.

---

## 3. Repository Structure

```
ppi-prediction/
├── README.md
├── requirements.txt
│
├── src/
│   ├── train.py
│   ├── embed.py
│   └── ...
│
├── data/
│   ├── proteins.fasta
│   └── pairs.csv
│
├── artifacts/
│   ├── embeddings/
│   │   └── esm2_mean_len1024.pt
│   └── run/
│       ├── history.csv
│       ├── y_test.npy
│       └── p_test.npy
│
├── figures/
│   ├── pipeline.png
│   ├── pr_curve.png
│   ├── roc_curve.png
│   └── confusion_matrix.png
│
├── ppi_val_curves.png
├── make_training_curve.py
├── make_pr_curve.py
├── make_roc_curve.py
├── make_confusion_matrix.py
│
└── report/
    └── PPI_IEEE_Journal_OneColumn_Final_Aplus.pdf
```

---

## 4. Environment Setup

```bash
conda create -n ppi python=3.10
conda activate ppi
pip install -r requirements.txt
```

---

## 5. Protein Embeddings

- **Model:** ESM-2  
- **Pooling strategy:** Mean pooling over token embeddings  
- **Embedding dimension:** 320  
- **Maximum sequence length:** 1024 tokens  

Embeddings are cached to disk to avoid recomputation.

---

## 6. Model Description

### Pair Representation
```
x = [e_A ; e_B]
```

### Classifier
- Multilayer perceptron (MLP)
- ReLU activations
- Dropout regularization
- Weighted binary cross-entropy loss

Training uses **AdamW** with early stopping on validation **PR-AUC**.

---

## 7. Running the Project

### Train the Model
```bash
python -m src.train \
  --data_csv data/pairs.csv \
  --fasta_path data/proteins.fasta \
  --emb_dir artifacts/embeddings \
  --pair_mode concat \
  --split_strategy protein \
  --epochs 30
```

Outputs are saved to `artifacts/run/`.

---

## 8. Generating Figures

```bash
python make_training_curve.py
python make_pr_curve.py
python make_roc_curve.py
python make_confusion_matrix.py
```

Generated figures:
- `ppi_val_curves.png`
- `figures/pr_curve.png`
- `figures/roc_curve.png`
- `figures/confusion_matrix.png`

---

## 9. Results (Best Run)

- **ROC-AUC:** 0.8609  
- **PR-AUC:** 0.8697  
- **F1-score:** 0.7645  
- **Precision:** 0.8235  
- **Recall:** 0.7134  

All metrics are reported on a protein-disjoint test set.

---

## 10. Final Report

The final report is located at:
```
report/PPI_report.pdf
```

---

## 11. Reproducibility Notes

To reproduce all results:
- Use STRING organism **9606**
- Apply the protein-disjoint split
- Cache embeddings before training
- Run the commands listed above

---

## 12. Acknowledgment

This project was completed as part of a graduate-level course and is intended for academic evaluation and learning purposes.
