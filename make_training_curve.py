import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("artifacts/run/history.csv")

plt.figure(figsize=(6,4))
plt.plot(df["epoch"], df["val_pr_auc"], label="Validation PR-AUC")
plt.plot(df["epoch"], df["val_roc_auc"], label="Validation ROC-AUC")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Performance vs Epoch")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ppi_val_curves.png", dpi=300)
plt.close()
