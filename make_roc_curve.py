import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

y = np.load("artifacts/run/y_test.npy")
p = np.load("artifacts/run/p_test.npy")

fpr, tpr, _ = roc_curve(y, p)
auc = roc_auc_score(y, p)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/roc_curve.png", dpi=300)
plt.close()
