import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

y = np.load("artifacts/run/y_test.npy")
p = np.load("artifacts/run/p_test.npy")

precision, recall, _ = precision_recall_curve(y, p)
ap = average_precision_score(y, p)

plt.figure(figsize=(6,4))
plt.plot(recall, precision, linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision–Recall Curve (AP = {ap:.3f})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/pr_curve.png", dpi=300)
plt.close()
