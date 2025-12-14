import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y = np.load("artifacts/run/y_test.npy")
p = np.load("artifacts/run/p_test.npy")

y_pred = (p >= 0.5).astype(int)
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-interact", "Interact"]
)

plt.figure(figsize=(5,5))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Threshold = 0.5)")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=300)
plt.close()
