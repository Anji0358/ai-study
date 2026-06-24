import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

np.random.seed(1)

X_xor = np.random.randn(200, 2)

y_xor = np.logical_xor(
    X_xor[:, 0] > 0,
    X_xor[:, 1] > 0
)

y_xor = np.where(y_xor, 1, -1)

svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)

plot_decision_regions(X_xor, y_xor, classifier=svm)

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()