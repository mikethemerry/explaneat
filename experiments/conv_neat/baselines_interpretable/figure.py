"""Figure: sparse-LR weight maps (2x5) and the depth-6 tree's top levels (10k-row models)."""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.tree import plot_tree  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M = lambda n: pickle.load(open(os.path.join(HERE, "models", n + ".pkl"), "rb"))  # noqa: E731
lr, tree = M("lr_l1_10k"), M("tree_d6_10k")

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.9], hspace=0.35)
W = lr.coef_
v = np.abs(W).max()
for k in range(10):
    ax = fig.add_subplot(gs[k // 5, k % 5])
    im = ax.imshow(W[k].reshape(28, 28), cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title(f"digit {k}: {int((W[k] != 0).sum())} non-zero", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
cax = fig.add_axes([0.92, 0.55, 0.012, 0.3])
fig.colorbar(im, cax=cax, label="weight (red: evidence for the digit)")
fig.text(0.5, 0.955, f"Sparse (L1) logistic regression, C={lr.C}, 10k rows: {int((W != 0).sum())} of 7,840 pixel weights non-zero",
         ha="center", fontsize=13)
ax = fig.add_subplot(gs[2, :])
plot_tree(tree, max_depth=2, ax=ax, filled=True, impurity=False, proportion=False, fontsize=8,
          class_names=[str(i) for i in range(10)],
          feature_names=[f"px(r{i // 28},c{i % 28})" for i in range(784)], label="root")
ax.set_title(f"CART depth 6 (10k rows), top 3 levels of {tree.get_n_leaves()} leaves; split = pixel intensity threshold",
             fontsize=12)
out = os.path.join(HERE, "baselines_figure.png")
fig.savefig(out, dpi=110, bbox_inches="tight")
print(out)
