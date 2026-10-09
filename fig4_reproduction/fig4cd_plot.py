# Fig 4.cd: The evolution of 1D representations predicted by the effective theory or obtained from neural network training (shown in (c) and (d) respectively) agree creditably well.
# First I need to pick A dimension above r_c as nothing below it seems to converge. 
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)
import numpy as np
import matplotlib.pyplot as plt
from train_add import train_add # We get torch from this also, hopefully
import time

t0 = time.time()
dic = train_add(train_num=45)  # seed=58 implicit; p=10, steps=5000, eff_steps=5000 defaults
print("elapsed", time.time()-t0)

# Computing the Hessian Matrix
p = dic["p"]
seed = dic["seed"]
all_num = dic["all_num"]
train_num = dic["train_num"]
eta = dic["eta_repr"]
output_dim = 30
_ = np.random.seed(seed)
_ = np.random.normal(0, 1, size=(2 * p - 1, output_dim)) * dic["label_scale"]
train_id = np.random.choice(all_num, train_num, replace=False)

# Rebuilding A rows like train_add.py
D0_id = []
for i in range(p):
    for j in range(i, p):
        D0_id.append((i, j))
P0_num = 0
for i in range(all_num):
    for j in range(i + 1, all_num):
        if np.sum(D0_id[i]) == np.sum(D0_id[j]):
            P0_num += 1
A = []
idx = 0
pair_of = {}
for i in range(all_num):
    for j in range(i + 1, all_num):
        if np.sum(D0_id[i]) == np.sum(D0_id[j]):
            pair_of[idx] = (D0_id[i], D0_id[j])
            idx += 1
for k in range(P0_num):
    (i, j), (m, n) = pair_of[k]
    x = np.zeros(p)
    x[i] += 1
    x[j] += 1
    x[m] -= 1
    x[n] -= 1
    A.append(x)
A = np.array(A).astype(int)

# P0(D): parallelograms with both endpoints in D (Also like train_add.py)
P0D_id = []
ii = 0
for i in range(all_num):
    for j in range(i + 1, all_num):
        if np.sum(D0_id[i]) == np.sum(D0_id[j]):
            if i in train_id and j in train_id:
                P0D_id.append(ii)
            ii += 1

E_init = np.array(dic["repr_eff"][0], dtype=float)
Z0 = float(np.sum(E_init ** 2))
H_code = 2.0 * (A.T @ A) / Z0
eigs_code = np.linalg.eigh(H_code)[0]
lam3_code = float(eigs_code[2])
n_h_code = float(1.0 / (lam3_code * eta)) if lam3_code > 1e-12 else float("inf")
mat = A[np.array(P0D_id)]
H_paper = 2.0 * (mat.T @ mat) / (len(P0D_id) * Z0)
eigs_paper = np.linalg.eigh(H_paper)[0]
lam3_paper = float(eigs_paper[2])
n_h_paper = float(1.0 / (lam3_paper * eta)) if lam3_paper > 1e-12 else float("inf")
lam3, n_h, eigs = lam3_code, n_h_code, eigs_code
print("code-consistent:  eigs[:5]=%s lam3=%.6f eta=%.1e n_h=%.1f 3n_h=%.1f" % (
    np.array2string(eigs_code[:5], precision=6), lam3_code, eta, n_h_code, 3 * n_h_code))
print("paper-consistent: eigs[:5]=%s lam3=%.6f eta=%.1e n_h=%.1f 3n_h=%.1f" % (
    np.array2string(eigs_paper[:5], precision=6), lam3_paper, eta, n_h_paper, 3 * n_h_paper))
print("USING code-consistent arrow (matches this run's loss); paper variant is report contrast.")
print("NN RQI step %d vs 3n_h %.0f | eff loss ~0 by ~2000" % (dic["iter_rqi"], 3 * n_h))

# --- side-by-side plot, same x-axis ---
steps = dic["steps"]
XMAX = 2000  # paper panels run 0-2000; keeps axvline from stretching the axis
fig = plt.figure(figsize=(12, 5))
for k, key, title in [(1, "repr_eff", "(c) effective theory"), (2, "repr_normalized_nn", "(d) NN")]:
    ax = plt.subplot(1, 2, k)
    reprs = np.array(dic[key])
    # repr_eff has eff_steps rows; align x to its own length if shorter
    xx = np.arange(reprs.shape[0])
    for i in range(p):
        ax.plot(xx, reprs[:, i])
        ax.text(0, reprs[0, i], str(i), fontsize=9)
    ax.set_xlim(0, XMAX)
    for i in range(p):
        ax.text(XMAX, reprs[min(XMAX, reprs.shape[0] - 1), i], str(i), fontsize=9)
    if np.isfinite(n_h):
        ax.axvline(3 * n_h, linestyle="--")
        ax.text(3 * n_h, 2.1, "3n_h", fontsize=10)
    ax.set_xlabel("optimization step")
    ax.set_ylabel("1D representation")
    ax.set_title(title)
plt.tight_layout()
plt.savefig("fig4cd_repro.png", dpi=150)
print("saved fig4cd_repro.png")

