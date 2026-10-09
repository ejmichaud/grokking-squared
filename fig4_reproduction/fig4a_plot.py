import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = np.genfromtxt("fig4a_theory_n50.csv", delimiter=",", names=True)
m = d["fraction"] >= 0.15
plt.figure(figsize=(5, 5))
plt.plot(d["fraction"][m], d["P_linear"][m], "o-")
plt.fill_between(d["fraction"][m], d["ci_lo"][m], d["ci_hi"][m], alpha=0.25)
plt.axvline(0.4, linestyle="--")
plt.xlabel("training data fraction"); plt.ylabel("P(linear structure, n0=2)")
plt.savefig("fig4a_repro.png", dpi=150)
f, P = d["fraction"], d["P_linear"]
i = np.argmax(P >= 0.5)
rc = f[i-1] + (0.5 - P[i-1]) * (f[i] - f[i-1]) / (P[i] - P[i-1])
w = f[np.argmax(P >= 0.9)] - f[np.argmax(P >= 0.1)]
print(f"theory r_c (P=0.5, linear interp) = {rc:.3f}; 10%-90% width = {w:.3f}")

