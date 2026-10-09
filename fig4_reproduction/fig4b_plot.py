# Fig 4.b: Empirical results display a phase transition of RQI around rc = 0.4, in agreement with the theory (the blue line shows the median of multiple random seeds). 
import numpy as np
from matplotlib import pyplot as plt

# Hardcoding parameters
path4 = "./results/fig4/"
train_nums = [5,10,15,20,25,30,35,40,45,50,54]
seeds = [0,1,2]
all_num = 55  # p=10
NONCONV = 9999

xs, ys = [], []
for tn in train_nums:
    for s in seeds:
        v = float(np.loadtxt(path4+f"rqistep_num_{tn}_seed_{s}.txt"))
        xs.append(tn/all_num)
        ys.append(v)
xs = np.array(xs); ys = np.array(ys)

# Sanity check - Do we get the data generated?
assert len(xs)==33
print("nonconv:", (ys>=9999).sum(), "conv:", (ys<9999).sum())

fracs = np.array(sorted(set(xs)))
meds = np.array([np.median(ys[xs==f]) for f in fracs])

for f,m in zip(fracs, meds):
    print(f"{f:.3f} median {m:.0f}")

all_conv = [f for f in fracs if np.all(ys[xs==f] < NONCONV)]
none_conv = [f for f in fracs if np.all(ys[xs==f] >= NONCONV)]
lo = max(none_conv) if len(none_conv) else 0.0
hi = min(all_conv) if len(all_conv) else 1.0
print(f"bracket: never up to {lo:.3f}, always from {hi:.3f}")
print(f"r_c ~ {(lo+hi)/2:.2f} +/- {(hi-lo)/2:.2f}")

# Plotting time: Seaborn to make it pretty but probably matplotlib because it is easier:
conv = ys < NONCONV
plt.figure(figsize=(5,5))
plt.scatter(xs[conv], ys[conv], label="reached RQI>0.95")
plt.scatter(xs[~conv], ys[~conv], marker="^", color="red", label="didn't reach")
plt.plot(fracs, meds)  # blue median like paper
plt.yscale("log"); plt.ylim(2e2, 1.2e4)
plt.axvline(0.4, linestyle="--")
plt.xlabel("training data fraction"); plt.ylabel("Steps to RQI>0.95")
plt.legend(); plt.savefig("fig4b_repro.png")
