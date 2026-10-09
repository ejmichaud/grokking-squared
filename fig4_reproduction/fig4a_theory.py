import numpy as np, sys

def build_A(D, p):
    rows = []
    for a in range(len(D)):
        for b in range(a + 1, len(D)):
            (i, j), (m, n) = D[a], D[b]
            if i + j == m + n:
                r = np.zeros(p); r[i] += 1; r[j] += 1; r[m] -= 1; r[n] -= 1
                rows.append(r)
    return np.array(rows) if rows else np.zeros((0, p))

def nullity(A, p, tol=1e-8):
    if len(A) == 0:
        return p
    return int((np.linalg.eigvalsh(A.T @ A) < tol).sum())

def wilson(k, n, z=1.96):
    ph = k / n; d = 1 + z**2 / n
    c = (ph + z**2 / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h

if __name__ == "__main__":
# Testing against p=4 to see if it matches what was done by hand (will maybe add to report)
    D1 = [(0, 2), (1, 1), (0, 3)]
    D2 = [(0, 2), (1, 1), (1, 3), (2, 2)]
    assert nullity(build_A(D1, 4), 4) == 3
    assert nullity(build_A(D2, 4), 4) == 2
    print("passed")

    p = 10
    pairs = [(i, j) for i in range(p) for j in range(i, p)]
    N = len(pairs)  # 55
    n_ds = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    out = open(f"fig4a_theory_n{n_ds}.csv", "w")
    out.write("train_num,fraction,n_ds,n_linear,P_linear,ci_lo,ci_hi,mean_n0\n")
    rng = np.random.default_rng(0)
    for k in range(1, N):
        n0s = []
        for _ in range(n_ds):
            idx = rng.choice(N, k, replace=False)
            D = [pairs[t] for t in idx]
            n0s.append(nullity(build_A(D, p), p))
        n0s = np.array(n0s); kk = int((n0s == 2).sum())
        lo, hi = wilson(kk, n_ds)
        out.write(f"{k},{k/N:.4f},{n_ds},{kk},{kk/n_ds:.4f},{lo:.4f},{hi:.4f},{n0s.mean():.3f}\n")
    out.close()

