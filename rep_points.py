"""Representative points other than quantization, for a weighted sample (x, q) with q summing to 1, and the
discrepancies used to compare them with WAC.
  support_points  Mak and Joseph (2018): K equal-weight points minimizing the energy distance, by their convex-concave
                  (majorization-minimization) iteration
  herding         Chen, Welling and Smola (2010): greedy kernel herding over the observations, Gaussian kernel
  archetypes      Cutler and Breiman (1994): K convex combinations of the observations, by their alternating least squares
"""
import numpy as np
import ot
from scipy.optimize import nnls
from scipy.spatial.distance import cdist


def support_points(x, q, K, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    z = x[rng.choice(len(x), K, replace=False, p=q)] + 1e-3 * rng.standard_normal((K, x.shape[1]))
    for _ in range(iters):
        D = np.maximum(cdist(z, x), 1e-10)                                   # K x n
        E = z[:, None] - z[None]
        R = np.linalg.norm(E, axis=2) + np.eye(K)
        z_new = ((q / D) @ x + (E / R[..., None]).sum(1) / K) / (q / D).sum(1)[:, None]
        if np.abs(z_new - z).max() < 1e-8:
            break
        z = z_new
    return z


def bandwidth(x, q, m=2000, seed=0):
    """Median heuristic on a weighted subsample."""
    s = x[np.random.default_rng(seed).choice(len(x), m, p=q)]
    return np.median(cdist(s, s)[np.triu_indices(m, 1)])


def herding(x, q, K, h):
    G = np.exp(-cdist(x, x, 'sqeuclidean') / (2 * h ** 2))
    mean, idx = G @ q, []
    for t in range(K):
        idx.append(int(np.argmax(mean - G[:, idx].sum(1) / (t + 1))))
    return x[idx]


def _simplex_ls(M, b, big=200.0):
    """argmin over the simplex of ||M u - b||, by NNLS with the sum-to-one constraint as a heavily weighted row."""
    return nnls(np.r_[M, np.full((1, M.shape[1]), big)], np.r_[b, big])[0]


def archetypes(x, q, K, iters=200, seed=0, tol=1e-7):
    """Cutler and Breiman's (1994, Section 4) alternating algorithm for min sum_i q_i ||x_i - alpha_i A||^2, alpha_i in the
    simplex, A = B x with the rows of B in the simplex: the memberships alpha given A, then the archetypes one at a time
    given alpha, cycling until a pass lowers the loss by less than tol; returns (archetypes A, memberships alpha, loss)."""
    A, loss = x[np.random.default_rng(seed).choice(len(x), K, replace=False, p=q)], np.inf
    rss = lambda: q @ ((x - alpha @ A) ** 2).sum(1)
    for _ in range(iters):
        alpha = np.array([_simplex_ls(A.T, xi) for xi in x])
        new = rss()
        while True:
            for l in range(K):
                s = q * alpha[:, l]
                if s @ alpha[:, l] == 0:          # redundant archetype: move it to the worst-fitted observation
                    A[l] = x[((x - alpha @ A) ** 2).sum(1).argmax()]
                else:
                    A[l] = x.T @ _simplex_ls(x.T, s @ (x - alpha @ A + np.outer(alpha[:, l], A[l])) / (s @ alpha[:, l]))
            old, new = new, rss()
            if old - new < tol:
                break
        if loss - new < tol:
            break
        loss = new
    return A, alpha, new


def discrepancies(x, q, z, w, h):
    """W_2^2 with weights w (exact OT), quantization error (Voronoi weights), energy distance and squared MMD with the
    Gaussian kernel of bandwidth h, all against the weighted sample (x, q)."""
    C, w = cdist(x, z, "sqeuclidean"), w / w.sum()
    Dxz, Dzz, Dxx = np.sqrt(C), cdist(z, z), cdist(x, x)
    k = lambda D: np.exp(-D ** 2 / (2 * h ** 2))
    return dict(W2=float(np.sum(ot.emd(q, w, C, numItermax=10 ** 7) * C)), quant=float(q @ C.min(1)),
                energy=float(2 * q @ Dxz @ w - w @ Dzz @ w - q @ Dxx @ q),
                mmd=float(w @ k(Dzz) @ w - 2 * q @ k(Dxz) @ w + q @ k(Dxx) @ q))
