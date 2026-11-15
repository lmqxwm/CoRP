"""Wasserstein coarsened posterior over representative points: model, exact transport and the rung updates shared by
the bridge sampler (bridge_sampler.py) and the trans-dimensional sampler (trans_sampler.py).

Target   pi(K, z, w | x) ∝ Pi(K) Pi(z | K) Pi(w | K) exp[-Lam{W(z, w)}],   W(z, w) = W_p^p(P_n, sum_j w_j delta_{z_j}).
A rung with inverse temperature beta = lambda / lambda_n in [0, 1] targets
         Pi(z | K) Pi(w | K) exp{-beta E(K, z, w)},   E = Lam(W) - log Pi(K),
so beta = 0 is the conditional prior with K uniform on 1..K_max and beta = 1 is the target."""
import numpy as np
import ot
from scipy.integrate import quad
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import betaln, gammaln


# ---------------------------------------------------------------- tolerance laws: Lam = -log F and its derivative

class Exp:
    def __init__(self, rate):
        self.rate = rate

    def Lam(self, t):
        return self.rate * t

    def dLam(self, t):
        return self.rate


class Lomax:
    def __init__(self, alpha, scale):
        self.alpha, self.scale = alpha, scale

    def Lam(self, t):
        return self.alpha * np.log1p(t / self.scale)

    def dLam(self, t):
        return self.alpha / (self.scale + t)


class Weibull:
    def __init__(self, shape, scale):
        self.shape, self.scale = shape, scale

    def Lam(self, t):
        return (t / self.scale) ** self.shape

    def dLam(self, t):
        return self.shape / self.scale * (t / self.scale) ** (self.shape - 1)


# ---------------------------------------------------------------- priors (log densities)

def penalty(c):
    """log Pi(K) = -c K."""
    return lambda K: -c * K


def poisson(mu):
    """K - 1 ~ Poisson(mu)."""
    return lambda K: (K - 1) * np.log(mu) - gammaln(K)


class GaussianAtoms:
    """z_j iid N(mean, sd^2 I_d)."""

    def __init__(self, mean, sd):
        self.mean, self.sd = np.atleast_1d(np.asarray(mean, float)), float(sd)

    def __call__(self, z):
        return -0.5 * ((z - self.mean) ** 2).sum() / self.sd ** 2 - z.size * np.log(self.sd * np.sqrt(2 * np.pi))


class Dirichlet:
    """w ~ Dir(alpha, ..., alpha)."""

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        K, a = len(w), self.alpha
        return gammaln(K * a) - K * gammaln(a) + (a - 1) * np.log(w).sum()


# ---------------------------------------------------------------- model and exact transport

class Model:
    """Data x (n or n x d), cost exponent p >= 1, tolerance law, log priors, counts 1..K_max, optional row masses
    (survey weights, default 1/n).  The atom update uses the Gaussian prior of the atoms, or for another atom prior the
    Gaussian guide N(data mean, (2 sbar)^2 I), sbar = average coordinate standard deviation."""

    def __init__(self, x, p=2.0, radius=None, prior_K=None, prior_z=None, prior_w=None, K_max=10, weights=None):
        x = np.asarray(x, float)
        self.x = x[:, None] if x.ndim == 1 else x
        self.n, self.d = self.x.shape
        self.q = np.full(self.n, 1 / self.n) if weights is None else np.asarray(weights, float) / np.sum(weights)
        self.p, self.radius, self.K_max = p, radius, K_max
        sbar = self.x.std(0).mean()
        guide = GaussianAtoms(self.q @ self.x, 2 * sbar)
        self.prior_K = prior_K or (lambda K: 0.0)
        self.prior_z = prior_z or guide
        self.prior_w = prior_w or Dirichlet(1.0)
        if isinstance(self.prior_z, GaussianAtoms):
            guide = self.prior_z
        self.guide_mean, self.guide_prec = guide.mean, guide.sd ** -2
        self.h_c = 0.1 * sbar                                    # distance floor of the atom update for p < 2
        self.ridge = 0.01 * sbar                                 # ridge of the split offset covariance
        self.order = np.argsort(self.x[:, 0], kind='stable')     # for the d = 1 monotone coupling
        self.x_sorted, self.cum_q = self.x[self.order, 0], np.r_[0.0, np.cumsum(self.q[self.order])]

    def cost_matrix(self, z):
        """D_ij = ||x_i - z_j||^p."""
        D = cdist(self.x, z, 'sqeuclidean')
        return D if self.p == 2 else D ** (self.p / 2)

    def voronoi_masses(self, z):
        return np.bincount(self.cost_matrix(z).argmin(1), self.q, minlength=len(z))

    def transport(self, z, w):
        """(W, optimal plan Gamma, per-atom costs D_j = sum_i Gamma_ij D_ij): monotone coupling for d = 1, network
        simplex (ot.emd) for d >= 2."""
        if self.d > 1:
            D = self.cost_matrix(z)
            G = ot.emd(self.q, w / w.sum(), D, numItermax=10 ** 7)
            Dj = (G * D).sum(0)
            return Dj.sum(), G, Dj
        n, K, zz = self.n, len(w), z[:, 0]
        oz = np.argsort(zz, kind='stable')
        cw = np.r_[0.0, np.cumsum(w[oz])]
        cw /= cw[-1]
        breaks = np.unique(np.r_[cw, self.cum_q])                # overlaps of the quantile intervals of x and z
        mid, mass = 0.5 * (breaks[:-1] + breaks[1:]), np.diff(breaks)
        i = np.minimum(np.searchsorted(self.cum_q, mid, 'right') - 1, n - 1)
        k = np.minimum(np.searchsorted(cw, mid, 'right') - 1, K - 1)
        G = np.zeros((n, K))
        np.add.at(G, (self.order[i], oz[k]), mass)
        Dj = np.bincount(oz[k], mass * np.abs(self.x_sorted[i] - zz[oz][k]) ** self.p, minlength=K)
        return Dj.sum(), G, Dj


def quantizer(M, K, n_init=10, seed=0):
    """(empirical distortion, z): the best of n_init Lloyd runs from k-means++ seeds (weighted p-centres for p != 2)."""
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    for _ in range(n_init):
        z = M.x[[rng.choice(M.n, p=M.q)]]
        for _ in range(K - 1):
            pr = M.q * M.cost_matrix(z).min(1)
            z = np.r_[z, M.x[[rng.choice(M.n, p=pr / pr.sum())]]]
        for _ in range(100):
            cell = M.cost_matrix(z).argmin(1)
            z_new = z.copy()
            for j in range(K):
                xj, qj = M.x[cell == j], M.q[cell == j]
                if len(xj):
                    c = qj @ xj / qj.sum()
                    for _ in range(0 if M.p == 2 else 10):      # Weiszfeld-type steps for the p-centre
                        v = qj * np.maximum(np.linalg.norm(xj - c, axis=1), 1e-12) ** (M.p - 2)
                        c = v @ xj / v.sum()
                    z_new[j] = c
            if np.allclose(z_new, z):
                break
            z = z_new
        L = M.q @ M.cost_matrix(z).min(1)
        if L < best[0]:
            best = (L, z)
    return best


def lloyd_start(M, K, seed=0):
    """Chain start (z, w): the best Lloyd quantizer and its Voronoi masses, moved off the boundary of the simplex."""
    z = quantizer(M, K, seed=seed)[1]
    w = M.voronoi_masses(z) + 1e-3
    return z, w / w.sum()


def relabel(z_ref, z):
    """Permutation of the rows of z that best matches z_ref."""
    return linear_sum_assignment(cdist(z_ref, z, 'sqeuclidean'))[1]


# ---------------------------------------------------------------- one rung

def log_beta_pdf(u, a, b):
    return (a - 1) * np.log(u) + (b - 1) * np.log1p(-u) - betaln(a, b)


class Rung:
    """Current state (K, z, w) of one rung, its transport cost W, plan, per-atom costs and log target, and the burn-in
    moment sums that set the weight concentration kappa_K.  One sweep updates every atom (zplan), the weights (wdir)
    and, if split_merge, the count."""
    A_U, C1, C2 = 2.0, 1.6, 0.5          # u ~ Beta(2, 2); offset mean c1 sqrt(gamma_1) e_1; offset covariance c2^2 S_j

    def __init__(self, M, beta, rng, z, w):
        self.M, self.beta, self.rng = M, beta, rng
        self.kappa_sums = np.zeros((2, M.K_max + 1))   # per K: sums of sum_j v_j (1 - v_j) and ||w - v(z)||^2
        self.adapt = True
        self.z, self.w = np.asarray(z, float).reshape(-1, M.d), np.asarray(w, float)
        self.logpost, self.W, self.plan, self.Dj = self.evaluate(self.z, self.w)

    def copy(self, beta):
        new = Rung.__new__(Rung)
        new.__dict__.update(self.__dict__)
        new.kappa_sums, new.beta = self.kappa_sums.copy(), beta
        new.logpost = new.log_target(new.z, new.w, new.W)
        return new

    def energy(self):
        return self.M.radius.Lam(self.W) - self.M.prior_K(len(self.w))

    def log_target(self, z, w, W):
        M = self.M
        return M.prior_z(z) + M.prior_w(w) + self.beta * (M.prior_K(len(w)) - M.radius.Lam(W))

    def evaluate(self, z, w):
        """(log target, W, plan, per-atom costs) of a proposed state, None outside the support."""
        if len(w) > self.M.K_max or np.any(w <= 0):
            return None
        W, G, Dj = self.M.transport(z, w)
        return self.log_target(z, w, W), W, G, Dj

    def mh(self, z, w, new, log_r):
        """Metropolis-Hastings step to (z, w) with evaluation new and log proposal ratio log_r."""
        if new is not None and np.log(self.rng.random()) < new[0] - self.logpost + log_r:
            self.z, self.w = z, w
            self.logpost, self.W, self.plan, self.Dj = new

    def kappa(self, K):
        s = self.kappa_sums[:, K]
        return max(s[0] / s[1] - 1, 1.0) if s[1] > 0 else self.M.n

    # -- atoms
    def surrogate(self, z, W, G, j):
        """Precision P_j and mean mu_j of the Gaussian surrogate of the log target in z_j with the plan G fixed."""
        M = self.M
        omega = (self.beta * M.radius.dLam(W) * M.p * max(M.p - 1, 1) * G[:, j]
                 * np.maximum(np.linalg.norm(M.x - z[j], axis=1), M.h_c) ** (M.p - 2))
        P = M.guide_prec + omega.sum()
        return P, (M.guide_prec * M.guide_mean + omega @ M.x) / P

    def update_atom(self, j):
        z, d = self.z, self.M.d
        P, mu = self.surrogate(z, self.W, self.plan, j)
        z_new = z.copy()
        z_new[j] = mu + self.rng.standard_normal(d) / np.sqrt(P)
        new = self.evaluate(z_new, self.w)
        if new is None:
            return
        P_new, mu_new = self.surrogate(z_new, new[1], new[2], j)
        log_r = (0.5 * (P * ((z_new[j] - mu) ** 2).sum() - P_new * ((z[j] - mu_new) ** 2).sum())
                 + 0.5 * d * np.log(P_new / P))
        self.mh(z_new, self.w, new, log_r)

    # -- weights
    def update_weights(self):
        """Dirichlet independence proposal Dir{kappa_K v(z) + 1/2} centred at the Voronoi masses v(z)."""
        K, w = len(self.w), self.w
        v = self.M.voronoi_masses(self.z)
        if self.adapt:
            self.kappa_sums[:, K] += (v @ (1 - v), ((w - v) ** 2).sum())
        a = self.kappa(K) * v + 0.5
        w_new = self.rng.dirichlet(a)
        new = self.evaluate(self.z, w_new)
        if new is not None:
            self.mh(self.z, w_new, new, ((a - 1) * (np.log(w) - np.log(w_new))).sum())

    # -- count: split / merge
    def split_prob(self, K):
        return 1.0 if K == 1 else 0.0 if K == self.M.K_max else 0.5

    @staticmethod
    def parent_probs(Dj):
        """p_s(j) = 1/(2K) + D_j / (2 sum_k D_k), uniform if sum_k D_k = 0."""
        s = Dj.sum()
        return 0.5 / len(Dj) + 0.5 * Dj / s if s > 0 else np.full(len(Dj), 1 / len(Dj))

    @staticmethod
    def log_pair_prob(z, a, b):
        """p_m({a, b}) = 1/(K(K-1)) + {1(b = nn(a)) + 1(a = nn(b))} / (2K): half a uniform pair, half a uniform atom with
        its nearest neighbour."""
        K = len(z)
        D = cdist(z, z, 'sqeuclidean')
        np.fill_diagonal(D, np.inf)
        nn = D.argmin(1)
        return np.log(1 / (K * (K - 1)) + (int(nn[a] == b) + int(nn[b] == a)) / (2 * K))

    def offset_law(self, G, j, w_j):
        """(mu_s, Cholesky factor of Sigma_j): mu_s = c1 sqrt(gamma_1) e_1 from the leading eigenpair of the plan-weighted
        covariance S_j of the cell of atom j, Sigma_j = c2^2 S_j + ridge^2 I."""
        x, g = self.M.x, G[:, j] / w_j
        xbar = g @ x
        S = ((x - xbar) * g[:, None]).T @ (x - xbar)
        eig, vec = np.linalg.eigh(S)
        return (self.C1 * np.sqrt(max(eig[-1], 0.0)) * vec[:, -1],
                np.linalg.cholesky(self.C2 ** 2 * S + self.M.ridge ** 2 * np.eye(self.M.d)))

    def log_offset_density(self, Delta, mu, chol):
        """log g_j(Delta), g_j = N(mu_s, Sigma_j) / 2 + N(-mu_s, Sigma_j) / 2."""
        a, b = np.linalg.solve(chol, Delta - mu), np.linalg.solve(chol, Delta + mu)
        return (np.logaddexp(-0.5 * a @ a, -0.5 * b @ b) - np.log(2) - np.log(np.diag(chol)).sum()
                - 0.5 * self.M.d * np.log(2 * np.pi))

    def log_ratio_split(self, K, w, G, Dj, j, u, Delta, z_split, a, b, law=None):
        """log r of splitting atom j of the K-atom state (w, G, Dj) into the atoms a, b of z_split; a merge uses -1 times
        this ratio evaluated at the merged state."""
        mu, chol = law or self.offset_law(G, j, w[j])
        return (np.log(1 - self.split_prob(K + 1)) + self.log_pair_prob(z_split, a, b) + np.log((K + 1) / 2 * w[j])
                - np.log(self.split_prob(K)) - np.log(self.parent_probs(Dj)[j]) - log_beta_pdf(u, self.A_U, self.A_U)
                - self.log_offset_density(Delta, mu, chol))

    def update_count(self):
        rng, z, w, d = self.rng, self.z, self.w, self.M.d
        K = len(w)
        if rng.random() < self.split_prob(K):
            j = rng.choice(K, p=self.parent_probs(self.Dj))
            u = rng.beta(self.A_U, self.A_U)
            law = mu, chol = self.offset_law(self.plan, j, w[j])
            Delta = (mu if rng.random() < 0.5 else -mu) + chol @ rng.standard_normal(d)
            a, b = rng.choice(K + 1, 2, replace=False)               # positions of the two children
            rest = np.setdiff1d(np.arange(K + 1), [a, b])
            z_new, w_new = np.empty((K + 1, d)), np.empty(K + 1)
            z_new[rest], w_new[rest] = np.delete(z, j, 0), np.delete(w, j)
            z_new[a], z_new[b] = z[j] + (1 - u) * Delta, z[j] - u * Delta
            w_new[a], w_new[b] = u * w[j], (1 - u) * w[j]
            new = self.evaluate(z_new, w_new)
            if new is not None:
                self.mh(z_new, w_new, new, self.log_ratio_split(K, w, self.plan, self.Dj, j, u, Delta, z_new, a, b, law))
        else:
            if rng.random() < 0.5:
                a, b = rng.choice(K, 2, replace=False)
            else:
                D = cdist(z, z, 'sqeuclidean')
                np.fill_diagonal(D, np.inf)
                a = rng.integers(K)
                b = int(D[a].argmin())
            u = w[a] / (w[a] + w[b])
            j = rng.integers(K - 1)                                  # position of the merged atom
            rest = np.delete(np.arange(K), [a, b])
            z_new = np.insert(z[rest], j, u * z[a] + (1 - u) * z[b], 0)
            w_new = np.insert(w[rest], j, w[a] + w[b])
            new = self.evaluate(z_new, w_new)
            if new is not None:
                self.mh(z_new, w_new, new, -self.log_ratio_split(K - 1, w_new, new[2], new[3], j, u, z[a] - z[b], z, a, b))

    def sweep(self, split_merge):
        for j in range(len(self.w)):
            self.update_atom(j)
        if len(self.w) > 1:
            self.update_weights()
        if split_merge:
            self.update_count()


# ---------------------------------------------------------------- non-reversible parallel tempering

def place_ladder(betas, rejection, N_max):
    """2 B + 1 rungs (at most N_max) with equal estimated swap rejection, B = sum of the rejection rates (Syed et al.
    2022); the cumulative rejection is interpolated linearly in beta^(1/5) between the old rungs."""
    cum = np.r_[0.0, np.cumsum(np.maximum(rejection, 1e-3))]
    N = int(np.clip(np.ceil(2 * cum[-1]) + 1, 2, N_max))
    return np.interp(np.linspace(0, cum[-1], N), cum, betas ** 0.2) ** 5


def run(M, n_iter, init, split_merge=True, temper=True, burn=0.3, thin=1, seed=0, N0=6, N_max=48):
    """n_iter iterations; each updates every rung by one sweep and then swaps adjacent rungs in DEO order (even pairs at
    even t, odd pairs at odd t).  In the first burn x n_iter iterations the ladder is re-placed after rounds of 10, 20,
    40, ... iterations and kappa_K adapts; both are then frozen.  Returns, for the last rung after burn-in, the trace of
    K and every thin-th state (z, w) with its cost W, plus the ladder and the swap acceptance rates."""
    rng = np.random.default_rng(seed)
    betas = np.linspace(0.0, 1.0, N0) ** 5 if temper else np.ones(1)
    rungs = [Rung(M, beta, rng, *init) for beta in betas]
    swaps = np.zeros((N_max, 2))                       # accepted and attempted swaps of rungs l, l + 1
    n_burn, next_round, round_len = int(burn * n_iter), 10, 10
    K_trace, draws, W_draws = [], [], []
    for t in range(1, n_iter + 1):
        for r in rungs:
            r.sweep(split_merge)
        for l in range(t % 2, len(rungs) - 1, 2):
            A, B = rungs[l], rungs[l + 1]
            swaps[l, 1] += 1
            if np.log(rng.random()) < (B.beta - A.beta) * (B.energy() - A.energy()):
                state = lambda r: (r.z, r.w, r.W, r.plan, r.Dj)
                (A.z, A.w, A.W, A.plan, A.Dj), (B.z, B.w, B.W, B.plan, B.Dj) = state(B), state(A)
                A.logpost, B.logpost = A.log_target(A.z, A.w, A.W), B.log_target(B.z, B.w, B.W)
                swaps[l, 0] += 1
        if temper and t == next_round and t <= n_burn:
            new_betas = place_ladder(betas, 1 - swaps[:len(rungs) - 1, 0] / np.maximum(swaps[:len(rungs) - 1, 1], 1), N_max)
            rungs = [rungs[int(np.argmin(np.abs(betas - beta)))].copy(beta) for beta in new_betas]
            betas, swaps = new_betas, np.zeros((N_max, 2))
            round_len *= 2
            next_round = t + round_len
        if t == n_burn:
            for r in rungs:
                r.adapt = False
            swaps[:] = 0
        if t > n_burn:
            last = rungs[-1]
            K_trace.append(len(last.w))
            if (t - n_burn) % thin == 0:
                draws.append((last.z.copy(), last.w.copy()))
                W_draws.append(last.W)
    return dict(K=np.array(K_trace), draws=draws, W=np.array(W_draws), betas=betas,
                swap_acc=swaps[:len(rungs) - 1, 0] / np.maximum(swaps[:len(rungs) - 1, 1], 1))


# ---------------------------------------------------------------- conditional calibration at a fixed count

def calibrate(M, draws, K, design=None):
    """Sandwich-calibrated draws of theta = (z, w_1..w_{K-1}) from the draws with K atoms.  Draws are relabelled to the
    best Lloyd quantizer; J^{-1} = S / c_n with S their covariance; the influence of x in the Voronoi cell j of zbar is
    J^{-1}[:, z_j] p ||x - zbar_j||^{p-2} (x - zbar_j) + e_{w_j}; Omega is its q-weighted covariance, or with
    design = (strata, psu) n times the with-replacement linearization variance; theta~ = thetabar + A (theta - thetabar)
    with A S A' = Omega / n."""
    z_ref = quantizer(M, K, 20)[1]
    T = np.array([np.r_[z[o].ravel(), w[o][:K - 1]] for z, w in draws if len(w) == K for o in [relabel(z_ref, z)]])
    theta_bar, S, kd = T.mean(0), np.atleast_2d(np.cov(T.T)), K * M.d
    z_bar = theta_bar[:kd].reshape(K, M.d)
    m = kd + K - 1
    D = M.cost_matrix(z_bar)
    L, cell = M.q @ D.min(1), D.argmin(1)
    a = M.radius.dLam(L)
    F = lambda s, k: s ** k * np.exp(-(M.radius.Lam(L + s * s / (2 * a)) - M.radius.Lam(L)))
    c_n = quad(F, 0, np.inf, args=(m + 1,))[0] / (a * m * quad(F, 0, np.inf, args=(m - 1,))[0])
    J_inv = S / c_n
    psi = np.zeros((M.n, m))
    for j in range(K):
        v = M.x[cell == j] - z_bar[j]
        psi[cell == j] = M.p * np.linalg.norm(v, axis=1, keepdims=True) ** (M.p - 2) * v @ J_inv[:, j * M.d:(j + 1) * M.d].T
        if j < K - 1:
            psi[cell == j, kd + j] += 1
    if design is None:
        Omega = np.atleast_2d(np.cov(psi.T, aweights=M.q, bias=True))
    else:
        strata, psu = design
        tq, Omega = M.q[:, None] * (psi - M.q @ psi), 0.0
        for h in np.unique(strata):
            totals = np.array([tq[(strata == h) & (psu == k)].sum(0) for k in np.unique(psu[strata == h])])
            dev = totals - totals.mean(0)
            Omega = Omega + len(totals) / (len(totals) - 1) * dev.T @ dev
        Omega = M.n * np.atleast_2d(Omega)
    e, U = np.linalg.eigh(S)
    eo, Uo = np.linalg.eigh(Omega / M.n)
    A = Uo @ np.diag(np.sqrt(np.maximum(eo, 0))) @ Uo.T @ U @ np.diag(e ** -0.5) @ U.T
    T_adj = theta_bar + (T - theta_bar) @ A.T
    return dict(z_bar=z_bar, w_bar=np.r_[theta_bar[kd:], 1 - theta_bar[kd:].sum()], S=S, Omega=Omega, c_n=c_n,
                z=T_adj[:, :kd].reshape(-1, K, M.d), w=np.c_[T_adj[:, kd:], 1 - T_adj[:, kd:].sum(1)])
