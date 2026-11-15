"""Simulation study for simulation.ipynb: data-generating processes, population quantities, the comparison methods
(k-means with the elbow rule, Gaussian mixture with BIC, Gaussian mixture of finite mixtures) and one job per simulated
data set.

DGPs in R^2, each truncated to a box B (rejection sampling):
  gmm   sum_k pi_k N(mu_k, Sigma_k) with unequal weights and covariances
  t     sum_k pi_k t_3(mu_k, Sigma_k / 3)  (component covariance Sigma_k before truncation)
  ring  radius 2 + N(0, 0.2^2), angle density ∝ 1 + 0.5 cos a + 0.3 sin 2a + 0.3 cos 4a
Truncation (compact convex support, (D3)) and the asymmetry of the angle density (unique optimum, (D4)) are needed
only for the asymptotic normality theorem; the count results need neither."""
import numpy as np
from kneed import KneeLocator
from numba import njit
from scipy.cluster.vq import kmeans2
from scipy.special import gammaln, logsumexp, multigammaln, softmax
from scipy.stats import invwishart
from helper_functions import Model, Exp, penalty, GaussianAtoms, Dirichlet, quantizer, lloyd_start, relabel, calibrate, run
from bridge_sampler import bridge_sampler, evidence
import rep_points as rp

LAM0, KMAX, KFIT = 0.5, 16, 8                       # lambda_0, K_max of the WAC sampler and of the elbow rule, largest K reported and searched by GMM-BIC
MU = np.array([[0., 0.], [3., 0.], [1.5, 2.5]])
PI = np.array([.5, .3, .2])
SIG = np.array([[[1.0, 0.3], [0.3, 0.6]], [[0.5, 0.0], [0.0, 0.5]], [[0.4, -0.1], [-0.1, 0.8]]])
BOX = {'gmm': ((-3, -3), (6, 6)), 't': ((-3, -3), (6, 6)), 'ring': ((-3, -3), (3, 3))}
KDAG = {'gmm': 3, 't': 3, 'ring': 4}
SEED = {'gmm': 1, 't': 2, 'ring': 3}
ANGLE = lambda a: 1 + 0.5 * np.cos(a) + 0.3 * np.sin(2 * a) + 0.3 * np.cos(4 * a)


def _draw(rng, dgp, n):
    if dgp == 'ring':
        a = rng.uniform(0, 2 * np.pi, n)
        a = a[rng.random(n) * 2.1 < ANGLE(a)]
        r = 2 + 0.2 * rng.standard_normal(len(a))
        return np.c_[r * np.cos(a), r * np.sin(a)]
    k = rng.choice(3, n, p=PI)
    e = np.einsum('nij,nj->ni', np.linalg.cholesky(SIG)[k], rng.standard_normal((n, 2)))
    return MU[k] + (e if dgp == 'gmm' else e / np.sqrt(3) * np.sqrt(3 / rng.chisquare(3, n))[:, None])


def gen(rng, dgp, n):
    x = np.empty((0, 2))
    while len(x) < n:
        y = _draw(rng, dgp, 2 * n)
        x = np.r_[x, y[np.all((y >= BOX[dgp][0]) & (y <= BOX[dgp][1]), 1)]]
    return x[:n]


def population(dgp, seed=1):
    """L*_K (K = 1..9, 2 x 10^5 draws), c = geometric mean of the two cost gaps around K†, gamma, and theta* = (z*, w*)
    at K† from 10^6 draws (best of 30 Lloyd starts), with the spread of the 30 optima as a (D4) check."""
    rng = np.random.default_rng(seed)
    M = Model(gen(rng, dgp, 200000), 2)
    L = np.array([quantizer(M, k, n_init=5)[0] for k in range(1, 10)])
    K, g = KDAG[dgp], -np.diff(L)
    c = float(np.sqrt(g[K - 2] * g[K - 1]))
    crit = np.sort(L[:KFIT] + c * np.arange(1, KFIT + 1))
    Mb = Model(gen(rng, dgp, 10 ** 6), 2)
    sols = sorted((quantizer(Mb, K, n_init=1, seed=s) for s in range(30)), key=lambda t: t[0])
    z = sols[0][1]
    spread = max(np.abs(s[1][relabel(z, s[1])] - z).max() for s in sols if s[0] - sols[0][0] < 1e-4)
    return dict(L=L, c=c, gamma=crit[1] - crit[0], K=K, z=z, w=Mb.voronoi_masses(z), share_best=np.mean([s[0] - sols[0][0] < 1e-4 for s in sols]),
                spread=spread, P0=gen(rng, dgp, 20000))


def theta(z, w):
    return np.r_[z.ravel(), w[:-1]]


def elbow(L):
    """Elbow of the cost curve L_1, ..., L_K by Kneedle (Satopaa et al. 2011, package kneed, sensitivity S = 1)."""
    return int(KneeLocator(np.arange(1, len(L) + 1), L, curve='convex', direction='decreasing').knee)


def em(rng, x, K, restarts=4, iters=200):
    """Full-covariance Gaussian mixture by EM: (log-likelihood, means, weights, covariances, responsibilities)."""
    best, d = None, x.shape[1]
    for _ in range(restarts):
        m, S, p = x[rng.choice(len(x), K, replace=False)], np.array([np.cov(x.T)] * K), np.full(K, 1 / K)
        for _ in range(iters):
            lr = np.empty((len(x), K))
            for j in range(K):
                C = np.linalg.cholesky(S[j])
                u = np.linalg.solve(C, (x - m[j]).T)
                lr[:, j] = np.log(p[j]) - .5 * (u ** 2).sum(0) - np.log(np.diag(C)).sum() - d / 2 * np.log(2 * np.pi)
            mx = lr.max(1, keepdims=True)
            r = np.exp(lr - mx)
            ll = (np.log(r.sum(1)) + mx[:, 0]).sum()
            r /= r.sum(1, keepdims=True)
            nk = r.sum(0) + 1e-10
            p, m = nk / len(x), r.T @ x / nk[:, None]
            S = np.array([(r[:, j, None] * (x - m[j])).T @ (x - m[j]) / nk[j] + 1e-6 * np.eye(d) for j in range(K)])
        if best is None or ll > best[0]:
            best = (ll, m, p, S, r)
    return best


def gmm_bic(rng, x):
    fits = [em(rng, x, K) for K in range(1, KFIT + 1)]
    bic = [-2 * f[0] + (6 * K - 1) * np.log(len(x)) for K, f in enumerate(fits, 1)]
    return int(np.argmin(bic)) + 1, fits[int(np.argmin(bic))], np.array(bic)


def data(dgp, n, rep):
    rng = np.random.default_rng([SEED[dgp], n, rep])
    return rng, gen(rng, dgp, n)


def make_model(x, c):
    n = len(x)
    return Model(x, 2, Exp(n * LAM0), penalty(n * LAM0 * c), GaussianAtoms(np.zeros(2), 10.0), Dirichlet(1.0), K_max=KMAX)


def job(a):
    """One data set: (dgp, n, rep, pop, full).  full: Pi(K | x) by bridge sampling (bridge_sampler, K_max = KMAX) plus
    k-means elbow and GMM-BIC counts; otherwise only the two fixed-K chains at K† (as in bridge_sampler).  Both: sandwich
    calibration at K† and coverage of theta* by the 95% posterior, sandwich and k-means bootstrap (B = 200) intervals,
    and the posterior-mean transport allocations.  The population quantities in pop (c, K†, theta*) enter the fit only
    through the count prior (c) and the count K† at which the conditional summaries are computed; theta* is used
    afterwards to label the atoms and to score coverage."""
    dgp, n, rep, P, full = a
    rng, x = data(dgp, n, rep)
    K = P['K']
    M = make_model(x, P['c'])
    o = dict(dgp=dgp, n=n, rep=rep)
    if full:
        pK, ev = bridge_sampler(M)
        o.update(pK=pK, logZ={k: e['logZ'] for k, e in ev.items()}, delta={k: e['delta'] for k, e in ev.items()},
                 re={k: e['e'] for k, e in ev.items()})
        draws = ev[K]['runs'][0]['draws'] + ev[K]['runs'][1]['draws']
        Lhat = np.array([quantizer(M, k)[0] for k in range(1, KMAX + 1)])
        o.update(K_km=elbow(Lhat), Lhat=Lhat, K_gmm=gmm_bic(rng, x)[0])
    else:
        draws = sum((run(M, 2000, lloyd_start(M, K, s), split_merge=False, temper=False, seed=s)['draws'] for s in (0, 1)), [])
    cal = calibrate(M, draws, K)
    perm = relabel(P['z'], cal['z_bar'])                                  # calibrate's labels -> theta* labels
    raw = np.array([theta(z[q][perm], w[q][perm]) for z, w in draws for q in [relabel(cal['z_bar'], z)]])
    adj = np.c_[cal['z'][:, perm].reshape(len(cal['z']), -1), cal['w'][:, perm][:, :K - 1]]
    zk = quantizer(M, K)[1]
    zk = zk[relabel(P['z'], zk)]
    th_hat = theta(zk, M.voronoi_masses(zk))
    boot = []
    for _ in range(200):
        Mb = Model(x[rng.integers(n, size=n)])
        zb = quantizer(Mb, K, n_init=3, seed=int(rng.integers(1 << 30)))[1]
        zb = zb[relabel(zk, zb)]
        boot.append(theta(zb, Mb.voronoi_masses(zb)))
    th = theta(P['z'], P['w'])
    cover = lambda D: ((np.percentile(D, 2.5, 0) <= th) & (th <= np.percentile(D, 97.5, 0)))
    hw = lambda D: np.diff(np.percentile(D, [2.5, 97.5], 0), axis=0)[0] / 2
    sub = [draws[i] for i in rng.choice(len(draws), 100, replace=False)]
    alloc = np.zeros((n, K))
    for z, w in sub:
        q = relabel(P['z'], z)
        alloc += M.transport(z[q], w[q])[1] * n / len(sub)
    o.update(cov_post=cover(raw), cov_sand=cover(adj), cov_boot=cover(np.array(boot)), hw_post=hw(raw), hw_sand=hw(adj),
             hw_boot=hw(np.array(boot)), th_hat=th_hat, post_mean=raw.mean(0), post_sd=raw.std(0), soft=np.mean(alloc.max(1) < 0.9))
    if rep == 0:
        o.update(x=x, alloc=alloc, raw=raw, adj=adj, boot=np.array(boot))
    return o


def path_job(a):
    """log Z_K at one count on one data set (two fixed-K chains), for Pi(K | x) as a function of c: (dgp, n, rep, pop, K).
    run_sims.py uses K = 1..KMAX; the counts above KFIT are not reported one by one, but they keep the probabilities of
    K <= KFIT correct at small c."""
    dgp, n, rep, P, K = a
    M = make_model(data(dgp, n, rep)[1], P['c'])
    lz = [evidence(M, K, max(2000, 100 * (3 * K - 1)), s)['logZ'] for s in (0, 1)]
    return dict(dgp=dgp, n=n, rep=rep, K=K, logZ=np.logaddexp(*lz) - np.log(2), delta=abs(lz[0] - lz[1]))


@njit(cache=True)
def _lm(m, s, Q, G, B0, u0, kappa0, nu0):
    """log of Gamma(gamma + m) times the marginal likelihood of a cluster of m points with sum s and sum of squares Q, up to
    lm0 (see mfm) per cluster and a constant per point."""
    v = s + u0
    return G[m] - (nu0 + m) / 2 * np.linalg.slogdet(B0 + Q - np.outer(v, v) / (kappa0 + m))[1]


@njit(cache=True)
def _seq(x, i, j, rest, logu, to_i, draw, G, B0, u0, kappa0, nu0):
    """Sequential allocation of the points rest to the clusters {i} and {j} (Dahl 2003): the side of each point, drawn with
    the log uniforms logu if draw and given by to_i otherwise, the log probability of these sides and the sum of _lm over
    the two clusters."""
    m, s = np.ones(2, np.int64), np.stack((x[i], x[j]))
    Q = np.stack((np.outer(x[i], x[i]), np.outer(x[j], x[j])))
    l = np.array([_lm(1, s[0], Q[0], G, B0, u0, kappa0, nu0), _lm(1, s[1], Q[1], G, B0, u0, kappa0, nu0)])
    side, lq, new = np.empty(len(rest), np.bool_), 0.0, np.empty(2)
    for r in range(len(rest)):
        y = x[rest[r]]
        yy = np.outer(y, y)
        for c in range(2):
            new[c] = _lm(m[c] + 1, s[c] + y, Q[c] + yy, G, B0, u0, kappa0, nu0)
        dl = new[1] - l[1] - new[0] + l[0]                                      # log odds of joining j
        side[r] = logu[r] < -np.logaddexp(0.0, dl) if draw else to_i[r]
        c = 0 if side[r] else 1
        lq -= np.logaddexp(0.0, dl if side[r] else -dl)
        m[c], l[c] = m[c] + 1, new[c]
        s[c] += y
        Q[c] += yy
    return side, lq, l[0] + l[1]


def mfm(rng, x, a=-np.log(0.9), sweeps=3000, burn=1000, sm=3, starts=(1, 10), gamma=1.0, kappa0=0.01, K_max=30):
    """Bayesian Gaussian mixture with a prior on the number of components (mixture of finite mixtures, Miller and Harrison
    2018): Pi(K) ∝ exp(-a K) on K = 1..K_max (the default a is the Geometric(0.1) prior of their package BayesianMixtures.jl,
    a = n lambda_0 c is the count prior of CoRP), weights Dir(gamma, ..., gamma) given K, and on each component the
    normal-inverse-Wishart prior Sigma_k ~ IW(nu0 = d + 2, Psi0 = Cov(x) / 4), mu_k | Sigma_k ~ N(x_bar, Sigma_k / kappa0).
    One chain per entry of starts, started at k-means with that many clusters.  Each sweep is one telescoping sweep
    (Fruehwirth-Schnatter, Malsiner-Walli and Gruen 2021: parameters, K, weights, allocations) followed by sm
    sequentially-allocated split-merge proposals on the partition with the parameters integrated out (Dahl 2003), which
    change the number of filled components also when the prior leaves no empty ones.  Returns the posterior of the number
    of filled components K+ from the pooled chains (pK) and from each chain (pK_chain), its mode K and, given K+ = K, the
    draws of the component means and of the weights of the filled components (renormalised), relabelled to a common
    reference."""
    n, d = x.shape
    b0, Psi0, nu0 = x.mean(0), np.cov(x.T) / 4, d + 2
    ks = np.arange(1, K_max + 1)
    lV = np.array([logsumexp((-a * ks + gammaln(ks + 1) - gammaln(ks - t + 1) + gammaln(gamma * ks) - gammaln(gamma * ks + n))[t - 1:])
                   for t in ks] + [-np.inf])                                    # log V_n(t), t = 1..K_max, up to a constant

    def niw(nk, xb, S):                                                         # posterior draw of (mu_k, Sigma_k); prior if nk = 0
        kn, dev = kappa0 + nk, xb - b0
        Sig = invwishart.rvs(nu0 + nk, Psi0 + S + kappa0 * nk / kn * np.outer(dev, dev), random_state=rng)
        return rng.multivariate_normal((kappa0 * b0 + nk * xb) / kn, Sig / kn), Sig

    B0, u0 = Psi0 + kappa0 * np.outer(b0, b0), kappa0 * b0
    G = np.arange(n + 1)
    G = multigammaln((nu0 + G) / 2, d) + gammaln(gamma + G) - d / 2 * np.log(kappa0 + G)

    lm0 = d / 2 * np.log(kappa0) + nu0 / 2 * np.linalg.slogdet(Psi0)[1] - multigammaln(nu0 / 2, d) - gammaln(gamma)   # constant of _lm per cluster
    H = (G, B0, u0, kappa0, float(nu0))

    kept = []
    for chain, K0 in enumerate(starts):
        lab = kmeans2(x, K0, minit='++', seed=rng)[1]
        for sweep in range(sweeps + 1):
            cl, cnt = np.unique(lab, return_counts=True)                       # filled components
            Kp = len(cl)
            comp = [niw(nk, xk.mean(0), (xk - xk.mean(0)).T @ (xk - xk.mean(0))) for k, nk in zip(cl, cnt) for xk in [x[lab == k]]]
            Ks = np.arange(Kp, K_max + 1)
            K = rng.choice(Ks, p=softmax(-a * Ks + gammaln(Ks + 1) - gammaln(Ks - Kp + 1) + gammaln(gamma * Ks) - gammaln(gamma * Ks + n)))
            eta = rng.dirichlet(np.r_[gamma + cnt, np.full(K - Kp, gamma)])
            if sweep > burn:
                kept.append((chain, Kp, np.array([m for m, _ in comp]), eta[:Kp] / eta[:Kp].sum()))
            comp += [niw(0, b0, 0) for _ in range(K - Kp)]
            L = np.empty((n, K))
            for k, (m, Sig) in enumerate(comp):
                C = np.linalg.cholesky(Sig)
                u = np.linalg.solve(C, (x - m).T)
                L[:, k] = np.log(eta[k]) - 0.5 * (u ** 2).sum(0) - np.log(np.diag(C)).sum()
            P = np.exp(L - L.max(1, keepdims=True))
            lab = np.minimum((P.cumsum(1) < rng.random((n, 1)) * P.sum(1, keepdims=True)).sum(1), K - 1)
            for _ in range(sm):                                                 # split the cluster of i and j, or merge their two clusters
                i, j = rng.choice(n, 2, replace=False)
                t, same = len(np.unique(lab)), lab[i] == lab[j]
                rest = rng.permutation(np.flatnonzero(((lab == lab[i]) | (lab == lab[j])) & (np.arange(n) != i) & (np.arange(n) != j)))
                side, lq, l2 = _seq(x, i, j, rest, np.log(rng.random(len(rest))), lab[rest] == lab[i], same, *H)
                xs = x[np.r_[i, j, rest]]
                t -= not same                                                   # number of clusters in the merged state
                r = lV[t] - lV[t - 1] + l2 + lm0 - _lm(len(xs), xs.sum(0), xs.T @ xs, *H) - lq   # log of (split over merged) / proposal probability of the split
                if same and np.log(rng.random()) < r:
                    lab[np.r_[j, rest[~side]]] = lab.max() + 1
                elif not same and np.log(rng.random()) < -r:
                    lab[lab == lab[j]] = lab[i]
    ch, Kps = np.array([c for c, _, _, _ in kept]), np.array([k for _, k, _, _ in kept])
    post = lambda v: {int(k): float(np.mean(v == k)) for k in np.unique(v)}
    K = int(np.bincount(Kps).argmax())
    sel = [(m, w) for _, k, m, w in kept if k == K]
    ref = kmeans2(np.concatenate([m for m, _ in sel]), K, minit='++', seed=rng)[0]
    perm = [relabel(ref, m) for m, _ in sel]
    post_m, post_p = np.array([m[q] for (m, _), q in zip(sel, perm)]), np.array([w[q] for (_, w), q in zip(sel, perm)])
    return dict(pK=post(Kps), pK_chain=[post(Kps[ch == c]) for c in range(len(starts))], K=K, post_m=post_m, post_p=post_p, m=post_m.mean(0), p=post_p.mean(0))


def mfm_job(a):
    """MFM on one data set, (dgp, n, rep, P): with P = None the Geometric(0.1) count prior, with P = pop[dgp] the count prior of
    CoRP.  The count posterior and its mode; the conditional draws are kept for rep 0."""
    dgp, n, rep, P = a
    rng, x = data(dgp, n, rep)
    r = mfm(rng, x) if P is None else mfm(rng, x, n * LAM0 * P['c'])
    return dict(dgp=dgp, n=n, rep=rep, **(r if rep == 0 else dict(K=r['K'], pK=r['pK'], pK_chain=r['pK_chain'])))


def rep_job(a):
    """Other representative points at K† on one data set, (dgp, n, rep, P, th): support points (best of 5 starts), kernel
    herding and archetypes (best of 5 starts), next to the CoRP posterior mean and the k-means solution in th (the
    post_mean and th_hat of job).  Each K†-point measure is scored against 5000 draws of P_0 by rep_points.discrepancies,
    with the median-heuristic bandwidth of those draws for the kernel; the points are kept for rep 0."""
    dgp, n, rep, P, th = a
    x, K, y = data(dgp, n, rep)[1], P['K'], P['P0'][:5000]
    q, qy, eq = np.full(n, 1 / n), np.full(len(y), 1 / len(y)), np.full(K, 1 / K)
    h = rp.bandwidth(y, qy)
    R = {m: (t[:2 * K].reshape(K, 2), np.r_[t[2 * K:], 1 - t[2 * K:].sum()]) for m, t in th.items()}
    R['support points'] = (min((rp.support_points(x, q, K, seed=s) for s in range(5)), key=lambda z: rp.discrepancies(x, q, z, eq, h)['energy']), eq)
    R['kernel herding'] = (rp.herding(x, q, K, h), eq)
    A, alpha, _ = min((rp.archetypes(x, q, K, seed=s) for s in range(5)), key=lambda r: r[2])
    R['archetypes'] = (A, q @ alpha)
    return dict(dgp=dgp, n=n, rep=rep, disc={m: rp.discrepancies(y, qy, z, w, h) for m, (z, w) in R.items()}, **(dict(points=R) if rep == 0 else {}))
