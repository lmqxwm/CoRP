"""Bridge sampler (the main algorithm): Pi(K | x) by bridge sampling from fixed-K chains.

Pi(K | x) ∝ Pi(K) Z_K,  Z_K = ∫ Pi(z, w | K) exp[-Lam{W(z, w)}] dz dw_1..dw_{K-1} = K! Z_K^A,
where A_K = {the identity is an optimal assignment of z to the best Lloyd quantizer zhat_K} (Pi(z, w | K) must be
exchangeable in the labels).  In t = (z, log(w_1/w_K), ..., log(w_{K-1}/w_K)) the integrand of Z_K^A is
f_K(t) = Pi(z, w | K) exp[-Lam{W(z, w)}] prod_j w_j 1{t in A_K}."""
from functools import partial
import numpy as np
from scipy.special import gammaln, logsumexp
from helper_functions import lloyd_start, relabel, run


def evidence(M, K, n_iter, seed=0, temper=False):
    """log Z_K and its relative standard error from one fixed-K chain (atom and weight updates, tempered if temper)
    started at zhat_K.  The first half of the post-burn-in draws fits the Gaussian proposal q_K; the second half and as
    many draws from q_K enter the Meng-Wong fixed point."""
    rng = np.random.default_rng([seed, 1])                             # proposal draws: a stream separate from the chain's
    z_ref, w0 = lloyd_start(M, K, seed)
    chain = run(M, n_iter, (z_ref, w0), split_merge=False, temper=temper, seed=seed)
    kd = K * M.d

    def to_t(z, w):
        return np.r_[z.ravel(), np.log(w[:-1] / w[-1])]

    def log_f(z, w, W):
        return M.prior_z(z) + M.prior_w(w) - M.radius.Lam(W) + np.log(w).sum()

    # chain draws, relabelled into A_K; their cost W is known from the chain
    perms = [relabel(z_ref, z) for z, w in chain['draws']]
    T = np.array([to_t(z[o], w[o]) for (z, w), o in zip(chain['draws'], perms)])
    lf_draws = np.array([log_f(z, w, W) for (z, w), W in zip(chain['draws'], chain['W'])])

    half = len(T) // 2
    mean, chol = T[:half].mean(0), np.linalg.cholesky(np.atleast_2d(np.cov(T[:half].T)))

    def log_q(X):
        return (-0.5 * (np.linalg.solve(chol, (X - mean).T) ** 2).sum(0) - np.log(np.diag(chol)).sum()
                - len(mean) / 2 * np.log(2 * np.pi))

    def log_f_t(t):
        z, u = t[:kd].reshape(K, M.d), np.r_[t[kd:], 0.0]
        if np.any(relabel(z_ref, z) != np.arange(K)):
            return -np.inf
        w = np.exp(u - logsumexp(u))
        return log_f(z, w, M.transport(z, w)[0])

    T_prop = mean + rng.standard_normal((len(T) - half, len(mean))) @ chol.T
    l_draw = lf_draws[half:] - log_q(T[half:])                         # log f / q at the draws
    l_prop = np.array([log_f_t(t) for t in T_prop]) - log_q(T_prop)    # log f / q at the proposals
    logZ = np.median(l_draw)
    for _ in range(1000):                                              # Meng-Wong fixed point for log Z_K^A
        new = logsumexp(l_prop - np.logaddexp(l_prop, logZ)) - logsumexp(-np.logaddexp(l_draw, logZ))
        if abs(new - logZ) < 1e-10:
            break
        logZ = new
    # relative standard error (Fruhwirth-Schnatter 2004), the draw term inflated by its batch-means autocorrelation time
    f_prop, f_draw = np.exp(l_prop - np.logaddexp(l_prop, logZ)), np.exp(logZ - np.logaddexp(l_draw, logZ))
    b = int(np.sqrt(len(f_draw)))
    tau = b * f_draw[:len(f_draw) // b * b].reshape(-1, b).mean(1).var() / f_draw.var()
    re = np.sqrt(f_prop.var() / f_prop.mean() ** 2 / len(f_prop)
                 + max(tau, 1.0) * f_draw.var() / f_draw.mean() ** 2 / len(f_draw))
    return dict(logZ=gammaln(K + 1) + logZ, re=re, draws=chain['draws'], n_iter=n_iter)


def posterior_K(ev, prior_K):
    Ks = sorted(ev)
    lp = np.array([ev[k]['logZ'] + prior_K(k) for k in Ks])
    return dict(zip(Ks, np.exp(lp - logsumexp(lp))))


def bridge_sampler(M, tau=0.1, eps=1e-3, eta=1e-4, rounds=3, seeds=(0, 1), map=map):
    """Pi(K | x) and, per evaluated K, {logZ, delta = |log Z^(1) - log Z^(2)|, e = max relative error, temper, runs}.
    K = 1, 2, ... with N_K = max(2000, 100 (Kd + K - 1)) iterations per chain, until the mode is at least two counts back
    and two consecutive counts have Pi(K) Z_K < eta x the largest so far (at most K_max).  Counts with mass > eps and
    delta > 2 tau or e > tau are rerun 4x longer, up to rounds times, and then with tempered fixed-K chains.
    map(f, Ks, n_iters, seeds) runs the chains, e.g. lambda f, *a: pool.starmap(f, zip(*a))."""
    ev, S = {}, len(seeds)

    def fit(Ks, n_iters, temper):
        out = list(map(partial(evidence, M, temper=temper), [k for k in Ks for _ in seeds],
                       [n for n in n_iters for _ in seeds], list(seeds) * len(Ks)))
        for i, k in enumerate(Ks):
            runs = out[i * S:(i + 1) * S]
            logZ = np.array([r['logZ'] for r in runs])
            ev[k] = dict(logZ=logsumexp(logZ) - np.log(S), delta=np.ptp(logZ), e=max(r['re'] for r in runs),
                         temper=temper, runs=runs)

    for K in range(1, M.K_max + 1):
        fit([K], [max(2000, 100 * (K * M.d + K - 1))], False)
        lp = {k: e['logZ'] + M.prior_K(k) for k, e in ev.items()}
        top = max(lp.values())
        if max(lp, key=lp.get) <= K - 2 and max(lp[K - 1], lp[K]) < top + np.log(eta):
            break

    def failing():
        p = posterior_K(ev, M.prior_K)
        return [k for k in ev if p[k] > eps and (ev[k]['delta'] > 2 * tau or ev[k]['e'] > tau)]

    for _ in range(rounds):
        Ks = failing()
        if not Ks:
            break
        fit(Ks, [4 * ev[k]['runs'][0]['n_iter'] for k in Ks], False)
    Ks = failing()
    if Ks:
        fit(Ks, [ev[k]['runs'][0]['n_iter'] for k in Ks], True)
    return posterior_K(ev, M.prior_K), ev
