# CoRP: coarsened representative points

Posterior inference on a small set of weighted representative points $Q_\theta=\sum_{j=1}^Kw_j\delta_{z_j}$, $\theta=(K,z,w)$, under the Wasserstein coarsened posterior

$$\Pi(K,z,w\mid x)\propto\Pi(K)\,\Pi(z\mid K)\,\Pi(w\mid K)\exp[-\Lambda\{W_p^p(\hat P_n,Q_\theta)\}],$$

with exact transport (monotone coupling for $d=1$, network simplex `ot.emd` for $d\ge2$). The main algorithm is the bridge sampler. A trans-dimensional sampler is provided as an alternative.

| file | content |
|---|---|
| `helper_functions.py` | model, tolerance laws (`Exp`, `Lomax`, `Weibull`), priors, transport, Lloyd quantizer, one-rung updates (zplan, wdir, split/merge), non-reversible parallel tempering `run`, `calibrate` |
| `bridge_sampler.py` | main algorithm: $\Pi(K\mid x)$ by bridge sampling from fixed-$K$ chains |
| `trans_sampler.py` | alternative: trans-dimensional sampler with adaptive non-reversible tempering |
| `sim_utils.py`, `run_sims.py`, `simulation.ipynb` | simulation study: `run_sims.py` caches results in `results/`, the notebook reads and plots them |
| `rep_points.py` | other representative points (support points, kernel herding, archetypes) and the criteria that score them, used by the simulation study and the PFAS notebook |
| `PFAS data/` | NHANES PFAS application: `helper_xpt.py` (XPORT reader), `PFAS_analysis.ipynb`, cached results and figures (the data files are downloaded separately, see below) |

Requires `numpy`, `scipy` and `pot`. The simulation study also uses `kneed` (elbow rule) and `numba` (split-merge moves of the MFM sampler).

## Bridge sampler (main algorithm)

`bridge_sampler(M)` computes the count posterior $\Pi(K\mid x)\propto\Pi(K)Z_K$ from chains with $K$ fixed, where

$$Z_K=\int\Pi(z,w\mid K)\exp[-\Lambda\{W_p^p(\hat P_n,Q_\theta)\}]\,dz\,dw$$

is the normalizing constant at count $K$. No move changes the count, so a chain cannot get stuck at one count.

1. **Fixed-count chains.** For $K=1,2,\dots$, two chains start at the best Lloyd quantizer with its Voronoi masses and run $N_K=\max\{2000,100(Kd+K-1)\}$ iterations, the first 30% as burn-in. Each iteration applies two updates, both accepted or rejected by a Metropolis–Hastings step with the exact posterior.
   - **zplan** updates each atom in turn. It proposes from the Gaussian quadratic surrogate with the transport plan held fixed.
   - **wdir** draws $w'\sim\mathrm{Dir}\{\kappa_Kv(z)+1/2\}$, with $v(z)$ the Voronoi masses. Here $\kappa_K$ is moment-matched during burn-in and then frozen.
2. **Evidence.** $Z_K$ is estimated from each chain by bridge sampling (Meng and Wong, 1996). The draws are relabelled to the region $A_K$ in which the identity is the optimal assignment to the Lloyd quantizer, so that $Z_K=K!\,Z_K^{A}$. The first half of the draws fits a Gaussian proposal in $t=(z,\log(w_1/w_K),\dots,\log(w_{K-1}/w_K))$, and the second half and as many proposal draws enter the fixed-point equation. The estimates of $Z_K$ from the two chains are averaged.
3. **Stopping.** Counts are evaluated in increasing order until the mode is at least two counts back and two consecutive counts have $\Pi(K)Z_K$ below $10^{-4}$ times the largest value so far, and at most up to $K_{\max}$.
4. **Checks.** A count with posterior mass above $10^{-3}$ whose two chains differ by more than 0.2 in $\log Z_K$, or whose relative error exceeds 0.1, is rerun with four times as many iterations, up to three times, and after that with tempered fixed-count chains.

The function returns $\Pi(K\mid x)$ and, for each evaluated count, $\log Z_K$, the gap between the two chains, the relative error and the draws of $(z,w)$. The chains of different counts and seeds are independent, so they run in parallel through `map`. $Z_K$ does not involve the count prior, so one set of runs gives the count posterior under every count prior through `posterior_K(ev, prior_K)`. Given a count, `calibrate` returns the sandwich-calibrated draws of the locations and weights.

```python
import numpy as np
from multiprocessing import Pool
from helper_functions import Model, Exp, penalty, calibrate
from bridge_sampler import bridge_sampler, posterior_K

n, lam0 = len(x), 0.5
M = Model(x, p=2, radius=Exp(n * lam0), prior_K=penalty(n * c), K_max=10)

with Pool(8) as pool:                                                          # fixed-K chains in parallel, then bridge sampling
    pK, ev = bridge_sampler(M, map=lambda f, *a: pool.starmap(f, zip(*a)))
K = max(pK, key=pK.get)                                                        # posterior mode of the count
cal = calibrate(M, ev[K]['runs'][0]['draws'] + ev[K]['runs'][1]['draws'], K)   # sandwich-calibrated z, w at K
pK2 = posterior_K(ev, penalty(n * c2))                                          # count posterior under another price, no rerun
```

## Trans-dimensional sampler (alternative)

`trans_sampler(M, n_iter)` runs one chain over $(K,z,w)$. Each iteration applies zplan and wdir as above and then a split/merge move, and the chain is tempered.

- **Split/merge** changes $K$. The split weight fraction is $u\sim\mathrm{Beta}(2,2)$, and the split direction is the principal axis of the parent's transport cell. A merge picks a uniform pair with probability 1/2, and otherwise a uniform atom together with its nearest neighbor.
- **Tempering** is non-reversible parallel tempering. The ladder starts from $\beta=u^5$ with 6 rungs. During burn-in (the first 30%) it is re-placed after rounds of 10, 20, 40, … iterations, with $2\hat B+1$ rungs (at most 48) of equal estimated swap rejection. After burn-in the ladder and $\kappa_K$ are frozen.

```python
from trans_sampler import trans_sampler

r = trans_sampler(M, n_iter=5000, seed=0)
r['pK'], r['draws'], r['betas'], r['swap_acc']
```

The two samplers give the same count posterior when the ladder is adequate. A run of the trans-dimensional sampler should be checked: start several chains from different counts, for example with `init=(x[rows], np.full(K0, 1 / K0))` for $K_0=1,K_{\max},2,5$, and compare their $\Pi(K\mid x)$ with each other and with the bridge sampler. It also has to be rerun for every count prior. We recommend the bridge sampler in practice.

## PFAS data

The NHANES data files are not part of this repository. Before running `PFAS data/PFAS_analysis.ipynb`, download the two files of the August 2021 to August 2023 cycle from the CDC and put them in the folder `PFAS data/`, next to the notebook, under their original names:

- `PFAS_L.xpt` (perfluoroalkyl and polyfluoroalkyl substances): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/PFAS_L.xpt
- `DEMO_L.xpt` (demographics and survey weights): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt

The cached results were computed from the files downloaded on 2026-11-11.

## Reproducibility

The folders `results/`, `PFAS data/results/` and `figures/` are not tracked in this repository. `run_sims.py` (modes `pop`, `main`, `path`, `extra`, `rep`, `mfm`, `mfmc`, in this order) and `PFAS_analysis.ipynb` with `RECOMPUTE = True` recreate the results, and the two notebooks write the figures.

Every random draw comes from a `numpy` generator with a fixed seed. Rerunning `run_sims.py`, or a notebook with `RECOMPUTE = True`, reproduces the cached results bit for bit (the order of the records in `results/*.pkl` depends on the worker pool, their content does not).

- Samplers: `quantizer`, `lloyd_start`, `run`, `trans_sampler` and `evidence` take `seed` (default 0). `bridge_sampler` runs its chains with `seeds=(0, 1)`, and `evidence` draws the bridge proposals from the separate stream `[seed, 1]`.
- Simulation: `sim_utils.data(dgp, n, rep)` gives each data set its own generator, seeded by the triple `[SEED[dgp], n, rep]` so that no two data sets share a stream, which also drives the bootstrap, the GMM starts and the MFM sampler of that data set. The population quantities use `population(dgp, seed=1)`.
- PFAS: the chains and the evidence use seeds 0 and 1, bootstrap replicate `b` uses seed `b`, the censored imputations use seed 0, and the starts of the comparison methods use seeds 0, 1, ….

The cached results were produced with Python 3.13.14, `numpy` 2.4.6, `scipy` 1.18.0, `pot` 0.9.6.post1, `kneed` 0.9.0 and `numba` 0.65.1.

## Validation

All runs use the defaults above, with no tuning per setting. The validation scripts named below are not part of this repository.

Exact $\Pi(K\mid x)$ (`check_exact.py`): V1 ($d=1$) and B2 ($d=2$), four chains of the trans-dimensional sampler from $K_0=1,K_{\max},2,5$, and the bridge sampler.

| setting | trans-dimensional TV pooled (worst chain) / CPU s per chain, 2500 (V1) or 5000 (B2) iterations | rungs, min swap acc | bridge TV / CPU s |
|---|---|---|---|
| V1 | 0.028 (0.054) / 22 | 9, 0.38 | 0.019 / 1118 |
| B2 | 0.015 (0.027) / 145 | 15, 0.45 | 0.027 / 850 |

Ten design settings (`check_bench.py`): Gaussian ($d=2,10$), heteroscedastic, skewed, ring, helix ($d=8$), general $F$ and $p=1,3$. On all ten, the bridge sampler agrees with the old bridge sampler (TV 0.000) and with long tempered runs of all moves (TV $\le0.018$, where the largest value is on skew2_gen, whose tempered reference has not converged).

Four settings never used in design (`check_transfer.py`), against the old bridge sampler. The trans-dimensional sampler uses four chains of 3000 iterations.

| setting | reference mode (mass) | bridge TV / CPU s | trans-dimensional TV pooled (worst chain) / CPU s per chain |
|---|---|---|---|
| g20het: $d=20$, heterogeneous Gaussian | $K=3$ (1.000) | 0.000 / 785 | 0.000 (0.000) / 702 |
| unif2: uniform, Weibull radius | $K=4$ (0.773) | 0.003 / 408 | 0.014 (0.026) / 421 |
| banana3: Lomax radius, $t_3$ atom prior, Dir(0.5) | $K=4$ (0.999) | 0.000 / 107 | 0.001 (0.001) / 332 |
| lap6: Laplace components, Poisson × penalty $K$ prior | $K=4$ (1.000) | 0.000 / 174 | 0.000 (0.000) / 636 |

The ladder chose 12–17 rungs on unif2 and banana3, with minimum swap acceptance 0.27–0.50.

On g20het (`g20_ladder.py`), the estimated barrier is $\hat B\approx20$, so the ladder needs about 40–50 rungs.

- With the old cap of 24, a few rung pairs have swap acceptance 0.01–0.04.
- With the cap of 48, the ladder has 43–48 rungs and swap acceptance about 0.5, at about twice the cost.
- All three variants (cap 24, cap 48, untempered) give $P(K=3)=1$, and the untempered chain does so in 25 s.

For a large $d$ with a unimodal count posterior, the bridge sampler, or the trans-dimensional sampler without tempering, is therefore much cheaper.

PFAS was used as the test set and was not seen during design (`pfas_trans.py`, `pfas_evidence.py`). The model is that of `PFAS data/PFAS_analysis.ipynb`, with $n=2551$, $d=5$ and survey weights.

- The bridge sampler is reproducible. With 12000-iteration chains and four seeds, $\log Z_4$ and $\log Z_5$ agree to 0.03 (relative errors 0.007 and 0.013), and they move the 3000-iteration $P(K=5\mid K\in\{4,5\})$ only from 0.335 to 0.316.
- At $c=0.174$ (count posterior $K=4$ with mass 1), the trans-dimensional sampler is exact with 25–27 rungs (TV 0.000 for every chain).
- At $c=0.106$ (bridge: $K=4$ 0.68, $K=5$ 0.32), four chains of 3000 iterations give $K=5$ 0.18 (TV 0.159, worst chain 0.370), although swap acceptance is 0.40–0.44. The split/merge moves between $K=4$ and $5$ mix too slowly at this $n$.
- The bridge sampler is therefore the reliable route when the count posterior is spread over neighbouring $K$ at a large $n$.
