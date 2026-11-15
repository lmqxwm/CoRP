"""Runs the simulation study and caches results in results/ (simulation.ipynb reads them).
  python run_sims.py pop          -> results/pop.pkl
  python run_sims.py main [w]     -> results/main.pkl: n in NS; reps 0..R_FULL-1 full (bridge sampling), R_FULL..R-1 K†-only
  python run_sims.py path [w]     -> results/path.pkl: log Z_K, K = 1..16, rep 0 at n = 1000 (one job per count)
  python run_sims.py mfm [w]      -> results/mfm.pkl: MFM count posterior on the 20 full data sets per (DGP, n), draws kept for rep 0
  python run_sims.py mfmc [w]     -> results/mfmc.pkl: the same with the count prior of CoRP in place of the package default
  python run_sims.py rep [w]      -> results/rep.pkl: support points, kernel herding and archetypes at K† on the full data sets of main.pkl
  python run_sims.py extra [w]    -> results/extra.pkl: 120 more K†-only data sets (reps R..R+119) for the truncated GMM at n = 4000"""
import os, sys, pickle
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
from multiprocessing import Pool
import sim_utils as su

NS, R, R_FULL = (250, 500, 1000, 2000, 4000), 60, 20
HERE = os.path.dirname(os.path.abspath(__file__))
f = lambda name: os.path.join(HERE, 'results', name)
os.makedirs(f(''), exist_ok=True)

if __name__ == '__main__':
    mode, w = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 7
    if mode == 'pop':
        with Pool(3) as pool:
            pickle.dump(dict(zip(su.KDAG, pool.map(su.population, su.KDAG))), open(f('pop.pkl'), 'wb'))
        sys.exit()
    pop = pickle.load(open(f('pop.pkl'), 'rb'))
    fn, jobs = {'main': (su.job, [(d, n, r, pop[d], r < R_FULL) for n in NS for r in range(R) for d in su.KDAG]),
                'path': (su.path_job, [(d, 1000, 0, pop[d], K) for K in range(su.KMAX, 0, -1) for d in su.KDAG]),
                'mfm': (su.mfm_job, [(d, n, r, None) for n in NS[::-1] for r in range(R_FULL) for d in su.KDAG]),
                'mfmc': (su.mfm_job, [(d, n, r, pop[d]) for n in NS[::-1] for r in range(R_FULL) for d in su.KDAG]),
                'extra': (su.job, [('gmm', 4000, r, pop['gmm'], False) for r in range(R, R + 120)]),
                'rep': (su.rep_job, [(o['dgp'], o['n'], o['rep'], pop[o['dgp']], {'CoRP': o['post_mean'], 'k-means': o['th_hat']})
                                     for o in (pickle.load(open(f('main.pkl'), 'rb')) if mode == 'rep' else []) if 'pK' in o])}[mode]
    out = []
    with Pool(w) as pool:
        for o in pool.imap_unordered(fn, jobs):
            out.append(o)
            pickle.dump(out, open(f(f'{mode}.pkl'), 'wb'))
            print(o['dgp'], o['n'], o.get('rep'), o.get('pK', o.get('K', '')), flush=True)
    if mode == 'path':                                  # one record per data set: {K: log Z_K} and the two-chain gaps
        pickle.dump([dict(dgp=d, n=1000, rep=0, **{k: {o['K']: o[k] for o in out if o['dgp'] == d} for k in ('logZ', 'delta')})
                     for d in su.KDAG], open(f('path.pkl'), 'wb'))
