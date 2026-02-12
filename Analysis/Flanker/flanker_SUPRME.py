import numpy as np
import pandas as pd
from RunDEMC import Model, Param, dists
from RunDEMC import Hierarchy, HyperPrior
from RunDEMC.io import save_results, load_results
from joblib import Parallel, delayed
import pandas as pd
import flanker_score_and_map as fsm
try:
    import scoop
    from scoop import futures
except ImportError:
    print "Error loading scoop, reverting to joblib."
    scoop = None
from lca_conf import *


# default params
def_params = dict(r=.1,
                  p=1.0,
                  sd0=2.0,
                  bin_lim=.5,
                  in_bin=1.5,
                  out_bin=2.5,
                  sd_min=.01,
                  K=.1,
                  L=.5,
                  U=0.0,
                  eta=1.,
                  t0=.25,
                  thresh=1.,
                  alpha=0.,
                  max_time=5.0,
                  truncate=True,
                  dt=.01, tau=.1)
max_rt = 2
log_shift = .05

# define model evaluation functions for RunDEMC
def eval_mod(params, param_names, bdat=None, verbose=False, log_shift=log_shift):
    # use global dat if none based in
    if bdat is None:
        bdat = ddat

    # set up params
    mod_params = def_params.copy()
    mod_params.update({x: params[n]
                       for n, x in enumerate(param_names)})

    # set up bins to match different conditions
    dbin = {}
    dbin['+'] = {
             'bins': np.array([[-(mod_params['out_bin']), -(mod_params['in_bin'])],
                          [-(mod_params['in_bin']), -(mod_params['bin_lim'])],
                          [-(mod_params['bin_lim']), mod_params['bin_lim']],
                          [mod_params['bin_lim'], mod_params['in_bin']],
                          [mod_params['in_bin'], mod_params['out_bin']]], dtype=np.float32),
             'bin_ind': np.array([0,0,0,0,0], dtype=np.int32),
             'nbins': 5}
    dbin['='] = {
             'bins': np.array([[-(mod_params['out_bin']), -(mod_params['in_bin'])],
                          [-(mod_params['in_bin']), -(mod_params['bin_lim'])],
                          [-(mod_params['bin_lim']), mod_params['bin_lim']],
                          [mod_params['bin_lim'], mod_params['in_bin']],
                          [mod_params['in_bin'], mod_params['out_bin']]], dtype=np.float32),
             'bin_ind': np.array([1,0,0,0,1], dtype=np.int32),
             'nbins': 5}
    dbin['~'] = {
             'bins': np.array([[-(mod_params['out_bin']), -(mod_params['in_bin'])],
                          [-(mod_params['in_bin']), -(mod_params['bin_lim'])],
                          [-(mod_params['bin_lim']), mod_params['bin_lim']],
                          [mod_params['bin_lim'], mod_params['in_bin']],
                          [mod_params['in_bin'], mod_params['out_bin']]], dtype=np.float32),
             'bin_ind': np.array([0,1,0,1,0], dtype=np.int32),
             'nbins': 5}


    mod_params['x_init'] = np.ones(len(dbin), dtype=np.float32)*(mod_params['thresh']*float(1/3.))

    # loop over conditions, modeling appropriately
    ll = 0.0
    for c in bdat.keys():
        lca = SpotLCA(nitems=2, nbins=dbin[c]['nbins'],
                      nsims=20000, log_shift=log_shift, nreps=1)
        mod_params['bins'] = dbin[c]['bins']
        mod_params['bin_ind'] = dbin[c]['bin_ind']
        ll += lca.log_like(mod_params, data=bdat[c])
        if ll == -np.inf:
            return ll
    return ll

# this is the required def for RunDEMC
def eval_fun(pop, *args):
    bdat = args[1]
    pnames = args[2]
    if scoop and scoop.IS_RUNNING:
        likes = list(futures.map(eval_mod, [indiv for indiv in pop],
                                 [pnames]*len(pop), [bdat]*len(pop)))
    else:
        # use joblib
        likes = Parallel(n_jobs=20)(delayed(eval_mod)(indiv, pnames, bdat)
                                  for indiv in pop)
    return np.array(likes)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", type=str,
                        help="session identifier")
    args = parser.parse_args()
    s = args.datafile
    subject = s.replace('flanker_', '').replace('flkr-', '').replace('.csv','')
    dat = pd.read_csv(s)
    print("Fitting model to %s"%s)

    # prepare the data for the model
    log_shift = .05
    ddat = {}
    for c in ['+', '=', '~']:
        ind = (dat.condition == c) & (dat.rt < max_rt)
        d = {'rt':np.log(np.array(dat[ind].rt)+log_shift),
             'resp': np.array(~dat[ind]['correct'], dtype=np.int)}
        ddat[c] = d

    # store minimum RT
    min_rt = dat[(dat['rt']>=0.)]['rt'].min()

    # define priors
    params = [Param(name='r',
                display_name=r'$r$',
                prior=dists.normal(-2.0,1.0),
                transform=lambda x: dists.invlogit(x)*(20.-0.)+(0.)
                ),
          Param(name='p',
                display_name=r'$p$',
                prior=dists.normal(-0.8,1.2),
                transform=lambda x: dists.invlogit(x)*(20.-0.)+(0.)
                ),
          Param(name='sd0',
                display_name=r'$\sigma_0$',
                prior=dists.normal(-1.0,1.2),
                transform=lambda x: dists.invlogit(x)*(30.-0.1)+(0.1)
                ),
          Param(name='K',
                display_name=r'$K$',
                prior=dists.normal(0.0, 1.4),
                transform=dists.invlogit),
          Param(name='L',
                display_name=r'$L$',
                prior=dists.normal(0.0, 1.4),
                transform=dists.invlogit),
          Param(name='thresh',
                display_name=r'$\theta$',
                prior=dists.normal(-1.0,1.2),
                transform=lambda x: dists.invlogit(x)*(30.-0.0)+(0.0)
                ),
          Param(name='alpha',
                display_name=r'$\alpha$',
                prior=dists.normal(-1.0,1.2),
                transform=lambda x: dists.invlogit(x)*(30.-0.0)+(0.0)
                ),
          Param(name='t0',
                display_name=r'$t_0$',
                prior=dists.normal(-.2,1.2),
                transform=lambda x: dists.invlogit(x)*(min_rt-0.0)+(0.0)
                ),
            ]

    # grab the param names
    pnames = [p.name for p in params]
    print pnames
    mod = Model('subjid', params=params,
            like_fun=eval_fun,
            like_args=('subjid', ddat, pnames),
            num_chains=90,init_multiplier=4,
            purify_every=5,
            verbose=True)

    out_file = 'flkr_res_' + subject + '.tgz'
    print "Burn In"
    for mm in range(16):
        mod(25, burnin=True, migration_prob=0.0)
        save_results(out_file, mod)
        print "Burn In: " + str(mm)
    print "Sampling"
    for mm in range(64):
        mod(25, burnin=False, migration_prob=0.0)
        save_results(out_file, mod)
        print "Sampleing: " + str(mm)
        
    # simulate map params and score
    burnin=400
    res = load_results(out_file)
    params = fsm._get_best_params(res, burnin)
    map_res, sim_rts, sim_corrects = fsm._run_map_sims(res, subject, burnin)
    flkr_score = fsm.FlkrScorer(max_rt=1.863647, min_rt=0.528287, include_parts=True)
    scores = fsm._score_wrapper(subject, ddat, conditions + ['total'], sim_rts, sim_corrects, flkr_score=flkr_score)
    scores.update(params)
    scores = pd.DataFrame(scores, index=[0]).set_index('sub_id').reset_index()
    scores.to_csv('flkr_map' + subject + '.csv')
