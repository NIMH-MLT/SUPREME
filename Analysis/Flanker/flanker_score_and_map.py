from lca_conf import SpotLCA
from RunDEMC.io import load_results
from RunDEMC import Model, Param, dists
from pathlib2 import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from joblib import Parallel, delayed

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 2000)


class FlkrScorer():
    """Class to instatiate a flanker scorer.
    Done as a class to allow precalculation of some parts of the score equation.
    """
    def __init__(self, max_rt=3, min_rt=0.15, include_parts=False):
        """Init the flkr_score object and precalculate score components
        """
        self.max_rt = max_rt
        self.min_rt = min_rt
        self.max_rt_numerator_term = np.log(max_rt + 1.)
        self.denom_term = (np.log(max_rt + 1.) - np.log(min_rt + 1.))
        self.include_parts = include_parts
    
    def __call__(self, correct, rts, n_trials=None):
        """Calculate flanker score
        """
        if n_trials is None:
            n_trials = len(rts)
        n_trials = np.float(n_trials)
        correct = correct[pd.notnull(rts)]
        accuracy = ((correct.astype(bool).sum()/n_trials - 0.5)) / 0.5
        rts = rts[pd.notnull(rts)]
        speed = ((self.max_rt_numerator_term - np.log(rts + 1.)) / self.denom_term).sum()/n_trials
        score = speed * accuracy * 100
        if self.include_parts:
            return score, accuracy, speed
        else:
            return score
    
def test_FlkrScorer():
    rt = np.arange(0,3, 0.1)
    corr=np.ones_like(rt)
    n_trials = len(rt)
    ex_score = 45.02209646984963
    ex_accuracy = 1.
    ex_speed = ex_score / 100
    flkr_score = FlkrScorer(max_rt=3, min_rt=0.15, include_parts=True)
    score, accuracy, speed = flkr_score(corr, rt, n_trials)
    assert np.isclose(ex_score, score)
    assert np.isclose(ex_accuracy, accuracy)
    assert np.isclose(ex_speed, speed)
    
    rt = np.arange(0,3, 0.1)
    corr=np.hstack([np.ones_like(rt), np.ones_like(rt)])
    corr[np.array([23,5,22, 0,7])] = 0
    rt = np.hstack([rt, rt])
    rt[2] = np.nan
    rt[5] = np.nan
    n_trials = len(rt)
    ex_score = 33.68073882916825
    ex_accuracy = 0.8
    ex_speed = 0.4210092353646031
    flkr_score = FlkrScorer(max_rt=3, min_rt=0.15, include_parts=True)
    score, accuracy, speed = flkr_score(corr, rt)
    assert np.isclose(ex_score, score)
    assert np.isclose(ex_accuracy, accuracy)
    assert np.isclose(ex_speed, speed)

    score, accuracy, speed = flkr_score(corr, rt, n_trials)
    assert np.isclose(ex_score, score)
    assert np.isclose(ex_accuracy, accuracy)
    assert np.isclose(ex_speed, speed)

    flkr_score = FlkrScorer(max_rt=3, min_rt=0.15)
    score = flkr_score(corr, rt, n_trials)
    assert np.isclose(ex_score, score)

    
def _load_flkr_data(subject, csv_dir, max_rt=3, log_shift=0.05):
    s = csv_dir / ('flkr-' + subject + '.csv')
    
    # load participant data
    dat = pd.read_csv(s)
    ddat = {}
    all_rts = []
    all_trts = []
    all_resps = []
    all_nresps = 0
    for c in ['+', '=', '~']:
        ind = (dat.condition == c) & (dat.rt < max_rt) & dat.keep
        crt = dat[ind].rt.values
        ctrt = np.log(np.array(dat[ind].rt)+log_shift)
        cresp = np.array(~dat[ind]['correct'], dtype=np.int)
        nresp = ((dat.condition == c) & dat.keep).astype(int).sum()
        d = {
            'ort': crt,
            'rt': ctrt,
            'resp': cresp,
            'nresp': nresp
        }
        all_rts.append(crt)
        all_trts.append(ctrt)
        all_resps.append(cresp)
        all_nresps+=nresp
        
        ddat[c] = d
    # add total condition
    d = {
        'ort': np.hstack(all_rts),
        'rt': np.hstack(all_trts),
        'resp': np.hstack(all_resps),
        'nresp': all_nresps,
    }
    ddat['total'] = d
    return ddat


def _get_best_params(res, burnin):
    # simulate outcomes for map parameters
    best_ind = res['weights'][burnin:].argmax()
    indiv = [res['particles'][burnin:, :, i].ravel()[best_ind]
            for i in range(res['particles'].shape[-1])]
    params = {p:v for p, v in zip(res['param_names'], indiv)}
    return params


def _run_map_sims(res, sub_id, burnin, nsims=10000, log_shift=0.05, default_params=None):
    conditions = ['+', '=', '~']
    params = _get_best_params(res, burnin)

    if default_params is None:
        from flanker_SUPRME import def_params as default_params
    mod_params = default_params
    mod_params.update(params)


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

    out_times = {'total':[]}
    corrects = {'total': []}
    map_res = []
    for c in conditions:
        lca = SpotLCA(nitems=2, nbins=dbin[c]['nbins'],
                      nsims=nsims, log_shift=log_shift, nreps=1)
        mod_params['bins'] = dbin[c]['bins']
        mod_params['bin_ind'] = dbin[c]['bin_ind']
        out_time, x_ind, x_out, conf = lca.simulate(**mod_params)
        out_time = out_time[x_ind != -1]
        x_correct = (x_ind == 0).astype(bool)[x_ind != -1]
        out_times[c] = out_time
        out_times['total'].extend(list(out_time))
        corrects[c] = x_correct
        corrects['total'].extend(list(x_correct))
        for ot, corr in zip(out_time, x_correct):
            map_res.append(dict(sub_id=sub_id, condition=c, rt=ot, correct=corr,))
    out_times['total'] = np.array(out_times['total'])
    corrects['total'] = np.array(corrects['total'])
    return map_res, out_times, corrects


def _score_wrapper(sub_id, ddat, conditions, sim_rts, sim_corrects, max_rt=3, flkr_score=None):
    nsims = len(sim_rts[conditions[0]])
    # calculate scores on real data
    dat_scores = {
        'sub_id': sub_id,
    }
    for c in conditions:
        correct = (ddat[c]['resp'] == 0).astype(bool)
        rts = ddat[c]['rt']
        nresp = ddat[c]['nresp']
        dat_scores[c], dat_scores[c + '_accscore'], dat_scores[c + '_spdscore'] = flkr_score(correct, rts, nresp)
        dat_scores[c + '_nonresp'] = nresp - len(rts)
        dat_scores[c + '_wrong'] = nresp - correct.sum()
        dat_scores[c + '_acc'] = correct.mean()
        dat_scores[c + '_rt'] = rts.mean()
        dat_scores[c + '_rt_min'] = rts.min()
        dat_scores[c + '_rt_max'] = rts.max()

        
    # calculate scores for sims
    for c in conditions:
        rts = sim_rts[c]
        if c == 'total':
            nresp = nsims * 3
        else:
            nresp = nsims
        ind = rts < max_rt
        rts = rts[ind]
        correct = sim_corrects[c][ind]
        dat_scores[c + '_map'], dat_scores[c + '_accscore_map'], dat_scores[c + '_spdscore_map']  = flkr_score(correct, rts, nresp)
        dat_scores[c + '_nonresp_map'] = nresp - len(rts)
        dat_scores[c + '_wrong_map'] = nresp - correct.sum()
        dat_scores[c + '_acc_map'] = correct.mean()
        dat_scores[c + '_rt_map'] = rts.mean()
        dat_scores[c + '_rt_min_map'] = rts.min()
        dat_scores[c + '_rt_max_map'] = rts.max()
    return dat_scores
