from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.linalg import LinAlgError

uniDist = ss.uniform(scale=0.5)


@dataclass
class RDTSubspace:
    reldists: np.ndarray = None
    nq: int = None
    nr: int = None


@dataclass
class RDTSpace:
    name: str = None
    subspaces: tuple = None
    reldists: np.ndarray = None
    nq: int = None
    nr: int = None
    ks_pval: float = None
    stat: float = None
    nulldist: np.ndarray = None
    reldist_pval: float = None
    ecdf_corr: float = None
    direction: str = None


def calc_pvalue(nulldist: np.ndarray, stat: float, how: str ='right') -> float:
    if len(nulldist) < 2:
        raise ValueError('`nulldist` must have multiple elements.')
    if how not in ('right', 'left'):
        raise ValueError("how is not one of 'left', 'right'")
    try:
        kde = ss.gaussian_kde(nulldist)
        if how == 'right':
            pvalue = kde.integrate_box_1d(stat, np.inf)
        else:
            pvalue = kde.integrate_box_1d(np.NINF, stat)
            
    except LinAlgError:
        mean_value = nulldist.mean()
        if stat < mean_value:
            if how == 'right':
                pvalue = 1
            else:
                pvalue = 0
        else:
            if how == 'right':
                pvalue = 0
            else:
                pvalue = 1
    return pvalue


def calc_reldists(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    q_mod = q
    if np.min(q) < np.min(r):
        q_mod = q_mod[1:]
    if np.max(q) > np.max(r):
        q_mod = q_mod[:-1]
    indices = np.searchsorted(r, q_mod) - 1
    rel_dists = np.min(np.stack((q_mod - r[indices], r[indices + 1] - q_mod), axis=1), axis=1) / np.abs(r[indices + 1] - r[indices])
    return rel_dists


def reldist_ks_test(rel_dists: np.ndarray) -> float:
    return ss.ks_1samp(rel_dists, uniDist.cdf).pvalue


def integrate(func: Callable, low: float =0, high: float =0.5, steps: int=50) -> float:
    step_size = (high - low) / steps
    x = np.arange(low, high, step_size)
    return np.sum(func(x) * step_size)


def calc_reldist_stat(ecdf: Callable, steps: int =50) -> float:
    func = lambda x: np.abs(ecdf(x) - uniDist.cdf(x))
    return integrate(func, steps=steps)


def null_reldist_stats(size: int, steps: int =50) -> float:
    return calc_reldist_stat(ECDF(uniDist.rvs(size)), steps=steps)


def reldist_test(dfq: pd.DataFrame, dfr: pd.DataFrame, spaces: tuple =None, subspaces: tuple =None, permutations: int =100, output: str ='full') -> pd.DataFrame:
    subspaces = set(dfq['chrom']) | set(dfr['chrom'])
    spaces = list(subspaces) + ['whole']
    
    subspaces_data = {subspace: RDTSubspace() for subspace in subspaces}
    for subspace, subspace_data in subspaces_data.items():
        sub_dfq = dfq.query('chrom == @subspace')
        sub_dfr = dfr.query('chrom == @subspace')
        nq = sub_dfq.shape[0]
        nr = sub_dfr.shape[0]
        subspace_data.nq = nq
        subspace_data.nr = nr
        if nq == 0 or nr == 0:
            subspace_data.reldists = np.array([])
        else:
            subr_centers = ((sub_dfr['start'] + sub_dfr['end']) // 2).to_numpy()
            subq_centers = ((sub_dfq['start'] + sub_dfq['end']) // 2).to_numpy()
            subspace_data.reldists = calc_reldists(subq_centers, subr_centers)

    spaces_data = [RDTSpace(name=space) for space in spaces]
    
    for space_data in spaces_data:
        space = space_data.name
        if space == 'whole':
            space_data.subspaces = subspaces
        elif space in subspaces:
            space_data.subspaces = (space,)
        else:
            pass
    
        space_reldists = np.concatenate(tuple(subspaces_data[subspace].reldists for subspace in space_data.subspaces))
        space_data.reldists = space_reldists
        space_data.nq = sum(subspaces_data[subspace].nq for subspace in space_data.subspaces)
        space_data.nr = sum(subspaces_data[subspace].nr for subspace in space_data.subspaces)
    
        space_reldists = space_data.reldists
        if len(space_reldists) == 0:
            space_data.ks_pval = 1
            space_data.stat = None
            space_data.nulldist = np.array(())
            space_data.reldist_pval = 1
            space_data.ecdf_corr = None
            space_data.direction = 'undefined'
        else:
            space_data.ks_pval = reldist_ks_test(space_reldists)
            space_data.stat = calc_reldist_stat(ECDF(space_reldists), space_reldists.shape[0])
            space_data.nulldist = np.array([null_reldist_stats(space_reldists.shape[0]) for _ in range(permutations)])
            
            permut_pval = calc_pvalue(space_data.nulldist, space_data.stat)
            if permut_pval < 0.5:
                space_data.reldist_pval = permut_pval * 2
            else:
                space_data.reldist_pval = (1 - permut_pval) * 2
            
            ecdf_corr = integrate(lambda x: ECDF(space_reldists)(x) - uniDist.cdf(x)) / integrate(lambda x: uniDist.cdf(x))
            space_data.ecdf_corr = ecdf_corr
            
            if ecdf_corr > 0:
                space_data.direction = 'attraction'
            elif ecdf_corr < 0:
                space_data.direction = 'repulsion'
            else:
                space_data.direction = 'indifferent'
    
    result_df = pd.DataFrame(spaces_data)
    if output == 'full':
        return result_df
    elif output == 'simple':
        return result_df.loc[:, ["name", "subspaces", "nq", "nr", "ks_pval", "stat", "reldist_pval", "ecdf_corr", "direction"]]
    else:
        raise ValueError('`output` must be one of "full", "simple".')
