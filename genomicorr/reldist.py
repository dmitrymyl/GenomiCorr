from dataclasses import dataclass
from typing import Callable, Collection

import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.distributions.empirical_distribution import ECDF
from .utils import calc_pvalue, parse_spaces, process_result


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


def integrate(func: Callable, low: float = 0, high: float = 0.5, steps: int = 50) -> float:
    step_size = (high - low) / steps
    x = np.arange(low, high, step_size)
    return np.sum(func(x) * step_size)


def calc_reldist_stat(ecdf: Callable, steps: int = 50) -> float:
    func = lambda x: np.abs(ecdf(x) - uniDist.cdf(x))
    return integrate(func, steps=steps)


def null_reldist_stats(size: int, steps: int = 50) -> float:
    return calc_reldist_stat(ECDF(uniDist.rvs(size)), steps=steps)


def process_reldist_subspaces(dfq: pd.DataFrame,
                              dfr: pd.DataFrame,
                              subspaces: Collection[str],
                              subspace_col: str = 'chrom') -> dict:

    subspaces_data = {subspace: RDTSubspace() for subspace in subspaces}
    for subspace, subspace_data in subspaces_data.items():
        sub_dfq = dfq.query(f'{subspace_col} == @subspace')
        sub_dfr = dfr.query(f'{subspace_col} == @subspace')
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
    return subspaces_data


def process_reldist_spaces(subspaces_data: Collection,
                           spaces: Collection,
                           permutations: int) -> Collection:
    subspaces = tuple(subspaces_data.keys())
    spaces_data = [RDTSpace(name=space) for space in spaces]
    

    # TODO: spaces will be refactored
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
    return spaces_data


reldist_simple_cols = ("name", "subspaces", "nq", "nr", "ks_pval", "stat", "reldist_pval", "ecdf_corr", "direction")


def reldist_test(dfq: pd.DataFrame,
                 dfr: pd.DataFrame,
                 spaces: tuple = None,
                 subspaces: tuple = None,
                 permutations: int = 100,
                 output: str = 'full') -> pd.DataFrame:

    total_chroms = set(dfq['chrom']) | set(dfr['chrom'])
    # TODO: refactor spaces and subspaces
    spaces, subspaces = parse_spaces(total_chroms, spaces, subspaces)

    subspaces_data = process_reldist_subspaces(dfq, dfr, subspaces)
    spaces_data = process_reldist_spaces(subspaces_data, spaces, permutations)
    
    result_df = pd.DataFrame(spaces_data)
    return process_result(result_df, output, reldist_simple_cols)
