from typing import Collection
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .utils import calc_pvalue


@dataclass
class ADTSubspace:
    absdists: np.ndarray = None
    stat: float = None
    nulldist: np.ndarray = None
    nq: int = None
    nr: int = None
    nobs: int = None
    chromsize: int = None

@dataclass
class ADTSpace:
    name: str = None
    subspaces: tuple = None
    nq: int = None
    nr: int = None
    stat: float = None
    nulldist: np.ndarray = None
    pval: float = None
    direction: str = None


def randomize_centers(chromsize: int, n: int) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.integers(low=0, high=chromsize, size=n)


def calc_absdists(q: np.ndarray, r: np.ndarray, chromsize: int) -> np.ndarray:
    return np.min(np.abs(np.subtract.outer(q, r)), axis=1) * r.shape[0] / chromsize


def calc_absdist_stat(absdists: np.ndarray) -> float:
    return absdists.mean()


def process_absdist_subspaces(dfq: pd.DataFrame,
                              dfr: pd.DataFrame,
                              chromsizes: dict,
                              subspaces: Collection[str],
                              subspace_col: str = 'chrom',
                              permutations: int = 100) -> dict:
    subspaces_data = {subspace: ADTSubspace() for subspace in subspaces}
    for subspace, subspace_data in subspaces_data.items():
        sub_dfq = dfq.query(f'{subspace_col} == @subspace')
        sub_dfr = dfr.query(f'{subspace_col} == @subspace')
        nq = sub_dfq.shape[0]
        nr = sub_dfr.shape[0]
        chromsize = chromsizes.get(subspace, max(sub_dfq['end'].max(), sub_dfr['end'].max()))
        subspace_data.nq = nq
        subspace_data.nr = nr
        subspace_data.chromsize = chromsize
        if nq == 0 or nr == 0:
            subspace_data.absdists = np.array([])
            subspace_data.nobs = 0
        else:
            subr_centers = ((sub_dfr['start'] + sub_dfr['end']) // 2).to_numpy()
            subq_centers = ((sub_dfq['start'] + sub_dfq['end']) // 2).to_numpy()
            absdists = calc_absdists(subq_centers, subr_centers, chromsize)
            subspace_data.absdists = absdists
            subspace_data.nobs = len(absdists)
            subspace_data.stat = calc_absdist_stat(absdists)
            subspace_data.nulldist = np.array([calc_absdist_stat(calc_absdists(randomize_centers(chromsize, subq_centers.shape[0]), subr_centers, chromsize)) for _ in range(permutations)])
    return subspaces_data


def process_absdist_spaces(subspaces_data: Collection,
                           spaces: Collection,
                           permutations: int) -> Collection:
    spaces_data = [ADTSpace(name=space_name, subspaces=space_subspaces)
                   for space_name, space_subspaces in spaces.items()]
    for space_data in spaces_data:
        space_data.nq = sum(subspaces_data[subspace].nq for subspace in space_data.subspaces)
        space_data.nr = sum(subspaces_data[subspace].nr for subspace in space_data.subspaces)

        n_obs = sum(subspaces_data[subspace].nobs for subspace in space_data.subspaces)
        if n_obs == 0:
            space_data.stat = None
            space_data.nulldist = np.array([])
            space_data.pval = 1
            space_data.direction = 'undefined'
            
        else:
            space_data.stat = sum(subspaces_data[subspace].stat
                                for subspace in space_data.subspaces
                                if subspaces_data[subspace].nobs > 0) / n_obs
            
            space_data.nulldist = np.array([subspaces_data[subspace].nulldist * subspaces_data[subspace].nobs
                                            for subspace in space_data.subspaces
                                            if subspaces_data[subspace].nobs > 0]).sum(axis=0) / n_obs
            absdist_pval = calc_pvalue(space_data.nulldist, space_data.stat, how='left')
            if absdist_pval < 0.5:
                space_data.pval = absdist_pval * 2
                space_data.direction = 'attraction'
            else:
                space_data.pval = (1 - absdist_pval) * 2
                space_data.direction = 'repulsion'
    return spaces_data


absdist_simple_cols = ("name", "subspaces", "nq", "nr", "stat", "pval", "direction")
