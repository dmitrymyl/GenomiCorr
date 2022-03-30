from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import scipy.stats as ss
import bioframe as bf


__all__ = ['process_proj_subspaces', 'process_proj_spaces']


@dataclass
class ProjSubspace:
    nq: int = None
    nr: int = None
    hits: int = None
    p0: float = None
    chromsize: int = None


@dataclass
class ProjSpace:
    name: str = None
    subspaces: Tuple[str] = None
    nq: int = None
    nr: int = None
    hits: int = None
    p0: float = None
    pval: float = None
    ratio: float = None
    direction: str = "undefined"


def process_proj_subspaces(dfq: pd.DataFrame,
                           dfr: pd.DataFrame,
                           chromsizes: Dict[str, int],
                           subspaces: Tuple[str],
                           subspace_col: str = 'chrom') -> Dict[str, ProjSubspace]:
    subspaces_data = {subspace: ProjSubspace() for subspace in subspaces}
    for subspace, subspace_data in subspaces_data.items():
        sub_dfq = dfq.query(f'{subspace_col} == @subspace')
        sub_dfr = dfr.query(f'{subspace_col} == @subspace')
        nq = sub_dfq.shape[0]
        nr = sub_dfr.shape[0]
        chromsize = chromsizes.get(subspace, max(sub_dfq['end'].max(), sub_dfr['end'].max()))
        subspace_data.nq = nq
        subspace_data.nr = nr
        subspace_data.chromsize = chromsize
        subq_centers = ((sub_dfq['start'] + sub_dfq['end']) // 2).to_numpy()
        subr_starts = sub_dfr['start'].to_numpy()
        subr_ends = sub_dfr['end'].to_numpy()

        if nr == 0:
            projection_stat = 0
            p0 = 0
        else:
            p0 = (bf.merge(sub_dfr)['end'] - bf.merge(sub_dfr)['start']).sum() / chromsize
            if nq == 0:
                projection_stat = 0
            else:
                projection_stat = np.sum(np.greater_equal.outer(subq_centers, subr_starts) & np.less.outer(subq_centers, subr_ends))

        subspace_data.hits = projection_stat
        subspace_data.p0 = p0
    return subspaces_data


def process_proj_spaces(subspaces_data: Dict[str, ProjSubspace],
                        spaces: Dict[str, Tuple[str]]) -> Tuple[ProjSpace]:
    spaces_data = tuple(ProjSpace(name=space_name, subspaces=space_subspaces)
                       for space_name, space_subspaces in spaces.items())
    for space_data in spaces_data:
        space_data.nq = sum(subspaces_data[subspace].nq
                            for subspace in space_data.subspaces)
        space_data.nr = sum(subspaces_data[subspace].nr
                            for subspace in space_data.subspaces)
        space_data.hits = sum(subspaces_data[subspace].hits
                              for subspace in space_data.subspaces)
        space_data.p0 = sum(subspaces_data[subspace].p0 * subspaces_data[subspace].chromsize
                            for subspace in space_data.subspaces) / sum(subspaces_data[subspace].chromsize
                                                                        for subspace in space_data.subspaces)

        if space_data.nq > 0:
            projection_test = ss.binomtest(space_data.hits, space_data.nq, space_data.p0)
            pval, observed = projection_test.pvalue, projection_test.proportion_estimate

            if space_data.p0 > 0:
                ratio = observed / space_data.p0
                if ratio < 1:
                    direction = 'repulsion'
                else:
                    direction = 'attraction'
            else:
                ratio = None
                direction = "undefined"
        else:
            pval = 1
            ratio = 0
            direction = "repulsion"
        
        space_data.pval = pval
        space_data.ratio = ratio
        space_data.direction = direction
    return spaces_data
        

