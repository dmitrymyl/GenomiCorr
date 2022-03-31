from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Union, cast
import numpy as np
import pandas as pd
import bioframe as bf
from .utils import calc_pvalue, NDArrayInt, NDArrayFloat, I_gen


def arr_inter_union(qstarts: NDArrayInt,
                    qends: NDArrayInt,
                    rstarts: NDArrayInt,
                    rends: NDArrayInt,
                    mergeq: bool = True,
                    merger: bool = True) -> Tuple[int, int]:
    if len(qstarts) != len(qends):
        raise ValueError
    if len(rstarts)  != len(rends):
        raise ValueError
    
    if len(qstarts) == 0:
        return 0, (rends - rstarts).sum()
    if len(rstarts) == 0:
        return 0, (qends - qstarts).sum()

    if mergeq:
        _, qstarts_m, qends_m = bf.arrops.merge_intervals(qstarts, qends)
    else:
        qstarts_m = qstarts
        qends_m = qends
    
    if merger:
        _, rstarts_m, rends_m = bf.arrops.merge_intervals(rstarts, rends)
    else:
        rstarts_m = rstarts
        rends_m = rends
    
    dfq_m = pd.DataFrame({'chrom': 'chr', 'start': qstarts_m, 'end': qends_m})
    dfr_m = pd.DataFrame({'chrom': 'chr', 'start': rstarts_m, 'end': rends_m})
    overlap = bf.overlap(dfq_m, dfr_m, how='outer', return_overlap=True)
    n_inter = int((overlap['overlap_end'] - overlap['overlap_start']).sum())
    n_union = int((overlap.loc[:, ['end', 'end_']].max(axis=1) - overlap.loc[:, ['start', 'start_']].min(axis=1)).sum())
    return n_inter, n_union


def calc_ji_stat(inter: I_gen,
                 union: I_gen) -> Union[float, NDArrayFloat]:
    return inter / union


def rearrange_intervals(starts: NDArrayInt,
                        ends: NDArrayInt,
                        chromsize: int) -> Tuple[NDArrayInt, NDArrayInt]:
    rng = np.random.default_rng()

    right_end = ends.max()
    widths = ends - starts
    addgap = starts.min() + chromsize - right_end
    gaps = starts[1:] - ends[:-1]
    gaps = np.append(gaps, addgap)

    rng.shuffle(widths)
    rng.shuffle(gaps)

    starting_gap = gaps[-1]
    if starting_gap > 0:
        starting_point = rng.integers(0, starting_gap, 1)[0]
    elif starting_gap < 0:
        starting_point = rng.integers(starting_gap, 0, 1)[0]

    new_starts = np.cumsum(np.insert(np.sum(np.stack((widths[:-1], gaps[:-1])), axis=0), 0, starting_point))
    new_ends = new_starts + widths
    overlist = new_starts < 0
    add_starts = chromsize + new_starts[overlist]
    add_ends = np.ones(overlist.sum(), dtype=int) * chromsize
    new_starts[overlist] = 0
    new_starts = np.concatenate((new_starts, add_starts))
    new_ends = np.concatenate((new_ends, add_ends))

    sorted_order = np.argsort(new_starts)
    return new_starts[sorted_order], new_ends[sorted_order]


def permute_intervals(starts: NDArrayInt,
                      ends: NDArrayInt,
                      chromsize: int) -> Tuple[NDArrayInt, NDArrayInt]:
    rng = np.random.default_rng()

    # right_end = ends.max()
    widths = ends - starts
    new_starts = rng.integers(0, high=chromsize, size=widths.shape[0], endpoint=False)
    new_ends = new_starts + widths
    overlist = new_ends > chromsize
    lag = new_ends[overlist] - chromsize
    new_ends[overlist] = chromsize
    add_starts = np.zeros(overlist.sum(), dtype='int')
    add_ends = lag
    new_starts = np.concatenate((new_starts, add_starts))
    new_ends = np.concatenate((new_ends, add_ends))
    sorted_order = np.argsort(new_starts)
    return new_starts[sorted_order], new_ends[sorted_order]


@dataclass
class JaccardSubspace:
    nq: int = 0
    nr: int = 0
    inter: int = 0
    union: int = 0
    null_inters: NDArrayInt = np.array([], dtype='int')
    null_unions: NDArrayInt = np.array([], dtype='int')
    chromsize: int = 0


@dataclass
class JaccardSpace:
    name: str = ""
    subspaces: Tuple[str, ...] = ("",)
    nq: int = 0
    nr: int = 0
    inter: int = 0
    union: int = 0
    null_inters: NDArrayInt = np.array([], dtype='int')
    null_unions: NDArrayInt = np.array([], dtype='int')
    ji: float = 0
    null_jis: NDArrayFloat = np.array([], dtype='float')
    pval: float = 1
    direction: str = "undefined"


def process_jaccard_subspaces(dfq: pd.DataFrame,
                              dfr: pd.DataFrame,
                              chromsizes: Dict[str, int],
                              subspaces: Tuple[str, ...],
                              subspace_col: str = 'chrom',
                              permutations: int = 100,
                              permute: Union[str,
                                             Callable[[NDArrayInt, NDArrayInt, int],
                                                      Tuple[NDArrayInt, NDArrayInt]]] = "rearrange") -> Dict[str, JaccardSubspace]:
    subspaces_data = {subspace: JaccardSubspace() for subspace in subspaces}
    if permute == "rearrange":
        permutation_func = rearrange_intervals
    elif permute == "permute":
        permutation_func = permute_intervals
    elif callable(permute):
        permutation_func = cast(Callable, permute)
    else:
        raise ValueError("`permute` is neither 'rearrange', 'permute' or callable.")

    for subspace, subspace_data in subspaces_data.items():
        sub_dfq = dfq.query(f'{subspace_col} == @subspace')
        sub_dfr = dfr.query(f'{subspace_col} == @subspace')
        nq = sub_dfq.shape[0]
        nr = sub_dfr.shape[0]
        chromsize = chromsizes.get(subspace, max(sub_dfq['end'].max(), sub_dfr['end'].max()))
        
        subspace_data.nq = nq
        subspace_data.nr = nr
        subspace_data.chromsize = chromsize
        null_inters: Union[Tuple[int, ...], NDArrayInt]
        null_unions: Union[Tuple[int, ...], NDArrayInt]

        if subspace_data.nq == 0 and subspace_data.nr == 0:
            inter = 0
            union = 0
            null_inters = np.zeros(permutations, dtype=int)
            null_unions = np.zeros(permutations, dtype=int)
        elif subspace_data.nq == 0:
            inter = 0
            union = nr
            null_inters = np.zeros(permutations, dtype=int)
            null_unions = np.ones(permutations, dtype=int) * nr
        elif subspace_data.nr == 0:
            inter = 0
            union = nq
            null_inters = np.zeros(permutations, dtype=int)
            null_unions = np.ones(permutations, dtype=int) * nq
        else:
            sub_dfr = bf.merge(sub_dfr)
            subq_starts = sub_dfq['start'].to_numpy()
            subq_ends = sub_dfq['end'].to_numpy()
            subr_starts = sub_dfr['start'].to_numpy()
            subr_ends = sub_dfr['end'].to_numpy()
            
            inter, union = arr_inter_union(subq_starts,
                                           subq_ends,
                                           subr_starts,
                                           subr_ends,
                                           merger=False)
            
            result = tuple(arr_inter_union(*permutation_func(subq_starts, subq_ends, chromsize),
                                           subr_starts,
                                           subr_ends,
                                           merger=False)
                           for _ in range(permutations))

            null_inters, null_unions = zip(*result)

        subspace_data.inter = inter
        subspace_data.union = union
        subspace_data.null_inters = np.array(null_inters, dtype='int')
        subspace_data.null_unions = np.array(null_unions, dtype='int')
        
    return subspaces_data


def process_jaccard_spaces(subspaces_data: Dict[str, JaccardSubspace],
                           spaces: Dict[str, Tuple[str, ...]]) -> Tuple[JaccardSpace, ...]:
    spaces_data = tuple(JaccardSpace(name=space_name, subspaces=space_subspaces)
                       for space_name, space_subspaces in spaces.items())
    for space_data in spaces_data:
        space_data.nq = sum(subspaces_data[subspace].nq
                            for subspace in space_data.subspaces)
        space_data.nr = sum(subspaces_data[subspace].nr
                            for subspace in space_data.subspaces)
        space_data.inter = sum(subspaces_data[subspace].inter
                            for subspace in space_data.subspaces)
        space_data.union = sum(subspaces_data[subspace].union
                            for subspace in space_data.subspaces)
        space_data.null_inters = np.array([subspaces_data[subspace].null_inters
                                           for subspace in space_data.subspaces]).sum(axis=0)
        space_data.null_unions = np.array([subspaces_data[subspace].null_unions
                                           for subspace in space_data.subspaces]).sum(axis=0)
        if space_data.union > 0:
            space_data.ji = cast(float, calc_ji_stat(space_data.inter, space_data.union))
            space_data.null_jis = cast(NDArrayFloat, calc_ji_stat(space_data.null_inters, space_data.null_unions))
            pval = calc_pvalue(space_data.null_jis, space_data.ji)

            if pval < 0.5:
                space_data.pval = pval * 2
                space_data.direction = 'attraction'
            else:
                space_data.pval = (1 - pval) * 2
                space_data.direction = 'repulsion'
        else:
            space_data.ji = float('nan')
            space_data.null_jis = np.array([])
            space_data.pval = 1
            space_data.direction = "undefined"
    return spaces_data


jaccard_simple_cols = ("nq",
                       "nr",
                       "subspaces",
                       "inter",
                       "union",
                       "ji",
                       "pval",
                       "direction")
