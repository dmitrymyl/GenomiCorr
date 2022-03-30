from typing import Callable, Dict, List, Tuple, Union

import pandas as pd

from .absdist import (absdist_simple_cols, process_absdist_spaces,
                      process_absdist_subspaces)
from .jaccard import (jaccard_simple_cols, process_jaccard_spaces,
                      process_jaccard_subspaces)
from .projection import process_proj_spaces, process_proj_subspaces
from .reldist import (process_reldist_spaces, process_reldist_subspaces,
                      reldist_simple_cols)
from .utils import parse_spaces, process_result

SpaceInput = Union[None, List[str], Tuple[str], Dict[str, Tuple[str]]]
SubspaceInput = Union[None, List[str], Tuple[str]]


__all__ = ['reldist_test', 'absdist_test', 'proj_test', 'jaccard_test']


def reldist_test(dfq: pd.DataFrame,
                 dfr: pd.DataFrame,
                 spaces: SpaceInput = None,
                 subspaces: SubspaceInput = None,
                 subspace_col: str = "chrom",
                 whole: bool = True,
                 permutations: int = 100,
                 output: str = 'full') -> pd.DataFrame:

    if subspace_col not in dfq.columns:
        raise ValueError(f"`subspace_col` is not in `dfq` columns.")
    if subspace_col not in dfr.columns:
        raise ValueError(f"`subspace_col` is not in `dfr` columns.")
    present_subspaces = tuple(set(dfq[subspace_col]) | set(dfr[subspace_col]))
    subspaces, spaces = parse_spaces(present_subspaces, spaces, subspaces, whole=whole)

    subspaces_data = process_reldist_subspaces(dfq, dfr, subspaces, subspace_col=subspace_col)
    spaces_data = process_reldist_spaces(subspaces_data, spaces, permutations)
    
    result_df = pd.DataFrame(spaces_data)
    return process_result(result_df, output, reldist_simple_cols)


def absdist_test(dfq: pd.DataFrame,
                 dfr: pd.DataFrame,
                 chromsizes: Dict[str, int],
                 spaces: SpaceInput = None,
                 subspaces: SubspaceInput = None,
                 subspace_col: str = "chrom",
                 whole: bool = True,
                 permutations: int = 100,
                 output: str = 'full') -> pd.DataFrame:

    if subspace_col not in dfq.columns:
        raise ValueError(f"`subspace_col` is not in `dfq` columns.")
    if subspace_col not in dfr.columns:
        raise ValueError(f"`subspace_col` is not in `dfr` columns.")
    present_subspaces = tuple(set(dfq[subspace_col]) | set(dfr[subspace_col]))
    subspaces, spaces = parse_spaces(present_subspaces, spaces, subspaces, whole=whole)
    extra_subspaces = set(subspaces) - set(chromsizes.keys())
    if len(extra_subspaces) > 0:
        raise ValueError(f"`subspaces` contains extra items not present in `chromsizes`: {extra_subspaces}.")

    subspaces_data = process_absdist_subspaces(dfq, dfr, chromsizes, subspaces, subspace_col, permutations)
    spaces_data = process_absdist_spaces(subspaces_data, spaces)

    result_df = pd.DataFrame(spaces_data)
    return process_result(result_df, output, absdist_simple_cols)


def proj_test(dfq: pd.DataFrame,
                 dfr: pd.DataFrame,
                 chromsizes: Dict[str, int],
                 spaces: SpaceInput = None,
                 subspaces: SubspaceInput = None,
                 subspace_col: str = "chrom",
                 whole: bool = True) -> pd.DataFrame:

    if subspace_col not in dfq.columns:
        raise ValueError(f"`subspace_col` is not in `dfq` columns.")
    if subspace_col not in dfr.columns:
        raise ValueError(f"`subspace_col` is not in `dfr` columns.")
    present_subspaces = tuple(set(dfq[subspace_col]) | set(dfr[subspace_col]))
    subspaces, spaces = parse_spaces(present_subspaces, spaces, subspaces, whole=whole)
    extra_subspaces = set(subspaces) - set(chromsizes.keys())
    if len(extra_subspaces) > 0:
        raise ValueError(f"`subspaces` contains extra items not present in `chromsizes`: {extra_subspaces}.")

    subspaces_data = process_proj_subspaces(dfq, dfr, chromsizes, subspaces, subspace_col)
    spaces_data = process_proj_spaces(subspaces_data, spaces)

    result_df = pd.DataFrame(spaces_data)
    return process_result(result_df, output='full')


def jaccard_test(dfq: pd.DataFrame,
                 dfr: pd.DataFrame,
                 chromsizes: Dict[str, int],
                 spaces: SpaceInput = None,
                 subspaces: SubspaceInput = None,
                 subspace_col: str = "chrom",
                 whole: bool = True,
                 permutations: int = 100,
                 permute: Union[str, Callable] = "rearrange",
                 output: str = 'full') -> pd.DataFrame:

    if subspace_col not in dfq.columns:
        raise ValueError(f"`subspace_col` is not in `dfq` columns.")
    if subspace_col not in dfr.columns:
        raise ValueError(f"`subspace_col` is not in `dfr` columns.")
    present_subspaces = tuple(set(dfq[subspace_col]) | set(dfr[subspace_col]))
    subspaces, spaces = parse_spaces(present_subspaces, spaces, subspaces, whole=whole)
    extra_subspaces = set(subspaces) - set(chromsizes.keys())
    if len(extra_subspaces) > 0:
        raise ValueError(f"`subspaces` contains extra items not present in `chromsizes`: {extra_subspaces}.")

    subspaces_data = process_jaccard_subspaces(dfq, dfr, chromsizes, subspaces, subspace_col, permutations, permute)
    spaces_data = process_jaccard_spaces(subspaces_data, spaces)

    result_df = pd.DataFrame(spaces_data)
    return process_result(result_df, output, jaccard_simple_cols)
