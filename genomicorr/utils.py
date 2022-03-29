import numpy as np
import scipy.stats as ss
from scipy.linalg import LinAlgError


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