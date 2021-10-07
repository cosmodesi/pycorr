"""Implement high-level interface to estimate 2-point correlation function."""

import numpy as np

from .estimator import get_estimator
from .pair_counter import TwoPointCounter, AnalyticTwoPointCounter
from .utils import BaseClass


def TwoPointCorrelationFunction(mode, edges, data_positions1, data_positions2=None, randoms_positions1=None, randoms_positions2=None,
                                data_weights1=None, data_weights2=None, randoms_weights1=None, randoms_weights2=None,
                                estimator='auto', boxsize=None, **kwargs):
    r"""
    Compute pair counts and correlation function estimation.

    Parameters
    ----------
    mode : string
        Type of correlation function, one of:
        
        - "theta": as a function of angle (in degree) between two galaxies
        - "s": as a function of distance between two galaxies
        - "smu": as a function of distance between two galaxies and cosine angle :math:`\mu`
                 w.r.t. the line-of-sight
        - "rppi": as a function of distance transverse (:math:`r_{p}`) and parallel (:math:`\pi`)
                 to the line-of-sight
        - "rp": same as "rppi", without binning in :math:`\pi`

    edges : tuple, array
        Tuple of bin edges (arrays), for the first (e.g. :math:`r_{p}`)
        and optionally second (e.g. :math:`\pi`) dimensions.
        In case of single-dimension binning (e.g. ``mode`` is "theta", "s" or "rp"),
        the single array of bin edges can be provided directly.

    data_positions1 : array
        Positions in the first data catalog. Typically of shape (3, N), but can be (2, N) when ``mode`` is "theta".

    data_positions2 : array, default=None
        Optionally, for cross-correlations, data positions in the second catalog. See ``data_positions1``.

    randoms_positions1 : array, default=None
        Optionally, positions of the random catalog representing the first selection function.
        If no randoms are provided, and estimator is "auto", or "natural",
        :class:`NaturalTwoPointEstimator` will be used to estimate the correlation function,
        with analytical pair counts for R1R2.

    randoms_positions2 : array, default=None
        Optionally, for cross-correlations, positions of the random catalog representing the second selection function.
        See ``randoms_positions1``.

    data_weights1 : array, default=None
        Weights of the first catalog. Not required if ``weight_type`` is either ``None`` or "auto".

    data_weights2 : array, default=None
        Optionally, for cross-pair counts, weights in the second catalog. See ``data_weights1``.

    randoms_weight1 : array, default=None
        Optionally, weights of the random catalog representing the first selection function. See ``data_weights1``.

    randoms_weights2 : array, default=None
        Optionally, for cross-correlations, weights of the random catalog representing the second selection function.
        See ``randoms_weights1``.

    estimator : string, default='auto'
        Estimator name, one of ["auto", "natural", "landyszalay"].
        If "auto", "landyszalay" will be chosen if random catalog(s) is/are provided.

    boxsize : array, float, default=None
        For periodic wrapping, the side-length(s) of the periodic cube.

    kwargs : dict
        Other arguments for pair counter, see :class:`BaseTwoPointCounterEngine`.

    Returns
    -------
    estimator : BaseTwoPointEstimator
        Estimator with correlation function estimation :attr:`BaseTwoPointEstimator.corr`
        at separations :attr:`BaseTwoPointEstimator.sep`.
    """
    has_randoms = randoms_positions1 is not None
    Estimator = get_estimator(estimator, has_cross=has_randoms)

    autocorr = data_positions2 is None or (data_positions2 is data_positions1 and data_weights2 is data_weights1)

    if autocorr:
        data_positions2 = data_positions1
        data_weights2 = data_weights1
        randoms_positions2 = randoms_positions1
        randoms_weights2 = randoms_weights1

    positions = {'D1':data_positions1, 'D2':data_positions2, 'R1':randoms_positions1, 'R2':randoms_positions2}
    weights = {'D1':data_weights1, 'D2':data_weights2, 'R1':randoms_weights1, 'R2':randoms_weights2}

    pairs = {}
    for label1,label2 in Estimator.requires(autocorr=True):
        if label1+label2 == 'R1R2' and not has_randoms:
            pairs[label1+label2] = AnalyticTwoPointCounter(mode, edges, boxsize,
                                                           n1=positions[label1][0].size, positions2=positions[label2][0].size)

        pairs[label1+label2] = TwoPointCounter(mode, edges, positions[label1], positions2=positions[label2],
                                               weights1=weights[label1], weights2=weights[label2], **kwargs)

    return Estimator(**pairs)
