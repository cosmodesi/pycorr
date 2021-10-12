"""Implement high-level interface to estimate 2-point correlation function."""

import logging
import numpy as np

from .estimator import get_estimator
from .pair_counter import TwoPointCounter, AnalyticTwoPointCounter
from .utils import BaseClass


def TwoPointCorrelationFunction(mode, edges, data_positions1, data_positions2=None, randoms_positions1=None, randoms_positions2=None,
                                data_weights1=None, data_weights2=None, randoms_weights1=None, randoms_weights2=None,
                                D1D2_twopoint_weights=None, D1R2_twopoint_weights=None, D2R1_twopoint_weights=None, R1R2_twopoint_weights=None,
                                R1R2=None, estimator='auto', boxsize=None, mpicomm=None, **kwargs):
    r"""
    Compute pair counts and correlation function estimation.

    Parameters
    ----------
    mode : string
        Type of correlation function, one of:

        - "theta": as a function of angle (in degree) between two particles
        - "s": as a function of distance between two particles
        - "smu": as a function of distance between two particles and cosine angle :math:`\mu`
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

    D1D2_twopoint_weights : WeightTwoPointEstimator, default=None
        Weights to be applied to each pair of particles between first and second data catalogs.
        A :class:`WeightTwoPointEstimator` instance or any object with attribute arrays ``sep``
        (separations) and ``weight`` (weight at given separation).

    D1R2_twopoint_weights : WeightTwoPointEstimator, default=None
        Weights to be applied to each pair of particles between first data catalog and second randoms catalog.
        See ``D1D2_twopoint_weights``.

    D2R1_twopoint_weights : WeightTwoPointEstimator, default=None
        Weights to be applied to each pair of particles between second data catalog and first randoms catalog.
        See ``D1D2_twopoint_weights``.

    R1R2_twopoint_weights : WeightTwoPointEstimator, default=None
        Weights to be applied to each pair of particles between first and second randoms catalogs.
        See ``D1D2_twopoint_weights``.

    R1R2 : BaseTwoPointCounter, default=None
        Precomputed R1R2 pairs; e.g. useful when running on many mocks with same randoms catalog.

    estimator : string, default='auto'
        Estimator name, one of ["auto", "natural", "landyszalay", "weight"].
        If "auto", "landyszalay" will be chosen if random catalog(s) is/are provided.

    boxsize : array, float, default=None
        For periodic wrapping, the side-length(s) of the periodic cube.

    mpicomm : MPI communicator, default=None
        The MPI communicator, when running over multiple MPI processes.

    kwargs : dict
        Other arguments for pair counter, see :class:`BaseTwoPointCounter`.

    Returns
    -------
    estimator : BaseTwoPointEstimator
        Estimator with correlation function estimation :attr:`BaseTwoPointEstimator.corr`
        at separations :attr:`BaseTwoPointEstimator.sep`.
    """
    logger = logging.getLogger('TwoPointCorrelationFunction')
    log = mpicomm is None or mpicomm.rank == 0

    has_randoms = randoms_positions1 is not None
    Estimator = get_estimator(estimator, has_cross=has_randoms)
    if log: logger.info('Using estimator {}.'.format(Estimator.__name__))

    autocorr = data_positions2 is None or (data_positions2 is data_positions1 and data_weights2 is data_weights1)

    if autocorr:
        data_positions2 = data_positions1
        data_weights2 = data_weights1
        randoms_positions2 = randoms_positions1
        randoms_weights2 = randoms_weights1

    positions = {'D1':data_positions1, 'D2':data_positions2, 'R1':randoms_positions1, 'R2':randoms_positions2}
    weights = {'D1':data_weights1, 'D2':data_weights2, 'R1':randoms_weights1, 'R2':randoms_weights2}
    twopoint_weights = {'D1D2':D1D2_twopoint_weights, 'D1R2':D1R2_twopoint_weights, 'D2R1':D2R1_twopoint_weights, 'R1R2':R1R2_twopoint_weights}
    precomputed = {'R1R2': R1R2}

    pairs = {}
    for label1,label2 in Estimator.requires(autocorr=True):
        label12 = label1 + label2
        pre = precomputed.get(label12, None)
        if pre is not None:
            if log: logger.info('Using precomputed pair counts {}.'.format(label12))
            pairs[label12] = pre
            continue
        if label12 == 'R1R2' and not has_randoms:
            if log: logger.info('Analytically computing pair counts {}.'.format(label12))
            pairs[label12] = AnalyticTwoPointCounter(mode, edges, boxsize,
                                                           n1=positions[label1][0].size, positions2=positions[label2][0].size)
        else:
            if log: logger.info('Computing pair counts {}.'.format(label12))
            pairs[label12] = TwoPointCounter(mode, edges, positions[label1], positions2=positions[label2],
                                                   weights1=weights[label1], weights2=weights[label2], twopoint_weights=twopoint_weights[label12],
                                                   mpicomm=mpicomm, **kwargs)
    return Estimator(**pairs)
