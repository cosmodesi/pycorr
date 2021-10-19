"""Implement high-level interface to estimate 2-point correlation function."""

import logging
import numpy as np

from .estimator import get_estimator
from .twopoint_counter import TwoPointCounter, AnalyticTwoPointCounter
from .utils import BaseClass


def TwoPointCorrelationFunction(mode, edges, data_positions1, data_positions2=None, randoms_positions1=None, randoms_positions2=None,
                                data_weights1=None, data_weights2=None, randoms_weights1=None, randoms_weights2=None, R1R2=None, estimator='auto',
                                D1D2_twopoint_weights=None, D1R2_twopoint_weights=None, D2R1_twopoint_weights=None, R1R2_twopoint_weights=None,
                                boxsize=None, mpicomm=None, **kwargs):
    r"""
    Compute pair counts and correlation function estimation.

    Parameters
    ----------
    mode : string
        Type of correlation function, one of:

            - "theta": as a function of angle (in degree) between two particles
            - "s": as a function of distance between two particles
            - "smu": as a function of distance between two particles and cosine angle :math:`\mu` w.r.t. the line-of-sight
            - "rppi": as a function of distance transverse (:math:`r_{p}`) and parallel (:math:`\pi`) to the line-of-sight
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
        See ``weight_type``.

    data_weights2 : array, default=None
        Optionally, for cross-pair counts, weights in the second catalog. See ``data_weights1``.

    randoms_weight1 : array, default=None
        Optionally, weights of the random catalog representing the first selection function. See ``data_weights1``.

    randoms_weights2 : array, default=None
        Optionally, for cross-correlations, weights of the random catalog representing the second selection function.
        See ``randoms_weights1``.

    R1R2 : BaseTwoPointCounter, default=None
        Precomputed R1R2 pairs; e.g. useful when running on many mocks with same randoms catalog.

    estimator : string, default='auto'
        Estimator name, one of ["auto", "natural", "landyszalay", "weight"].
        If "auto", "landyszalay" will be chosen if random catalog(s) is/are provided.

    bin_type : string, default='auto'
        Binning type for first dimension, e.g. :math:`r_{p}` when ``mode`` is "rppi".
        Set to ``lin`` for speed-up in case of linearly-spaced bins.
        In this case, the bin number for a pair separated by a (3D, projected, angular...) separation
        ``sep`` is given by ``(sep - edges[0])/(edges[-1] - edges[0])*(len(edges) - 1)``,
        i.e. only the first and last bins of input edges are considered.
        Then setting ``output_sepavg`` is virtually costless.
        For non-linear binning, set to "custom".
        "auto" allows for auto-detection of the binning type:
        linear binning will be chosen if input edges are
        within ``rtol = 1e-05`` (relative tolerance) *or* ``atol = 1e-08``
        (absolute tolerance) of the array
        ``np.linspace(edges[0], edges[-1], len(edges))``.

    position_type : string, default='auto'
        Type of input positions, one of:

            - "rd": RA/Dec in degree, only if ``mode`` is "theta"
            - "rdd": RA/Dec in degree, distance, for any ``mode``
            - "xyz": Cartesian positions

    weight_type : string, default='auto'
        The type of weighting to apply to provided weights. One of:

            - ``None``: no weights are applied.
            - "product_individual": each pair is weighted by the product of weights :math:`w_{1} w_{2}`.
            - "inverse_bitwise": each pair is weighted by :math:`(1 + \mathrm{nrealizations})/(1 + \mathrm{popcount}(w_{1} \& w_{2}))`.
               Multiple bitwise weights can be provided as a list.
               Individual weights can additionally be provided as float arrays, and angular upweights with ``twopoint_weights``.
               In case of cross-correlations with floating weights, bitwise weights are automatically turned to IIP weights,
               i.e. :math:`(1 + \mathrm{nrealizations})/(1 + \mathrm{popcount}(w_{1}))`.
            - "auto": automatically choose weighting based on input ``weights1`` and ``weights2``,
               i.e. ``None`` when ``weights1`` and ``weights2`` are ``None``,
               "inverse_bitwise" if onf of input weights is integer, else "product_individual".

    nrealizations : int, default=None
        In case ``weight_type`` is "inverse_bitwise", the number of realizations,
        *not* counting in current realization.
        If ``None``, will be set to the number of bits in input weights.

    D1D2_twopoint_weights : WeightTwoPointEstimator, default=None
        Weights to be applied to each pair of particles between first and second data catalogs.
        A :class:`WeightTwoPointEstimator` instance or any object with arrays ``sep``
        (separations) and ``weight`` (weight at given separation) as attributes
        (i.e. to be accessed through ``twopoint_weights.sep``, ``twopoint_weights.weight``)
        or as keys (i.e. ``twopoint_weights['sep']``, ``twopoint_weights['weight']``)
        or as element (i.e. ``sep, weight = twopoint_weights``)

    D1R2_twopoint_weights : WeightTwoPointEstimator, default=None
        Weights to be applied to each pair of particles between first data catalog and second randoms catalog.
        See ``D1D2_twopoint_weights``.

    D2R1_twopoint_weights : WeightTwoPointEstimator, default=None
        Weights to be applied to each pair of particles between second data catalog and first randoms catalog.
        See ``D1D2_twopoint_weights``.

    R1R2_twopoint_weights : WeightTwoPointEstimator, default=None
        Weights to be applied to each pair of particles between first and second randoms catalogs.
        See ``D1D2_twopoint_weights``.

    los : string, default='midpoint'
        Line-of-sight to be used when ``mode`` is "smu", "rppi" or "rp"; one of:

            - "midpoint": the mean position of the pair: :math:`\mathbf{\eta} = (\mathbf{r}_{1} + \mathbf{r}_{2})/2`
            - "x", "y" or "z": cartesian axis

    boxsize : array, float, default=None
        For periodic wrapping, the side-length(s) of the periodic cube.

    output_sepavg : bool, default=True
        Set to ``False`` to *not* calculate the average separation for each bin.
        This can make the pair counts faster if ``bin_type`` is "custom".
        In this case, :attr:`sep` will be set the midpoint of input edges.

    dtype : string, np.dtype, default=None
        Array type for positions and weights.
        If ``None``, defaults to type of first array of positions.

    nthreads : int
        Number of OpenMP threads to use.

    mpicomm : MPI communicator, default=None
        The MPI communicator, if input positions and weights are MPI-scattered.

    kwargs : dict
        Pair-counter engine-specific options.

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

    autocorr = data_positions2 is None

    positions = {'D1':data_positions1, 'D2':data_positions2, 'R1':randoms_positions1, 'R2':randoms_positions2}
    weights = {'D1':data_weights1, 'D2':data_weights2, 'R1':randoms_weights1, 'R2':randoms_weights2}
    twopoint_weights = {'D1D2':D1D2_twopoint_weights, 'D1R2':D1R2_twopoint_weights, 'D2R1':D2R1_twopoint_weights, 'R1R2':R1R2_twopoint_weights}
    precomputed = {'R1R2':R1R2}

    pairs = {}
    for label1, label2 in Estimator.requires(autocorr=(not has_randoms) or randoms_positions2 is None):
        label12 = label1 + label2
        pre = precomputed.get(label12, None)
        if pre is not None:
            if log: logger.info('Using precomputed pair counts {}.'.format(label12))
            pairs[label12] = pre
            continue
        if label12 == 'R1R2' and not has_randoms:
                if log: logger.info('Analytically computing pair counts {}.'.format(label12))
                size2 = size1 = len(positions[label1.replace('R','D')][0])
                if not autocorr:
                    size2 = len(positions[label2.replace('R','D')][0])
                pairs[label12] = AnalyticTwoPointCounter(mode, edges, boxsize, size1=size1, size2=size2)
        else:
            if log: logger.info('Computing pair counts {}.'.format(label12))
            # label12 is D1R2, but we only have R1, so swith label2 to R1
            if autocorr and label12 == 'D1R2':
                label2 = 'R1'
            pairs[label12] = TwoPointCounter(mode, edges, positions[label1], positions2=positions[label2],
                                                   weights1=weights[label1], weights2=weights[label2], twopoint_weights=twopoint_weights[label12],
                                                   boxsize=boxsize, mpicomm=mpicomm, **kwargs)
    return Estimator(**pairs)
