"""Implement base pair counter, to be extended when implementing a new engine."""

import os
import numpy as np

from .utils import BaseClass, BaseMetaClass
from . import utils


class TwoPointCounterError(Exception):

    """Exception raised when issue with pair counting."""


def get_twopoint_counter(engine='corrfunc'):
    """
    Return :class:`BaseTwoPointCounter` subclass corresponding
    to input engine name.

    Parameters
    ----------
    engine : string, default='corrfunc'
        Name of pair counter engine, one of ["corrfunc", "analytical"].

    Returns
    -------
    pair_counter : type
        Pair counter class.
    """
    if isinstance(engine, str):

        if engine.lower() == 'analytic':
            return AnalyticTwoPointCounter

        if engine.lower() == 'corrfunc':
            from .corrfunc import CorrfuncTwoPointCounter
            return CorrfuncTwoPointCounter

        raise TwoPointCounterError('Unknown engine {}.'.format(engine))

    return engine


class MetaTwoPointCounter(BaseMetaClass):

    """Metaclass to return correct pair counter engine."""

    def __call__(cls, *args, engine='corrfunc', **kwargs):
        return get_twopoint_counter(engine)(*args, **kwargs)


class TwoPointCounter(metaclass=MetaTwoPointCounter):
    """
    Entry point to pair counter engines.

    Parameters
    ----------
    engine : string, default='corrfunc'
        Name of pair counter engine, one of ["corrfunc", "analytical"].

    args : list
        Arguments for pair counter engine, see :class:`BaseTwoPointCounter`.

    kwargs : dict
        Arguments for pair counter engine, see :class:`BaseTwoPointCounter`.

    Returns
    -------
    engine : BaseTwoPointCounter
    """
    @classmethod
    def load(cls, filename):
        cls.log_info('Loading {}.'.format(filename))
        state = np.load(filename, allow_pickle=True)[()]
        return get_twopoint_counter(state.pop('name')).from_state(state)


class BaseTwoPointCounter(BaseClass):
    """
    Base class for pair counters.
    Extend this class to implement a new pair counter engine.

    Attributes
    ----------
    sep : array
        Array of separation values.

    wcounts : array
        (Optionally weighted) pair-counts.
    """
    def __init__(self, mode, edges, positions1, positions2=None, weights1=None, weights2=None,
                bin_type='auto', position_type='auto', weight_type='auto', nrealizations=None,
                twopoint_weights=None, los='midpoint', boxsize=None, output_sepavg=True, dtype=None,
                nthreads=None, mpicomm=None, **kwargs):
        r"""
        Initialize :class:`BaseTwoPointCounter`, and run actual pair counts
        (calling :meth:`run`), setting :attr:`wcounts` and :attr:`sep`.

        Parameters
        ----------
        mode : string
            Type of pair counts, one of:

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

        positions1 : list, array
            Positions in the first catalog. Typically of shape (3, N), but can be (2, N) when ``mode`` is "theta".

        positions2 : list, array, default=None
            Optionally, for cross-pair counts, positions in the second catalog. See ``positions1``.

        weights1 : array, list, default=None
            Weights of the first catalog. Not required if ``weight_type`` is either ``None`` or "auto".
            See ``weight_type``.

        weights2 : array, list, default=None
            Optionally, for cross-pair counts, weights in the second catalog. See ``weights1``.

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

        twopoint_weights : WeightTwoPointEstimator, default=None
            Weights to be applied to each pair of particles.
            A :class:`WeightTwoPointEstimator` instance or any object with arrays ``sep``
            (separations) and ``weight`` (weight at given separation) as attributes
            (i.e. to be accessed through ``twopoint_weights.sep``, ``twopoint_weights.weight``)
            or as keys (i.e. ``twopoint_weights['sep']``, ``twopoint_weights['weight']``)
            or as element (i.e. ``sep, weight = twopoint_weights``)

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
            If ``None``, defaults to type of first ``positions1`` array.

        nthreads : int
            Number of OpenMP threads to use.

        mpicomm : MPI communicator, default=None
            The MPI communicator, if input positions and weights are MPI-scattered.

        kwargs : dict
            Pair counter engine-specific options.
        """
        self.mode = mode
        self.nthreads = nthreads
        if nthreads is None:
            self.nthreads = int(os.getenv('OMP_NUM_THREADS','1'))
        self.mpicomm = mpicomm
        self._set_positions(positions1, positions2, position_type=position_type, dtype=dtype)
        self._set_weights(weights1, weights2, weight_type=weight_type, nrealizations=nrealizations, twopoint_weights=twopoint_weights)
        self._set_edges(edges, bin_type=bin_type)
        self._set_boxsize(boxsize)
        self._set_los(los)
        self.output_sepavg = output_sepavg
        self.attrs = kwargs
        self.wnorm = self.normalization()

        self.run()
        self._set_default_sep()


    def run(self):
        """
        Method that computes the actual pair counts and set :attr:`wcounts` and :attr:`sep`,
        to be implemented in your new engine.
        """
        raise NotImplementedError('Implement method "run" in your {}'.format(self.__class__.__name__))

    def _set_edges(self, edges, bin_type='auto'):
        if np.ndim(edges[0]) == 0:
            edges = (edges,)
        self.edges = tuple(edges)
        if self.mode in ['smu','rppi']:
            if not self.ndim == 2:
                raise TwoPointCounterError('A tuple of edges should be provided to pair counter in mode {}'.format(self.mode))
        else:
            if not self.ndim == 1:
                raise TwoPointCounterError('Only one edge array should be provided to pair counter in mode {}'.format(self.mode))
        self._set_bin_type(bin_type)

    def _set_bin_type(self, bin_type):
        self.bin_type = bin_type.lower()
        allowed_bin_types = ['lin', 'custom', 'auto']
        if self.bin_type not in allowed_bin_types:
            raise TwoPointCounterError('bin type should be one of {}'.format(allowed_bin_types))
        if self.bin_type == 'auto':
            edges = self.edges[0]
            if np.allclose(edges, np.linspace(edges[0], edges[-1], len(edges))):
                self.bin_type = 'lin'

    @property
    def shape(self):
        """Return shape of obtained pair counts :attr:`wcounts`."""
        return tuple(len(edges) - 1 for edges in self.edges)

    @property
    def ndim(self):
        """Return binning dimensionality."""
        return len(self.edges)

    @property
    def periodic(self):
        """Whether periodic wrapping is used (i.e. :attr:`boxsize` is not ``None``)."""
        return self.boxsize is not None

    @property
    def with_mpi(self):
        """Whether to use MPI."""
        if not hasattr(self, 'mpicomm'): self.mpicomm = None
        return self.mpicomm is not None and self.mpicomm.size > 1

    def _set_positions(self, positions1, positions2=None, position_type='auto', dtype=None):
        position_type = position_type.lower()
        if position_type == 'auto':
            if self.mode == 'theta': position_type = 'rd'
            else: position_type = 'xyz'
        self.dtype = dtype

        def check_shape(positions):
            for ip, p in enumerate(positions):
                # cast to the input dtype if exists (may be set by previous weights)
                positions[ip] = np.asarray(p, dtype=self.dtype)
            size = len(positions[0])
            self.dtype = positions[0].dtype
            if not np.issubdtype(self.dtype, np.floating):
                raise TwoPointCounterError('Input position arrays should be of floating type, not {}'.format(self.dtype))
            for p in positions[1:]:
                if len(p) != size:
                    raise TwoPointCounterError('All position arrays should be of the same size')
                if p.dtype != self.dtype:
                    raise TwoPointCounterError('All position arrays should be of the same type, you can e.g. provide dtype')
            if position_type != 'auto' and len(positions) != len(position_type):
                raise TwoPointCounterError('For position type = {}, please provide a list of {:d} arrays for positions'.format(position_type, len(positions)))
            if self.mode == 'theta':
                if position_type == 'xyz':
                    positions = utils.cartesian_to_sky(positions, degree=True)[:2]
                elif position_type in ['rdd', 'rdz']:
                    positions = list(positions)
                elif position_type != 'rd':
                    raise TwoPointCounterError('For mode = {}, position type should be one of ["xyz", "rdz", "rd"]'.format(self.mode))
            else:
                if position_type == 'rdd':
                    positions = utils.sky_to_cartesian(positions, degree=True)
                elif position_type != 'xyz':
                    raise TwoPointCounterError('For mode = {}, position type should be one of ["xyz", "rdd"]'.format(self.mode))
            return positions

        self.positions1 = list(positions1)
        self.positions1 = check_shape(self.positions1)

        self.autocorr = positions2 is None
        if self.autocorr:
            self.positions2 = None
        else:
            self.positions2 = list(positions2)
            self.positions2 = check_shape(self.positions2)

    def _set_weights(self, weights1, weights2=None, weight_type='auto', nrealizations=None, twopoint_weights=None):

        if weight_type is not None: weight_type = weight_type.lower()
        allowed_weight_types = [None, 'auto', 'product_individual', 'inverse_bitwise']
        if weight_type not in allowed_weight_types:
            raise TwoPointCounterError('weight_type should be one of {}'.format(allowed_weight_types))

        if self.autocorr:
            if weights2 is not None:
                raise TwoPointCounterError('weights2 are provided, but not positions2')

        if weights1 is None:
            if weights2 is not None:
                raise TwoPointCounterError('weights2 are provided, but not weights1')
        else:
            if self.autocorr:
                if weights2 is not None:
                    raise TwoPointCounterError('weights2 are provided, but not positions2')
            else:
                if weights2 is None:
                    raise TwoPointCounterError('weights1 are provided, but not weights2')

        self.nrealizations = nrealizations
        self.n_bitwise_weights = 0
        if weight_type is None:
            self.weights1 = self.weights2 = None
        else:

            def check_shape(weights, size):
                if weights is None or len(weights) == 0:
                    return None, 0
                if np.ndim(weights[0]) == 0:
                    weights = [weights]
                individual_weights = []
                bitwise_weights = []
                for w in weights:
                    if len(w) != size:
                        raise TwoPointCounterError('All weight arrays should be of the same size as position arrays')
                    if np.issubdtype(w.dtype, np.integer):
                        if weight_type == 'product_individual':
                            individual_weights.append(w)
                        else: # certainly bitwise weight
                            bitwise_weights.append(w)
                    else:
                        individual_weights.append(w)
                # any integer array bit size will be a multiple of 8
                bitwise_weights = utils.reformat_bitarrays(*bitwise_weights, dtype=np.uint8)
                n_bitwise_weights = len(bitwise_weights)
                weights = bitwise_weights
                if individual_weights:
                    weights += [np.prod(individual_weights, axis=0, dtype=self.dtype)]
                return weights, n_bitwise_weights

            self.weights1, self.n_bitwise_weights = check_shape(weights1, len(self.positions1[0]))
            if self.nrealizations is None:
                self.nrealizations = self.n_bitwise_weights * 8
            elif self.n_bitwise_weights and self.nrealizations > self.n_bitwise_weights * 8:
                raise TwoPointCounterError('Provided number of realizations is {:d}, '
                                       'more than actual number of bits in bitwise weights {:d}'.format(self.nrealizations, self.n_bitwise_weights * 8))
            self.weights2 = weights2
            if not self.autocorr:
                self.weights2, n_bitwise_weights2 = check_shape(weights2, len(self.positions2[0]))

                if n_bitwise_weights2 != self.n_bitwise_weights:

                    def wiip(weights):
                        return (1. + self.nrealizations)/(1. + utils.popcount(*weights))

                    if n_bitwise_weights2 == 0:
                        indweights = self.weights1[self.n_bitwise_weights:]
                        self.weights1 = [wiip(self.weights1[:self.n_bitwise_weights])*(indweights or 1)]
                        self.n_bitwise_weights = 0
                        self.log_info('Setting IIP weights for first catalog.')
                    elif self.n_bitwise_weights == 0:
                        self.nrealizations = n_bitwise_weights2 * 8
                        indweights = self.weights2[n_bitwise_weights2:]
                        self.weights2 = [wiip(self.weights2[:n_bitwise_weights2])*(indweights or 1)]
                        self.log_info('Setting IIP weights for second catalog.')
                    else:
                        raise TwoPointCounterError('Incompatible length of bitwise weights: {:d} and {:d} bytes'.format(self.n_bitwise_weights, n_bitwise_weights2))

        self.twopoint_weights = twopoint_weights
        if twopoint_weights is not None:
            from collections import namedtuple
            TwoPointWeight = namedtuple('TwoPointWeight', ['sep', 'weight'])
            try:
                sep = twopoint_weights.sep
                weight = twopoint_weights.weight
            except AttributeError:
                try:
                    sep = twopoint_weights['sep']
                    weight = twopoint_weights['weight']
                except IndexError:
                    sep, weight = twopoint_weights
            # just to make sure we use the correct dtype
            self.twopoint_weights = TwoPointWeight(sep=np.cos(np.radians(np.ravel(sep)[::-1]), dtype=self.dtype),
                                                   weight=np.array(np.ravel(weight)[::-1], dtype=self.dtype))


    def _mpi_decompose(self):
        if self.with_mpi:
            smoothing = np.max(self.edges[0])
            if self.mode == 'theta':
                smoothing = 2 * np.sin(0.5 * np.deg2rad(smoothing))
            elif self.mode == 'rppi':
                smoothing = np.sqrt(smoothing**2 + np.max(self.edges[1])**2)
            elif self.mode == 'rp':
                smoothing = np.inf
            from . import mpi
            return mpi.domain_decompose(self.mpicomm, smoothing, self.positions1, weights1=self.weights1,
                                 positions2=self.positions2, weights2=self.weights2, boxsize=self.boxsize)
        return (self.positions1, self.weights1), (self.positions2, self.weights2)

    def _set_default_sep(self):
        mid = [(edges[1:] + edges[:-1])/2. for edges in self.edges]
        sep = list(np.meshgrid(*mid, indexing='ij'))
        if self.output_sepavg:
            sep[0] = self.sep
        self.sep = sep

    def _set_los(self, los):
        self.los = los
        allowed_los = ['midpoint', 'endpoint', 'firstpoint', 'x', 'y', 'z']
        if self.los not in allowed_los:
            raise TwoPointCounterError('los should be one of {}'.format(allowed_los))
        if self.periodic and self.mode != 's':
            allowed_los = ['x', 'y', 'z']
            if self.los not in allowed_los:
                raise TwoPointCounterError('When boxsize is provided, los should be one of {}'.format(allowed_los))

    def _set_boxsize(self, boxsize):
        self.boxsize = boxsize
        if self.periodic:
            self.boxsize = np.empty(3, dtype='f8')
            self.boxsize[:] = boxsize

    def normalization(self):
        r"""
        Return pair count normalization, i.e., in case of cross-correlation:

        .. math::

            \left(\sum_{i=1}^{N_{1}} w_{1,i}\right) \left(\sum_{j=1}^{N_{2}} w_{2,j}\right)

        with the sums running over the weights of the first and second catalogs, and in case of auto-correlation:

        .. math::

            \left(\sum_{i=1}^{N_{1}} w_{1,i}\right)^{2} - \sum_{i=1}^{N_{1}} w_{1,i}^{2}

        """
        if self.weights1 is None:
            size1 = size2 = len(self.positions1[0])
            if not self.autocorr: size2 = len(self.positions2[0])
            if self.with_mpi:
                size1, size2 = self.mpicomm.allreduce(size1), self.mpicomm.allreduce(size2)
            if self.autocorr:
                return size1 * (size1 - 1)
            return size1 * size2

        if self.n_bitwise_weights:

            def binned_weights(weights):
                indweights = weights[self.n_bitwise_weights:]
                if indweights: indweights = np.prod(indweights, axis=0)
                else: indweights = None
                w = np.bincount(utils.popcount(*weights[:self.n_bitwise_weights]) + 1,
                                weights=indweights, minlength=self.nrealizations + 2)
                if self.with_mpi:
                    w = self.mpicomm.allreduce(w)
                c = np.flatnonzero(w)
                return w[c], c

            w1, c1 = binned_weights(self.weights1)
            if self.autocorr:
                w2, c2 = w1, c1
            else:
                w2, c2 = binned_weights(self.weights2)
            joint = utils.joint_occurences(self.nrealizations + 1, max_occurences=max(c1.max(), c2.max()))
            sumw_cross = 0
            for c1_, w1_ in zip(c1, w1):
                for c2_, w2_ in zip(c2, w2):
                    sumw_cross += w1_ * w2_ * (joint[c1_][c2_] if c2_ <= c1_ else joint[c2_][c1_])
            sumw_auto = 0
            if self.autocorr:
                w1sq = np.bincount(utils.popcount(*self.weights1[:self.n_bitwise_weights]) + 1,
                                   weights=self.weights1[-1]**2, minlength=self.nrealizations + 2)[c1]
                if self.with_mpi:
                    w1sq = self.mpicomm.allreduce(w1sq)
                for c1_, w1sq_ in zip(c1, w1sq):
                    sumw_auto += joint[c1_][c1_] * w1sq_
            return sumw_cross - sumw_auto

        # individual_weights
        if self.autocorr:
            sumw_cross, sumw_auto = self.weights1[0].sum(), (self.weights1[0]**2).sum()
            if self.with_mpi:
                sumw_cross, sumw_auto = self.mpicomm.allreduce(sumw_cross), self.mpicomm.allreduce(sumw_auto)
            return sumw_cross**2 - sumw_auto
        sumw1, sumw2 = self.weights1[0].sum(), self.weights2[0].sum()
        if self.with_mpi:
            sumw1, sumw2 = self.mpicomm.allreduce(sumw1), self.mpicomm.allreduce(sumw2)
        return sumw1 * sumw2

    def normalized_wcounts(self):
        """Return normalized pair counts, i.e. :attr:`wcounts` divided by :meth:`normalization`."""
        return self.wcounts/self.wnorm

    def rebin(self, factor=1):
        """
        Rebin pair counts, by factor(s) ``factor``.
        A tuple must be provided in case :attr:`ndim` is greater than 1.
        Input factors must divide :attr:`shape`.
        """
        if np.ndim(factor) == 0:
            factor = (factor,)
        if len(factor) != self.ndim:
            raise TwoPointCounterError('Provide a rebinning factor for each dimension')
        new_shape = tuple(s//f for s,f in zip(self.shape, factor))
        wcounts = self.wcounts
        self.wcounts = utils.rebin(wcounts, new_shape, statistic=np.sum)
        self.sep = [utils.rebin(sep*wcounts, new_shape, statistic=np.sum)/self.wcounts for sep in self.sep]

    def __getstate__(self):
        state = {}
        for name in ['name', 'autocorr', 'sep', 'wcounts', 'wnorm', 'edges', 'mode', 'bin_type',
                     'boxsize', 'los', 'output_sepavg', 'attrs']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def save(self, filename):
        """Save pair counts to ``filename``."""
        if not self.with_mpi or self.mpicomm.rank == 0:
            super(BaseTwoPointCounter, self).save(filename)
        if self.with_mpi:
            self.mpicomm.Barrier()


class AnalyticTwoPointCounter(BaseTwoPointCounter):
    """
    Analytic pair counter. Assume periodic wrapping and no data weights.

    Attributes
    ----------
    sep : array
        Array of separation values.

    wcounts : array
        Analytical pair counts.
    """
    name = 'analytic'

    def __init__(self, mode, edges, boxsize, size1=10, size2=None, los='z'):
        r"""
        Initialize :class:`AnalyticTwoPointCounter`, and set :attr:`wcounts` and :attr:`sep`.

        Parameters
        ----------
        mode : string
            Pair counting mode, one of:

                - "s": pair counts as a function of distance between two particles
                - "smu": pair counts as a function of distance between two particles and cosine angle :math:`\mu` w.r.t. the line-of-sight
                - "rppi": pair counts as a function of distance transverse (:math:`r_{p}`) and parallel (:math:`\pi`) to the line-of-sight
                - "rp": same as "rppi", without binning in :math:`\pi`

        edges : tuple, array
            Tuple of bin edges (arrays), for the first (e.g. :math:`r_{p}`)
            and optionally second (e.g. :math:`\pi`) dimensions.
            In case of single-dimension binning (e.g. ``mode`` is "theta", "s" or "rp"),
            the single array of bin edges can be provided directly.

        boxsize : array, float
            The side-length(s) of the periodic cube.

        size1 : int, default=10
            Length of the first catalog.

        size2 : int, default=None
            Optionally, for cross-pair counts, length of second catalog.

        los : string, default='z'
            Line-of-sight to be used when ``mode`` is "rp", in case of non-cubic box;
            one of cartesian axes "x", "y" or "z".
        """
        self.mode = mode
        self._set_edges(edges)
        self._set_boxsize(boxsize)
        self._set_los(los)
        self.size1 = size1
        self.size2 = size2
        self.autocorr = size2 is None
        self.output_sepavg = False
        self.run()
        self._set_default_sep()
        self.wnorm = self.normalization()

    def run(self):
        """Set analytical pair counts."""
        if self.mode == 's':
            v = 4./3. * np.pi * self.edges[0]**3
            dv = np.diff(v, axis=0)
        elif self.mode == 'smu':
            # we bin in abs(mu)
            v = 4./3. * np.pi * self.edges[0][:,None]**3 * self.edges[1]
            dv = np.diff(np.diff(v, axis=0), axis=-1)
        elif self.mode == 'rppi':
            # height is double pimax
            v = 2. * np.pi * self.edges[0][:,None]**2 * self.edges[1]
            dv = np.diff(np.diff(v, axis=0), axis=1)
        elif self.mode == 'rp':
            v = np.pi * self.edges[0][:,None]**2 * self.boxsize['xyz'.index(self.los)]
            dv = np.diff(v, axis=0)
        else:
            raise PairCounterError('No analytic randoms provided for mode {}'.format(self.mode))
        self.wcounts = self.normalization()*dv/self.boxsize.prod()

    def normalization(self):
        """
        Return pair count normalization, i.e., in case of cross-correlation ``size1 * size2``,
        and in case of auto-correlation ``size1 * (size1 - 1)``.
        """
        if self.autocorr:
            return self.size1 * (self.size1 - 1)
        return self.size1 * self.size2