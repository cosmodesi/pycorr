"""Implements base two-point counter, to be extended when implementing a new engine."""

import os
import numpy as np

from .utils import BaseClass, get_mpi
from . import utils


def _make_array(value, shape, dtype='f8'):
    # Return numpy array filled with value
    toret = np.empty(shape, dtype=dtype)
    toret[...] = value
    return toret


class TwoPointCounterError(Exception):

    """Exception raised when issue with two-point counting."""


def get_twopoint_counter(engine='corrfunc'):
    """
    Return :class:`BaseTwoPointCounter`-subclass corresponding to input engine name.

    Parameters
    ----------
    engine : string, default='corrfunc'
        Name of two-point counter engine, one of ["corrfunc", "analytic"].

    Returns
    -------
    counter : type
        Two-point counter class.
    """
    if isinstance(engine, str):

        if engine.lower() == 'analytic':
            return AnalyticTwoPointCounter

        if engine.lower() == 'corrfunc':
            from .corrfunc import CorrfuncTwoPointCounter
            return CorrfuncTwoPointCounter

        raise TwoPointCounterError('Unknown engine {}.'.format(engine))

    return engine


class MetaTwoPointCounter(type(BaseClass)):

    """Metaclass to return correct two-point counter engine."""

    def __call__(cls, *args, engine='corrfunc', **kwargs):
        return get_twopoint_counter(engine)(*args, **kwargs)


class TwoPointCounter(BaseClass, metaclass=MetaTwoPointCounter):
    """
    Entry point to two-point counter engines.

    Parameters
    ----------
    engine : string, default='corrfunc'
        Name of two-point counter engine, one of ["corrfunc", "analytical"].

    args : list
        Arguments for two-point counter engine, see :class:`BaseTwoPointCounter`.

    kwargs : dict
        Arguments for two-point counter engine, see :class:`BaseTwoPointCounter`.

    Returns
    -------
    engine : BaseTwoPointCounter
    """
    @classmethod
    def from_state(cls, state):
        """Return new two point counter based on state dictionary."""
        state = state.copy()
        cls = get_twopoint_counter(state.pop('name'))
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new


def _vlogical_and(*arrays):
    # & between any number of arrays
    toret = arrays[0].copy()
    for array in arrays[1:]: toret &= array
    return toret


def _nan_to_zero(array):
    # Replace nans with 0s
    array = array.copy()
    mask = np.isnan(array)
    array[mask] = 0.
    return array


def get_inverse_probability_weight(*weights, noffset=1, nrealizations=None, default_value=0., dtype='f8'):
    r"""
    Return inverse probability weight given input bitwise weights.
    Inverse probability weight is computed as::math:`\mathrm{nrealizations}/(\mathrm{noffset} + \mathrm{popcount}(w_{1} \& w_{2} \& ...))`.
    If denominator is 0, weight is set to default_value.

    Parameters
    ----------
    weights : int arrays
        Bitwise weights.

    noffset : int, default=1
        The offset to be added to the bitwise counts in the denominator (defaults to 1).

    nrealizations : int, default=None
        Number of realizations (defaults to the number of bits in input weights plus one).

    default_value : float, default=0.
        Default weight value, if the denominator is zero (defaults to 0).

    dtype : string, np.dtype
        Type for output weight.

    Returns
    -------
    weight : array
        IIP weight.
    """
    if nrealizations is None:
        nrealizations = get_default_nrealizations(weights[0])
    #denom = noffset + sum(utils.popcount(w1 & w2) for w1, w2 in zip(*weights))
    denom = noffset + sum(utils.popcount(_vlogical_and(*weight)) for weight in zip(*weights))
    mask = denom == 0
    denom[mask] = 1
    toret = np.empty_like(denom, dtype=dtype)
    toret[...] = nrealizations/denom
    toret[mask] = default_value
    return toret


def _format_positions(positions, mode='auto', position_type='xyz', dtype=None, mpicomm=None, mpiroot=None):
    # Format input array of positions
    # position_type in ["xyz", "rdd", "pos"]
    mode = mode.lower()
    position_type = position_type.lower()
    if position_type == 'auto':
        if mode in ['theta', 'angular']: position_type = 'rd'
        else: position_type = 'xyz'

    def __format_positions(positions, position_type=position_type, dtype=dtype):
        if position_type == 'pos': # array of shape (N, 3)
            positions = np.asarray(positions, dtype=dtype)
            if positions.shape[-1] != 3:
                return None, 'For position type = {}, please provide a (N, 3) array for positions'.format(position_type)
            position_type = 'xyz'
            positions = positions.T
        # Array of shape (3, N)
        for ip, p in enumerate(positions):
            # Cast to the input dtype if exists (may be set by previous weights)
            positions[ip] = np.asarray(p, dtype=dtype)

        size = len(positions[0])
        dtype = positions[0].dtype
        if not np.issubdtype(dtype, np.floating):
            return None, 'Input position arrays should be of floating type, not {}'.format(dtype)
        for p in positions[1:]:
            if len(p) != size:
                return None, 'All position arrays should be of the same size'
            if p.dtype != dtype:
                return None, 'All position arrays should be of the same type, you can e.g. provide dtype'
        if position_type != 'auto' and len(positions) != len(position_type):
            return None, 'For position type = {}, please provide a list of {:d} arrays for positions (found {:d})'.format(position_type, len(position_type), len(positions))

        if mode in ['theta', 'angular']:
            if position_type == 'xyz':
                positions = utils.cartesian_to_sky(positions, degree=True)[:2]
            elif position_type in ['rdd', 'rdz']:
                positions = list(positions)[:2]
            elif position_type != 'rd':
                return None, 'For mode = {}, position type should be one of ["xyz", "rdz", "rd"]'.format(mode)
        else:
            if position_type == 'rdd':
                positions = utils.sky_to_cartesian(positions, degree=True)
            elif position_type != 'xyz':
                return None, 'For mode = {}, position type should be one of ["pos", "xyz", "rdd"]'.format(mode)
        return positions, None

    error = None
    if positions is not None and (position_type == 'pos' or not all(position is None for position in positions)):
        positions, error = __format_positions(positions) # return error separately to raise on all processes
    if mpicomm is not None:
        error = mpicomm.allgather(error)
    else:
        error = [error]
    errors = [err for err in error if err is not None]
    if errors:
        raise TwoPointCounterError(errors[0])
    if mpiroot is not None and mpicomm.bcast(positions is not None if mpicomm.rank == mpiroot else None, root=mpiroot):
        n = mpicomm.bcast(len(positions) if mpicomm.rank == mpiroot else None, root=mpiroot)
        if mpicomm.rank != mpiroot: positions = [None]*n
        positions = [get_mpi().scatter_array(position, mpicomm=mpicomm, root=mpiroot) for position in positions]
    return positions


def _format_weights(weights, weight_type='auto', size=None, dtype=None, mpicomm=None, mpiroot=None):
    # Format input weights, as a list of n_bitwise_weights uint8 arrays, and optionally a float array for individual weights.
    # Return formated list of weights, and n_bitwise_weights.

    def __format_weights(weights, weight_type=weight_type, dtype=dtype):
        if weights is None or all(weight is None for weight in weights):
            return [], 0
        if np.ndim(weights[0]) == 0:
            weights = [weights]
        individual_weights = []
        bitwise_weights = []
        for w in weights:
            if np.issubdtype(w.dtype, np.integer):
                if weight_type == 'product_individual': # enforce float individual weight
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
            weights += [np.prod(individual_weights, axis=0, dtype=dtype)]
        return weights, n_bitwise_weights

    weights, n_bitwise_weights = __format_weights(weights)
    if mpiroot is not None and mpicomm.bcast(weights is not None if mpicomm.rank == mpiroot else None, root=mpiroot):
        n = mpicomm.bcast(len(weights) if mpicomm.rank == mpiroot else None, root=mpiroot)
        if mpicomm.rank != mpiroot: weights = [None]*n
        weights = [get_mpi().scatter_array(weight, mpicomm=mpicomm, root=mpiroot) for weight in weights]
        n_bitwise_weights = mpicomm.bcast(n_bitwise_weights, root=mpiroot)

    if size is not None:
        if not all(len(weight) == size for weight in weights):
            raise TwoPointCounterError('All weight arrays should be of the same size as position arrays')
    return weights, n_bitwise_weights


class BaseTwoPointCounter(BaseClass):
    """
    Base class for two-point counters.
    Extend this class to implement a new two-point counter engine.

    Attributes
    ----------
    wcounts : array
        (Optionally weighted) two-point counts.

    wnorm : float
        Two-point count normalization.
    """
    def __init__(self, mode, edges, positions1, positions2=None, weights1=None, weights2=None,
                 bin_type='auto', position_type='auto', weight_type='auto', weight_attrs=None,
                 twopoint_weights=None, los='midpoint', boxsize=None, compute_sepavg=True, dtype=None,
                 nthreads=None, mpicomm=None, mpiroot=None, **kwargs):
        r"""
        Initialize :class:`BaseTwoPointCounter`, and run actual two-point counts
        (calling :meth:`run`), setting :attr:`wcounts` and :attr:`sep`.

        Parameters
        ----------
        mode : string
            Type of two-point counts, one of:

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
            See ``position_type``.

        positions2 : list, array, default=None
            Optionally, for cross-two-point counts, positions in the second catalog. See ``positions1``.

        weights1 : array, list, default=None
            Weights of the first catalog. Not required if ``weight_type`` is either ``None`` or "auto".
            See ``weight_type``.

        weights2 : array, list, default=None
            Optionally, for cross-two-point counts, weights in the second catalog. See ``weights1``.

        bin_type : string, default='auto'
            Binning type for first dimension, e.g. :math:`r_{p}` when ``mode`` is "rppi".
            Set to ``lin`` for speed-up in case of linearly-spaced bins.
            In this case, the bin number for a pair separated by a (3D, projected, angular...) separation
            ``sep`` is given by ``(sep - edges[0])/(edges[-1] - edges[0])*(len(edges) - 1)``,
            i.e. only the first and last bins of input edges are considered.
            Then setting ``compute_sepavg`` is virtually costless.
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
                - "xyz": Cartesian positions, shape (3, N)
                - "pos": Cartesian positions, shape (N, 3).

        weight_type : string, default='auto'
            The type of weighting to apply to provided weights. One of:

                - ``None``: no weights are applied.
                - "product_individual": each pair is weighted by the product of weights :math:`w_{1} w_{2}`.
                - "inverse_bitwise": each pair is weighted by :math:`\mathrm{nrealizations}/(\mathrm{noffset} + \mathrm{popcount}(w_{1} \& w_{2}))`.
                   Multiple bitwise weights can be provided as a list.
                   Individual weights can additionally be provided as float arrays.
                   In case of cross-correlations with floating weights, bitwise weights are automatically turned to IIP weights,
                   i.e. :math:`\mathrm{nrealizations}/(\mathrm{noffset} + \mathrm{popcount}(w_{1}))`.
                - "auto": automatically choose weighting based on input ``weights1`` and ``weights2``,
                   i.e. ``None`` when ``weights1`` and ``weights2`` are ``None``,
                   "inverse_bitwise" if one of input weights is integer, else "product_individual".

            In addition, angular upweights can be provided with ``twopoint_weights``.

        weight_attrs : dict, default=None
            Dictionary of weighting scheme attributes. In case ``weight_type`` is "inverse_bitwise",
            one can provide "nrealizations", the total number of realizations (*including* current one;
            defaulting to the number of bits in input weights plus one);
            "noffset", the offset to be added to the bitwise counts in the denominator (defaulting to 1)
            and "default_value", the default value of pairwise weights if the denominator is zero (defaulting to 0).
            One can also provide "nalways", stating the number of bits systematically set to 1 (defaulting to 0),
            and "nnever", stating the number of bits systematically set to 0 (defaulting to 0).
            These will only impact the normalization factors.
            For example, for the "zero-truncated" estimator (arXiv:1912.08803), one would use noffset = 0, nalways = 1, nnever = 0.

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

        compute_sepavg : bool, default=True
            Set to ``False`` to *not* calculate the average separation for each bin.
            This can make the two-point counts faster if ``bin_type`` is "custom".
            In this case, :attr:`sep` will be set the midpoint of input edges.

        dtype : string, np.dtype, default=None
            Array type for positions and weights.
            If ``None``, defaults to type of first ``positions1`` array.

        nthreads : int, default=None
            Number of OpenMP threads to use.

        mpicomm : MPI communicator, default=None
            The MPI communicator, to MPI-distribute calculation.

        mpiroot : int, default=None
            In case ``mpicomm`` is provided, if ``None``, input positions and weights are assumed to be scattered across all ranks.
            Else the MPI rank where input positions and weights are gathered.

        kwargs : dict
            Two-point counter engine-specific options.
        """
        self.mode = mode.lower()
        self.nthreads = nthreads
        if nthreads is None:
            self.nthreads = int(os.getenv('OMP_NUM_THREADS','1'))
        self.mpicomm = mpicomm
        if self.mpicomm is None and mpiroot is not None:
            raise TwoPointCounterError('mpiroot is not None, but no mpicomm provided')
        self._set_positions(positions1, positions2, position_type=position_type, dtype=dtype, mpiroot=mpiroot)
        self._set_weights(weights1, weights2, weight_type=weight_type, twopoint_weights=twopoint_weights, weight_attrs=weight_attrs, mpiroot=mpiroot)
        self._set_edges(edges, bin_type=bin_type)
        self._set_boxsize(boxsize)
        self._set_los(los)
        self.compute_sepavg = compute_sepavg
        self.attrs = kwargs
        self.wnorm = self.normalization()
        self._set_zeros()
        if self.size1 * self.size2:
            self.run()
        del self.positions1, self.positions2, self.weights1, self.weights2

    def run(self):
        """
        Method that computes the actual two-point counts and set :attr:`wcounts` and :attr:`sep`,
        to be implemented in your new engine.
        """
        raise NotImplementedError('Implement method "run" in your {}'.format(self.__class__.__name__))

    def _set_zeros(self):
        self._set_default_separation()
        self.wcounts = np.zeros_like(self.sep)
        self.ncounts = np.zeros_like(self.sep, dtype='i8')

    def _set_edges(self, edges, bin_type='auto'):
        if np.ndim(edges[0]) == 0:
            edges = (edges,)
        self.edges = [np.array(edge, dtype='f8') for edge in edges]
        if self.mode in ['smu','rppi']:
            if not self.ndim == 2:
                raise TwoPointCounterError('A tuple of edges should be provided to two-point counter in mode {}'.format(self.mode))
        else:
            if not self.ndim == 1:
                raise TwoPointCounterError('Only one edge array should be provided to two-point counter in mode {}'.format(self.mode))
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
        """Return shape of obtained counts :attr:`wcounts`."""
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

    def _set_positions(self, positions1, positions2=None, position_type='auto', dtype=None, mpiroot=None):
        self.positions1 = _format_positions(positions1, mode=self.mode, position_type=position_type, dtype=dtype, mpicomm=self.mpicomm, mpiroot=mpiroot)
        self.positions2 = _format_positions(positions2, mode=self.mode, position_type=position_type, dtype=dtype, mpicomm=self.mpicomm, mpiroot=mpiroot)
        self.autocorr = self.positions2 is None
        self.dtype = self.positions1[0].dtype

        self.size1 = self.size2 = len(self.positions1[0])
        if not self.autocorr: self.size2 = len(self.positions2[0])
        if self.with_mpi:
            self.size1, self.size2 = self.mpicomm.allreduce(self.size1), self.mpicomm.allreduce(self.size2)

    def _set_weights(self, weights1, weights2=None, weight_type='auto', twopoint_weights=None, weight_attrs=None, mpiroot=None):

        if weight_type is not None: weight_type = weight_type.lower()
        allowed_weight_types = [None, 'auto', 'product_individual', 'inverse_bitwise']
        if weight_type not in allowed_weight_types:
            raise TwoPointCounterError('weight_type should be one of {}'.format(allowed_weight_types))
        self.weight_type = weight_type

        weight_attrs = weight_attrs or {}
        self.weight_attrs = {}

        self.n_bitwise_weights = 0
        if weight_type is None:
            self.weights1 = self.weights2 = []

        else:

            self.weight_attrs.update(nalways=weight_attrs.get('nalways', 0), nnever=weight_attrs.get('nnever', 0))
            noffset = weight_attrs.get('noffset', 1)
            default_value = weight_attrs.get('default_value', 0.)
            self.weight_attrs.update(noffset=noffset, default_value=default_value)

            self.weights1, n_bitwise_weights1 = _format_weights(weights1, weight_type=self.weight_type, size=len(self.positions1[0]), dtype=self.dtype, mpicomm=self.mpicomm, mpiroot=mpiroot)

            def get_nrealizations(n_bitwise_weights):
                nrealizations = weight_attrs.get('nrealizations', None)
                if nrealizations is None:
                    nrealizations = n_bitwise_weights * 8 + 1
                return nrealizations

            if self.autocorr:

                nrealizations = get_nrealizations(n_bitwise_weights1)
                self.weight_attrs.update(nrealizations=nrealizations)
                self.weights2 = self.weights1
                self.n_bitwise_weights = n_bitwise_weights1

            else:
                self.weights2, n_bitwise_weights2 = _format_weights(weights2, weight_type=self.weight_type, size=len(self.positions2[0]), dtype=self.dtype, mpicomm=self.mpicomm, mpiroot=mpiroot)

                if n_bitwise_weights2 == n_bitwise_weights1:

                    nrealizations = get_nrealizations(n_bitwise_weights1)
                    self.n_bitwise_weights = n_bitwise_weights1

                else:

                    if n_bitwise_weights2 == 0:
                        indweights = self.weights1[n_bitwise_weights1] if len(self.weights1) > n_bitwise_weights1 else 1.
                        nrealizations = get_nrealizations(n_bitwise_weights1)
                        self.weights1 = [get_inverse_probability_weight(self.weights1[:n_bitwise_weights1], nrealizations=nrealizations,
                                                                        noffset=noffset, default_value=default_value, dtype=self.dtype)*indweights]
                        self.n_bitwise_weights = 0
                        self.log_info('Setting IIP weights for first catalog.')
                    elif n_bitwise_weights1 == 0:
                        indweights = self.weights2[n_bitwise_weights2] if len(self.weights2) > n_bitwise_weights2 else 1.
                        nrealizations = get_nrealizations(n_bitwise_weights2)
                        self.weights2 = [get_inverse_probability_weight(self.weights2[:n_bitwise_weights2], nrealizations=nrealizations,
                                                                        noffset=noffset, default_value=default_value, dtype=self.dtype)*indweights]
                        self.n_bitwise_weights = 0
                        self.log_info('Setting IIP weights for second catalog.')
                    else:
                        raise TwoPointCounterError('Incompatible length of bitwise weights: {:d} and {:d} bytes'.format(n_bitwise_weights1, n_bitwise_weights2))

                self.weight_attrs.update(nrealizations=nrealizations)

        if len(self.weights1) == len(self.weights2) + 1:
            self.weights2.append(np.ones(len(self.positions1), dtype=self.dtype))
        elif len(self.weights1) == len(self.weights2) - 1:
            self.weights1.append(np.ones(len(self.positions2), dtype=self.dtype))
        elif len(self.weights1) != len(self.weights2):
            raise ValueError('Something fishy happened with weights; number of weights1/weights2 is {:d}/{:d}'.format(len(self.weights1),len(self.weights2)))

        self.twopoint_weights = twopoint_weights
        self.cos_twopoint_weights = None
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
            self.cos_twopoint_weights = TwoPointWeight(sep=np.cos(np.radians(sep[::-1]), dtype=self.dtype),
                                                       weight=np.array(weight[::-1], dtype=self.dtype))

    def _mpi_decompose(self):
        if self.with_mpi:
            smoothing = np.max(self.edges[0])
            if self.mode == 'theta':
                smoothing = 2 * np.sin(0.5 * np.deg2rad(smoothing))
            elif self.mode == 'rppi':
                smoothing = np.sqrt(smoothing**2 + np.max(self.edges[1])**2)
            elif self.mode == 'rp':
                smoothing = np.inf
            return get_mpi().domain_decompose(self.mpicomm, smoothing, self.positions1, weights1=self.weights1,
                                              positions2=self.positions2, weights2=self.weights2, boxsize=self.boxsize)
        return (self.positions1, self.weights1), (self.positions2, self.weights2)

    def _set_default_separation(self):
        mid = [(edges[1:] + edges[:-1])/2. for edges in self.edges]
        self.seps = list(np.meshgrid(*mid, indexing='ij'))

    def _set_los(self, los):
        self.los = los.lower()
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
            self.boxsize = _make_array(boxsize, 3, dtype='f8')

    def normalization(self):
        r"""
        Return two-point count normalization, i.e., in case of cross-correlation:

        .. math::

            \left(\sum_{i=1}^{N_{1}} w_{1,i}\right) \left(\sum_{j=1}^{N_{2}} w_{2,j}\right)

        with the sums running over the weights of the first and second catalogs, and in case of auto-correlation:

        .. math::

            \left(\sum_{i=1}^{N_{1}} w_{1,i}\right)^{2} - \sum_{i=1}^{N_{1}} w_{1,i}^{2}

        """
        if not self.weights1:

            if self.autocorr:
                return self.size1 * (self.size1 - 1)
            return self.size1 * self.size2

        if self.n_bitwise_weights:

            noffset = self.weight_attrs['noffset']
            nrealizations = self.weight_attrs['nrealizations']
            nalways = self.weight_attrs['nalways'] - self.weight_attrs['nnever']

            def binned_weights(weights, pow=1):
                indweights = weights[self.n_bitwise_weights:]
                if indweights: indweights = np.prod(indweights, axis=0)**pow
                else: indweights = None
                w = np.bincount(utils.popcount(*weights[:self.n_bitwise_weights]),
                                weights=indweights, minlength=self.n_bitwise_weights * 8 + 1)
                if self.with_mpi:
                    w = self.mpicomm.allreduce(w)
                c = np.flatnonzero(w)
                return w[c], c

            w1, c1 = binned_weights(self.weights1)
            if self.autocorr:
                w2, c2 = w1, c1
            else:
                w2, c2 = binned_weights(self.weights2)
            joint = utils.joint_occurences(nrealizations, max_occurences=noffset+max(c1.max(), c2.max()), noffset=noffset + nalways)
            sumw_cross = 0
            for c1_, w1_ in zip(c1, w1):
                for c2_, w2_ in zip(c2, w2):
                    sumw_cross += w1_ * w2_ * (joint[c1_ - nalways][c2_ - nalways] if c2_ <= c1_ else joint[c2_- nalways][c1_- nalways])
            sumw_auto = 0
            if self.autocorr:
                w1sq, c1sq = binned_weights(self.weights1, pow=2)
                for c1sq_, w1sq_ in zip(c1sq, w1sq):
                    sumw_auto += joint[c1sq_ - nalways][c1sq_ - nalways] * w1sq_
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

    @property
    def sep(self):
        """Array of separation values of first dimension (e.g. :math:`s` if :attr:`mode` is "smu")."""
        return self.seps[0]

    @sep.setter
    def sep(self, sep):
        self.seps[0] = sep

    def normalized_wcounts(self):
        """Return normalized two-point counts, i.e. :attr:`wcounts` divided by :meth:`normalization`."""
        return self.wcounts/self.wnorm

    def rebin(self, factor=1):
        """
        Rebin two-point counts, by factor(s) ``factor``.
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
        if hasattr(self, 'ncounts'):
            self.ncounts = utils.rebin(self.ncounts, new_shape, statistic=np.sum)
        self.edges = [edges[::f] for edges, f in zip(self.edges, factor)]
        if self.compute_sepavg:
            self.seps = [utils.rebin(_nan_to_zero(sep)*wcounts, new_shape, statistic=np.sum)/self.wcounts for sep in self.seps]
        else:
            self._set_default_separation()

    def __getstate__(self):
        state = {}
        for name in ['name', 'autocorr', 'seps', 'ncounts', 'wcounts', 'wnorm', 'size1', 'size2', 'edges', 'mode', 'bin_type',
                     'boxsize', 'los', 'compute_sepavg', 'weight_attrs', 'attrs']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def save(self, filename):
        """Save two-point counts to ``filename``."""
        if not self.with_mpi or self.mpicomm.rank == 0:
            super(BaseTwoPointCounter, self).save(filename)
        if self.with_mpi:
            self.mpicomm.Barrier()


class AnalyticTwoPointCounter(BaseTwoPointCounter):
    """
    Analytic two-point counter. Assume periodic wrapping and no data weights.

    Attributes
    ----------
    wcounts : array
        Analytical two-point counts.
    """
    name = 'analytic'

    def __init__(self, mode, edges, boxsize, size1=10, size2=None, los='z'):
        r"""
        Initialize :class:`AnalyticTwoPointCounter`, and set :attr:`wcounts` and :attr:`sep`.

        Parameters
        ----------
        mode : string
            Two-point counting mode, one of:

                - "s": two-point counts as a function of distance between two particles
                - "smu": two-point counts as a function of distance between two particles and cosine angle :math:`\mu` w.r.t. the line-of-sight
                - "rppi": two-point counts as a function of distance transverse (:math:`r_{p}`) and parallel (:math:`\pi`) to the line-of-sight
                - "rp": same as "rppi", without binning in :math:`\pi`

        edges : tuple, array
            Tuple of bin edges (arrays), for the first (e.g. :math:`r_{p}`)
            and optionally second (e.g. :math:`\pi`) dimensions.
            In case of single-dimension binning (e.g. ``mode`` is "theta", "s" or "rp"),
            the single array of bin edges can be provided directly.

        boxsize : array, float
            The side-length(s) of the periodic cube.

        size1 : int, default=2
            Length of the first catalog.

        size2 : int, default=None
            Optionally, for cross-two-point counts, length of second catalog.

        los : string, default='z'
            Line-of-sight to be used when ``mode`` is "rp", in case of non-cubic box;
            one of cartesian axes "x", "y" or "z".
        """
        self.mode = mode.lower()
        self._set_edges(edges)
        self._set_boxsize(boxsize)
        self._set_los(los)
        self.size1 = size1
        self.size2 = size2
        self.autocorr = size2 is None
        self.compute_sepavg = False
        self._set_default_separation()
        self.run()
        self.wnorm = self.normalization()

    def run(self):
        """Set analytical two-point counts."""
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
            raise TwoPointCounterError('No analytic randoms provided for mode {}'.format(self.mode))
        self.wcounts = self.normalization()*dv/self.boxsize.prod()

    def normalization(self):
        """
        Return two-point count normalization, i.e., in case of cross-correlation ``size1 * size2``,
        and in case of auto-correlation ``size1 * (size1 - 1)``.
        """
        if self.autocorr:
            return self.size1 * (self.size1 - 1)
        return self.size1 * self.size2
