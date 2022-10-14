"""
Implements methods to perform jackknife estimates of the correlation function covariance matrix:

- subsamplers, to split footprint into subregions
- two-point counters, to run jackknife two-point realizations
- two-point estimators, using the jackknife realizations to estimate a covariance matrix.
"""

import numpy as np

from .utils import BaseClass, get_mpi, TaskManager, _get_box, _make_array, _nan_to_zero
from .twopoint_counter import BaseTwoPointCounter, TwoPointCounter, TwoPointCounterError, _format_positions
from .twopoint_estimator import BaseTwoPointEstimator, TwoPointEstimatorError
from . import utils


class BaseSubsampler(BaseClass):
    """
    Base class for subsamplers. Extend this class to implement a new subsampler;
    in particular, one should implement :meth:`BaseSubsampler.run` and :meth:`BaseSubsampler.label`,
    which provides a subsample label to input positions.
    """
    def __init__(self, mode, positions, weights=None, nsamples=8, position_type='auto', dtype=None, mpicomm=None, mpiroot=None, **kwargs):
        """
        Initialize :class:`BaseSubsampler`.

        Parameters
        ----------
        mode : How to divide space, one of:

            - "angular": on the sky
            - "3d": in Cartesian, 3D space

        positions : list, array
            Positions of (typically randoms) particles to define subsamples.
            Typically of shape (3, N), but can be (2, N) when ``mode`` is "angular".
            See ``position_type``.

        weights : array, default=None
            Optionally, weights of (typically randoms) particles to define subsamples.

        nsamples : int, default=8
            Number of subsamples to define.

        position_type : string, default='auto'
            Type of input positions, one of:

                - "rd": RA/Dec in degree, only if ``mode`` is "angular"
                - "rdd": RA/Dec in degree, distance, for any ``mode``
                - "xyz": Cartesian positions, shape (3, N)
                - "pos": Cartesian positions, shape (N, 3).

        dtype : string, np.dtype, default=None
            Array type for positions and weights.
            If ``None``, defaults to type of ``positions`` array.

        mpicomm : MPI communicator, default=None
            The MPI communicator, to MPI-distribute calculation.

        mpiroot : int, default=None
            In case ``mpicomm`` is provided, if ``None``, input positions and weights are assumed to be scattered across all ranks.
            Else the MPI rank where input positions and weights are gathered.

        kwargs : dict
            Subsampler engine-specific options.
        """
        self.mpicomm = mpicomm
        self.mode = mode.lower()
        self.position_type = position_type.lower()
        self.dtype = dtype
        self.positions = self._format_positions(positions, copy=False, mpicomm=self.mpicomm, mpiroot=mpiroot)
        self.weights = weights
        if weights is not None: self.weights = np.array(weights, dtype=self.dtype, copy=False)
        self.nsamples = nsamples
        self.attrs = kwargs
        self.run()

    @property
    def with_mpi(self):
        """Whether to use MPI."""
        if not hasattr(self, 'mpicomm'): self.mpicomm = None
        return self.mpicomm is not None and self.mpicomm.size > 1

    def _format_positions(self, positions, position_type=None, dtype=None, copy=False, mpicomm=None, mpiroot=None):
        # Format input positions
        position_type = self.position_type if position_type is None else position_type.lower()
        dtype = self.dtype if dtype is None else dtype
        positions = _format_positions(positions, position_type=position_type, mode=self.mode, dtype=dtype, copy=False, mpicomm=mpicomm, mpiroot=mpiroot)
        if self.mode == 'angular':
            positions = utils.sky_to_cartesian(positions + [1.], degree=True, dtype=positions[0].dtype)  # project onto unit sphere
        return positions

    def label(self, positions, position_type=None):
        """Method that returns subsample labels corresponding to input positions."""
        raise NotImplementedError('Implement method "label" in your {}'.format(self.__class__.__name__))


class BoxSubsampler(BaseSubsampler):

    """Basic subsampler, that divides a box into subboxes in 3D space."""

    def __init__(self, positions=None, boxsize=None, boxcenter=None, nsamples=8, position_type='auto', dtype=None, mpicomm=None, mpiroot=None):
        """
        Initialize :class:`BoxSubsampler`.

        Parameters
        ----------
        positions : list, array
            If ``boxsize`` and / or ``boxcenter`` is ``None``, use these positions
            to determine ``boxsize`` and / or ``boxcenter``.
            Typically of shape (3, N), see ``position_type``.

        boxsize : array, float, default=None
            Physical size of the box.
            If not provided, see ``positions``.

        boxcenter : array, float, default=None
            Box center.
            If not provided, see ``positions``.

        nsamples : int, tuple, default=8
            Total number of subsamples to define.
            Can be a 3-tuple, corresponding to the number of divisions along each x, y, z axis.

        position_type : string, default='auto'
            Type of input positions, one of:

                - "rdd": RA/Dec in degree, distance, shape (3, N)
                - "xyz": Cartesian positions, shape (3, N)
                - "pos": Cartesian positions, shape (N, 3).

        dtype : string, np.dtype, default=None
            Array type for positions and weights.
            If ``None``, defaults to type of ``positions`` array.

        mpicomm : MPI communicator, default=None
            The MPI communicator, to MPI-distribute calculation.

        mpiroot : int, default=None
            In case ``mpicomm`` is provided, if ``None``, input positions are assumed to be scattered across all ranks.
            Else the MPI rank where input positions are gathered.
        """
        ndim = 3
        self.mpicomm = mpicomm
        self.mode = '3d'
        self.position_type = position_type.lower()
        self.dtype = dtype

        if boxsize is None or boxcenter is None:
            if positions is None:
                raise ValueError('positions must be provided if boxsize or boxcenter is not provided')
            positions = self._format_positions(positions, copy=False, mpicomm=self.mpicomm, mpiroot=mpiroot)
            posmin, posmax = _get_box(positions)
            if self.with_mpi:
                posmin = np.min(mpicomm.allgather(posmin), axis=0)
                posmax = np.max(mpicomm.allgather(posmax), axis=0)
            if boxsize is None:
                boxsize = (posmax - posmin) * (1. + 1e-9)
            if boxcenter is None:
                boxcenter = (posmin + posmax) / 2.

        self.boxsize = _make_array(boxsize, ndim, dtype='f8')
        self.boxcenter = _make_array(boxcenter, ndim, dtype='f8')
        if isinstance(nsamples, (list, tuple)):
            if len(nsamples) != ndim:
                raise ValueError('nsamples must be a list/tuple of size {:d}'.format(ndim))
            self.nsamples = tuple(nsamples)
        else:
            nsamples = int(nsamples)
            self.nsamples = (int(nsamples**(1. / ndim) + 0.5),) * ndim
            if nsamples != np.prod(self.nsamples):
                raise ValueError('Number of regions must be a power of {:d}'.format(ndim))

        self.run()

    def run(self):
        """Set edges for binning along each axis."""
        offset = self.boxcenter - self.boxsize / 2.
        self.edges = [o + np.linspace(0, b, n) for o, b, n in zip(offset, self.boxsize, self.nsamples)]

    def label(self, positions, position_type=None):
        """
        Return subsample labels given input positions.

        Parameters
        ----------
        positions : list, array
            Positions to which labels will be attributed.
            Typically of shape (3, N), see ``position_type``.

        position_type : string, default='auto'
            Type of input positions, one of:

                - "rdd": RA/Dec in degree, distance, shape (3, N)
                - "xyz": Cartesian positions, shape (3, N)
                - "pos": Cartesian positions, shape (N, 3).

        Returns
        -------
        labels : array
            Labels corresponding to input ``positions``.
        """
        positions = self._format_positions(positions, position_type=position_type, copy=False, mpicomm=None, mpiroot=None)
        ii = []
        for edge, p in zip(self.edges, positions):
            tmp = np.searchsorted(edge, p, side='right', sorter=None) - 1
            if not np.all((tmp >= 0) & (tmp < len(edge) - 1)):
                raise ValueError('Some input positions outside of bounding box')
            ii.append(tmp)
        return np.ravel_multi_index(tuple(ii), tuple(len(edge) - 1 for edge in self.edges), mode='raise', order='C')


class KMeansSubsampler(BaseSubsampler):

    """Subsampler using k-means scikit-learn algorithm to group particles together."""

    def __init__(self, mode, positions, nside=None, random_state=None, **kwargs):
        """
        Initialize :class:`KMeansSubsampler`.

        Parameters
        ----------
        mode : How to divide space, one of:

            - "angular": on the sky
            - "3d": in Cartesian, 3D space

        positions : list, array
            Positions of (typically randoms) particles to define subsamples.
            Typically of shape (3, N), but can be (2, N) when ``mode`` is "angular".

        nside : int, default=None
            Only if mode is "angular".
            If not ``None``, Healpix ``nside`` to pixelate input positions and weights.
            Smaller ``nside`` allows faster runtime, but coarser angular binning.
            If ``None``, no Healpix pixelation is performed.

        random_state : int, np.random.RandomState instance, default=None
            Determines random number generation for centroid initialization.

        kwargs : dict
            Other arguments, see :class:`BaseSubsampler`.
            One can also provide arguments for :class:`sklearn.cluster.KMeans`.
        """
        mode = mode.lower()
        self.nside = nside
        if self.nside is not None and mode != 'angular':
            raise ValueError('Healpix (nside = {:d}) can only be used with mode == angular'.format(self.nside))
        self.nest = False
        self.random_state = random_state
        super(KMeansSubsampler, self).__init__(mode, positions, **kwargs)

    def run(self):
        """Set :attr:`kmeans` instance to group particles together."""
        from sklearn import cluster
        if self.nside is not None:
            self.nside = int(self.nside)
            import healpy as hp
            pix = hp.vec2pix(self.nside, *self.positions, nest=self.nest)
            weights = w = np.bincount(pix, weights=self.weights, minlength=hp.nside2npix(self.nside))
            if self.with_mpi:
                weights = np.empty_like(weights)
                self.mpicomm.Allreduce(w, weights)
            pix = np.flatnonzero(weights)
            weights = weights[pix]
            positions = np.asarray(hp.pix2vec(self.nside, pix, nest=self.nest)).T
        else:
            positions, weights = np.asarray(self.positions).T, self.weights
            if self.with_mpi:
                positions = get_mpi().gather_array(positions, root=None, mpicomm=self.mpicomm)  # WARNING: bcast on all ranks
                if weights is not None:
                    weights = get_mpi().gather_array(weights, root=None, mpicomm=self.mpicomm)
        self.kmeans = cluster.KMeans(n_clusters=self.nsamples, random_state=self.random_state, **self.attrs)
        self.kmeans.fit(positions, sample_weight=weights)

    def label(self, positions, position_type=None):
        """
        Return subsample labels given input positions.

        Parameters
        ----------
        positions : list, array
            Positions of (typically randoms) particles to define subsamples.
            Typically of shape (3, N), but can be (2, N) when ``mode`` is "angular".
            See ``position_type``.

        position_type : string, default='auto'
            Type of input positions, one of:

                - "rd": RA/Dec in degree, only if ``mode`` is "angular"
                - "rdd": RA/Dec in degree, distance, for any ``mode``
                - "xyz": Cartesian positions, shape (3, N)
                - "pos": Cartesian positions, shape (N, 3).

        Returns
        -------
        labels : array
            Labels corresponding to input ``positions``.
        """
        positions = self._format_positions(positions, position_type=position_type, copy=False, mpicomm=None, mpiroot=None)
        if self.nside is not None:
            import healpy as hp
            pix = hp.vec2pix(self.nside, *positions, nest=self.nest)
            positions = hp.pix2vec(self.nside, pix, nest=self.nest)
        return self.kmeans.predict(np.asarray(positions).T)


class JackknifeTwoPointCounter(BaseTwoPointCounter):

    """Perform jackknife two-point counts."""

    name = 'jackknife'
    _result_names = ['auto', 'cross12', 'cross21']

    def __init__(self, mode, edges, positions1, samples1, weights1=None, positions2=None, samples2=None, weights2=None,
                 bin_type='auto', position_type='auto', weight_type='auto', weight_attrs=None,
                 twopoint_weights=None, los='midpoint', boxsize=None, compute_sepsavg=True, dtype=None,
                 nthreads=None, mpicomm=None, mpiroot=None, nprocs_per_real=1, samples=None, **kwargs):
        r"""
        Initialize :class:`JackknifeTwoPointCounter`.

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
            and optionally second (e.g. :math:`\pi > 0`, :math:`\mu \in [-1, 1]`) dimensions.
            In case of single-dimension binning (e.g. ``mode`` is "theta", "s" or "rp"),
            the single array of bin edges can be provided directly.
            Edges are inclusive on the low end, exclusive on the high end,
            i.e. a pair separated by :math:`s` falls in bin `i` if ``edges[i] <= s < edges[i+1]``.
            In case ``mode`` is "smu" however, the first :math:`\mu`-bin is exclusive on the low end
            (increase the :math:`\mu`-range by a tiny value to include :math:`\mu = \pm 1`).
            Pairs at separation :math:`s = 0` are included in the :math:`\mu = 0` bin.
            In case of auto-correlation (no ``positions2`` provided), auto-pairs (pairs of same objects) are not counted.
            In case of cross-correlation, all pairs are counted.
            In any case, duplicate objects (with separation zero) will be counted.

        positions1 : list, array
            Positions in the first catalog. Typically of shape (3, N), but can be (2, N) when ``mode`` is "theta".
            See ``position_type``.

        samples1 : array
            Labels of subsamples for the first catalog.

        weights1 : array, list, default=None
            Weights of the first catalog. Not required if ``weight_type`` is either ``None`` or "auto".
            See ``weight_type``.

        positions2 : list, array, default=None
            Optionally, for cross-two-point counts, positions in the second catalog. See ``positions1``.

        samples2 : list, array, default=None
            Optionally, for cross-two-point counts, labels in the second catalog. See ``samples1``.

        weights2 : array, list, default=None
            Optionally, for cross-two-point counts, weights in the second catalog. See ``weights1``.

        bin_type : string, default='auto'
            Binning type for first dimension, e.g. :math:`r_{p}` when ``mode`` is "rppi".
            Set to ``lin`` for speed-up in case of linearly-spaced bins.
            In this case, the bin number for a pair separated by a (3D, projected, angular...) separation
            ``sep`` is given by ``(sep - edges[0])/(edges[-1] - edges[0])*(len(edges) - 1)``,
            i.e. only the first and last bins of input edges are considered.
            Then setting ``compute_sepsavg`` is virtually costless.
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

        compute_sepsavg : bool, default=True
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

        nprocs_per_real : int, default=1
            In case ``mpicomm`` is provided, the number of MPI processes to devote to the calculation of two-point counts for each jackknife realization.
            If ``nprocs_per_real`` is e.g. 1 (default), the parallelization is exclusively on the jackknife realizations.
            If ``nprocs_per_real`` is e.g. 2, the parallelization is on the jackknife realizations (with ``mpicomm.size // n_procs_per_real`` realizations treated in parallel)
            and the counts within each jackknife realization use 2 MPI processes.

        samples : list, array, default=None
            Whether to restrict jackknife counts to these subsamples. This may be useful to manually distribute the calculation.
            At the end of the computation, the different :class:`JackknifeTwoPointCounter` instances can be concatenated with :meth:`concatenate`.

        kwargs : dict
            Two-point counter engine-specific options.
        """
        self.attrs = kwargs
        self.mpicomm = mpicomm
        if self.mpicomm is None and mpiroot is not None:
            raise TwoPointCounterError('mpiroot is not None, but no mpicomm provided')
        self._set_nthreads(nthreads)
        self._set_mode(mode)
        self.nprocs_per_real = nprocs_per_real
        self._set_boxsize(boxsize)
        self._set_edges(edges, bin_type=bin_type)
        self._set_los(los)
        self._set_compute_sepsavg(compute_sepsavg)
        self._set_positions(positions1, positions2, position_type=position_type, dtype=dtype, copy=False, mpiroot=mpiroot)
        self._set_weights(weights1, weights2, weight_type=weight_type, twopoint_weights=twopoint_weights, weight_attrs=weight_attrs, copy=False, mpiroot=mpiroot)
        self._set_samples(samples1, samples2, mpiroot=mpiroot)
        self._set_zeros()
        self._set_reversible()
        self.auto, self.cross12, self.cross21 = {}, {}, {}
        self.run(samples=samples)
        del self.positions1, self.positions2, self.weights1, self.weights2, self.samples1, self.samples2

    def _set_samples(self, samples1, samples2=None, mpiroot=None):

        def _format_samples(samples):
            if samples is not None:
                samples = np.asarray(samples)
            if self.with_mpi and mpiroot is not None and self.mpicomm.bcast(samples is not None, root=mpiroot):
                samples = get_mpi().scatter_array(samples, mpicomm=self.mpicomm, root=mpiroot)
            return samples

        self.samples2 = self.samples1 = _format_samples(samples1)
        if not self.autocorr:
            self.samples2 = _format_samples(samples2)
            if self.samples2 is None:
                raise ValueError('samples2 must be provided in case of cross-correlation')

    def _set_sum(self):
        # Set global :attr:`wcounts` (and :attr:`sep`) based on all jackknife realizations.
        if not self.auto:
            self.wnorm = 0.
            self._set_zeros()
            self._set_reversible()
            return
        for counts in self.cross12.values(): break
        for name in ['is_reversible', 'compute_sepsavg']:
            setattr(self, name, getattr(counts, name))
        self.is_reversible = self.autocorr or self.is_reversible
        self.edges = counts.edges.copy()  # useful when rebinning
        for name in ['wcounts', 'wnorm', 'ncounts']:
            if hasattr(counts, name):
                setattr(self, name, sum(getattr(r, name) for r in self.auto.values()) + sum(getattr(r, name) for r in self.cross12.values()))
        if hasattr(self, 'ncounts'):
            self.wcounts[self.ncounts == 0] = 0.
        self._set_default_seps()  # reset self.seps to default
        for idim, compute_sepavg in enumerate(self.compute_sepsavg):
            if compute_sepavg:
                self.seps[idim] = np.sum([_nan_to_zero(r.seps[idim]) * r.wcounts for r in self.auto.values()], axis=0) + np.sum([_nan_to_zero(r.seps[idim]) * r.wcounts for r in self.cross12.values()], axis=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.seps[idim] /= self.wcounts

    def run(self, samples=None):
        """Run jackknife two-point counts."""
        if samples is None:
            samples = np.unique(self.samples1)
            if self.with_mpi:
                if not self.autocorr:
                    samples = np.unique(np.concatenate([samples, self.samples2], axis=0))
                samples = np.unique(get_mpi().gather_array(samples, root=None))
        if np.ndim(samples) == 0:
            samples = [samples]

        for ii in samples:
            self.auto[ii] = self.cross12[ii] = self.cross21[ii] = None

        with TaskManager(nprocs_per_task=self.nprocs_per_real, use_all_nprocs=True, mpicomm=self.mpicomm) as tm:

            def _mpi_distribute_arrays(*arrays):
                # Send array(s) to the root = 0 rank of each subcommunicator
                if self.with_mpi:
                    mpiroot_worker = tm.self_worker_ranks[0] if len(tm.self_worker_ranks) else -1
                    mpiroot_workers = list(np.unique(self.mpicomm.allgather(mpiroot_worker))[1:])

                    def _gather_array(array):
                        tmp = [get_mpi().gather_array(array, mpicomm=self.mpicomm, root=mpiroot) for mpiroot in mpiroot_workers]
                        if mpiroot_worker == -1:
                            return None
                        return tmp[mpiroot_workers.index(mpiroot_worker)]

                    return [_gather_array(array) for array in arrays]
                return arrays

            positions2 = positions1 = _mpi_distribute_arrays(*self.positions1)
            weights2 = weights1 = _mpi_distribute_arrays(*self.weights1)
            samples2 = samples1 = _mpi_distribute_arrays(self.samples1)[0]

            if not self.autocorr:
                positions2 = _mpi_distribute_arrays(*self.positions2)
                weights2 = _mpi_distribute_arrays(*self.weights2)
                samples2 = _mpi_distribute_arrays(self.samples2)[0]

            for ii in tm.iterate(samples):
                mask2 = mask1 = samples1 == ii
                spositions1, sweights1 = None, None
                spositions2, sweights2 = None, None
                is_root = not self.with_mpi or tm.mpicomm.rank == 0
                if is_root:
                    spositions1 = [position[mask1] for position in positions1]
                    sweights1 = [weight[mask1] for weight in weights1]
                if not self.autocorr:
                    mask2 = samples2 == ii
                    if is_root:
                        spositions2 = [position[mask2] for position in positions2]
                        sweights2 = [weight[mask2] for weight in weights2]
                mpiroot = 0 if self.with_mpi else None
                kwargs = {name: getattr(self, name) for name in ['bin_type', 'weight_attrs', 'twopoint_weights', 'boxsize', 'compute_sepsavg', 'nthreads']}
                kwargs['los'] = self.los_type
                kwargs['position_type'] = 'rd' if self.mode == 'theta' else 'xyz'
                kwargs.update(self.attrs)
                tmp = TwoPointCounter(self.mode, edges=self.edges, positions1=spositions1, weights1=sweights1, positions2=spositions2, weights2=sweights2, mpicomm=tm.mpicomm, mpiroot=mpiroot, **kwargs)
                if is_root:
                    self.auto[ii] = tmp
                if is_root:
                    spositions2 = [position[~mask2] for position in positions2]
                    sweights2 = [weight[~mask2] for weight in weights2]
                tmp = TwoPointCounter(self.mode, edges=self.edges, positions1=spositions1, weights1=sweights1, positions2=spositions2, weights2=sweights2, mpicomm=tm.mpicomm, mpiroot=mpiroot, **kwargs)
                if is_root:
                    self.cross12[ii] = tmp
                if self.autocorr and tmp.is_reversible:
                    tmp = tmp.reverse()
                    if is_root:
                        self.cross21[ii] = tmp
                else:
                    if is_root:
                        spositions1 = [position[~mask1] for position in positions1]
                        sweights1 = [weight[~mask1] for weight in weights1]
                        spositions2 = [position[mask2] for position in positions2]
                        sweights2 = [weight[mask2] for weight in weights2]
                    tmp = TwoPointCounter(self.mode, edges=self.edges, positions1=spositions1, weights1=sweights1, positions2=spositions2, weights2=sweights2, mpicomm=tm.mpicomm, mpiroot=mpiroot, **kwargs)
                    if is_root:
                        self.cross21[ii] = tmp

        if self.with_mpi:
            # Let us broadcast results to all ranks
            for name in self._result_names:
                results = getattr(self, name)
                for ii in samples:
                    cls, mpiroot_worker, state, state_arrays = None, None, None, {name: None for name in ['wcounts', 'ncounts', 'seps']}
                    if results[ii] is not None:
                        cls = results[ii].__class__
                        mpiroot_worker = self.mpicomm.rank
                        state = results[ii].__getstate__()
                        state_arrays = {name: state.pop(name, None) for name in state_arrays}
                    for mpiroot_worker in self.mpicomm.allgather(mpiroot_worker):
                        if mpiroot_worker is not None: break
                    cls = self.mpicomm.bcast(cls, root=mpiroot_worker)
                    state = self.mpicomm.bcast(state, root=mpiroot_worker)
                    # We separate out large arrays to bypass 2 Gb limit (actually not for the moment)
                    for key, value in state_arrays.items():
                        if self.mpicomm.bcast(value is not None, root=mpiroot_worker):
                            if key == 'seps':
                                if value is None: value = self.seps
                                # state[key] = [get_mpi().broadcast_array(val, mpicomm=self.mpicomm, root=mpiroot_worker) for val in value]
                                state[key] = [self.mpicomm.bcast(val, root=mpiroot_worker) for val in value]
                            else:
                                # state[key] = get_mpi().broadcast_array(value, mpicomm=self.mpicomm, root=mpiroot_worker)
                                state[key] = self.mpicomm.bcast(value, root=mpiroot_worker)
                    results[ii] = cls.from_state(state)

        self._set_sum()

    @property
    def realizations(self):
        """List of jackknife realizations, corresponding to input samples."""
        return list(self.auto.keys())

    @property
    def nrealizations(self):
        """Number of jackknife realizations."""
        return len(self.auto)

    def realization(self, ii, correction='mohammad21'):
        """
        Return jackknife realization ``ii``.

        Parameters
        ----------
        ii : int
            Label of jackknife realization.

        correction : string, default='mohammad'
            Correction to apply to computed counts.
            If ``None``, no correction is applied.
            Else, if "mohammad21", rescale cross-pairs by factor eq. 27 in arXiv:2109.07071.
            Else, rescale cross-pairs by provided correction factor.

        Returns
        -------
        counts : BaseTwoPointCounter
            Two-point counts for realization ``ii``.
        """
        alpha = 1.
        if isinstance(correction, str):
            if correction == 'mohammad21':
                # arXiv https://arxiv.org/pdf/2109.07071.pdf eq. 27
                alpha = self.nrealizations / (2. + np.sqrt(2) * (self.nrealizations - 1))
            else:
                raise TwoPointCounterError('Unknown jackknife correction {}'.format(correction))
        elif correction is not None:
            alpha = float(correction)
        state = self.auto[ii].__getstate__()
        for name in ['wcounts', 'wnorm', 'ncounts']:
            if hasattr(self, name):
                state[name] = getattr(self, name) - getattr(self.auto[ii], name) - alpha * (getattr(self.cross12[ii], name) + getattr(self.cross21[ii], name))
        if 'ncounts' in state:
            state['wcounts'][state['ncounts'] == 0] = 0.
        state['seps'] = state['seps'].copy()
        for idim, compute_sepavg in enumerate(self.compute_sepsavg):
            if compute_sepavg:
                state['seps'][idim] = _nan_to_zero(self.seps[idim]) * self.wcounts - _nan_to_zero(self.auto[ii].seps[idim]) * self.auto[ii].wcounts - alpha * (_nan_to_zero(self.cross12[ii].seps[idim]) * self.cross12[ii].wcounts + _nan_to_zero(self.cross21[ii].seps[idim]) * self.cross21[ii].wcounts)
                with np.errstate(divide='ignore', invalid='ignore'):
                    state['seps'][idim] /= state['wcounts']
                # The above may lead to rounding errors
                # such that seps may be non-zero even if wcounts is zero.
                mask = np.ones_like(state['seps'][idim], dtype='?')
                for name in ['ncounts', 'wcounts']:
                    if name in state:
                        mask &= state[name] != 0  # if ncounts / wcounts computed, good indicator of whether pairs exist or not
                        break
                # For more robustness we restrict to those separations which lie in between the lower and upper edges
                mask &= np.apply_along_axis(lambda x: (x >= self.edges[idim][:-1]) & (x <= self.edges[idim][1:]), idim, state['seps'][idim])
                state['seps'][idim][~mask] = np.nan
        for name in ['size1', 'size2']:
            state[name] = getattr(self, name) - getattr(self.auto[ii], name)
        return self.auto[ii].__class__.from_state(state)

    def cov(self, **kwargs):
        """
        Return jackknife covariance (of flattened counts).

        Parameters
        ----------
        kwargs : dict
            Optional arguments for :meth:`realization`.

        Returns
        -------
        cov : array
            Covariance matrix.
        """
        return (self.nrealizations - 1) * np.cov([self.realization(ii, **kwargs).normalized_wcounts().ravel() for ii in self.realizations], rowvar=False, ddof=0)

    def slice(self, *slices):
        """
        Slice counts in place. If slice step is not 1, use :meth:`rebin`.
        For example:

        .. code-block:: python

            counts.slice(slice(0, 10, 2), slice(0, 6, 3)) # rebin by factor 2 (resp. 3) along axis 0 (resp. 1), up to index 10 (resp. 6)
            counts[:10:2,:6:3] # same as above, but return new instance.

        """
        for name in self._result_names:
            for r in getattr(self, name).values(): r.slice(*slices)
        # Cannot do super(JackknifeTwoPointCounter, self).slice(*slices), as this would call self.rebin()
        tmp = BaseTwoPointCounter.__new__(BaseTwoPointCounter)
        tmp.__dict__.update(self.__dict__)
        tmp.slice(*slices)
        self.__dict__.update(tmp.__dict__)
        # self._set_sum()

    def rebin(self, factor=1):
        """
        Rebin two-point counts, by factor(s) ``factor``.
        A tuple must be provided in case :attr:`ndim` is greater than 1.
        Input factors must divide :attr:`shape`.

        Warning
        -------
        If current instance is the result of :meth:`concatenate_x`,
        rebinning is exact only if ``factor`` divides each of the constant-:attr:`wnorm` chunks.
        """
        for name in self._result_names:
            for r in getattr(self, name).values(): r.rebin(factor=factor)
        super(JackknifeTwoPointCounter, self).rebin(factor=factor)
        # self._set_sum()

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate input :class:`JackknifeTwoPointCounter` instances;
        typically used when calculation has been split into different samples,
        see argument ``samples`` of :meth:`__init__`.
        """
        if not others:
            raise TwoPointCounterError('Provide at least one {} instance.'.format(cls.__name__))
        new = others[0].copy()
        if np.ndim(new.wnorm) > 1:
            import warnings
            warnings.warn('Calling concatenate after concatenate_x & rebin will yield slightly incorrect wcounts at the boundaries of the x-concatenated wcounts; if you did not call rebin, you can ignore this message')
        for name in cls._result_names:
            tmp = {}
            for other in others: tmp.update(getattr(other, name))
            setattr(new, name, tmp)
        new._set_sum()
        return new

    def extend(self, other):
        """Extend current instance with another; see :meth:`concatenate`."""
        new = self.concatenate(self, other)
        self.__dict__.update(new.__dict__)

    @classmethod
    def concatenate_x(cls, *others):
        """
        Concatenate input two-point counts along :attr:`sep`;
        see :meth:`BaseTwoPointCounter.concatenate_x`.
        """
        # new = others[0].copy()
        new = super(JackknifeTwoPointCounter, cls).concatenate_x(*[other for other in others])
        for name in cls._result_names:
            tmp = getattr(new, name)
            for k in tmp:
                tmp[k] = tmp[k].concatenate_x(*[getattr(other, name)[k] for other in others])
        # new._set_sum()
        return new

    def normalize(self, wnorm):
        """
        Rescale both :attr:`wcounts` and :attr:`wnorm` such that new :attr:`wnorm` matches ``wnorm``.
        This is useful when combining counts in various regions.

        Parameters
        ----------
        wnorm : float
            New normalization :attr:`wnorm`.

        Returns
        -------
        new : JackknifeTwoPointCounter
            Normalized counts.
        """
        new = super(JackknifeTwoPointCounter, self).normalize(wnorm)
        factor = wnorm / self.wnorm
        for name in self._result_names:
            tmp = getattr(new, name)
            for k in tmp:
                # tmp[k] = tmp[k].normalize(wnorm)
                tmp[k] = tmp[k].normalize(factor * tmp[k].wnorm)
        # new._set_sum()
        return new

    @classmethod
    def sum(cls, *others):
        """Sum input two-point counts, see :meth:`BaseTwoPointCounter.sum`."""
        # new = self.copy()
        new = super(JackknifeTwoPointCounter, cls).sum(*others)
        for name in cls._result_names:
            tmp = getattr(new, name)
            for k in tmp:
                tmp[k] = tmp[k].sum(*[getattr(other, name)[k] for other in others])
        # new._set_sum()
        return new

    def __copy__(self):
        new = super(JackknifeTwoPointCounter, self).__copy__()
        for name in self._result_names:
            setattr(new, name, {ii: r.__copy__() for ii, r in getattr(self, name).items()})
        return new

    def reverse(self):
        new = super(JackknifeTwoPointCounter, self).reverse()  # a deepcopy with swapped size1, size2, and reversed counts
        if not self.autocorr:
            for name in self._result_names:
                setattr(new, name, {k: r.reverse() for k, r in getattr(self, name).items()})
            new.cross12, new.cross21 = new.cross21, new.cross12
            # new._set_sum()
        return new

    def wrap(self):
        new = super(JackknifeTwoPointCounter, self).wrap()  # a deepcopy with counts
        for name in self._result_names:
            setattr(new, name, {k: r.wrap() for k, r in getattr(self, name).items()})
        # new._set_sum()
        return new

    def __getstate__(self):
        state = super(JackknifeTwoPointCounter, self).__getstate__()
        for name in self._result_names:
            state[name] = {ii: r.__getstate__() for ii, r in getattr(self, name).items()}
        return state

    def __setstate__(self, state, load=False):
        super(JackknifeTwoPointCounter, self).__setstate__(state=state, load=load)
        for name in self._result_names:
            setattr(self, name, {ii: TwoPointCounter.from_state(s, load=load) for ii, s in getattr(self, name).items()})


class JackknifeTwoPointEstimator(BaseTwoPointEstimator):

    """Extend :class:`BaseTwoPointEstimator` with methods to handle jackknife realizations."""

    name = 'jackknife'

    @property
    def realizations(self):
        """List of jackknife realizations."""
        return self.XX.realizations

    @property
    def nrealizations(self):
        """Number of jackknife realizations."""
        return self.XX.nrealizations

    def realization(self, ii, **kwargs):
        """
        Return jackknife realization ``ii``.

        Parameters
        ----------
        ii : int
            Label of jackknife realization.

        kwargs : dict
            Optional arguments for :meth:`JackknifeTwoPointCounter.realization`.

        Returns
        -------
        estimator : BaseTwoPointEstimator
            Two-point estimator for realization ``ii``.
        """
        cls = self.__class__.__bases__[0]
        kw = {}
        for name in self.count_names:
            counts = getattr(self, name)
            try:
                kw[name] = counts.realization(ii, **kwargs)
            except AttributeError:
                kw[name] = counts  # in case counts are not jackknife, e.g. analytic randoms (but that'd be wrong!)
        return cls(**kw)

    def cov(self, **kwargs):
        cov = (self.nrealizations - 1) * np.cov([self.realization(ii, **kwargs).corr.ravel() for ii in self.realizations], rowvar=False, ddof=0)
        return np.atleast_2d(cov)

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate input :class:`JackknifeTwoPointEstimator` instances;
        typically used when calculation has been split into different samples,
        see argument ``samples`` of :meth:`JackknifeTwoPointCounter.__init__`.
        """
        if not others:
            raise TwoPointEstimatorError('Provide at least one {} instance.'.format(cls.__name__))
        kw = {}
        new = others[0]
        for name in new.count_names:
            cls_counts = getattr(new, name).__class__
            try:
                tmp = cls_counts.concatenate(*[getattr(other, name) for other in others])
            except AttributeError:  # in case counts are not jackknife, e.g. analytic randoms
                tmp = getattr(new, name)
            kw[name] = tmp
        cls = others[0].__class__
        return cls(**kw)

    def extend(self, other):
        """Extend current instance with another; see :meth:`concatenate`."""
        new = self.concatenate(self, other)
        self.__dict__.update(new.__dict__)

    def __setstate__(self, state, load=False):
        kwargs = {}
        counts = set(self.requires(with_reversed=True, with_shifted=True, join='')) | set(self.requires(with_reversed=True, with_shifted=False, join=''))  # most general list
        for name in counts:
            if name in state:
                if 'jackknife' in state[name].get('name', ''):
                    kwargs[name] = JackknifeTwoPointCounter.from_state(state[name], load=load)
                else:
                    kwargs[name] = TwoPointCounter.from_state(state[name], load=load)
        self.__init__(**kwargs)


# Generate all estimators, e.g. JackknifeNaturalTwoPointEstimator
for name, cls in list(BaseTwoPointEstimator._registry.items()):

    if name not in ['base', 'jackknife']:
        name_cls = 'Jackknife{}'.format(cls.__name__)
        globals()[name_cls] = type(BaseTwoPointEstimator)(name_cls, (cls, JackknifeTwoPointEstimator), {'name': 'jackknife-{}'.format(name), '__module__': __name__})
