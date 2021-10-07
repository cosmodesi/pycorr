"""Implement base pair counter, to be extended when implementing a new engine."""

import os
import numpy as np

from .utils import BaseClass
from . import utils


class PairCounterError(Exception):

    """Exception raised when issue with pair counting."""


class BaseTwoPointCounterEngine(BaseClass):
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
                bin_type='auto', position_type='auto', weight_type='auto', los='midpoint',
                boxsize=None, output_sepavg=True, nthreads=None, **kwargs):
        r"""
        Initialize :class:`BaseTwoPointCounterEngine`, and run actual pair counts
        (calling :meth:`run`), setting :attr:`wcounts` and :attr:`sep`.

        Parameters
        ----------
        mode : string
            Type of pair counts, one of:

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

        positions1 : list, array
            Positions in the first catalog. Typically of shape (3, N), but can be (2, N) when ``mode`` is "theta".

        positions2 : list, array, default=None
            Optionally, for cross-pair counts, positions in the second catalog. See ``positions1``.

        weights1 : array, default=None
            Weights of the first catalog. Not required if ``weight_type`` is either ``None`` or "auto".

        weights2 : array, default=None
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
            The type of weighting to apply. One of:

            - ``None``: no weights are applied.
            - "pair_product": each pair is weighted by the product of weights.
            - "auto": automatically choose weighting based on input ``weights1`` and ``weights2``,
               i.e. set to ``None`` when ``weights1`` and ``weights2`` are ``None``,
               else ``pair_product``.

        los : string, default='midpoint'
            Line-of-sight to be used when ``mode`` is "smu", "rppi" or "rp"; one of:

            - "midpoint": the mean position of the pair: :math:`\eta = (\mathbf{r}_{1} + \mathbf{r}_{2})/2`
            - "x", "y" or "z": cartesian axis

        boxsize : array, float, default=None
            For periodic wrapping, the side-length(s) of the periodic cube.

        output_sepavg : bool, default=True
            Set to ``False`` to *not* calculate the average separation for each bin.
            This can make the pair counts faster if ``bin_type`` is "custom".
            In this case, :attr:`sep` will be set the midpoint of input edges.

        nthreads : int
            Number of OpenMP threads to use.

        kwargs : dict
            Pair-counter engine-specific options.
        """
        self.mode = mode
        self.nthreads = nthreads
        if nthreads is None:
            self.nthreads = int(os.getenv('OMP_NUM_THREADS','1'))

        self._set_positions(positions1, positions2, position_type=position_type)
        self._set_weights(weights1, weights2, weight_type=weight_type)
        self._set_edges(edges, bin_type=bin_type)
        self._set_los(los)
        self._set_boxsize(boxsize)

        self.output_sepavg = output_sepavg
        self.attrs = kwargs

        self.run()

        if not self.output_sepavg:
            self._set_default_sep()

        self.norm = self.normalization()

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
                raise PairCounterError('A tuple of edges should be provided to pair counter in mode {}'.format(self.mode))
        else:
            if not self.ndim == 1:
                raise PairCounterError('Only one edge array should be provided to pair counter in mode {}'.format(self.mode))
        self._set_bin_type(bin_type)

    def _set_bin_type(self, bin_type):
        self.bin_type = bin_type.lower()
        allowed_bin_types = ['lin', 'custom', 'auto']
        if self.bin_type not in allowed_bin_types:
            raise PairCounterError('bin type should be one of {}'.format(allowed_bin_types))
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
        return self.boxsize is None

    def _set_positions(self, positions1, positions2=None, position_type='auto'):
        position_type = position_type.lower()
        if position_type == 'auto':
            if self.mode == 'theta': position_type = 'rd'
            else: position_type = 'xyz'

        def check_positions(positions):
            if self.mode == 'theta':
                if position_type == 'xyz':
                    positions = utils.cartesian_to_sky(positions)[1:]
                elif position_type == 'rdz':
                    positions = positions[:2]
                elif position_type != 'rd':
                    raise PairCounterError('For mode = {}, position type should be one of ["xyz", "rdz", "rd"]'.format(self.mode))
                if len(positions) != 2:
                    raise PairCounterError('For mode = {}, please provide a list of 2 arrays for positions'.format(self.mode))
            else:
                if position_type == 'rdd':
                    positions = utils.sky_to_cartesian(positions)
                elif position_type != 'xyz':
                    raise PairCounterError('For mode = {}, position type should be one of ["xyz", "rdd"]'.format(self.mode))
                if len(positions) != 3:
                    raise PairCounterError('For mode = {}, please provide a list of 3 arrays for positions'.format(self.mode))
            size = len(positions[0])
            dtype = positions[0].dtype
            for p in positions[1:]:
                if len(p) != size:
                    raise PairCounterError('All position arrays should be of the same size')
                if p.dtype != dtype:
                    raise PairCounterError('All position arrays should be of the same type')
            return positions

        self.positions1 = list(positions1)
        self.positions1 = check_positions(self.positions1)

        self.autocorr = positions2 is None
        if self.autocorr:
            self.positions2 = [None]*len(self.positions1)
        else:
            self.positions2 = list(positions2)
            self.positions2 = check_positions(self.positions2)

    def _set_weights(self, weights1, weights2=None, weight_type='auto'):

        self._set_weight_type(weight_type)

        if self.autocorr:
            if weights2 is not None:
                raise PairCounterError('weights2 are provided, but not positions2')

        if weights1 is None:
            if weights2 is not None:
                raise PairCounterError('weights2 are provided, but not weights1')
        else:
            if self.autocorr:
                if weights2 is not None:
                    raise PairCounterError('weights2 are provided, but not positions2')
            else:
                if weights2 is None:
                    raise PairCounterError('weights1 are provided, but not weights2')

        if self.weight_type == 'auto':
            if weights1 is None:
                self.weight_type = None
            else:
                self.weight_type = 'pair_product'

        if self.weight_type is None:
            self.weights1 = self.weights2 = None
        else:

            def check_weights(weights, size):
                if len(weights) != size:
                    raise PairCounterError('Weight array should be of the same length as position arrays')

            self.weights1 = weights1
            check_weights(self.weights1, len(self.positions1[0]))
            self.weights2 = weights2
            if not self.autocorr:
                check_weights(self.weights2, len(self.positions2[0]))

    def _set_default_sep(self):
        edges = self.edges[0]
        sep = (edges[1:] + edges[:-1])/2.
        if self.ndim == 2:
            self.sep = np.empty(self.shape, dtype='f8')
            self.sep[...] = sep
        else:
            self.sep = sep

    def _set_los(self, los):
        self.los = los
        allowed_los = ['midpoint', 'endpoint', 'firstpoint', 'x', 'y', 'z']
        if self.los not in allowed_los:
            raise PairCounterError('los should be one of {}'.format(allowed_los))

    def _set_boxsize(self, boxsize):
        self.boxsize = boxsize
        if self.periodic:
            self.boxsize = np.empty(3, dtype='f8')
            self.boxsize[:] = boxsize

    def _set_weight_type(self, weight_type=None):
        self.weight_type = weight_type
        allowed_weight_types = [None, 'auto', 'pair_product']
        if self.weight_type not in allowed_weight_types:
            raise PairCounterError('weight_type should be one of {}'.format(allowed_weight_types))

    def normalization(self):
        r"""
        Return pair count normalization, i.e., in case of cross-correlation:

        .. math::

            \left(\sum_{i=1}^{N_{1}} w_{1,i}\right) \left(\sum_{j=1}^{N_{2}} w_{2,j}\right)

        with the sums running over the weights of the first and second catalogs, and in case of auto-correlation:

        .. math::

            \left(\sum_{i=1}^{N_{1}} w_{1,i}\right)^{2} - \sum_{i=1}^{N_{1}} w_{1,i}^{2}

        """
        if self.weight_type is None:
            if self.autocorr:
                return len(self.positions1[0]) * (len(self.positions1[0]) - 1)
            return len(self.positions1[0]) * len(self.positions2[0])
        if self.autocorr:
            return self.weights1.sum()**2 - (self.weights1**2).sum()
        return self.weights1.sum()*self.weights2.sum()

    def normalized_wcounts(self):
        """Return normalized pair counts, i.e. :attr:`wcounts` divided by :meth:`normalization`."""
        return self.wcounts/self.norm

    def __getstate__(self):
        state = {}
        for name in ['sep', 'wcounts', 'edges', 'mode', 'bin_type', 'weight_type',
                    'los', 'periodic', 'boxsize', 'output_sepavg']:
            state[name] = getattr(self, name)
        return state

    def rebin(self, factor=1):
        """
        Rebin pair counts, by factor(s) ``factor``.
        A tuple must be provided in case :attr:`ndim` is greater than 1.
        Input factors must divide :attr:`shape`.
        """
        if np.ndim(factor) == 0:
            factor = (factor,)
        if len(factor) != self.ndim:
            raise PairCounterError('Provide a rebinning factor for each dimension')
        new_shape = tuple(s//f for s,f in zip(self.shape, factor))
        self.wcounts = utils.rebin(self.wcounts, statistic=np.sum)
        self.sep = utils.rebin(self.sep*self.wcounts, statistic=np.sum)/self.wcounts


def TwoPointCounter(*args, engine='corrfunc', **kwargs):
    """
    Entry point to pair counter engines.

    Parameters
    ----------
    engine : string
        Name of pair counter engine, one of ["corrfunc"].

    args : list
        Arguments for pair counter engine, see :class:`BaseTwoPointCounterEngine`.

    kwargs : dict
        Arguments for pair counter engine, see :class:`BaseTwoPointCounterEngine`.

    Returns
    -------
    engine : BaseTwoPointCounterEngine
    """
    if isinstance(engine, str):

        if engine.lower() == 'corrfunc':
            from .corrfunc import CorrfuncTwoPointCounterEngine
            return CorrfuncTwoPointCounterEngine(*args, **kwargs)

        raise PairCounterError('Unknown engine {}.'.format(engine))

    return engine


class AnalyticTwoPointCounter(BaseTwoPointCounterEngine):
    """
    Analytic pair counter. Assume periodic wrapping and no data weights.

    Attributes
    ----------
    sep : array
        Array of separation values.

    wcounts : array
        Analytical pair counts.
    """
    def __init__(self, mode, edges, boxsize, n1=10, n2=None, los='z'):
        """
        Initialize :class:`AnalyticTwoPointCounter`, and set :attr:`wcounts` and :attr:`sep`.

        Parameters
        ----------
        mode : string
            Pair counting mode, one of:

            - "s": pair counts as a function of distance between two galaxies
            - "smu": pair counts as a function of distance between two galaxies and cosine angle :math:`\mu`
                     w.r.t. the line-of-sight
            - "rppi": pair counts as a function of distance transverse (:math:`r_{p}`) and parallel (:math:`\pi`)
                     to the line-of-sight
            - "rp": same as "rppi", without binning in :math:`\pi`

        edges : tuple, array
            Tuple of bin edges (arrays), for the first (e.g. :math:`r_{p}`)
            and optionally second (e.g. :math:`\pi`) dimensions.
            In case of single-dimension binning (e.g. ``mode`` is "theta", "s" or "rp"),
            the single array of bin edges can be provided directly.

        boxsize : array, float
            The side-length(s) of the periodic cube.

        n1 : int, default=10
            Length of the first catalog.

        n2 : int, default=None
            Optionally, for cross-pair counts, length of second catalog.

        los : string, default='z'
            Line-of-sight to be used when ``mode`` is "rp", in case of non-cubic box;
            one of cartesian axes "x", "y" or "z".
        """
        self.mode = mode
        self._set_edges(edges)
        self._set_boxsize(boxsize)
        self._set_los(los)
        self.n1 = n1
        self.n2 = n2
        self.autocorr = n2 is None
        self.run()
        self._set_default_sep()

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
        Return pair count normalization, i.e., in case of cross-correlation ``n1 * n2``,
        and in case of auto-correlation ``n1 * (n1 - 1)``.
        """
        if self.autocorr:
            return self.n1 * (self.n1 - 1)
        return self.n1 * self.n2
