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
    wcounts : array
        (Optionally weighted) pair-counts.

    sep : array
        Array of separation values.
    """

    def __init__(self, mode, edges, positions1, positions2=None, weights1=None, weights2=None,
                bin_type='auto', position_type='auto', weight_type='auto', los='midpoint',
                boxsize=None, output_sepavg=True, nthreads=None, **kwargs):
        """
        Initialize :class:`BaseTwoPointCounterEngine`.

        Parameters
        ----------
        mode : string
            Pair counting mode, one of:

            - "auto": pair counts as a function of angle (in degree) between two galaxies
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

        positions1 : list, array
            Positions in the first catalog. Typically of shape (3, N), but can be (2, N) when ``mode`` is "theta".

        positions2 : list, array, default=None
            Optionally, for cross-correlations, positions in the second catalog.

        weights1 : array, default=None
            Weights of the first catalog. Not required if ``weight_type`` is either ``None`` or "auto".

        weights2 : array, default=None
            Optionally, for cross-correlations, weights in the second catalog
            (not required if ``weight_type`` is either ``None`` or "auto").

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
            Line-of-sight to be used when ``mode`` is "smu" or "rppi"; one of:
            - "midpoint": the mean position of the pair: :math:`\eta = (\mathbf{r}_{1} + \mathbf{r}_{2})/2`
            - "x", "y" or "z": cartesian axis

        boxsize : array, int
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
        return tuple(len(edges) - 1 for edges in self.edges)

    @property
    def ndim(self):
        return len(self.edges)

    @property
    def periodic(self):
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
        """Return pair count normalization."""
        if self.weight_type is None:
            if self.autocorr:
                return len(self.positions1[0]) * (len(self.positions1[0]) - 1)
            return len(self.positions1[0]) * len(self.positions2[0])
        if self.autocorr:
            return self.weights1.sum()**2 - (self.weights1**2).sum()
        return self.weights1.sum()*self.weights2.sum()

    def normalized_wcounts(self):
        """Return normalized pair counts."""
        return self.wcounts/self.norm

    def __getstate__(self):
        state = {}
        for name in ['sep', 'wcounts', 'edges', 'mode', 'bin_type', 'weight_type',
                    'los', 'periodic', 'boxsize', 'output_sepavg']:
            state[name] = getattr(self, name)
        return state

    def rebin(self, factor=1):
        if np.ndim(factor) == 0:
            factor = (factor,)
        if len(factor) != self.ndim:
            raise PairCounterError('Provide a rebinning factor for each dimension')
        new_shape = tuple(s//f for s,f in zip(self.shape, factor))
        self.wcounts = utils.rebin(self.wcounts, statistic=np.sum)
        self.sep = utils.rebin(self.sep*self.wcounts, statistic=np.sum)/self.wcounts


def TwoPointCounter(*args, engine='corrfunc', **kwargs):

    if isinstance(engine, str):

        if engine.lower() == 'corrfunc':
            from .corrfunc import CorrfuncTwoPointCounterEngine
            return CorrfuncTwoPointCounterEngine(*args, **kwargs)

        raise PairCounterError('Unknown engine {}.'.format(engine))

    return engine


class AnalyticTwoPointCounter(BaseTwoPointCounterEngine):

    def __init__(self, mode, edges, boxsize, n1=10, n2=None, los='z'):
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
        if self.autocorr:
            return self.n1 * (self.n1 - 1)
        return self.n1 * self.n2
