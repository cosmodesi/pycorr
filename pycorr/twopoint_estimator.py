"""Implements correlation function estimators, natural and Landy-Szalay."""

import os

import numpy as np
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from scipy import special

from .twopoint_counter import BaseTwoPointCounter, TwoPointCounter
from .utils import BaseClass
from . import utils


_default_ells = (0, 2, 4)
_default_wedges = (-1., -2. / 3, -1. / 3, 0., 1. / 3, 2. / 3, 1.)


def _format_ells(ells):
    isscalar = False
    isscalar = np.ndim(ells) == 0
    if isscalar: ells = (ells,)
    return ells, isscalar


def _format_wedges(wedges):
    isscalar = False
    if np.ndim(wedges[0]) == 0:
        isscalar = len(wedges) == 2
        wedges = list(zip(wedges[:-1], wedges[1:]))
    return wedges, isscalar


_default_wedges = _format_wedges(_default_wedges)[0]


def _get_project_mode(mode='auto', **kwargs):
    # Return projection mode depending on provided arguments
    if 'ell' in kwargs:
        kwargs['ells'] = kwargs.pop('ell')
    mode = mode.lower()
    if mode == 'auto':
        if 'ells' in kwargs:
            mode = 'poles'
        elif 'wedges' in kwargs:
            mode = 'wedges'
        elif 'pimax' in kwargs:
            mode = 'wp'
        else:
            mode = None
    return mode, kwargs


class TwoPointEstimatorError(Exception):

    """Exception raised when issue with estimator."""


def get_twopoint_estimator(estimator='auto', with_DR=True, with_jackknife=False):
    """
    Return :class:`BaseTwoPointEstimator` subclass corresponding
    to input estimator name.

    Parameters
    ----------
    estimator : string, default='auto'
        Estimator name, one of ["auto", "natural", "davispeebles", "landyszalay"].
        If "auto", "landyszalay" will be chosen if ``with_DR``, else "natural".

    with_DR : bool, default=True
        If estimator will be provided with two-point counts from data x random catalogs. See above.

    Returns
    -------
    estimator : type
        Estimator class.
    """
    if isinstance(estimator, str):

        if estimator == 'auto':
            estimator = {True: 'landyszalay', False: 'natural'}[with_DR]

        if with_jackknife:
            estimator = 'jackknife-{}'.format(estimator)

        try:
            estimator = BaseTwoPointEstimator._registry[estimator.lower()]
        except KeyError:
            raise TwoPointEstimatorError('Unknown estimator {}.'.format(estimator))

    return estimator


class TwoPointEstimator(BaseClass):
    """
    Entry point to two-point estimators.

    Parameters
    ----------
    estimator : string, default='landyszalay'
        Estimator name, one of ["auto", "natural", "davispeebles", "landyszalay"].

    args : list
        Arguments for two-point estimator, see :class:`TwoPointEstimator`.

    kwargs : dict
        Arguments for two-point estimator, see :class:`TwoPointEstimator`.

    Returns
    -------
    estimator : BaseTwoPointEstimator
        Estimator instance.
    """
    @staticmethod
    def from_state(state, load=False):
        """Return new estimator based on state dictionary."""
        cls = get_twopoint_estimator(state.pop('name'))
        new = cls.__new__(cls)
        new.__setstate__(state, load=load)
        return new


class RegisteredTwoPointEstimator(type(BaseClass)):

    """Metaclass registering :class:`BaseTwoPointEstimator`-derived classes."""

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.name] = cls
        return cls


class BaseTwoPointEstimator(BaseClass, metaclass=RegisteredTwoPointEstimator):
    """
    Base class for estimators.
    Extend this class to implement a new estimator.

    Attributes
    ----------
    corr : array
        Correlation function estimation.
    """
    name = 'base'

    def __init__(self, D1D2=None, R1R2=None, D1R2=None, R1D2=None, S1S2=None, D1S2=None, S1D2=None, S1R2=None):
        """
        Initialize :class:`BaseTwoPointEstimator`, and set correlation
        estimation :attr:`corr` (calling :meth:`run`).

        Parameters
        ----------
        D1D2 : BaseTwoPointCounter, default=None
            D1D2 two-point counts.

        R1R2 : BaseTwoPointCounter, default=None
            R1R2 two-point counts.

        D1R2 : BaseTwoPointCounter, default=None
            D1R2 two-point counts, e.g. for :class:`LandySzalayTwoPointEstimator`.

        R1D2 : BaseTwoPointCounter, default=None
            R1D2 two-point counts, e.g. for :class:`LandySzalayTwoPointEstimator`,
            in case of cross-correlation.

        S1S2 : BaseTwoPointCounter, default=None
            S1S2 two-point counts, e.g. with reconstruction, the Landy-Szalay estimator is commonly written:
            :math:`(D1D2 - D1S2 - S1D2 - S1S2)/R1R2`, with S1 and S2 shifted random catalogs.
            Defaults to ``R1R2``.

        D1S2 : BaseTwoPointCounter, default=None
            D1S2 two-point counts, see ``S1S2``. Defaults to ``D1R2``.

        S1D2 : BaseTwoPointCounter, default=None
            S1D2 two-point counts, see ``S1S2``. Defaults to ``R1D2``.

        S1R2 : BaseTwoPointCounter, default=None
            S1R2 two-point counts. Defaults to ``R1R2``.
        """
        self.with_shifted = S1S2 is not None or D1S2 is not None or S1D2 is not None or S1R2 is not None
        with_reversed = R1D2 is not None or S1D2 is not None
        for name in self.requires(with_reversed=with_reversed, with_shifted=self.with_shifted, join=''):
            counts = locals()[name]
            if locals()[name] is None:
                raise TwoPointEstimatorError('Counts {} must be provided'.format(name))
            setattr(self, name, counts)
        if not with_reversed:
            for name in self.requires(with_reversed=True, with_shifted=self.with_shifted, join=''):
                if not hasattr(self, name) and name in ['R1D2', 'S1D2']:
                    rname = ''.join([{'1': '2', '2': '1'}.get(nn, nn) for nn in name])
                    counts = getattr(self, rname[-2:] + rname[:2])
                    setattr(self, name, counts.reverse())
        self.run()

    def __getattr__(self, name):
        if name in ['S1S2', 'D1S2', 'S1D2', 'S1R2'] and not self.with_shifted:
            return getattr(self, name.replace('S', 'R'))
        raise AttributeError('Attribute {} does not exist'.format(name))

    @property
    def seps(self):
        """Array of separation values, if not set by :meth:`run`, taken from :attr:`R1R2` if provided, else :attr:`XX`."""
        if hasattr(self, '_seps'):
            return self._seps
        if hasattr(self, 'R1R2'):
            return self.R1R2.seps
        return self.XX.seps

    @property
    def sep(self):
        """Array of separation values of first dimension; if not set by :meth:`run`, taken from :attr:`R1R2` if provided, else :attr:`XX`."""
        return self.seps[0]

    def sepavg(self, *args, **kwargs):
        """Return average of separation for input axis; this is an 1D array of size :attr:`shape[axis]`."""
        if hasattr(self, 'R1R2'):
            return self.R1R2.sepavg(*args, **kwargs)
        return self.XX.sepavg(*args, **kwargs)

    @property
    def edges(self):
        """Edges for two-point count calculation, taken from :attr:`XX`."""
        return self.XX.edges

    @property
    def mode(self):
        """Two-point counting mode, taken from :attr:`XX`."""
        return self.XX.mode

    @property
    def shape(self):
        """Return shape of obtained correlation :attr:`corr`."""
        return tuple(len(edges) - 1 for edges in self.edges)

    @property
    def ndim(self):
        """Return binning dimensionality."""
        return len(self.edges)

    @classmethod
    def requires(cls, join=None, **kwargs):
        """List required counts."""
        toret = []
        for tu in cls._tuple_requires(**kwargs):
            if join is not None:
                toret.append(join.join(tu))
            else:
                toret.append(tu)
        return toret

    @classmethod
    def _tuple_requires(cls, with_reversed=False, with_shifted=False, join=None):
        toret = []
        toret.append(('D1', 'D2'))
        key = 'S' if with_shifted else 'R'
        toret.append(('D1', '{}2'.format(key)))
        if with_reversed:
            toret.append(('{}1'.format(key), 'D2'))
        toret.append(('{}1'.format(key), '{}2'.format(key)))
        if with_shifted:
            toret.append(('R1', 'R2'))
        return toret

    @property
    def count_names(self):
        """Return list of counts used in estimator."""
        return self.requires(with_reversed=True, with_shifted=self.with_shifted, join='')

    @property
    def XX(self):
        """Return first two-point counts."""
        return getattr(self, self.count_names[0])

    def __getitem__(self, slices):
        """Call :meth:`slice`."""
        new = self.copy()
        if isinstance(slices, tuple):
            new.slice(*slices)
        else:
            new.slice(slices)
        return new

    def select(self, *args, **kwargs):
        """
        Restrict estimator to provided coordinate limits in place.

        For example:

        .. code-block:: python

            estimator.select((0, 0.3)) # restrict first axis to (0, 0.3)
            estimator.select(None, (0, 0.2)) # restrict second axis to (0, 0.2)

        """
        BaseTwoPointCounter.select(self, *args, **kwargs)

    def slice(self, *args, **kwargs):
        """Slice estimator, by rebinning all two-point counts. See :meth:`BaseTwoPointCounter.slice`."""
        for name in self.count_names:
            getattr(self, name).slice(*args, **kwargs)
        self.run()

    def rebin(self, *args, **kwargs):
        """Rebin estimator, by rebinning all two-point counts. See :meth:`BaseTwoPointCounter.rebin`."""
        for name in self.count_names:
            getattr(self, name).rebin(*args, **kwargs)
        self.run()

    @classmethod
    def concatenate_x(cls, *others):
        """
        Concatenate input estimators along :attr:`sep` by concatenating their two-point counts;
        see :meth:`BaseTwoPointCounter.concatenate_x`.
        """
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        new = others[0].copy()
        for name in new.count_names:
            setattr(new, name, getattr(new, name).concatenate_x(*[getattr(other, name) for other in others]))
        new.run()
        return new

    def wrap(self):
        r"""Return new 'smu' or 'rppi' two-point estimators with 2nd coordinate wrapped to positive values, :math:`\mu > 0` or :math:`\pi > 0`."""
        new = self.copy()
        for name in new.count_names:
            setattr(new, name, getattr(new, name).wrap())
        new.run()
        return new

    def normalize(self, wnorm='XX'):
        """
        Renormalize all counts (:attr:`BaseTwoPointCounter.wcounts` and :attr:`BaseTwoPointCounter.wnorm`).
        This is useful when combining measurements in various regions.

        Parameters
        ----------
        wnorm : float, string, default='XX'
            If float, rescale all :attr:`BaseTwoPointCounter.wcounts` and :attr:`BaseTwoPointCounter.wnorm`
            such that :attr:`BaseTwoPointCounter.wnorm` matches ``wnorm``.
            Else, name of counts (e.g. 'D1D2', 'R1R2', etc.) to take ``wnorm`` from.
            'XX' is the first counts of the estimator (usually 'D1D2').

        Returns
        -------
        new : BaseTwoPointEstimator
            New estimator, with all counts renormalized.
            :attr:`corr` is expected to be exactly the same.
        """
        new = self.copy()
        if isinstance(wnorm, str):
            wnorm = getattr(self, wnorm).wnorm
        for name in new.count_names:
            setattr(new, name, getattr(new, name).normalize(wnorm=wnorm))
        return new

    @classmethod
    def sum(cls, *others):
        """
        Sum input estimators (their two-point counts, actually).
        See e.g. https://arxiv.org/pdf/1905.01133.pdf for a use case.
        Input two-point estimators must have same edges for this operation to make sense
        (no checks performed).
        """
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        new = others[0].copy()
        for name in new.count_names:
            setattr(new, name, getattr(new, name).sum(*[getattr(other, name) for other in others]))
        new.run()
        return new

    def __add__(self, other):
        return self.sum(self, other)

    def __radd__(self, other):
        if other == 0: return self.deepcopy()
        return self.__add__(other)

    def __iadd__(self, other):
        if other == 0: return self.deepcopy()
        return self.__add__(other)

    def deepcopy(self):
        import copy
        return copy.deepcopy(self)

    def __copy__(self):
        new = super(BaseTwoPointEstimator, self).__copy__()
        for name in self.count_names:
            setattr(new, name, getattr(self, name).__copy__())
        return new

    def __getstate__(self):
        state = {}
        for name in ['name']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for name in self.count_names:
            state[name] = getattr(self, name).__getstate__()
        return state

    def __setstate__(self, state, load=False):
        kwargs = {}
        counts = set(self.requires(with_reversed=True, with_shifted=True, join='')) | set(self.requires(with_reversed=True, with_shifted=False, join=''))  # most general list
        for name in counts:
            if name in state:
                kwargs[name] = TwoPointCounter.from_state(state[name], load=load)
        self.__init__(**kwargs)

    def save(self, filename):
        """Save estimator to ``filename``."""
        if not self.with_mpi or self.mpicomm.rank == 0:
            super(BaseTwoPointEstimator, self).save(filename)

    def get_corr(self, return_sep=False, return_cov=None, mode='auto', **kwargs):
        r"""
        Return correlation function, optionally its covariance estimate, if available.

        Parameters
        ----------
        return_sep : bool, default=False
            Whether (``True``) to return average separation(s) :attr:`sepavg`.

        return_cov : bool, default=None
            If ``True`` or ``None`` and estimator holds a covariance estimate :meth:`cov` (e.g. from jackknife realizations),
            return covariance estimate.
            If ``True`` and estimator does not have :meth:`cov`,
            raise :class:`TwoPointEstimatorError`.

        mode : str, default='auto'
            If 'poles', return multipoles.
            If 'wedges', return :math:`\mu`-wedges.
            If 'wp', return projected correlation function.
            If 'auto', and 'ells' provided (see ``kwargs``), return multipoles;
            else if 'wedges' provided, return :math:`\mu`-wedges.
            else if 'pimax' provided, return projected correlation function.

        kwargs : dict
            Optionally arguments for :func:`project_to_poles` (if :attr:`mode` is 'smu'), e.g. ``ells``,
            :func:`project_to_wedges` (if :attr:`mode` is 'smu'), e.g. ``wedges``,
            `project_to_wp` (if :attr:`mode` is 'rpppi'), e.g. ``pimax``, and :meth:`cov`.

        Returns
        -------
        sep : array
            Optionally, separation values.

        corr : array
            Estimated correlation function.

        cov : array
            Optionally, covariance estimate (of the flattened ``corr``), see ``return_cov``.
        """
        mode, kwargs = _get_project_mode(mode=mode, **kwargs)
        if mode == 'poles':
            return project_to_poles(self, return_sep=return_sep, return_cov=return_cov, **kwargs)
        if mode == 'wedges':
            return project_to_wedges(self, return_sep=return_sep, return_cov=return_cov, **kwargs)
        if mode == 'wp':
            return project_to_wp(self, return_sep=return_sep, return_cov=return_cov, **kwargs)
        for name in ['ells', 'wedges', 'pimax']: kwargs.pop(name, None)
        toret = []
        if return_sep:
            for axis in range(self.ndim): toret.append(self.sepavg(axis=axis))
        toret.append(self.corr.copy())
        if return_cov is False:
            return toret if len(toret) > 1 else toret[0]
        if hasattr(self, 'cov'):
            cov = self.cov(**kwargs)
            toret.append(cov)
        elif return_cov is True:
            raise TwoPointEstimatorError('Input estimator has no covariance')
        return toret if len(toret) > 1 else toret[0]

    def __call__(self, *seps, return_sep=False, return_std=None, mode='auto', **kwargs):
        """
        Return (1D) correlation function, optionally performing linear interpolation over :math:`sep`.

        Parameters
        ----------
        seps : float, array, default=None
            Separations where to interpolate the correlation function.
            Values outside :attr:`sepavg` are linearly extrapolated;
            outside :attr:`edges` are set to nan.
            Defaults to :attr:`sepavg` (no interpolation performed).

        return_sep : bool, default=False
            Whether (``True``) to return separation (see ``sep``).
            If ``None``, return separation if ``sep`` is ``None``.

        return_std : bool, default=None
            If ``True`` or ``None`` and estimator holds a covariance estimate :meth:`cov` (e.g. from jackknife realizations),
            return standard deviation estimate.
            If ``True`` and estimator does not have :meth:`cov`,
            raise :class:`TwoPointEstimatorError`.

        mode : str, default='auto'
            If 'poles', return multipoles.
            If 'wp', return projected correlation function.
            If 'auto', and 'ells' provided (see ``kwargs``), return multipoles;
            else if 'pimax' provided, return projected correlation function.

        kwargs : dict
            Optionally arguments for :func:`project_to_poles` (if :attr:`mode` is 'smu'), e.g. ``ells``,
            `project_to_wp` (if :attr:`mode` is 'rpppi'), e.g. ``pimax``, and :meth:`cov`.

        Returns
        -------
        sep : array
            Optionally, separation values.

        corr : array
            (Optionally interpolated) correlation function.

        std : array
            Optionally, (optionally interpolated) standard deviation estimate, see ``return_std``.
        """
        return_std = return_std or return_std is None and hasattr(self, 'cov')
        tmp = self.get_corr(return_sep=True, return_cov=return_std, mode=mode, **kwargs)
        ndim = len(tmp) - int(return_std) - 1
        sepsavg = tmp[:ndim]
        corr_std = [tmp[ndim]]
        if return_std: corr_std.append(tmp[ndim + 1])
        isscalar = corr_std[0].ndim == 1
        if return_std:
            # Turn covariance to standard deviation
            corr_std[-1] = np.diag(corr_std[-1])**0.5
            corr_std[-1].shape = corr_std[0].shape
        if return_sep is None:
            return_sep = bool(seps)
        if not seps:
            toret = []
            if return_sep: toret += list(sepsavg)
            toret += corr_std
            return toret if len(toret) > 1 else toret[0]
        seps = seps + tuple(self.sepavg(idim) for idim in range(len(seps), ndim))
        if len(seps) > ndim:
            raise TwoPointEstimatorError('Expected {:d} input separation arrays, got {:d}'.format(ndim, len(seps)))
        if isscalar and ndim == 1:
            corr_std = [array[None, :] for array in corr_std]
        if ndim == 1:
            mask_finite_seps = tuple(~np.isnan(sepavg) & ~np.isnan(corr_std[0]).any(axis=0) for axis, sepavg in enumerate(sepsavg))
        else:
            mask_finite_seps = tuple(~np.isnan(sepavg) & ~np.isnan(corr_std[0]).any(axis=tuple(ii for ii in range(ndim) if ii != axis)) for axis, sepavg in enumerate(sepsavg))
        sepsavg = tuple(sepavg[mask] for sepavg, mask in zip(sepsavg, mask_finite_seps))
        indices = (Ellipsis, mask_finite_seps[0]) if ndim == 1 else np.ix_(*mask_finite_seps)
        corr_std = [array[indices] for array in corr_std]
        seps = tuple(np.asarray(sep) for sep in seps)
        toret_shape = sum((sep.shape for sep in seps), tuple())
        if ndim == 1 and not isscalar: toret_shape = (len(corr_std[0]),) + toret_shape
        seps = tuple(sep.ravel() for sep in seps)
        shape = ((len(corr_std[0]),) if ndim == 1 else tuple()) + tuple(sep.size for sep in seps)
        toret_corr_std = [np.nan * np.zeros(shape, dtype=corr_std[0].dtype) for i in range(len(corr_std))]
        mask_seps = tuple((sep >= self.edges[idim][0]) & (sep <= self.edges[idim][-1]) for idim, sep in enumerate(seps))
        seps_masked = tuple(sep[mask] for sep, mask in zip(seps, mask_seps))
        indices = (Ellipsis, mask_seps[0]) if ndim == 1 else np.ix_(*mask_seps)

        if all(sep_masked.size for sep_masked in seps_masked) and all(sepavg.size for sepavg in sepsavg):
            if ndim == 1:

                def interp(array):
                    return np.array([UnivariateSpline(sepsavg[0], arr, k=1, s=0, ext='extrapolate')(seps_masked[0]) for arr in array], dtype=array.dtype)

            else:
                i_seps = tuple(np.argsort(sep) for sep in seps_masked)
                sseps_masked = tuple(sep[i_sep] for sep, i_sep in zip(seps_masked, i_seps))
                ii_seps = tuple(np.argsort(i_sep) for i_sep in i_seps)

                def interp(array):
                    return RectBivariateSpline(*sepsavg, array, kx=1, ky=1, s=0)(*sseps_masked, grid=True)[np.ix_(*ii_seps)]

            for toret_array, array in zip(toret_corr_std, corr_std): toret_array[indices] = interp(array)
        if isscalar and ndim == 1:
            toret_corr_std = [array[0] for array in toret_corr_std]
        for array in toret_corr_std: array.shape = toret_shape
        toret = []
        if return_sep: toret += list(seps)
        toret += toret_corr_std
        return toret if len(toret) > 1 else toret[0]

    def save_txt(self, filename, fmt='%.12e', delimiter=' ', header=None, comments='# ', return_std=None, mode='auto', **kwargs):
        """
        Save correlation function as txt file.

        Warning
        -------
        Attributes are not all saved, hence there is :meth:`load_txt` method.

        Parameters
        ----------
        filename : str
            File name.

        fmt : str, default='%.12e'
            Format for floating types.

        delimiter : str, default=' '
            String or character separating columns.

        header : str, list, default=None
            String that will be written at the beginning of the file.
            If multiple lines, provide a list of one-line strings.

        comments : str, default=' #'
            String that will be prepended to the header string.

        return_std : bool, default=None
            If ``True`` or ``None`` and estimator holds a covariance estimate :meth:`cov` (e.g. from jackknife realizations),
            save standard deviation estimate.
            If ``True`` and estimator does not have :meth:`cov`,
            raise :class:`TwoPointEstimatorError`.

        mode : str, default='auto'
            If 'poles', save multipoles.
            If 'wp', save projected correlation function.
            If 'auto', and 'ells' provided (see ``kwargs``), save multipoles;
            else if 'pimax' provided, save projected correlation function.

        kwargs : dict
            Optionally arguments for :func:`project_to_poles` (if :attr:`mode` is 'smu'), e.g. ``ells``,
            `project_to_wp` (if :attr:`mode` is 'rpppi'), e.g. ``pimax``, and :meth:`cov`.
        """
        if not self.with_mpi or self.mpicomm.rank == 0:
            self.log_info('Saving {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
            formatter = {'int_kind': lambda x: '%d' % x, 'float_kind': lambda x: fmt % x}
            # First, header
            if header is None: header = []
            elif isinstance(header, str): header = [header]
            else: header = list(header)
            attrs = {}
            for count in self.count_names:
                for name in ['size1', 'size2']:
                    attrs['.'.join([count, name])] = getattr(getattr(self, count), name, None)
            for name in ['mode', 'autocorr'] + list(attrs.keys()) + ['los_type', 'bin_type']:
                value = attrs.get(name, getattr(self.XX, name, None))
                if value is None:
                    value = 'None'
                elif any(name.startswith(key) for key in ['mode', 'los_type', 'bin_type']):
                    value = str(value)
                else:
                    value = np.array2string(np.array(value), separator=delimiter, formatter=formatter).replace('\n', '')
                header.append('{} = {}'.format(name, value))
            # Then, data
            mode, kwargs = _get_project_mode(mode=mode, **kwargs)
            if mode is None: mode = self.mode
            return_std = return_std or return_std is None and hasattr(self, 'cov')
            tmp = self(return_sep=True, return_std=return_std, mode=mode, **kwargs)
            ndim = len(tmp) - int(return_std) - 1
            if ndim == self.ndim:  # True, except in case 2d (smu, rppi) -> 1d (poles, wp)
                seps = tuple(self.seps[idim] for idim in range(ndim))
            else:
                seps = tmp[:ndim]
            corr = tmp[ndim]
            std = tmp[ndim + 1] if return_std else None
            names = {'poles': ('s',), 'wedges': ('s',), 'smu': ('s', 'mu'), 'wp': ('rp',), 'rppi': ('rp', 'pi')}.get(mode, (mode,))
            labels = []
            for name in names: labels += ['{}mid'.format(name), '{}avg'.format(name)]
            if mode == 'poles':
                ells = _format_ells(kwargs.get('ells', _default_ells))[0]
                labels += ['corr{:d}({})'.format(ell, names[0]) for ell in ells]
                if std is not None:
                    labels += ['std{:d}({})'.format(ell, names[0]) for ell in ells]
            elif mode == 'wedges':
                wedges = _format_wedges(kwargs.get('wedges', _default_wedges))[0]
                labels += ['corr({},[{:.2f},{:.2f}])'.format(names[0], *wedge) for wedge in wedges]
                if std is not None:
                    labels += ['std({},[{:.2f},{:.2f}])'.format(names[0], *wedge) for wedge in wedges]
            else:
                labels += ['corr({})'.format(','.join(names))]
                if std is not None:
                    labels += ['std({})'.format(','.join(names))]
            columns = []
            mids = np.meshgrid(*(self.sepavg(idim, method='mid') for idim in range(ndim)), indexing='ij')
            for idim, sep in enumerate(seps):
                columns += [mids[idim].flat, sep.flat]
            for column in corr.reshape((-1,) * (corr.ndim == ndim) + corr.shape):
                columns += [column.flat]
            if std is not None:
                for column in std.reshape((-1,) * (std.ndim == ndim) + std.shape):
                    columns += [column.flat]
            columns = [[np.array2string(value, formatter=formatter) for value in column] for column in columns]
            widths = [max(max(map(len, column)) - len(comments) * (icol == 0), len(label)) for icol, (column, label) in enumerate(zip(columns, labels))]
            widths[-1] = 0  # no need to leave a space
            header.append((' ' * len(delimiter)).join(['{:<{width}}'.format(label, width=width) for label, width in zip(labels, widths)]))
            widths[0] += len(comments)
            with open(filename, 'w') as file:
                for line in header:
                    file.write(comments + line + '\n')
                for irow in range(len(columns[0])):
                    file.write(delimiter.join(['{:<{width}}'.format(column[irow], width=width) for column, width in zip(columns, widths)]) + '\n')

    def plot(self, plot_std=None, mode='auto', ax=None, fn=None, kw_save=None, show=False, **kwargs):
        r"""
        Plot correlation function.

        Parameters
        ----------
        plot_std : bool, default=None
            If ``True`` or ``None`` and estimator holds a covariance estimate :meth:`cov` (e.g. from jackknife realizations),
            plot standard deviation estimate.
            If ``True`` and estimator does not have :meth:`cov`,
            raise :class:`TwoPointEstimatorError`.

        mode : str, default='auto'
            If 'poles', save multipoles.
            If 'wp', save projected correlation function.
            If 'auto', and 'ells' provided (see ``kwargs``), save multipoles;
            else if 'pimax' provided, save projected correlation function.

        ax : matplotlib.axes.Axes, default=None
            Axes where to plot samples. If ``None``, takes current axes.

        fn : string, default=None
            If not ``None``, file name where to save figure.

        kw_save : dict, default=None
            Optional arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            Whether to show figure.

        kwargs : dict
            Optionally arguments for :func:`project_to_poles` (if :attr:`mode` is 'smu'), e.g. ``ells``,
            `project_to_wp` (if :attr:`mode` is 'rpppi'), e.g. ``pimax``, and :meth:`cov`.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        mode, kwargs = _get_project_mode(mode=mode, **kwargs)
        if mode is None: mode = self.mode
        plot_std = plot_std or plot_std is None and hasattr(self, 'cov')
        tmp = self(return_sep=True, return_std=plot_std, mode=mode, **kwargs)
        if len(tmp) - int(plot_std) - 1 > 1:
            raise ValueError('Cannot plot > 1D correlation function')
        sep, corr = tmp[:2]
        std = tmp[2] if plot_std else None
        from matplotlib import pyplot as plt
        fig = None
        if ax is None: fig, ax = plt.subplots()

        def plot(x, y, yerr=None, **kwargs):
            if yerr is None:
                ax.plot(x, y, **kwargs)
            else:
                ax.errorbar(x, y, yerr=yerr, linestyle='none', marker='o', **kwargs)

        if mode == 'poles':
            ells = _format_ells(kwargs.get('ells', _default_ells))[0]
            for ill, ell in enumerate(ells):
                plot(sep, sep**2 * corr[ill], yerr=sep**2 * std[ill] if std is not None else None, label=r'$\ell = {:d}$'.format(ell))
            ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
            ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
            ax.legend()
        elif mode == 'wedges':
            wedges = _format_wedges(kwargs.get('wedges', _default_wedges))[0]
            for iwedge, wedge in enumerate(wedges):
                plot(sep, sep**2 * corr[iwedge], yerr=sep**2 * std[iwedge] if std is not None else None, label=r'${:.2f} < \mu < {:.2f}$'.format(*wedge))
            ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
            ax.set_ylabel(r'$s^{2} \xi(s, \mu)$ [$(\mathrm{Mpc}/h)^{2}$]')
            ax.legend()
        elif mode in ['rp', 'wp']:
            plot(sep, sep * corr, yerr=sep * std if std is not None else None)
            ax.set_xscale('log')
            ax.set_xlabel(r'$r_{p}$ [$\mathrm{Mpc}/h$]')
            ax.set_ylabel(r'$r_{p} w_{p}$ [$(\mathrm{{Mpc}}/h)^{{2}}$]')
        elif mode == 's':
            plot(sep, sep**2 * corr, yerr=sep**2 * std if std is not None else None)
            ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
            ax.set_ylabel(r'$s^{2} \xi(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        elif mode == 'theta':
            plot(sep, sep * corr, yerr=sep * std if std is not None else None)
            ax.set_xlabel(r'$\theta$ [deg]')
            ax.set_ylabel(r'$\theta w(\theta)$')
        else:
            raise ValueError('mode must be one of [poles, wedges, rp, s, theta]')
        ax.grid(True)
        if not self.with_mpi or self.mpicomm.rank == 0:
            if fn is not None:
                utils.savefig(fn, fig=fig, **(kw_save or {}))
            if show:
                plt.show()
        return ax


def _make_property(name):

    @property
    def func(self):
        return getattr(self.XX, name)

    return func


for name in ['with_mpi', 'mpicomm']:
    setattr(BaseTwoPointEstimator, name, _make_property(name))


class NaturalTwoPointEstimator(BaseTwoPointEstimator):

    name = 'natural'

    def run(self):
        """
        Set correlation function estimate :attr:`corr` based on the natural estimator
        :math:`(D1D2 - S1S2)/R1R2`.
        """
        nonzero = self.R1R2.wcounts != 0
        # init
        corr = np.empty_like(self.R1R2.wcounts, dtype='f8')
        corr[...] = np.nan

        # the natural estimator
        # (DD - RR) / RR
        DD = self.D1D2.normalized_wcounts()[nonzero]
        RR = self.R1R2.normalized_wcounts()[nonzero]
        SS = self.S1S2.normalized_wcounts()[nonzero]
        tmp = (DD - SS) / RR
        corr[nonzero] = tmp[...]
        self.corr = corr

    @classmethod
    def _tuple_requires(cls, with_reversed=False, with_shifted=False, join=None):
        toret = []
        toret.append(('D1', 'D2'))
        key = 'S' if with_shifted else 'R'
        toret.append(('{}1'.format(key), '{}2'.format(key)))
        if with_shifted:
            toret.append(('R1', 'R2'))
        return toret


class DavisPeeblesTwoPointEstimator(BaseTwoPointEstimator):

    name = 'davispeebles'

    def run(self):
        """
        Set correlation function estimate :attr:`corr` based on the Davis-Peebles estimator
        :math:`(D1D2 - D1S2)/D1R2`.
        """
        nonzero = self.D1R2.wcounts != 0
        # init
        corr = np.empty_like(self.D1R2.wcounts, dtype='f8')
        corr[...] = np.nan

        # the natural estimator
        # (DD - RR) / RR
        DD = self.D1D2.normalized_wcounts()[nonzero]
        DR = self.D1R2.normalized_wcounts()[nonzero]
        DS = self.D1S2.normalized_wcounts()[nonzero]
        tmp = (DD - DS) / DR
        corr[nonzero] = tmp[...]
        self.corr = corr

    @classmethod
    def _tuple_requires(cls, with_reversed=False, with_shifted=False, join=None):
        """Yield required two-point counts."""
        toret = []
        toret.append(('D1', 'D2'))
        key = 'S' if with_shifted else 'R'
        toret.append(('D1', '{}2'.format(key)))
        if with_shifted:
            toret.append(('D1', 'R2'))
        return toret


class LandySzalayTwoPointEstimator(BaseTwoPointEstimator):

    name = 'landyszalay'

    def run(self):
        """
        Set correlation function estimate :attr:`corr` based on the Landy-Szalay estimator
        :math:`(D1D2 - D1S2 - S1D2 - S1S2)/R1R2`.
        """
        nonzero = self.R1R2.wcounts != 0
        # init
        corr = np.empty_like(self.R1R2.wcounts, dtype='f8')
        corr[...] = np.nan

        # the Landy - Szalay estimator
        # (DD - DR - RD + RR) / RR
        DD = self.D1D2.normalized_wcounts()[nonzero]
        RR = self.R1R2.normalized_wcounts()[nonzero]
        DS = self.D1S2.normalized_wcounts()[nonzero]
        SD = self.S1D2.normalized_wcounts()[nonzero]
        SS = self.S1S2.normalized_wcounts()[nonzero]

        tmp = (DD - DS - SD + SS) / RR
        corr[nonzero] = tmp[...]
        self.corr = corr


class WeightTwoPointEstimator(NaturalTwoPointEstimator):

    name = 'weight'

    def run(self):
        """
        Set weight estimate :attr:`corr` following :math:`R1R2/D1D2`,
        typically used for angular weights, with R1R2 from parent sample and D1D2 from fibered sample.
        """
        nonzero = self.D1D2.wcounts != 0
        # init
        corr = np.ones_like(self.R1R2.wcounts, dtype='f8')

        DD = self.D1D2.normalized_wcounts()[nonzero]
        RR = self.R1R2.normalized_wcounts()[nonzero]
        tmp = RR / DD
        corr[nonzero] = tmp[...]
        self.corr = corr
        self._seps = [sep.copy() for sep in self.R1R2.seps]
        zero = self.R1R2.wcounts == 0.
        seps_mid = list(np.meshgrid(*self.R1R2._get_default_seps(), indexing='ij'))
        for sep, sep_mid in zip(self._seps, seps_mid):
            sep[zero] = sep_mid[zero]

    @classmethod
    def _yield_requires(cls, with_reversed=False, with_shifted=False, join=None):
        toret = []
        toret.append(('D1', 'D2'))
        toret.append(('R1', 'R2'))
        return toret

    @property
    def weight(self):
        """Another name for :attr:`corr`."""
        return self.corr


class ResidualTwoPointEstimator(BaseTwoPointEstimator):

    name = 'residual'

    def run(self):
        """
        Set correlation function estimate :attr:`corr` based on the residual estimator
        :math:`(D1R2 - S1R2)/R1R2`.
        """
        nonzero = self.R1R2.wcounts != 0
        # init
        corr = np.empty_like(self.R1R2.wcounts, dtype='f8')
        corr[...] = np.nan

        DR = self.D1R2.normalized_wcounts()[nonzero]
        SR = self.S1R2.normalized_wcounts()[nonzero]
        RR = self.R1R2.normalized_wcounts()[nonzero]
        tmp = (DR - SR) / RR
        corr[nonzero] = tmp[...]
        self.corr = corr

    @classmethod
    def _tuple_requires(cls, with_reversed=False, with_shifted=False, join=None):
        toret = []
        key = 'S' if with_shifted else 'R'
        toret.append(('D1', 'R2'))
        toret.append(('{}1'.format(key), 'R2'))
        if with_shifted:
            toret.append(('R1', 'R2'))
        return toret


def project_to_poles(estimator, ells=_default_ells, return_sep=True, return_cov=None, ignore_nan=False, **kwargs):
    r"""
    Project :math:`(s, \mu)` correlation function estimation onto Legendre polynomials.

    Parameters
    ----------
    estimator : BaseTwoPointEstimator
        Estimator for :math:`(s, \mu)` correlation function.

    ells : tuple, int, default=(0, 2, 4)
        Order of Legendre polynomial.

    return_sep : bool, default=True
        Whether (``True``) to return separation.

    return_cov : bool, default=None
        If ``True`` or ``None`` and input ``estimator`` holds (jackknife) :meth:`realization`,
        return covariance matrix estimate (for all successive ``ells``).
        If ``True`` and input ``estimator`` does not have :meth:`realization`,
        raise :class:`TwoPointEstimatorError`.

    ignore_nan : bool, default=False
        If ``True``, ignore NaN values of the correlation functions in the integration.

    kwargs : dict
        Optional arguments for :meth:`JackknifeTwoPointEstimator.realization`, when relevant.

    Returns
    -------
    sep : array
        Optionally, array of separation values.

    poles : array
        Correlation function multipoles.

    cov : array
        Optionally, covariance estimate (for all successive ``ells``), see ``return_cov``.
    """
    if getattr(estimator, 'mode', 'smu') != 'smu':
        raise TwoPointEstimatorError('Estimating multipoles is only possible in mode = "smu"')
    ells, isscalar = _format_ells(ells)
    muedges = estimator.edges[1]
    dmu = np.diff(muedges)
    sep = estimator.sepavg(axis=0)
    corr = []
    for ell in ells:
        # \sum_{i} \xi_{i} \int_{\mu_{i}}^{\mu_{i+1}} L_{\ell}(\mu^{\prime}) d\mu^{\prime}
        poly = special.legendre(ell).integ()(muedges)
        legendre = (2 * ell + 1) * (poly[1:] - poly[:-1])
        if ignore_nan:
            correll = np.empty(estimator.corr.shape[0], dtype=estimator.corr.dtype)
            for i_s, corr_s in enumerate(estimator.corr):
                mask_s = ~np.isnan(corr_s)
                correll[i_s] = np.sum(corr_s[mask_s] * legendre[mask_s], axis=-1) / np.sum(dmu[mask_s])
        else:
            correll = np.sum(estimator.corr * legendre, axis=-1) / np.sum(dmu)
        corr.append(correll)
    if isscalar:
        corr = corr[0]
    corr = np.array(corr)
    toret = []
    if return_sep: toret.append(sep)
    toret.append(corr)
    if return_cov is False:
        return toret if len(toret) > 1 else toret[0]
    try:
        realizations = [project_to_poles(estimator.realization(ii, **kwargs),
                                         ells=ells, return_sep=False, return_cov=False, ignore_nan=ignore_nan).ravel()
                        for ii in estimator.realizations]
        cov = (len(realizations) - 1) * np.cov(realizations, rowvar=False, ddof=0)
        toret.append(np.atleast_2d(cov))
    except AttributeError as exc:
        if return_cov is True:
            raise TwoPointEstimatorError('Input estimator has no jackknife realizations') from exc
    return toret if len(toret) > 1 else toret[0]


project_to_multipoles = project_to_poles  # for backward-compatibility


def project_to_wedges(estimator, wedges=_default_wedges, return_sep=True, return_cov=None, ignore_nan=False, **kwargs):
    r"""
    Project :math:`(s, \mu)` correlation function estimation onto wedges (integrating over :math:`\mu`).

    Parameters
    ----------
    estimator : BaseTwoPointEstimator
        Estimator for :math:`(s, \mu)` correlation function.

    wedges : tuple, default=(-1., -2./3, -1./3, 0., 1./3, 2./3, 1.)
        :math:`mu`-wedges.
        Single or list of tuples :math:`(\mu_{\mathrm{min}}, \mu_{\mathrm{max}})`,
        or :math:`\mu`-edges :math:`(\mu_{0}, ..., \mu_{n})`,

    return_sep : bool, default=True
        Whether (``True``) to return separation.

    return_cov : bool, default=None
        If ``True`` or ``None`` and input ``estimator`` holds (jackknife) :meth:`realization`,
        return covariance matrix estimate (for all successive ``ells``).
        If ``True`` and input ``estimator`` does not have :meth:`realization`,
        raise :class:`TwoPointEstimatorError`.

    ignore_nan : bool, default=False
        If ``True``, ignore NaN values of the correlation functions in the integration.

    kwargs : dict
        Optional arguments for :meth:`JackknifeTwoPointEstimator.realization`, when relevant.

    Returns
    -------
    sep : array
        Optionally, array of separation values.

    wedges : array
        Correlation function wedges.

    cov : array
        Optionally, covariance estimate (for all successive ``wedges``), see ``return_cov``.
    """
    if getattr(estimator, 'mode', 'smu') != 'smu':
        raise TwoPointEstimatorError('Estimating wedges is only possible in mode = "smu"')
    wedges, isscalar = _format_wedges(wedges)
    muedges = estimator.edges[1]
    mumid = (muedges[:-1] + muedges[1:]) / 2.
    dmu = np.diff(muedges)
    sep = estimator.sepavg(axis=0)
    corr = []
    for wedge in wedges:
        mask = (mumid >= wedge[0]) & (mumid < wedge[1])
        if ignore_nan:
            corrmu = np.empty(estimator.corr.shape[0], dtype=estimator.corr.dtype)
            for i_s, corr_s in enumerate(estimator.corr):
                mask_s = mask & ~np.isnan(corr_s)
                corrmu[i_s] = np.sum(corr_s[mask_s] * dmu[mask_s], axis=-1) / np.sum(dmu[mask_s])
        else:
            corrmu = np.sum(estimator.corr[:, mask] * dmu[mask], axis=-1) / np.sum(dmu[mask])
        corr.append(corrmu)
    if isscalar:
        corr = corr[0]
    corr = np.array(corr)
    toret = []
    if return_sep: toret.append(sep)
    toret.append(corr)
    if return_cov is False:
        return toret if len(toret) > 1 else toret[0]
    try:
        realizations = [project_to_wedges(estimator.realization(ii, **kwargs),
                                          wedges=wedges, return_sep=False, return_cov=False, ignore_nan=ignore_nan).ravel()
                        for ii in estimator.realizations]
        cov = (len(realizations) - 1) * np.cov(realizations, rowvar=False, ddof=0)
        toret.append(np.atleast_2d(cov))
    except AttributeError as exc:
        if return_cov is True:
            raise TwoPointEstimatorError('Input estimator has no jackknife realizations') from exc
    return toret if len(toret) > 1 else toret[0]


def project_to_wp(estimator, pimax=None, return_sep=True, return_cov=None, ignore_nan=False, **kwargs):
    r"""
    Integrate :math:`(r_{p}, \pi)` correlation function over :math:`\pi` to obtain :math:`w_{p}(r_{p})`.

    Parameters
    ----------
    estimator : BaseTwoPointEstimator
        Estimator for :math:`(r_{p}, \pi)` correlation function.

    pimax : float, tuple, default=None
        Upper bound for summation of :math:`\pi`, or tuple of (lower bound, upper bound).

    return_sep : bool, default=True
        Whether (``True``) to return separation.

    return_cov : bool, default=None
        If ``True`` or ``None`` and input ``estimator`` holds (jackknife) :meth:`realization`,
        return covariance matrix estimate (for all successive ``ells``).
        If ``True`` and input ``estimator`` does not have :meth:`realization`,
        raise :class:`TwoPointEstimatorError`.

    ignore_nan : bool, default=False
        If ``True``, ignore NaN values of the correlation functions in the integration.

    kwargs : dict
        Optional arguments for :meth:`JackknifeTwoPointEstimator.realization`, when relevant.

    Returns
    -------
    sep : array
        Optionally, array of separation values.

    wp : array
        Estimated :math:`w_{p}(r_{p})`.

    cov : array
        Optionally, covariance estimate, see ``return_cov``.
    """
    if getattr(estimator, 'mode', 'rppi') != 'rppi':
        raise TwoPointEstimatorError('Estimating projected correlation function is only possible in mode = "rppi"')

    if pimax is not None:
        tpimax = pimax
        if np.ndim(pimax) == 0:
            tpimax = (-pimax, pimax)
        estimator = estimator.copy()
        estimator.select(None, tpimax)

    sep = estimator.sepavg(axis=0)
    dpi = np.diff(estimator.edges[1])
    if ignore_nan:
        corr = np.empty(estimator.corr.shape[0], dtype=estimator.corr.dtype)
        for i_rp, corr_rp in enumerate(estimator.corr):
            mask_rp = ~np.isnan(corr_rp)
            corr[i_rp] = np.sum(corr_rp[mask_rp] * dpi[mask_rp], axis=-1) * np.sum(dpi) / np.sum(dpi[mask_rp])  # extra factor to correct for missing bins
    else:
        corr = np.sum(estimator.corr * dpi, axis=-1)
    toret = []
    if return_sep: toret.append(sep)
    toret.append(corr)
    if return_cov is False:
        return toret if len(toret) > 1 else toret[0]
    try:
        realizations = [project_to_wp(estimator.realization(ii, **kwargs), return_sep=False, return_cov=False, ignore_nan=ignore_nan) for ii in estimator.realizations]  # no need to provide pimax, as selection already performed
        cov = (len(realizations) - 1) * np.cov(realizations, rowvar=False, ddof=0)
        toret.append(np.atleast_2d(cov))
    except AttributeError as exc:
        if return_cov is True:
            raise TwoPointEstimatorError('Input estimator has no jackknife realizations') from exc
    return toret if len(toret) > 1 else toret[0]
