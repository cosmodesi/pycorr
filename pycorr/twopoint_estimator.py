"""Implements correlation function estimators, natural and Landy-Szalay."""

import os

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import special

from .twopoint_counter import BaseTwoPointCounter, TwoPointCounter
from .utils import BaseClass
from . import utils


_default_ells = (0, 2, 4)


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
            estimator = {True:'landyszalay', False:'natural'}[with_DR]

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
    def from_state(state):
        """Return new estimator based on state dictionary."""
        cls = get_twopoint_estimator(state.pop('name'))
        new = cls.__new__(cls)
        new.__setstate__(state)
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
    _require_randoms = False

    def __init__(self, D1D2=None, R1R2=None, D1R2=None, R1D2=None, S1S2=None, D1S2=None, S1D2=None):
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
        """
        self.with_shifted = S1S2 is not None or D1S2 is not None or S1D2 is not None
        self.with_reversed = R1D2 is not None or S1D2 is not None
        for name in self.count_names:
            counts = locals()[name]
            if locals()[name] is None:
                raise TwoPointEstimatorError('Counts {} must be provided'.format(name))
            setattr(self, name, counts)
        self.run()

    def __getattr__(self, name):
        if name in ['R1D2', 'S1D2'] and not self.with_reversed:
            name = ''.join([{'1':'2', '2':'1'}.get(nn, nn) for nn in name])
            counts = getattr(self, name[-2:] + name[:2])
            return counts.reversed()
        if name in ['S1S2', 'D1S2', 'S1D2'] and not self.with_shifted:
            return getattr(self, name.replace('S', 'R'))
        raise AttributeError('Attribute {} does not exist'.format(name))

    @property
    def sep(self):
        """Array of separation values of first dimension, taken from :attr:`R1R2` if provided, else :attr:`D1D2`."""
        if getattr(self, 'R1R2', None) is not None:
            return self.R1R2.sep
        return self.XX.sep

    @property
    def seps(self):
        """Array of separation values, taken from :attr:`R1R2` if provided, else :attr:`D1D2`."""
        if getattr(self, 'R1R2', None) is not None:
            return self.R1R2.seps
        return self.XX.seps

    def sepavg(self, *args, **kwargs):
        """Return average of separation for input axis; this is an 1D array of size :attr:`shape[axis]`."""
        if getattr(self, 'R1R2', None) is not None:
            return self.R1R2.sepavg(*args, **kwargs)
        return self.XX.sepavg(*args, **kwargs)

    @property
    def edges(self):
        """Edges for two-point count calculation, taken from :attr:`D1D2`."""
        return self.XX.edges

    @property
    def mode(self):
        """Two-point counting mode, taken from :attr:`D1D2`."""
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
    def requires(cls, with_reversed=False, with_shifted=False, join=None):
        """List required counts."""
        toret = []
        for tu in cls._tuple_requires(with_reversed=with_reversed, with_shifted=with_shifted):
            if join is not None:
                toret.append(join.join(tu))
            else:
                toret.append(tu)
        return toret

    @classmethod
    def _tuple_requires(cls, with_reversed=False, with_shifted=False, join=None):
        toret = []
        toret.append(('D1','D2'))
        key = 'S' if with_shifted else 'R'
        toret.append(('D1','{}2'.format(key)))
        if with_reversed:
            toret.append(('{}1'.format(key), 'D2'))
        toret.append(('{}1'.format(key),'{}2'.format(key)))
        if with_shifted:
            toret.append(('R1','R2'))
        return toret

    @property
    def count_names(self):
        """Return list of counts used in estimator."""
        return self.requires(with_reversed=self.with_reversed, with_shifted=self.with_shifted, join='')

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
        new = others[0].copy()
        for name in new.count_names:
            setattr(new, name, getattr(new, name).concatenate_x(*[getattr(other, name) for other in others]))
        new.run()
        return new

    @classmethod
    def sum(cls, *others):
        """
        Sum input estimators (their two-point counts, actually).
        See e.g. https://arxiv.org/pdf/1905.01133.pdf for a use case.
        """
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

    def __setstate__(self, state):
        kwargs = {}
        counts = set(self.requires(with_reversed=True, with_shifted=True, join='')) | set(self.requires(with_reversed=True, with_shifted=False, join='')) # most general list
        for name in counts:
            if name in state:
                kwargs[name] = TwoPointCounter.from_state(state[name])
        self.__init__(**kwargs)

    def save(self, filename):
        """Save estimator to ``filename``."""
        if not self.XX.with_mpi or self.XX.mpicomm.rank == 0:
            super(BaseTwoPointEstimator, self).save(filename)
        if self.XX.with_mpi:
            self.XX.mpicomm.Barrier()

    def get_corr(self, return_cov=None, **kwargs):
        """
        Return (1D) correlation function, optionally its jackknife covariance estimate, if available.

        Parameters
        ----------
        return_cov : bool, default=None
            If ``True`` or ``None`` and estimator holds jackknife realizations,
            return jackknife covariance estimate.
            If ``True`` and estimator does not hold jackknife realizations,
            raise :class:`TwoPointEstimatorError`.

        kwargs : dict
            Optionally, arguments for :func:`project_to_multipoles` (if :attr:`mode` is 'smu'), e.g. ``ells``
            and `project_to_wp` (if :attr:`mode` is 'rpppi'), e.g. ``pimax``.

        Returns
        -------
        sep : array
            Array of separation values.

        corr : array
            Estimated correlation function.

        cov : array
            Optionally, jackknife covariance estimate, see ``return_cov``.
        """
        if self.mode == 'smu':
            return project_to_multipoles(self, return_cov=return_cov, **kwargs)
        if self.mode == 'rppi':
            return project_to_wp(self, return_cov=return_cov, **kwargs)
        if return_cov is False:
            return self.sep, self.corr
        try:
            realizations = [self.realization(ii, **kwargs).corr for ii in self.realizations]
        except AttributeError as exc:
            if return_cov is True:
                raise TwoPointEstimatorError('Input estimator has no jackknife realizations') from exc
            return self.sep, self.corr
        cov = (len(realizations) - 1) * np.cov(realizations, rowvar=False, ddof=0)
        return self.sep, self.corr, cov

    def __call__(self, sep=None, return_std=None, **kwargs):
        """
        Return (1D) correlation function, optionally performing linear interpolation over :math:`sep`.

        Parameters
        ----------
        sep : float, array, default=None
            Separations where to interpolate the correlation function.
            Values outside :attr:`sepavg` are set to the first/last correlation function value;
            outside :attr:`edges[0]` to nan.
            Defaults to :attr:`sepavg` (no interpolation performed).

        return_std : bool, default=None
            If ``True`` or ``None`` and estimator holds jackknife realizations,
            return jackknife standard deviation estimate.
            If ``True`` and estimator does not hold jackknife realizations,
            raise :class:`TwoPointEstimatorError`.

        kwargs : dict
            Optionally, arguments for :func:`project_to_multipoles` (if :attr:`mode` is 'smu'), e.g. ``ells``
            and `project_to_wp` (if :attr:`mode` is 'rpppi'), e.g. ``pimax``.

        Returns
        -------
        sep : array
            If input ``sep`` is ``None``, array of separation values.

        corr : array
            (Optionally interpolated) correlation function.

        std : array
            Optionally, (optionally interpolated) jackknife standard deviation estimate, see ``return_std``.
        """
        tmp = self.get_corr(return_cov=return_std, **kwargs)
        if len(tmp) < 3: tmp = tmp + (None,)
        sepavg, corr, std = tmp
        isscalar = corr.ndim == 1
        if std is not None:
            std = np.diag(std)**0.5
            if not isscalar:
                std = np.array(np.array_split(std, len(corr)))
        if sep is None:
            if std is None:
                return sepavg, corr
            return sepavg, corr, std
        if isscalar:
            corr = corr[None, :]
            if std is not None: std = std[None, :]
        mask_finite_sep = ~np.isnan(sepavg) & ~np.isnan(corr).any(axis=0)
        sepavg, corr = sepavg[mask_finite_sep], corr[:, mask_finite_sep]
        sep = np.asarray(sep)
        toret_corr = np.nan * np.zeros((len(corr),) + sep.shape, dtype=sep.dtype)
        if std is not None: toret_std = toret_corr.copy()
        mask_sep = (sep >= self.edges[0][0]) & (sep <= self.edges[0][-1])
        sep = sep[mask_sep]
        if mask_sep.any():
            interp = lambda array: np.array([UnivariateSpline(sepavg, arr, k=1, s=0, ext='const')(sep) for arr in array], dtype=array.dtype)
            toret_corr[..., mask_sep] = interp(corr)
            if std is not None: toret_std[..., mask_sep] = interp(std)
        if isscalar:
            toret_corr = toret_corr[0]
            if std is not None: toret_std = toret_std[0]
        if std is None:
            return toret_corr
        return toret_corr, toret_std

    def save_txt(self, filename, fmt='%.12e', delimiter=' ', header=None, comments='# ', return_std=None, **kwargs):
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
            If ``True`` or ``None`` and estimator holds jackknife realizations,
            save jackknife standard deviation estimate.
            If ``True`` and estimator does not hold jackknife realizations,
            raise :class:`TwoPointEstimatorError`.

        kwargs : dict
            Arguments for :meth:`get_corr`.
        """
        if not self.XX.with_mpi or self.XX.mpicomm.rank == 0:
            self.log_info('Saving {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
            formatter = {'int_kind': lambda x: '%d' % x, 'float_kind': lambda x: fmt % x}
            if header is None: header = []
            elif isinstance(header, str): header = [header]
            else: header = list(header)
            attrs = {}
            for count in self.count_names:
                for name in ['size1', 'size2', 'wnorm']:
                    attrs['.'.join([count, name])] = getattr(getattr(self, count), name, None)
            for name in ['mode', 'autocorr'] + list(attrs.keys()) + ['los_type', 'bin_type']:
                value = attrs.get(name, getattr(self.XX, name, None))
                if value is None:
                    value = 'None'
                elif any(name.startswith(key) for key in ['mode', 'los_type', 'bin_type']):
                    value = str(value)
                else:
                    value = np.array2string(np.array(value), formatter=formatter).replace('\n', '')
                header.append('{} = {}'.format(name, value))
            name = {'smu': 's', 'rppi': 'rp'}.get(self.mode, self.mode)
            labels = ['{}mid'.format(name), '{}avg'.format(name)]
            tmp = self(sep=None, return_std=return_std, **kwargs)
            if len(tmp) < 3: tmp = tmp + (None,)
            sepavg, corr, std = tmp
            ells = kwargs.get('ells', _default_ells)
            isscalar = corr.ndim == 1
            if self.mode == 'smu':
                if isscalar: ells = [ells]
                labels += ['corr{:d}({})'.format(ell, name) for ell in ells]
                if std is not None:
                    labels += ['std{:d}({})'.format(ell, name) for ell in ells]
            else:
                labels += ['corr({})'.format(name)]
                if std is not None:
                    labels += ['std({})'.format(name)]
            columns = [(self.edges[0][:-1] + self.edges[0][1:])/2., sepavg]
            for column in corr.reshape((-1,)*isscalar + corr.shape):
                columns += [column.flat]
            if std is not None:
                for column in std.reshape((-1,)*isscalar + std.shape):
                    columns += [column.flat]
            columns = [[np.array2string(value, formatter=formatter) for value in column] for column in columns]
            widths = [max(max(map(len, column)) - len(comments) * (icol == 0), len(label)) for icol, (column, label) in enumerate(zip(columns, labels))]
            widths[-1] = 0 # no need to leave a space
            header.append((' '*len(delimiter)).join(['{:<{width}}'.format(label, width=width) for label, width in zip(labels, widths)]))
            widths[0] += len(comments)
            with open(filename, 'w') as file:
                for line in header:
                    file.write(comments + line + '\n')
                for irow in range(len(columns[0])):
                    file.write(delimiter.join(['{:<{width}}'.format(column[irow], width=width) for column, width in zip(columns, widths)]) + '\n')


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
        tmp = (DD - SS)/RR
        corr[nonzero] = tmp[...]
        self.corr = corr

    @classmethod
    def _tuple_requires(cls, with_reversed=False, with_shifted=False, join=None):
        toret = []
        toret.append(('D1','D2'))
        key = 'S' if with_shifted else 'R'
        toret.append(('{}1'.format(key),'{}2'.format(key)))
        if with_shifted:
            toret.append(('R1','R2'))
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
        tmp = (DD - DS)/DR
        corr[nonzero] = tmp[...]
        self.corr = corr

    @classmethod
    def _tuple_requires(cls, with_reversed=False, with_shifted=False, join=None):
        """Yield required two-point counts."""
        toret = []
        toret.append(('D1','D2'))
        key = 'S' if with_shifted else 'R'
        toret.append(('D1','{}2'.format(key)))
        if with_shifted:
            toret.append(('D1','R2'))
        return toret


class LandySzalayTwoPointEstimator(BaseTwoPointEstimator):

    name = 'landyszalay'
    _require_randoms = True

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

        tmp = (DD - DS - SD + SS)/RR
        corr[nonzero] = tmp[...]
        self.corr = corr


class WeightTwoPointEstimator(NaturalTwoPointEstimator):

    name = 'weight'
    _require_randoms = True

    def run(self):
        """
        Set weight estimate :attr:`corr` following :math:`R1R2/D1D2`,
        typically used for angular weights, with R1R2 from parent sample and D1D2 from fibered sample.
        """
        nonzero = self.D1D2.wcounts != 0
        # init
        corr = np.ones_like(self.R1R2.wcounts, dtype='f8')

        # the natural estimator
        # (DD - RR) / RR
        DD = self.D1D2.normalized_wcounts()[nonzero]
        RR = self.R1R2.normalized_wcounts()[nonzero]
        tmp = RR/DD
        corr[nonzero] = tmp[...]
        self.corr = corr

    @classmethod
    def _yield_requires(cls, with_reversed=False, with_shifted=False, join=None):
        toret = []
        toret.append(('D1','D2'))
        toret.append(('R1','R2'))
        return toret

    @property
    def weight(self):
        """Another name for :attr:`corr`."""
        return self.corr


class ResidualTwoPointEstimator(BaseTwoPointEstimator):

    name = 'residual'
    _require_randoms = True

    def run(self):
        """
        Set correlation function estimate :attr:`corr` based on the residual estimator
        :math:`(D1S2 - S1S2)/R1R2`.
        """
        nonzero = self.R1R2.wcounts != 0
        # init
        corr = np.empty_like(self.R1R2.wcounts, dtype='f8')
        corr[...] = np.nan

        DS = self.D1S2.normalized_wcounts()[nonzero]
        SS = self.S1S2.normalized_wcounts()[nonzero]
        RR = self.R1R2.normalized_wcounts()[nonzero]
        tmp = (DS - SS)/RR
        corr[nonzero] = tmp[...]
        self.corr = corr

    @classmethod
    def _tuple_requires(cls, with_reversed=False, with_shifted=False, join=None):
        toret = []
        key = 'S' if with_shifted else 'R'
        toret.append(('D1'.format(key),'{}2'.format(key)))
        toret.append(('{}1'.format(key),'{}2'.format(key)))
        if with_shifted:
            toret.append(('R1','R2'))
        return toret


def project_to_multipoles(estimator, ells=_default_ells, return_cov=None, **kwargs):
    r"""
    Project :math:`(s, \mu)` correlation function estimation onto Legendre polynomials.

    Parameters
    ----------
    estimator : BaseTwoPointEstimator
        Estimator for :math:`(s, \mu)` correlation function.
        If estimator holds jackknife realizations, also return jackknife covariance estimate.

    ells : tuple, int, default=(0,2,4)
        Order of Legendre polynomial.

    return_cov : bool, default=None
        If ``True`` or ``None`` and input ``estimator`` holds jackknife realizations,
        return jackknife covariance estimate (for all successive ``ells``).
        If ``True`` and input ``estimator`` does not hold jackknife realizations,
        raise :class:`TwoPointEstimatorError`.

    kwargs : dict
        Optional arguments for :meth:`JackknifeTwoPointEstimator.realization`, when relevant.

    Returns
    -------
    sep : array
        Array of separation values.

    poles : list
        List of correlation function multipoles.

    cov : array
        Optionally, jackknife covariance estimate (for all successive ``ells``), see ``return_cov``.
    """
    isscalar = np.ndim(ells) == 0
    if isscalar:
        ells = (ells,)
    ells = list(ells)
    muedges = estimator.edges[1]
    sep = estimator.sepavg(axis=0)
    poles = []
    for ill, ell in enumerate(ells):
        # \sum_{i} \xi_{i} \int_{\mu_{i}}^{\mu_{i+1}} L_{\ell}(\mu^{\prime}) d\mu^{\prime}
        poly = special.legendre(ell).integ()(muedges)
        legendre = (2*ell + 1) * (poly[1:] - poly[:-1])
        poles.append(np.sum(estimator.corr*legendre, axis=-1)/(muedges[-1] - muedges[0]))
    if isscalar:
        poles = poles[0]
    poles = np.array(poles)
    if return_cov is False:
        return sep, poles
    try:
        realizations = [np.concatenate(project_to_multipoles(estimator.realization(ii, **kwargs), ells=ells)[1]).T for ii in estimator.realizations]
    except AttributeError as exc:
        if return_cov is True:
            raise TwoPointEstimatorError('Input estimator has no jackknife realizations') from exc
        return sep, poles
    cov = (len(realizations) - 1) * np.cov(realizations, rowvar=False, ddof=0)
    return sep, poles, cov


def project_to_wp(estimator, pimax=None, return_cov=None, **kwargs):
    r"""
    Integrate :math:`(r_{p}, \pi)` correlation function over :math:`\pi` to obtain :math:`w_{p}(r_{p})`.

    Parameters
    ----------
    estimator : BaseTwoPointEstimator
        Estimator for :math:`(r_{p}, \pi)` correlation function.
        If estimator holds jackknife realizations, also return jackknife covariance estimate.

    pimax : float, default=None
        Upper bound for summation of :math:`\pi`.

    return_cov : bool, default=None
        If ``True`` or ``None`` and input ``estimator`` holds jackknife realizations,
        return jackknife covariance estimate.
        If ``True`` and input ``estimator`` does not hold jackknife realizations,
        raise :class:`TwoPointEstimatorError`.

    kwargs : dict
        Optional arguments for :meth:`JackknifeTwoPointEstimator.realization`, when relevant.

    Returns
    -------
    sep : array
        Array of separation values.

    wp : array
        Estimated :math:`w_{p}(r_{p})`.

    cov : array
        Optionally, jackknife covariance estimate, see ``return_cov``.
    """
    mask = Ellipsis
    if pimax is not None:
        estimator = estimator.copy()
        estimator.select(None, (0, pimax))
    sep = estimator.sepavg(axis=0)
    wp = 2.*np.sum(estimator.corr*np.diff(estimator.edges[1]), axis=-1)
    if return_cov is False:
        return sep, wp
    try:
        realizations = [project_to_wp(estimator.realization(ii, **kwargs))[1] for ii in estimator.realizations] # no need to provide pimax, as selection already performed
    except AttributeError as exc:
        if return_cov is True:
            raise TwoPointEstimatorError('Input estimator has no jackknife realizations') from exc
        return sep, wp
    cov = (len(realizations) - 1) * np.cov(realizations, rowvar=False, ddof=0)
    return sep, wp, cov
