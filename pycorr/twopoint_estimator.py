"""Implements correlation function estimators, natural and Landy-Szalay."""

import numpy as np
from scipy import special

from .twopoint_counter import TwoPointCounter
from .utils import BaseClass


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


class MetaTwoPointEstimator(type(BaseClass)):

    """Metaclass to return correct estimator."""

    def __call__(cls, *args, estimator='landyszalay', **kwargs):
        return get_twopoint_estimator(estimator=estimator)(*args, **kwargs)


class TwoPointEstimator(BaseClass, metaclass=MetaTwoPointEstimator):
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
    @classmethod
    def from_state(cls, state):
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


class BaseTwoPointEstimator(BaseClass,metaclass=RegisteredTwoPointEstimator):
    """
    Base class for estimators.
    Extend this class to implement a new estimator.

    Attributes
    ----------
    corr : array
        Correlation function estimation.
    """
    name = 'base'

    def __init__(self, D1D2=None, R1R2=None, D1R2=None, D2R1=None, S1S2=None, D1S2=None, D2S1=None):
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

        D2R1 : BaseTwoPointCounter, default=None
            D2R1 two-point counts, e.g. for :class:`LandySzalayTwoPointEstimator`,
            in case of cross-correlation.

        S1S2 : BaseTwoPointCounter, default=None
            S1S2 two-point counts, e.g. with reconstruction, the Landy-Szalay estimator is commonly written:
            :math:`(D1D2 - D1S2 - D2S1 - S1S2)/R1R2`, with S1 and S2 shifted random catalogs.
            Defaults to ``R1R2``.

        D1S2 : BaseTwoPointCounter, default=None
            D1S2 two-point counts, see ``S1S2``. Defaults to ``D1R2``.

        S1D2 : BaseTwoPointCounter, default=None
            S1D2 two-point counts, see ``S1S2``. Defaults to ``D2R1``.
        """
        self.D1D2 = D1D2
        self.R1R2 = R1R2
        self.D1R2 = D1R2
        self.D2R1 = D2R1
        self.S1S2 = S1S2
        self.D1S2 = D1S2
        self.D2S1 = D2S1
        if D2R1 is None: # D1 = D2 and R1 = R2
            self.D2R1 = D1R2
        if S1S2 is None:
            self.S1S2 = self.R1R2
        if D1S2 is None:
            self.D1S2 = self.D1R2
        if D2S1 is None:
            if D1S2 is not None: # autocorr
                self.D2S1 = self.D1S2
            else:
                self.D2S1 = self.D2R1
        self.run()

    @property
    def sep(self):
        """Array of separation values of first dimension, taken from :attr:`R1R2` if provided, else :attr:`D1D2`."""
        if self.R1R2 is not None:
            return self.R1R2.sep
        return self.D1D2.sep

    @property
    def seps(self):
        """Array of separation values, taken from :attr:`R1R2` if provided, else :attr:`D1D2`."""
        if self.R1R2 is not None:
            return self.R1R2.seps
        return self.D1D2.seps

    @property
    def edges(self):
        """Edges for two-point count calculation, taken from :attr:`D1D2`."""
        return self.D1D2.edges

    @property
    def mode(self):
        """Two-point counting mode, taken from :attr:`D1D2`."""
        return self.D1D2.mode

    @property
    def shape(self):
        """Return shape of obtained correlation :attr:`corr`."""
        return tuple(len(edges) - 1 for edges in self.edges)

    @property
    def ndim(self):
        """Return binning dimensionality."""
        return len(self.edges)

    @property
    def autocorr(self):
        """Is this an autocorrelation?"""
        return self.D1D2.autocorr

    @property
    def with_shifted(self):
        """Do we have "shifted" (e.g. for reconstruction) two-point counts?"""
        return self.S1S2 is not self.R1R2 or self.D1S2 is not self.D1R2 or self.D2S1 is not self.D2R1

    @classmethod
    def requires(cls, autocorr=False, with_shifted=False, join=None):
        """List required counts."""
        toret = []
        for tu in cls._tuple_requires(autocorr=autocorr, with_shifted=with_shifted):
            if join is not None:
                toret.append(join.join(tu))
            else:
                toret.append(tu)
        return toret

    @classmethod
    def _tuple_requires(cls, autocorr=False, with_shifted=False, join=None):
        toret = []
        toret.append(('D1','D2'))
        key = 'S' if with_shifted else 'R'
        toret.append(('D1','{}2'.format(key)))
        if not autocorr:
            toret.append(('D2','{}1'.format(key)))
        toret.append(('{}1'.format(key),'{}2'.format(key)))
        if with_shifted:
            toret.append(('R1','R2'))
        return toret

    @property
    def _count_names(self):
        """Return list of counts used in estimator."""
        return self.requires(autocorr=self.autocorr, with_shifted=self.with_shifted, join='')

    def rebin(self, *args, **kwargs):
        """Rebin estimator, by rebinning all two-point counts. See :meth:`BaseTwoPointCounter.rebin`."""
        for name in self._count_names:
            getattr(self, name).rebin(*args, **kwargs)
        self.run()

    def __getstate__(self):
        state = {}
        for name in ['name']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for name in self._count_names:
            state[name] = getattr(self, name).__getstate__()
        return state

    def __setstate__(self, state):
        kwargs = {}
        counts = set(self.requires(autocorr=False, with_shifted=True, join='')) | set(self.requires(autocorr=False, with_shifted=False, join='')) # most general list
        for name in counts:
            if name in state:
                kwargs[name] = TwoPointCounter.from_state(state[name])
        self.__init__(**kwargs)

    def save(self, filename):
        """Save estimator to ``filename``."""
        if not self.D1D2.with_mpi or self.D1D2.mpicomm.rank == 0:
            super(BaseTwoPointEstimator, self).save(filename)
        if self.D1D2.with_mpi:
            self.D1D2.mpicomm.Barrier()


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
    def _tuple_requires(cls, autocorr=False, with_shifted=False, join=None):
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
    def _tuple_requires(cls, autocorr=False, with_shifted=False, join=None):
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

    def run(self):
        """
        Set correlation function estimate :attr:`corr` based on the Landy-Szalay estimator
        :math:`(D1D2 - D1S2 - D2S1 - S1S2)/R1R2`.
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
        SD = self.D2S1.normalized_wcounts()[nonzero]
        SS = self.S1S2.normalized_wcounts()[nonzero]

        tmp = (DD - DS - SD + SS)/RR
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

        # the natural estimator
        # (DD - RR) / RR
        DD = self.D1D2.normalized_wcounts()[nonzero]
        RR = self.R1R2.normalized_wcounts()[nonzero]
        tmp = RR/DD
        corr[nonzero] = tmp[...]
        self.corr = corr

    @classmethod
    def _yield_requires(cls, autocorr=False, with_shifted=False, join=None):
        toret = []
        toret.append(('D1','D2'))
        toret.append(('R1','R2'))
        return toret

    @property
    def weight(self):
        """Another name for :attr:`corr`."""
        return self.corr


def project_to_multipoles(estimator, ells=(0,2,4), **kwargs):
    r"""
    Project :math:`(s, \mu)` correlation function estimation onto Legendre polynomials.

    Parameters
    ----------
    estimator : BaseTwoPointEstimator
        Estimator for :math:`(s, \mu)` correlation function.
        If estimator holds jackknife realizations, also return jackknife covariance estimate.

    ells : tuple, int, default=(0,2,4)
        Order of Legendre polynomial.

    kwargs : dict
        Optional arguments for :math:`JackknifeTwoPointEstimator.realization`, when relevant.

    Returns
    -------
    sep : array
        Array of separation values.

    poles : list
        List of correlation function multipoles.

    cov : array
        If input ``estimator`` holds jackknife realizations, jackknife covariance estimate (for all successive ``ells``).
    """
    if np.ndim(ells) == 0:
        ells = (ells,)
    ells = tuple(ells)
    sep = np.nanmean(estimator.sep, axis=-1)
    poles = []
    for ill,ell in enumerate(ells):
        dmu = np.diff(estimator.edges[1], axis=-1)
        poly = special.legendre(ell)(estimator.edges[1])
        legendre = (2*ell + 1) * (poly[1:] + poly[:-1])/2. * dmu
        poles.append(np.sum(estimator.corr*legendre, axis=-1)/np.sum(dmu))
    try:
        realizations = [np.concatenate(project_to_multipoles(estimator.realization(ii, **kwargs), ells=ells)[1]).T for ii in estimator.realizations]
    except AttributeError:
        return sep, poles
    cov = (len(realizations) - 1) * np.cov(realizations, rowvar=False, ddof=0)
    return sep, poles, cov


def project_to_wp(estimator, pimax=None, **kwargs):
    r"""
    Integrate :math:`(r_{p}, \pi)` correlation function over :math:`\pi` to obtain :math:`w_{p}(r_{p})`.

    Parameters
    ----------
    estimator : BaseTwoPointEstimator
        Estimator for :math:`(r_{p}, \pi)` correlation function.
        If estimator holds jackknife realizations, also return jackknife covariance estimate.

    pimax : float, default=None
        Upper bound for summation of :math:`\pi`.

    kwargs : dict
        Optional arguments for :math:`JackknifeTwoPointEstimator.realization`, when relevant.

    Returns
    -------
    sep : array
        Array of separation values.

    wp : array
        Estimated :math:`w_{p}(r_{p})`.

    cov : array
        If input ``estimator`` holds jackknife realizations, jackknife covariance estimate.
    """
    mask = Ellipsis
    if pimax is not None:
        mask = (estimator.edges[1] <= pimax)[:-1]
    sep = np.nanmean(estimator.sep[:,mask], axis=-1)
    wp = 2.*np.sum(estimator.corr[:,mask]*np.diff(estimator.edges[1])[mask], axis=-1)
    try:
        realizations = [project_to_wp(estimator.realization(ii, **kwargs), pimax=pimax)[1] for ii in estimator.realizations]
    except AttributeError:
        return sep, wp
    cov = (len(realizations) - 1) * np.cov(realizations, rowvar=False, ddof=0)
    return sep, wp, cov
