"""Implement correlation function estimators, natural and Landy-Szalay."""

import numpy as np
from scipy import special

from .twopoint_counter import get_twopoint_counter
from .utils import BaseClass


class EstimatorError(Exception):

    """Exception raised when issue with estimator."""


def get_estimator(estimator='auto', with_DR=True):
    """
    Return :class:`BaseTwoPointEstimator` subclass corresponding
    to input estimator name.

    Parameters
    ----------
    estimator : string, default='auto'
        Estimator name, one of ["auto", "natural", "davispeebles", "hamilton", "landyszalay"].
        If "auto", "landyszalay" will be chosen if ``with_DR``, else "natural".

    with_DR : bool, default=True
        If estimator will be provided with pair counts from data x random catalogs. See above.

    Returns
    -------
    estimator : type
        Estimator class.
    """
    if estimator == 'auto':
        estimator = {True:'landyszalay', False:'natural'}[with_DR]

    if isinstance(estimator, str):

        try:
            return BaseTwoPointEstimator._registry[estimator.lower()]
        except KeyError:
            raise EstimatorError('Unknown estimator {}.'.format(estimator))

    return estimator


class MetaTwoPointEstimator(type(BaseClass)):

    """Metaclass to return correct estimator."""

    def __call__(cls, *args, estimator='landyszalay', **kwargs):
        return get_estimator(estimator=estimator)(*args, **kwargs)


class TwoPointEstimator(metaclass=MetaTwoPointEstimator):
    """
    Entry point to two point estimators.

    Parameters
    ----------
    estimator : string, default='landyszalay'
        Estimator name, one of ["auto", "natural", "davispeebles", "hamilton", "landyszalay"].

    args : list
        Arguments for two point estimator, see :class:`TwoPointEstimator`.

    kwargs : dict
        Arguments for two point estimator, see :class:`TwoPointEstimator`.

    Returns
    -------
    estimator : BaseTwoPointEstimator
        Estimator instance.
    """
    @classmethod
    def load(cls, filename):
        cls.log_info('Loading {}.'.format(filename))
        state = np.load(filename, allow_pickle=True)[()]
        return get_estimator(state.pop('name')).from_state(state)


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

    def __init__(self, D1D2=None, R1R2=None, D1R2=None, D2R1=None, W1W2=None, D1W2=None):
        """
        Initialize :class:`BaseTwoPointEstimator`, and set correlation
        estimation :attr:`corr` (calling :meth:`run`).

        Parameters
        ----------
        D1D2 : BaseTwoPointCounter, default=None
            D1D2 pair counts.

        R1R2 : BaseTwoPointCounter, default=None
            R1R2 pair counts.

        D1R2 : BaseTwoPointCounter, default=None
            D1R2 pair counts, e.g. for :class:`LandySzalayTwoPointEstimator`.

        D2R1 : BaseTwoPointCounter, default=None
            D2R1 pair counts, e.g. for :class:`LandySzalayTwoPointEstimator`,
            in case of cross-correlation.

        W1W2 : BaseTwoPointCounter, default=None
            W1W2 pair counts, e.g. for :class:`LandySzalayTwoPointEstimator`,
            in case window (W) is different from randoms (R).
            e.g. with reconstruction, the Landy-Szalay estimator is commonly written:
            :math:`(DD - 2 DS - SS)/RR`. Within our conventions, this is: :math:`(DD - 2 DR - RR)/WW`
            (more generally, including cross-correlations, :math:`(D1D2 - D1R2 - D2R1 - R1R2)/W1W2`).

        D1W2 : BaseTwoPointCounter, default=None
            D1W2 pair counts, e.g. for :class:`DavisPeeblesTwoPointEstimator`,
            in case window (W) is different from randoms (R). See ``W1W2``.
        """
        self.D1D2 = D1D2
        self.R1R2 = R1R2
        self.D1R2 = D1R2
        self.D2R1 = D2R1
        self.W1W2 = W1W2
        self.D1W2 = D1W2
        if D2R1 is None: # D1 = D2 and R1 = R2
            self.D2R1 = D1R2
        if W1W2 is None:
            self.W1W2 = R1R2
        if D1W2 is None:
            self.D1W2 = self.D1R2
        self.run()

    @property
    def sep(self):
        """Array of separation values of first dimension, taken from :attr:`W1W2` (or :attr:`D1R2`) if provided, else :attr:`D1D2`."""
        if self.W1W2 is not None:
            return self.W1W2.sep
        return self.D1D2.sep

    @property
    def seps(self):
        """Array of separation values, taken from :attr:`W1W2` (or :attr:`D1R2`) if provided, else :attr:`D1D2`."""
        if self.W1W2 is not None:
            return self.R1R2.seps
        return self.D1D2.seps

    @property
    def edges(self):
        """Edges for pair count calculation, taken from :attr:`D1D2`."""
        return self.D1D2.edges

    @property
    def mode(self):
        """Pair counting mode, taken from :attr:`D1D2`."""
        return self.D1D2.mode

    @property
    def ndim(self):
        """Return binning dimensionality."""
        return len(self.edges)

    @property
    def autocorr(self):
        return self.D1D2.autocorr

    @property
    def with_window(self):
        return self.W1W2 is not self.R1R2 or self.D1W2 is not self.D1R2

    @classmethod
    def requires(cls, autocorr=False, window=False, join=None):
        """Yield required pair counts."""
        for tu in cls._tuple_requires(autocorr=autocorr, window=window):
            if join is not None:
                yield join.join(tu)
            else:
                yield tu

    @classmethod
    def _tuple_requires(cls, autocorr=False, window=False, join=None):
        yield 'D1','D2'
        yield 'D1','R2'
        if not autocorr:
            yield 'D2','R1'
        yield 'R1','R2'
        if window:
            yield 'W1','W2'

    def rebin(self, *args, **kwargs):
        """Rebin estimator, by rebinning all pair counts. See :meth:`BaseTwoPointCounter.rebin`."""
        for pair in self.requires(autocorr=self.autocorr, window=self.with_window, join=''):
            getattr(self, pair).rebin(*args, **kwargs)
        self.run()

    def __getstate__(self):
        state = {}
        for name in ['name']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for pair in self.requires(autocorr=self.autocorr, window=self.with_window, join=''):
            state[pair] = getattr(self, pair).__getstate__()
        return state

    def __setstate__(self, state):
        kwargs = {}
        for pair in self.requires(autocorr=False, window=True, join=''):
            if pair in state:
                pair_state = state[pair].copy()
                kwargs[pair] = get_twopoint_counter(pair_state.pop('name')).from_state(pair_state)
        self.__init__(**kwargs)


class NaturalTwoPointEstimator(BaseTwoPointEstimator):

    name = 'natural'

    def run(self):
        """
        Set correlation function estimate :attr:`corr` based on the natural estimator
        :math:`(D1D2 - R1R2)/W1W2`.
        """
        nonzero = self.W1W2.wcounts != 0
        # init
        corr = np.empty_like(self.W1W2.wcounts, dtype='f8')
        corr[...] = np.nan

        # the natural estimator
        # (DD - RR) / RR
        DD = self.D1D2.normalized_wcounts()[nonzero]
        RR = self.R1R2.normalized_wcounts()[nonzero]
        WW = self.W1W2.normalized_wcounts()[nonzero]
        tmp = (DD - RR)/WW
        corr[nonzero] = tmp[...]
        self.corr = corr

    @classmethod
    def _tuple_requires(cls, autocorr=False, window=False, join=None):
        yield 'D1','D2'
        yield 'R1','R2'
        if window:
            yield 'W1','W2'


class DavisPeeblesTwoPointEstimator(BaseTwoPointEstimator):

    name = 'davispeebles'

    def run(self):
        """
        Set correlation function estimate :attr:`corr` based on the Davis-Peebles estimator
        :math:`(D1D2 - D1R2)/D1W2`.
        """
        nonzero = self.D1W2.wcounts != 0
        # init
        corr = np.empty_like(self.D1W2.wcounts, dtype='f8')
        corr[...] = np.nan

        # the natural estimator
        # (DD - RR) / RR
        DD = self.D1D2.normalized_wcounts()[nonzero]
        DR = self.D1R2.normalized_wcounts()[nonzero]
        DW = self.D1W2.normalized_wcounts()[nonzero]
        tmp = (DD - DR)/DW
        corr[nonzero] = tmp[...]
        self.corr = corr

    @classmethod
    def _tuple_requires(cls, autocorr=False, window=False, join=None):
        """Yield required pair counts."""
        yield 'D1','D2'
        yield 'D1','R2'
        if window:
            yield 'D1','W2'


class HamiltonTwoPointEstimator(DavisPeeblesTwoPointEstimator):

    name = 'hamilton'

    def run(self):
        """
        Set correlation function estimate :attr:`corr` based on the Hamilton estimator
        :math:`(D1D2 - D1R2^{2})/D1W2^{2}`.
        """
        nonzero = self.D1W2.wcounts != 0
        # init
        corr = np.empty_like(self.D1W2.wcounts, dtype='f8')
        corr[...] = np.nan

        # the natural estimator
        # (DD - RR) / RR
        DD = self.D1D2.normalized_wcounts()[nonzero]
        DR = self.D1R2.normalized_wcounts()[nonzero]
        DW = self.D1W2.normalized_wcounts()[nonzero]
        tmp = (DD - DR**2)/DW**2
        corr[nonzero] = tmp[...]
        self.corr = corr


class LandySzalayTwoPointEstimator(BaseTwoPointEstimator):

    name = 'landyszalay'

    def run(self):
        """
        Set correlation function estimate :attr:`corr` based on the Landy-Szalay estimator
        :math:`(D1D2 - D1R2 - D2R1 - R1R2)/W1W2`.
        """
        nonzero = self.W1W2.wcounts != 0
        # init
        corr = np.empty_like(self.W1W2.wcounts, dtype='f8')
        corr[...] = np.nan

        # the Landy - Szalay estimator
        # (DD - DR - RD + RR) / RR
        DD = self.D1D2.normalized_wcounts()[nonzero]
        RR = self.R1R2.normalized_wcounts()[nonzero]
        DR = self.D1R2.normalized_wcounts()[nonzero]
        RD = self.D2R1.normalized_wcounts()[nonzero]
        WW = self.W1W2.normalized_wcounts()[nonzero]

        tmp = (DD - DR - RD - RR)/WW
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
    def _yield_requires(cls, autocorr=False, window=False, join=None):
        yield 'D1','D2'
        yield 'R1','R2'

    @property
    def weight(self):
        """Another name for :attr:`corr`."""
        return self.corr


def project_to_multipoles(estimator, ells=(0,2,4)):
    r"""
    Project :math:`(s, \mu)` correlation function estimation onto Legendre polynomials.

    Parameters
    ----------
    estimator : BaseTwoPointEstimator
        Estimator for :math:`(s, \mu)` correlation function.

    ells : tuple, int, default=(0,2,4)
        Order of Legendre polynomial.

    Returns
    -------
    sep : array
        Array of separation values.

    toret : list
        List of correlation function multipoles.
    """
    if np.ndim(ells) == 0:
        ells = (ells,)
    ells = tuple(ells)
    sep = np.nanmean(estimator.sep, axis=-1)
    toret = []
    for ill,ell in enumerate(ells):
        dmu = np.diff(estimator.edges[1], axis=-1)
        poly = special.legendre(ell)(estimator.edges[1])
        legendre = (2*ell + 1) * (poly[1:] + poly[:-1])/2. * dmu
        toret.append(np.sum(estimator.corr*legendre, axis=-1)/np.sum(dmu))
    return sep, toret


def project_to_wp(estimator, pimax=None):
    r"""
    Integrate :math:`(r_{p}, \pi)` correlation function over :math:`\pi`
    to obtain :math:`w_{p}(r_{p})`.

    Parameters
    ----------
    estimator : BaseTwoPointEstimator
        Estimator for :math:`(r_{p}, \pi)` correlation function.

    pimax : float, default=None
        Upper bound for summation of :math:`\pi`.

    Returns
    -------
    sep : array
        Array of separation values.

    toret : array
        Estimated :math:`w_{p}(r_{p})`.
    """
    mask = Ellipsis
    if pimax is not None:
        mask = (estimator.edges[1] <= pimax)[:-1]
    sep = np.nanmean(estimator.sep[:,mask], axis=-1)
    wp = 2.*np.sum(estimator.corr[:,mask]*np.diff(estimator.edges[1])[mask], axis=-1)
    return sep, wp
