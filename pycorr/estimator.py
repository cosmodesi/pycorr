"""Implement correlation function estimators, natural and Landy-Szalay."""

import numpy as np
from scipy import special

from .utils import BaseClass


class EstimatorError(Exception):

    """Exception raised when issue with estimator."""


class BaseTwoPointEstimator(BaseClass):
    """
    Base class for estimators.
    Extend this class to implement a new estimator.

    Attributes
    ----------
    corr : array
        Correlation function estimation.
    """
    def __init__(self, D1D2=None, R1R2=None, D1R2=None, D2R1=None):
        """
        Initialize :class:`BaseTwoPointEstimator`, and set correlation
        estimation :attr:`corr` (calling :meth:`run`).

        Parameters
        ----------
        D1D2 : BaseTwoPointCounterEngine, default=None
            D1D2 pair counts.

        R1R2 : BaseTwoPointCounterEngine, default=None
            R1R2 pair counts.

        D1R2 : BaseTwoPointCounterEngine, default=None
            D1R2 pair counts, e.g. for :class:`LandySzalayTwoPointEstimator`.

        D2R1 : BaseTwoPointCounterEngine, default=None
            D2R1 pair counts, e.g. for :class:`LandySzalayTwoPointEstimator`,
            in case of cross-correlation.
        """
        self.D1D2 = D1D2
        self.R1R2 = R1R2
        self.D1R2 = D1R2
        self.D2R1 = D2R1
        if D2R1 is None: # D1 = D2 and R1 = R2
            self.D2R1 = D1R2
        self.run()

    @property
    def sep(self):
        """Array of separation values, taken from :attr:`R1R2` if provided, else :attr:`D1D2`."""
        if self.R1R2 is not None:
            return self.R1R2.sep
        return self.D1D2.sep

    @property
    def edges(self):
        """Edges for pair count calculation, taken from :attr:`D1D2`."""
        return self.D1D2.edges

    @property
    def autocorr(self):
        """Whether correlation function is an autocorrelation, i.e. :attr:`D2R1` is :attr:`D1R2`."""
        return self.D2R1 is self.D1R2

    @classmethod
    def requires(cls, autocorr=False):
        """Yield required pair counts."""
        yield 'D1','D2'
        yield 'D1','R2'
        if not autocorr:
            yield 'D2','R1'
        yield 'R1','R2'

    def rebin(self, *args, **kwargs):
        """Rebin estimator, by rebinning all pair counts. See :meth:`BaseTwoPointCounterEngine.rebin`."""
        for pair in self.requires(autocorr=self.autocorr):
            getattr(self,pair).rebin(*args, **kwargs)
        self.run()

    def __getstate__(self):
        state = {}
        for pair in self.requires(autocorr=self.autocorr):
            state[pair] = getattr(self, pair).__getstate__()
        return state

    def __setstate__(self, state):
        kwargs = {}
        for pair, pair_state in state.items():
            kwargs[pair] = BaseTwoPointCounterEngine.from_state(pair_state)
        self.__init__(**kwargs)


class NaturalTwoPointEstimator(BaseTwoPointEstimator):

    def run(self):
        """
        Set correlation function estimate :attr:`corr` based on the natural estimator
        :math:`DD/RR - 1`.
        """
        nonzero = self.R1R2.wcounts != 0
        # init
        corr = np.empty_like(self.R1R2.wcounts,dtype='f8')
        corr[...] = np.nan

        # the natural estimator
        # (DD - RR) / RR
        DD = self.D1D2.normalized_wcounts()[nonzero]
        RR = self.R1R2.normalized_wcounts()[nonzero]
        tmp = DD/RR - 1
        corr[nonzero] = tmp[...]
        self.corr = corr

    @classmethod
    def requires(cls, autocorr=False):
        """Yield required pair counts."""
        yield 'D1','D2'
        yield 'R1','R2'


class LandySzalayTwoPointEstimator(BaseTwoPointEstimator):

    def run(self):
        """
        Set correlation function estimate :attr:`corr` based on the Landy-Szalay estimator
        :math:`(DD - DR - RD)/RR + 1`.
        """
        nonzero = self.R1R2.wcounts != 0
        # init
        corr = np.empty_like(self.R1R2.wcounts,dtype='f8')
        corr[...] = np.nan

        # the Landy - Szalay estimator
        # (DD - DR - RD + RR) / RR
        DD = self.D1D2.normalized_wcounts()[nonzero]
        RR = self.R1R2.normalized_wcounts()[nonzero]
        DR = self.D1R2.normalized_wcounts()[nonzero]
        RD = self.D2R1.normalized_wcounts()[nonzero]
        tmp = (DD - DR - RD)/RR + 1
        corr[nonzero] = tmp[...]
        self.corr = corr


def project_to_multipoles(estimator, ells=(0,2,4)):
    r"""
    Project :math:`(s, \mu)` correlation function estimation onto Legendre polynomials.

    Parameters
    ----------
    estimator : BaseTwoPointEstimator
        Estimator for :math:`(s, \mu)` correlation function.

    ells : tuple, int
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
    sep = np.mean(self.sep, axis=-1)
    toret = []
    for ill,ell in enumerate(ells):
        dmu = np.diff(self.edges[1], axis=-1)
        poly = special.legendre(ell)(edges)
        legendre = (2*ell + 1) * (poly[1:] + poly[:-1])/2. * dmu
        toret.append(np.sum(self.corr*legendre, axis=-1)/np.sum(dmu))
    return sep, toret


def get_estimator(estimator='auto', has_cross=True):
    """
    Return :class:`BaseTwoPointEstimator` subclass corresponding
    to input estimator name.

    Parameters
    ----------
    estimator : string, default='auto'
        Estimator name, one of ["auto", "natural", "landyszalay"].
        If "auto", "landyszalay" will be chosen if ``has_cross``,
        else "natural".

    has_randoms : bool, default=True
        If estimator will be provided with pair counts from data x randoms
        catalogs. See above.

    Returns
    -------
    estimator : type
        Estimator class.
    """
    if estimator == 'auto':
        estimator = {True:'landyszalay', False:'natural'}[has_cross]

    if isinstance(estimator, str):

        if estimator.lower() == 'natural':
            return NaturalTwoPointEstimator

        if estimator.lower() == 'landyszalay':
            return LandySzalayTwoPointEstimator

        raise EstimatorError('Unknown estimator {}.'.format(estimator))

    return estimator


def TwoPointEstimator(*args, estimator='landyszalay', **kwargs):
    """
    Return :class:`BaseTwoPointEstimator` instance corresponding
    to input estimator name.

    Parameters
    ----------
    estimator : string, default='landyszalay'
        Estimator name, one of ["natural", "landyszalay"].

    args : list
        Arguments for pair counter engine, see :class:`TwoPointEstimator`.

    kwargs : dict
        Arguments for pair counter engine, see :class:`TwoPointEstimator`.

    Returns
    -------
    estimator : BaseTwoPointEstimator
        Estimator instance.
    """
    return get_estimator(estimator=estimator)(*args, **kwargs)
