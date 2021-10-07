import numpy as np
from scipy import special

from .utils import BaseClass


class EstimatorError(Exception):

    pass


class BaseTwoPointEstimator(BaseClass):

    def __init__(self, D1D2=None, R1R2=None, D1R2=None, D2R1=None):
        self.D1D2 = D1D2
        self.R1R2 = R1R2
        self.D1R2 = D1R2
        self.D2R1 = D2R1
        if D2R1 is None: # D1 = D2 and R1 = R2
            self.D2R1 = D1R2
        self.run()

    @property
    def sep(self):
        if self.R1R2 is not None:
            return self.R1R2.sep
        return self.D1D2.sep

    @property
    def edges(self):
        return self.D1D2.edges

    @property
    def autocorr(self):
        return self.D2R1 == self.D1R2

    @classmethod
    def requires(cls, autocorr=False):
        yield 'D1','D2'
        yield 'D1','R2'
        if not autocorr:
            yield 'D2','R1'
        yield 'R1','R2'

    def rebin(self, *args, **kwargs):
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
        yield 'D1','D2'
        yield 'R1','R2'


class LandySzalayTwoPointEstimator(BaseTwoPointEstimator):

    def run(self):
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


def get_estimator(estimator='landyszalay', has_randoms=True):

    if estimator is None:
        estimator = {True:'landyszalay',False:'natural'}[has_randoms]

    if isinstance(estimator, str):

        if estimator.lower() == 'natural':
            return NaturalTwoPointEstimator

        if estimator.lower() == 'landyszalay':
            return LandySzalayTwoPointEstimator

        raise EstimatorError('Unknown estimator {}.'.format(estimator))

    return estimator


def TwoPointEstimator(*args, estimator='landyszalay', **kwargs):

    return get_estimator(estimator=estimator)(*args, **kwargs)
