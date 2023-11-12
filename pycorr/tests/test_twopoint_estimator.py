import os
import tempfile
import pytest

import numpy as np

from pycorr import (TwoPointCorrelationFunction, TwoPointEstimator, TwoPointCounter,
                    JackknifeTwoPointEstimator, project_to_poles, project_to_wp, setup_logging)
from pycorr.twopoint_estimator import TwoPointEstimatorError


def generate_catalogs(size=100, boxsize=(1000,) * 3, offset=(0, 0, 0), n_individual_weights=1, n_bitwise_weights=0, seed=42):
    rng = np.random.RandomState(seed=seed)
    toret = []
    for i in range(2):
        positions = [o + rng.uniform(0., 1., size) * b for o, b in zip(offset, boxsize)]
        weights = [rng.randint(0, 0xffffffff, size, dtype='i8') for i in range(n_bitwise_weights)]
        weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
        toret.append(positions + weights)
    return toret


def test_multipoles():

    class Estimator(object):

        def sepavg(self, axis=0):
            return estimator.sep[:, 0]

    from scipy import special

    estimator = Estimator()
    for muedges in [np.linspace(0., 1., 10), np.linspace(0., 1., 100), np.linspace(-1., 1., 100)]:
        edges = [np.linspace(0., 1., 10), muedges]
        ss, mumu = np.meshgrid(*[(e[:-1] + e[1:]) / 2. for e in edges], indexing='ij')
        estimator.sep, estimator.edges, estimator.corr = ss, edges, np.ones_like(ss)
        s, xi = project_to_poles(estimator, ells=(0, 2, 4))
        assert np.allclose(xi[0], 1., atol=1e-9)
        for xiell in xi[1:]: assert np.allclose(xiell, 0., atol=1e-9)

    edges = [np.linspace(0., 1., 10), np.linspace(0., 1., 100)]
    ss, mumu = np.meshgrid(*[(e[:-1] + e[1:]) / 2. for e in edges], indexing='ij')
    ells = (0, 2, 4)
    for ellin in ells[1:]:
        estimator.sep, estimator.edges, estimator.corr = ss, edges, special.legendre(ellin)(mumu)
        s, xi = project_to_poles(estimator, ells=ells)
        for ell, xiell in zip(ells, xi):
            assert np.allclose(xiell, 1. if ell == ellin else 0., atol=1e-3)


def test_estimator(mode='s'):

    from pycorr import KMeansSubsampler
    list_engine = ['corrfunc']
    edges = np.linspace(1, 100, 10)
    size = 1000
    cboxsize = (500,) * 3
    from collections import namedtuple
    TwoPointWeight = namedtuple('TwoPointWeight', ['sep', 'weight'])
    twopoint_weights = TwoPointWeight(np.logspace(-4, 0, 40), np.linspace(4., 1., 40))

    list_options = []

    for autocorr in [False, True]:
        for with_shifted in [False, True]:

            list_options.append({'autocorr': autocorr, 'with_shifted': with_shifted, 'weights_one': ['D1', 'R2']})
            if mode not in ['theta', 'rp']:
                list_options.append({'autocorr': autocorr, 'with_shifted': with_shifted, 'estimator': 'natural', 'boxsize': cboxsize, 'with_randoms': False})
            if mode == 'rppi':
                list_options.append({'autocorr': autocorr, 'with_shifted': with_shifted, 'estimator': 'natural', 'boxsize': cboxsize, 'with_randoms': False, 'edges': (np.linspace(0, 100, 21), np.linspace(0, 20, 21))})

            for estimator in ['natural', 'landyszalay', 'davispeebles', 'weight', 'residual']:

                list_options.append({'autocorr': autocorr, 'with_shifted': with_shifted, 'estimator': estimator})
                if estimator not in ['weight']:
                    list_options.append({'autocorr': autocorr, 'with_shifted': with_shifted, 'estimator': estimator})

                # pip
                list_options.append({'autocorr': autocorr, 'with_shifted': with_shifted, 'estimator': estimator, 'n_individual_weights': 0})
                list_options.append({'autocorr': autocorr, 'with_shifted': with_shifted, 'estimator': estimator, 'n_individual_weights': 1, 'n_bitwise_weights': 1, 'compute_sepsavg': False})
                list_options.append({'autocorr': autocorr, 'with_shifted': with_shifted, 'estimator': estimator, 'n_individual_weights': 1, 'n_bitwise_weights': 1})

                # twopoint weights
                list_options.append({'autocorr': autocorr, 'with_shifted': with_shifted, 'estimator': estimator, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'twopoint_weights': twopoint_weights})

                # los
                if mode in ['smu', 'rppi']:
                    list_options.append({'autocorr': autocorr, 'with_shifted': with_shifted, 'estimator': estimator, 'los': 'firstpoint', 'twopoint_weights': twopoint_weights})
                    list_options.append({'autocorr': autocorr, 'with_shifted': with_shifted, 'estimator': estimator, 'los': 'endpoint'})

                # selection
                if mode == 'smu':
                    list_options.append({'autocorr': autocorr, 'with_shifted': with_shifted, 'estimator': estimator, 'los': 'midpoint', 'selection_attrs': {'rp': (5., np.inf)}})
                    if estimator == 'natural':
                        list_options.append({'autocorr': autocorr, 'with_shifted': with_shifted, 'estimator': estimator, 'boxsize': cboxsize, 'with_randoms': False, 'selection_attrs': {'rp': (5., np.inf)}})

    mpi = False
    try:
        from pycorr import mpi
        print('Has MPI')
    except ImportError:
        pass
    if mpi:
        list_options.append({'mpicomm': mpi.COMM_WORLD})

    # list_options.append({'weight_type':'inverse_bitwise', 'n_bitwise_weights':2})
    ref_edges = np.linspace(0, 100, 11)
    if mode == 'smu':
        ref_edges = (ref_edges, np.linspace(-1, 1, 21))
    elif mode == 'rppi':
        ref_edges = (ref_edges, np.linspace(-20, 20, 41))
    elif mode == 'theta':
        ref_edges = np.linspace(1e-5, 10, 11)  # below 1e-5, self pairs are counted by Corrfunc

    for engine in list_engine:
        for options in list_options:
            print(mode, options)
            options = options.copy()
            compute_sepsavg = options.get('compute_sepsavg', True)
            edges = options.pop('edges', ref_edges)
            weights_one = options.pop('weights_one', [])
            n_individual_weights = options.pop('n_individual_weights', 1)
            n_bitwise_weights = options.pop('n_bitwise_weights', 0)
            twopoint_weights = options.pop('twopoint_weights', None)
            data1, data2 = generate_catalogs(size, boxsize=cboxsize, n_individual_weights=n_individual_weights, n_bitwise_weights=n_bitwise_weights, seed=42)
            randoms1, randoms2 = generate_catalogs(size, boxsize=cboxsize, n_individual_weights=n_individual_weights, n_bitwise_weights=n_bitwise_weights, seed=43)
            shifted1, shifted2 = generate_catalogs(size, boxsize=cboxsize, n_individual_weights=n_individual_weights, n_bitwise_weights=n_bitwise_weights, seed=44)
            autocorr = options.pop('autocorr', False)
            mpicomm = options.pop('mpicomm', None)
            with_randoms = options.pop('with_randoms', True)
            with_shifted = options.pop('with_shifted', False)
            boxsize = options.get('boxsize', None)
            los = options['los'] = options.get('los', 'x' if boxsize is not None else 'midpoint')
            options['position_type'] = 'xyz'
            npos = 3
            for label, catalog in zip(['D1', 'D2', 'R1', 'R2'], [data1, data2, randoms1, randoms2]):
                if label in weights_one:
                    catalog.append(np.ones_like(catalog[0]))

            subsampler = KMeansSubsampler(mode='theta', positions=data1[:npos], nsamples=5, position_type='xyz')
            data1.append(subsampler.label(data1[:npos]))
            data2.append(subsampler.label(data2[:npos]))
            randoms1.append(subsampler.label(randoms1[:npos]))
            randoms2.append(subsampler.label(randoms2[:npos]))
            shifted1.append(subsampler.label(shifted1[:npos]))
            shifted2.append(subsampler.label(shifted2[:npos]))

            def run_nojackknife(ii=None, same_shotnoise=False, **kwargs):
                data_positions1, data_weights1, data_samples1 = data1[:npos], data1[npos:-1], data1[-1]
                data_positions2, data_weights2, data_samples2 = data2[:npos], data2[npos:-1], data2[-1]
                randoms_positions1, randoms_weights1, randoms_samples1 = randoms1[:npos], randoms1[npos:-1], randoms1[-1]
                randoms_positions2, randoms_weights2, randoms_samples2 = randoms2[:npos], randoms2[npos:-1], randoms2[-1]
                shifted_positions1, shifted_weights1, shifted_samples1 = shifted1[:npos], shifted1[npos:-1], shifted1[-1]
                shifted_positions2, shifted_weights2, shifted_samples2 = shifted2[:npos], shifted2[npos:-1], shifted2[-1]
                if same_shotnoise:
                    data_weights2 = data_weights1
                    randoms_weights2 = randoms_weights1
                    shifted_weights2 = shifted_weights1
                if ii is not None:
                    mask = data_samples1 == ii
                    data_positions1, data_weights1 = [position[~mask] for position in data_positions1], [weight[~mask] for weight in data_weights1]
                    mask = data_samples2 == ii
                    data_positions2, data_weights2 = [position[~mask] for position in data_positions2], [weight[~mask] for weight in data_weights2]
                    mask = randoms_samples1 == ii
                    randoms_positions1, randoms_weights1 = [position[~mask] for position in randoms_positions1], [weight[~mask] for weight in randoms_weights1]
                    mask = randoms_samples2 == ii
                    randoms_positions2, randoms_weights2 = [position[~mask] for position in randoms_positions2], [weight[~mask] for weight in randoms_weights2]
                    mask = shifted_samples1 == ii
                    shifted_positions1, shifted_weights1 = [position[~mask] for position in shifted_positions1], [weight[~mask] for weight in shifted_weights1]
                    mask = shifted_samples2 == ii
                    shifted_positions2, shifted_weights2 = [position[~mask] for position in shifted_positions2], [weight[~mask] for weight in shifted_weights2]
                if twopoint_weights is not None:
                    kwargs['D1D2_twopoint_weights'] = kwargs['D1R2_twopoint_weights'] = twopoint_weights

                return TwoPointCorrelationFunction(mode=mode, edges=edges, engine=engine, data_positions1=data_positions1, data_positions2=None if autocorr else data_positions2,
                                                   data_weights1=data_weights1, data_weights2=None if autocorr and not same_shotnoise else data_weights2,
                                                   randoms_positions1=randoms_positions1 if with_randoms else None, randoms_positions2=None if autocorr or not with_randoms else randoms_positions2,
                                                   randoms_weights1=randoms_weights1 if with_randoms else None, randoms_weights2=None if autocorr and not same_shotnoise or not with_randoms else randoms_weights2,
                                                   shifted_positions1=shifted_positions1 if with_shifted else None, shifted_positions2=None if autocorr or not with_shifted else shifted_positions2,
                                                   shifted_weights1=shifted_weights1 if with_shifted else None, shifted_weights2=None if autocorr and not same_shotnoise or not with_shifted else shifted_weights2,
                                                   **options, **kwargs)

            def run_jackknife(pass_none=False, pass_zero=False, same_shotnoise=False, **kwargs):
                data_positions1, data_weights1, data_samples1 = data1[:npos], data1[npos:-1], data1[-1]
                data_positions2, data_weights2, data_samples2 = data2[:npos], data2[npos:-1], data2[-1]
                randoms_positions1, randoms_weights1, randoms_samples1 = randoms1[:npos], randoms1[npos:-1], randoms1[-1]
                randoms_positions2, randoms_weights2, randoms_samples2 = randoms2[:npos], randoms2[npos:-1], randoms2[-1]
                shifted_positions1, shifted_weights1, shifted_samples1 = shifted1[:npos], shifted1[npos:-1], shifted1[-1]
                shifted_positions2, shifted_weights2, shifted_samples2 = shifted2[:npos], shifted2[npos:-1], shifted2[-1]
                if same_shotnoise:
                    data_weights2 = data_weights1
                    randoms_weights2 = randoms_weights1
                    shifted_weights2 = shifted_weights1

                def get_zero(arrays):
                    return [array[:0] for array in arrays]

                if pass_zero:
                    data_positions1, data_weights1, data_samples1 = get_zero(data_positions1), get_zero(data_weights1), data_samples1[:0]
                    data_positions2, data_weights2, data_samples2 = get_zero(data_positions2), get_zero(data_weights2), data_samples2[:0]
                    randoms_positions1, randoms_weights1, randoms_samples1 = get_zero(randoms_positions1), get_zero(randoms_weights1), randoms_samples1[:0]
                    randoms_positions2, randoms_weights2, randoms_samples2 = get_zero(randoms_positions2), get_zero(randoms_weights2), randoms_samples2[:0]
                    shifted_positions1, shifted_weights1, shifted_samples1 = get_zero(shifted_positions1), get_zero(shifted_weights1), shifted_samples1[:0]
                    shifted_positions2, shifted_weights2, shifted_samples2 = get_zero(shifted_positions2), get_zero(shifted_weights2), shifted_samples2[:0]

                if twopoint_weights is not None:
                    kwargs['D1D2_twopoint_weights'] = kwargs['D1R2_twopoint_weights'] = twopoint_weights

                return TwoPointCorrelationFunction(mode=mode, edges=edges, engine=engine, data_positions1=None if pass_none else data_positions1, data_positions2=None if pass_none or autocorr else data_positions2,
                                                   data_weights1=None if pass_none else data_weights1, data_weights2=None if pass_none or autocorr and not same_shotnoise else data_weights2,
                                                   randoms_positions1=randoms_positions1 if with_randoms and not pass_none else None, randoms_positions2=None if pass_none or autocorr else randoms_positions2,
                                                   randoms_weights1=randoms_weights1 if with_randoms and not pass_none else None, randoms_weights2=None if (pass_none or autocorr) and not same_shotnoise or not with_randoms else randoms_weights2,
                                                   shifted_positions1=shifted_positions1 if with_shifted and not pass_none else None, shifted_positions2=None if pass_none or autocorr else shifted_positions2,
                                                   shifted_weights1=shifted_weights1 if with_shifted and not pass_none else None, shifted_weights2=None if (pass_none or autocorr) and not same_shotnoise or not with_shifted else shifted_weights2,
                                                   data_samples1=None if pass_none else data_samples1, data_samples2=None if pass_none or autocorr else data_samples2,
                                                   randoms_samples1=None if pass_none else randoms_samples1, randoms_samples2=None if pass_none or autocorr else randoms_samples2,
                                                   shifted_samples1=None if pass_none else shifted_samples1, shifted_samples2=None if pass_none or autocorr else shifted_samples2,
                                                   **options, **kwargs)

            def assert_allclose(res1, res2):
                tol = {'atol': 1e-8, 'rtol': 1e-6}
                assert np.allclose(res2.wcounts, res1.wcounts, **tol)
                assert np.allclose(res2.wnorm, res1.wnorm, **tol)

            def assert_allclose_estimators(res1, res2):
                assert np.allclose(res2.corr, res1.corr, equal_nan=True)
                assert np.allclose(res2.sep, res1.sep, equal_nan=True)
                assert not np.any(res1.sep == 0.)

            estimator_nojackknife = run_nojackknife()
            estimator_jackknife = run_jackknife()
            if compute_sepsavg:
                assert not np.allclose(estimator_nojackknife.sepavg(), estimator_nojackknife.sepavg(method='mid'), equal_nan=True)
            assert_allclose_estimators(estimator_jackknife, estimator_nojackknife)

            if autocorr and (n_individual_weights or n_bitwise_weights):
                estimator_same_shotnoise = run_nojackknife(same_shotnoise=True)
                for label1, label2 in estimator_same_shotnoise.requires(with_reversed=True, with_shifted=estimator_same_shotnoise.with_shifted, join=None):
                    if label1[:-1] == label2[:-1]:
                        label12 = label1 + label2
                        if getattr(estimator_same_shotnoise, label12).name != 'analytic':
                            assert getattr(estimator_same_shotnoise, label12).same_shotnoise
                            assert not getattr(estimator_same_shotnoise, label12).autocorr
                #print(estimator_same_shotnoise.count_names, estimator_nojackknife.count_names)
                #print(estimator_same_shotnoise.D1D2.wcounts / estimator_nojackknife.D1D2.wcounts, estimator_same_shotnoise.D1D2.wnorm, estimator_nojackknife.D1D2.wnorm)
                #print(estimator_same_shotnoise.D1S2.wcounts / estimator_nojackknife.D1S2.wcounts, estimator_same_shotnoise.D1S2.wnorm, estimator_nojackknife.D1S2.wnorm)
                #print(estimator_same_shotnoise.R1R2.wcounts / estimator_nojackknife.R1R2.wcounts, estimator_same_shotnoise.R1R2.wnorm, estimator_nojackknife.R1R2.wnorm)
                assert_allclose_estimators(estimator_same_shotnoise, estimator_nojackknife)
                estimator_same_shotnoise2 = estimator_same_shotnoise
                estimator_same_shotnoise = run_jackknife(same_shotnoise=True)
                for label1, label2 in estimator_same_shotnoise.requires(with_reversed=True, with_shifted=estimator_same_shotnoise.with_shifted, join=None):
                    if label1[:-1] == label2[:-1]:
                        label12 = label1 + label2
                        if getattr(estimator_same_shotnoise, label12).name != 'analytic':
                            assert getattr(estimator_same_shotnoise, label12).same_shotnoise
                            assert not getattr(estimator_same_shotnoise, label12).autocorr
                #for i in range(estimator_jackknife.nrealizations):
                    #print(i, estimator_same_shotnoise.realization(i).R1R2.wcounts / estimator_jackknife.realization(i).R1R2.wcounts)
                    #print(i, estimator_same_shotnoise.realization(i).D1R2.wcounts / estimator_jackknife.realization(i).D1R2.wcounts)
                    #print(i, estimator_same_shotnoise.realization(i).D1R2.size2, estimator_jackknife.realization(i).D1R2.size2)
                assert_allclose_estimators(estimator_same_shotnoise, estimator_same_shotnoise2)
                assert_allclose_estimators(estimator_same_shotnoise, estimator_jackknife)

            if mode in ['theta', 'rp']:
                assert estimator_jackknife.cov().shape == (estimator_jackknife.shape[0],) * 2

            nsplits = 10
            estimator_jackknife = JackknifeTwoPointEstimator.concatenate(*[run_jackknife(samples=samples) for samples in np.array_split(np.unique(data1[-1]), nsplits)])
            assert_allclose_estimators(estimator_jackknife, estimator_nojackknife)

            ii = data1[-1][0]
            estimator_nojackknife_ii = run_nojackknife(ii=ii)
            estimator_jackknife_ii = estimator_jackknife.realization(ii, correction=None)
            assert_allclose_estimators(estimator_jackknife_ii, estimator_nojackknife_ii)

            options_counts = options.copy()
            estimator = options_counts.pop('estimator', 'landyszalay')

            if estimator in ['landyszalay', 'davispeebles', 'natural', 'weight']:
                D1D2 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=data1[:npos], positions2=None if autocorr else data2[:npos],
                                       weights1=data1[npos:-1], weights2=None if autocorr else data2[npos:-1], twopoint_weights=twopoint_weights, **options_counts)
                assert_allclose(D1D2, estimator_jackknife.D1D2)
            if estimator in ['residual']:
                D1R2 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=data1[:npos], positions2=randoms1[:npos] if autocorr else randoms2[:npos],
                                       weights1=data1[npos:-1], weights2=randoms1[npos:-1] if autocorr else randoms2[npos:-1], twopoint_weights=twopoint_weights, **options_counts)
                assert_allclose(D1R2, estimator_jackknife.D1R2)
            if estimator in ['landyszalay', 'natural', 'residual'] and with_randoms:
                R1R2 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=randoms1[:npos], positions2=None if autocorr else randoms2[:npos],
                                       weights1=randoms1[npos:-1], weights2=None if autocorr else randoms2[npos:-1], **options_counts)
                assert_allclose(R1R2, estimator_jackknife.R1R2)
            if with_shifted:
                if estimator in ['residual']:
                    S1R2 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=shifted1[:npos], positions2=randoms1[:npos] if autocorr else randoms2[:npos],
                                           weights1=shifted1[npos:-1], weights2=randoms1[npos:-1] if autocorr else randoms2[npos:-1], **options_counts)
                    assert_allclose(S1R2, estimator_jackknife.S1R2)
                # For following computation
                tmp_randoms1 = shifted1
                tmp_randoms2 = shifted2
                tmp_twopoint_weights = None
            else:
                tmp_randoms1 = randoms1
                tmp_randoms2 = randoms2
                tmp_twopoint_weights = twopoint_weights
            if estimator in ['landyszalay', 'davispeebles']:
                D1R2 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=data1[:npos], positions2=tmp_randoms1[:npos] if autocorr else tmp_randoms2[:npos],
                                       weights1=data1[npos:-1], weights2=tmp_randoms1[npos:-1] if autocorr else tmp_randoms2[npos:-1], twopoint_weights=tmp_twopoint_weights, **options_counts)
                assert_allclose(D1R2, estimator_jackknife.D1S2)
                #assert estimator_jackknife.with_reversed == ((not autocorr or los in ['firstpoint', 'endpoint']) and estimator in ['landyszalay'])
                if (not autocorr or los in ['firstpoint', 'endpoint']) and estimator in ['landyszalay']:
                    R1D2 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=tmp_randoms1[:npos], positions2=data1[:npos] if autocorr else data2[:npos],
                                           weights1=tmp_randoms1[npos:-1], weights2=data1[npos:-1] if autocorr else data2[npos:-1], twopoint_weights=tmp_twopoint_weights if autocorr else None, **options_counts)
                    assert_allclose(R1D2, estimator_jackknife.S1D2)
                else:
                    assert_allclose(D1R2, estimator_jackknife.D1S2)
            if estimator in ['landyszalay', 'natural', 'weight'] and with_randoms:
                R1R2 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=tmp_randoms1[:npos], positions2=None if autocorr else tmp_randoms2[:npos],
                                       weights1=tmp_randoms1[npos:-1], weights2=None if autocorr else tmp_randoms2[npos:-1], **options_counts)
                assert_allclose(R1R2, estimator_jackknife.S1S2)
            if estimator in ['natural'] and not with_randoms:
                R1R2 = TwoPointCounter(mode=mode, edges=edges, engine='analytic', boxsize=estimator_jackknife.D1D2.boxsize,
                                       size1=estimator_jackknife.D1D2.size1, size2=None if estimator_jackknife.D1D2.autocorr else estimator_jackknife.D1D2.size2, los=options_counts['los'])
                assert_allclose(R1R2, estimator_jackknife.R1R2)

            for estimator in [estimator_nojackknife, estimator_jackknife]:
                estimator_renormalized = estimator.normalize(1.)
                for name in estimator_renormalized.count_names:
                    assert np.allclose(getattr(estimator_renormalized, name).wnorm, 1.)
                assert_allclose_estimators(estimator_renormalized, estimator_nojackknife)

                estimator_renormalized = estimator.normalize()
                for name in estimator_renormalized.count_names:
                    assert np.allclose(getattr(estimator_renormalized, name).wnorm, estimator_nojackknife.XX.wnorm)
                assert_allclose_estimators(estimator_renormalized, estimator_nojackknife)

                estimator_renormalized2 = estimator.normalize() + estimator.normalize()
                assert_allclose_estimators(estimator_renormalized2, estimator)

                if mode in ['smu', 'rppi'] and len(estimator.edges[1]) % 2 == 1:
                    estimator_wrapped = estimator.wrap()
                    assert np.all(estimator_wrapped.edges[1] >= 0.)
                    assert np.all(estimator_wrapped.seps[1][~np.isnan(estimator_wrapped.seps[1])] >= 0.)
                    assert np.allclose(np.nansum(estimator_wrapped.corr, axis=1), np.nansum(estimator_wrapped.corr, axis=1), equal_nan=True)

            if estimator_jackknife.mode == 'smu':
                sep, xiell = project_to_poles(estimator_nojackknife, ells=(0, 2, 4))
                sep, xiell, cov = project_to_poles(estimator_jackknife, ells=(0, 2, 4))
                assert cov.shape == (sum([len(xi) for xi in xiell]),) * 2

            if estimator_jackknife.mode == 'rppi':

                def get_sepavg(estimator, sepmax):
                    mid = [(edges[:-1] + edges[1:]) / 2. for edges in estimator.edges]
                    if not estimator.XX.compute_sepavg:
                        return mid[0]
                    mask = mid[1] <= sepmax
                    sep = estimator.seps[0].copy()
                    sep[np.isnan(sep)] = 0.
                    if getattr(estimator, 'R1R2', None) is not None and estimator.R1R2.compute_sepavg:
                        wcounts = estimator.R1R2.wcounts
                    else:
                        wcounts = estimator.XX.wcounts
                    with np.errstate(divide='ignore', invalid='ignore'):
                        return np.sum(sep[:, mask] * wcounts[:, mask], axis=-1) / np.sum(wcounts[:, mask], axis=-1)

                def ref_to_wp(estimator, pimax=40):
                    mid = (estimator.edges[1][:-1] + estimator.edges[1][1:]) / 2.
                    mask = (mid >= -pimax) & (mid <= pimax)
                    dpi = np.diff(estimator.edges[1])
                    return np.sum(estimator.corr[:, mask] * dpi[mask], axis=-1)

                pimax = 40
                sep, wp = project_to_wp(estimator_nojackknife, pimax=pimax)
                assert np.allclose(sep, get_sepavg(estimator_nojackknife, pimax), equal_nan=True)
                assert np.allclose(wp, ref_to_wp(estimator_nojackknife, pimax), equal_nan=True)
                sep, wp, cov = project_to_wp(estimator_jackknife)
                sep, wp, cov = project_to_wp(estimator_jackknife, pimax=pimax)
                assert np.allclose(sep, get_sepavg(estimator_jackknife, pimax), equal_nan=True)
                assert np.allclose(wp, ref_to_wp(estimator_jackknife, pimax), equal_nan=True)
                assert cov.shape == (len(sep),) * 2 == (len(wp),) * 2

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir = '_tests'
                fn = os.path.join(tmp_dir, 'tmp.npy')
                fn_txt = os.path.join(tmp_dir, 'tmp.txt')
                plot_fn = os.path.join(tmp_dir, 'tmp.png')
                for test, isjkn in zip([estimator_nojackknife, estimator_jackknife], [False, True]):
                    tmp = test(*[test.sepavg(idim) for idim in range(test.ndim)], return_sep=True, return_std=False)
                    if mode in ['smu', 'rppi']:
                        assert len(tmp) == 3
                    else:
                        _, tmp = tmp
                        assert np.allclose(tmp, test.corr, equal_nan=True)
                    assert np.isnan(test(-1., return_std=False)).all()
                    isep = test.shape[0] // 2
                    sep = test.edges[0][isep:isep + 2]
                    arrays = test(sep), test(sep[::-1])
                    if not isjkn: arrays = [[array] for array in arrays]
                    for array1, array2 in zip(*arrays): assert np.allclose(array1, array2[::-1], atol=0, equal_nan=True)
                    zero = test.corr.flat[0]
                    test.corr.flat[0] = np.nan  # to test ignore_nan
                    test[::test.shape[0]].get_corr()
                    if isjkn: assert test[::test.shape[0]].get_corr(return_sep=False)[1].ndim == 2
                    if test.mode == 'smu':
                        # smu
                        arrays = test(sep, [0., 0.4]), test(sep[::-1], [0.4, 0.])
                        if not isjkn: arrays = [[array] for array in arrays]
                        for array1, array2 in zip(*arrays): assert np.allclose(array1, array2[::-1, ::-1], atol=0, equal_nan=True)
                        test.save_txt(fn_txt)
                        tmp = np.loadtxt(fn_txt, unpack=True)
                        mids = np.meshgrid(*[test.sepavg(axis=axis, method='mid') for axis in range(test.ndim)], indexing='ij')
                        assert np.allclose([tt.reshape(test.shape) for tt in tmp[:4]], [mids[0], test.seps[0], mids[1], test.seps[1]], equal_nan=True)
                        assert np.allclose([tt.reshape(test.shape) for tt in tmp[4:]], test(return_sep=False), equal_nan=True)
                        # poles
                        if isjkn: assert test[::test.shape[0]].get_corr(ells=(0, 2), return_sep=False)[1].ndim == 2
                        assert np.isnan(test(ell=2)).any()
                        assert not np.isnan(test(ell=2, ignore_nan=True)).any()
                        _, mask = test.get_corr(ell=2, ignore_nan=True, return_mask=True, return_cov=False, return_sep=False)
                        assert not mask.all()
                        assert np.allclose(test(sep, ells=(0, 2, 4)), test(sep, mode='poles'), atol=0, equal_nan=True)
                        arrays = test(5., ell=2), test([5.] * 3, ell=2), test([5.] * 4, ells=(0, 2, 4)), test([5.] * 4, [0.1, 0.2])  # corr, and std if jackknife
                        if not isjkn: arrays = [[array] for array in arrays]
                        for array in arrays[0]: assert array.shape == ()
                        for array in arrays[1]: assert array.shape == (3, )
                        for array in arrays[2]: assert array.shape == (3, 4)
                        for array in arrays[3]: assert array.shape == (4, 2)
                        arrays = test(sep, ell=2), test(sep[::-1], ell=2), test(sep, ell=[0, 2]), test(sep[::-1], ell=[2, 0]), test(sep, [0., 0.4]), test(sep[::-1], [0.4, 0.])
                        if not isjkn: arrays = [[array] for array in arrays]
                        for array1, array2 in zip(*arrays[:2]): assert np.allclose(array1, array2[::-1], atol=0, equal_nan=True)
                        for array1, array2 in zip(*arrays[2:4]): assert np.allclose(array1, array2[::-1, ::-1], atol=0, equal_nan=True)
                        test.save_txt(fn_txt, ells=(0, 2))
                        tmp = np.loadtxt(fn_txt, unpack=True)
                        assert np.allclose(tmp[0], test.sepavg(method='mid'))
                        tmp2 = test(return_sep=True, ells=(0, 2))
                        assert np.allclose(tmp[1:], np.concatenate([tmp2[0][None, :]] + tmp2[1:], axis=0), equal_nan=True)
                        # wedges
                        if isjkn: assert test[::test.shape[0]].get_corr(wedges=(-1., -2. / 3, -1. / 3.), return_sep=False)[1].ndim == 2
                        assert np.isnan(test(wedges=(-1., 0.5))).any()
                        assert not np.isnan(test(wedges=(-1., 0.5), ignore_nan=True)).any()
                        _, mask = test.get_corr(wedges=(-1., 0.5), ignore_nan=True, return_mask=True, return_cov=False, return_sep=False)
                        assert not mask.all()
                        assert np.allclose(test(sep, wedges=(-1., -2. / 3, -1. / 3, 0., 1. / 3, 2. / 3, 1.)), test(sep, mode='wedges'), atol=0)
                        arrays = test(5., wedges=(-1., -0.8)), test([5.] * 3, wedges=(-1., -0.8)), test([sep[0]] * 4, wedges=(0.1, 0.3, 0.8)), test([sep[0]] * 4, wedges=((0.1, 0.3), (0.3, 0.8)))
                        if not isjkn: arrays = [[array] for array in arrays]
                        for array in arrays[0]: assert array.shape == ()
                        for array in arrays[1]: assert array.shape == (3, )
                        for array in arrays[2]: assert array.shape == (2, 4)
                        for array in arrays[3]: assert array.shape == (2, 4)
                        for array1, array2 in zip(*arrays[2:4]): assert np.allclose(array1, array2, atol=0, equal_nan=True)
                        arrays = test(sep, wedges=(-1., -0.8)), test(sep[::-1], wedges=(-1., -0.8)), test(sep, wedges=(0.1, 0.3, 0.8)), test(sep[::-1], wedges=((0.3, 0.8), (0.1, 0.3)))
                        if not isjkn: arrays = [[array] for array in arrays]
                        for array1, array2 in zip(*arrays[:2]): assert np.allclose(array1, array2[::-1], atol=0, equal_nan=True)
                        for array1, array2 in zip(*arrays[2:4]): assert np.allclose(array1, array2[::-1, ::-1], atol=0, equal_nan=True)
                        test.save_txt(fn_txt, wedges=(0.1, 0.3, 0.8))
                        tmp = np.loadtxt(fn_txt, unpack=True)
                        assert np.allclose(tmp[0], test.sepavg(method='mid'))
                        tmp2 = test(return_sep=True, wedges=(0.1, 0.3, 0.8))
                        assert np.allclose(tmp[1:], np.concatenate([tmp2[0][None, :]] + tmp2[1:], axis=0), equal_nan=True)
                        test.corr.flat[0] = zero
                        assert np.allclose(test[:, :10].corr, test.corr[:, :10], equal_nan=True)
                        test(return_sep=True, ells=(0, 2), rp=(10., np.inf))
                        test(return_sep=True, wedges=(-1., -0.8), rp=(10., np.inf))
                    elif test.mode == 'rppi':
                        # rppi
                        arrays = test(sep, [0., 0.4]), test(sep[::-1], [0.4, 0.])
                        if not isjkn: arrays = [[array] for array in arrays]
                        for array1, array2 in zip(*arrays): assert np.allclose(array1, array2[::-1, ::-1], atol=0, equal_nan=True)
                        test.save_txt(fn_txt)
                        tmp = np.loadtxt(fn_txt, unpack=True)
                        mids = np.meshgrid(*[test.sepavg(axis=axis, method='mid') for axis in range(test.ndim)], indexing='ij')
                        assert np.allclose([tt.reshape(test.shape) for tt in tmp[:4]], [mids[0], test.seps[0], mids[1], test.seps[1]], equal_nan=True)
                        assert np.allclose([tt.reshape(test.shape) for tt in tmp[4:]], test(return_sep=False), equal_nan=True)
                        # wp
                        if isjkn: assert test[::test.shape[0]].get_corr(pimax=20, return_sep=False)[1].ndim == 2
                        assert np.isnan(test(pimax=None)).any()
                        assert not np.isnan(test(pimax=None, ignore_nan=True)).any()
                        arrays = test(10., pimax=60), test([9.] * 4, pimax=60), test([9.] * 4, [10., 12.])
                        if not isjkn: arrays = [[array] for array in arrays]
                        for array in arrays[0]: assert array.shape == ()
                        for array in arrays[1]: assert array.shape == (4, )
                        for array in arrays[2]: assert array.shape == (4, 2)
                        arrays = test(sep, pimax=60), test(sep[::-1], pimax=60)
                        if not isjkn: arrays = [[array] for array in arrays]
                        for array1, array2 in zip(*arrays): assert np.allclose(array1, array2[::-1], atol=0, equal_nan=True)
                        assert np.allclose(test(sep, pimax=40), test(sep, mode='wp'), atol=0, equal_nan=True)
                        test.save_txt(fn_txt, pimax=40.)
                        tmp = np.loadtxt(fn_txt, unpack=True)
                        assert np.allclose(tmp[0], test.sepavg(method='mid'))
                        assert np.allclose(tmp[1:], test(pimax=40., return_sep=True), equal_nan=True)
                        test.corr.flat[0] = zero
                        assert np.allclose(test[:, :10].corr, test.corr[:, :10], equal_nan=True)
                    else: # theta, s, wp
                        with pytest.raises(TwoPointEstimatorError):
                            test(5., ell=2)
                        with pytest.raises(TwoPointEstimatorError):
                            test(10., pimax=60)
                        arrays = test(10.), test([9.] * 4)
                        if not isjkn: arrays = [[array] for array in arrays]
                        for array in arrays[0]: assert array.shape == ()
                        for array in arrays[1]: assert array.shape == (4, )
                        test.save_txt(fn_txt)
                        tmp = np.loadtxt(fn_txt, unpack=True)
                        assert np.allclose(tmp[0], test.sepavg(method='mid'))
                        assert np.allclose(tmp[1:], test(return_sep=True), equal_nan=True)
                        test.corr.flat[0] = zero
                    test.save(fn)
                    if test.mode == 'smu':
                        test.plot(mode='wedges')
                        if isjkn: test.plot(plot_std=True, mode='wedges', fn=plot_fn)
                        test.plot(mode='poles')
                        if isjkn: test.plot(plot_std=True, mode='poles', fn=plot_fn)
                    elif test.mode == 'rppi':
                        test.plot(mode='wp')
                        if isjkn: test.plot(plot_std=True, mode='wp', fn=plot_fn)
                    else:
                        test.plot()
                        if isjkn: test.plot(plot_std=True)
                    test2 = TwoPointEstimator.load(fn)
                    assert type(TwoPointCorrelationFunction.from_state(test2.__getstate__())) is type(test2)
                    test2 = TwoPointCorrelationFunction.load(fn)
                    assert test2.__class__ is test.__class__
                    assert test2.with_shifted is test.with_shifted
                    #assert test2.with_reversed is test.with_reversed
                    test3 = test2.copy()
                    test3.rebin((2, 5) if len(edges) == 2 else (2,))
                    assert test3.shape[0] == test2.shape[0] // 2
                    test2 = test2[::2, ::5] if len(edges) == 2 else test2[::2]
                    assert_allclose_estimators(test2, test3)
                    test2 = test3.copy()
                    sepmax = test2.edges[0][-1] / 2
                    test3.select((0, sepmax))
                    assert np.all(test3.sepavg(method='mid', axis=0) <= sepmax)
                    assert np.all((test3.sepavg(axis=0) <= test2.edges[0][test2.edges[0] >= sepmax][0]) | np.isnan(test3.sepavg(axis=0)))
                    test2 = test + test
                    assert np.allclose(test2.corr, test.corr, equal_nan=True)
                    if test2.ndim >= 2:
                        diff = np.diff(test2.edges[1])
                        if np.allclose(diff, diff[0]):
                            test2.select(None, (-np.inf, np.inf, diff[0] * 2.))
                            assert np.allclose(test2.edges[1], test.edges[1][::2])
                    test3 = test * 3
                    assert np.allclose(test3.corr, test.corr, equal_nan=True)
                    if hasattr(test, 'D1D2') and hasattr(test, 'R1R2'):
                        test1 = test.deepcopy()
                        test1.R1R2.wcounts += 1.
                        test2 = test + test1
                        assert np.allclose(test2.D1D2.wcounts, test.D1D2.wcounts)
                        assert test2.D1D2.size1 > 0
                        assert np.allclose(test2.R1R2.wcounts, 2 * test.R1R2.wcounts + 1.)
                        assert test2.R1R2.size1 == 0
                        assert np.allclose(test3.D1D2.wcounts, 3 * test.D1D2.wcounts)
                    test2 = test.concatenate_x(test[:test.shape[0] // 2], test[test.shape[0] // 2:])
                    assert np.allclose(test2.corr, test.corr, equal_nan=True)
                    if 'davispeebles' not in test.name:
                        test2 = run_jackknife(R1R2=test.R1R2)
                        assert_allclose_estimators(test2, test)

            if mpicomm is not None:
                test_mpi = run_jackknife(mpicomm=mpicomm, pass_none=mpicomm.rank != 0, mpiroot=0)
                assert_allclose_estimators(test_mpi, estimator_jackknife)

                test_mpi = run_jackknife(mpicomm=mpicomm, pass_zero=mpicomm.rank != 0, mpiroot=None)
                assert_allclose_estimators(test_mpi, estimator_jackknife)

                data1 = [mpi.scatter(d, mpiroot=0, mpicomm=mpicomm) for d in data1]
                data2 = [mpi.scatter(d, mpiroot=0, mpicomm=mpicomm) for d in data2]
                randoms1 = [mpi.scatter(d, mpiroot=0, mpicomm=mpicomm) for d in randoms1]
                randoms2 = [mpi.scatter(d, mpiroot=0, mpicomm=mpicomm) for d in randoms2]
                test_mpi = run_nojackknife(mpicomm=mpicomm)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    fn = test_mpi.XX.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
                    fn_txt = test_mpi.XX.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.txt'), root=0)
                    test_mpi.save(fn)
                    test_mpi.save_txt(fn_txt)
                    test_mpi.mpicomm.Barrier()
                    test_mpi = TwoPointEstimator.load(fn)
                    fn = os.path.join(tmp_dir, 'tmp.npy')
                    test_mpi.save(fn)

                assert_allclose_estimators(test_mpi, estimator_jackknife)


def test_s():
    boxsize = [1000.] * 3
    edges = np.linspace(0., 50, 51)
    positions = generate_catalogs(100000, boxsize=boxsize, n_individual_weights=0, n_bitwise_weights=0, seed=42)[0]
    for i in range(100):
        print(i)
        result = TwoPointCorrelationFunction('s', edges, data_positions1=positions, engine='corrfunc', los='x', boxsize=boxsize, position_type='xyz', isa='fallback', nthreads=1)



if __name__ == '__main__':

    setup_logging()

    #test_s()
    test_multipoles()
    for mode in ['theta', 's', 'smu', 'rppi', 'rp']:
        test_estimator(mode=mode)
