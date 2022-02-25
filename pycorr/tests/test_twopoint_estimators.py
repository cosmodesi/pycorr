import os
import tempfile

import numpy as np

from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, TwoPointCounter,\
                   JackknifeTwoPointEstimator, project_to_multipoles, project_to_wp, setup_logging


def generate_catalogs(size=100, boxsize=(1000,)*3, offset=(0,0,0), n_individual_weights=1, n_bitwise_weights=0, seed=42):
    rng = np.random.RandomState(seed=seed)
    toret = []
    for i in range(2):
        positions = [o + rng.uniform(0., 1., size)*b for o, b in zip(offset, boxsize)]
        weights = [rng.randint(0, 0xffffffff, size, dtype='i8') for i in range(n_bitwise_weights)]
        weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
        toret.append(positions+weights)
    return toret


def test_multipoles():

    class Estimator(object):

        def sepavg(self, axis=0):
            return estimator.sep[:,0]

    from scipy import special

    estimator = Estimator()
    for muedges in [np.linspace(0., 1., 10), np.linspace(0., 1., 100), np.linspace(-1., 1., 100)]:
        edges = [np.linspace(0., 1., 10), muedges]
        ss, mumu = np.meshgrid(*[(e[:-1] + e[1:])/2. for e in edges], indexing='ij')
        estimator.sep, estimator.edges, estimator.corr = ss, edges, np.ones_like(ss)
        s, xi = project_to_multipoles(estimator, ells=(0,2,4))
        assert np.allclose(xi[0], 1., atol=1e-9)
        for xiell in xi[1:]: assert np.allclose(xiell, 0., atol=1e-9)

    edges = [np.linspace(0., 1., 10), np.linspace(0., 1., 100)]
    ss, mumu = np.meshgrid(*[(e[:-1] + e[1:])/2. for e in edges], indexing='ij')
    ells = (0, 2, 4)
    for ellin in ells[1:]:
        estimator.sep, estimator.edges, estimator.corr = ss, edges, special.legendre(ellin)(mumu)
        s, xi = project_to_multipoles(estimator, ells=ells)
        for ell, xiell in zip(ells, xi):
            assert np.allclose(xiell, 1. if ell == ellin else 0., atol=1e-3)


def test_estimator(mode='s'):

    from pycorr import KMeansSubsampler

    list_engine = ['corrfunc']
    edges = np.linspace(1, 100, 10)
    size = 500
    boxsize = (1000,)*3
    list_options = []
    list_options.append({'weights_one':['D1', 'R2']})
    list_options.append({'estimator':'natural'})
    if mode not in ['theta', 'rp']:
        list_options.append({'estimator':'natural', 'boxsize':boxsize, 'with_randoms':False})
        list_options.append({'autocorr':True, 'estimator':'natural', 'boxsize':boxsize, 'with_randoms':False})
    list_options.append({'estimator':'davispeebles'})
    list_options.append({'estimator':'weight'})
    list_options.append({'with_shifted':True})
    list_options.append({'with_shifted':True, 'autocorr':True})
    if mode == 'smu':
        list_options.append({'los':'firstpoint', 'autocorr':True})
        list_options.append({'los':'endpoint', 'autocorr':True})
    list_options.append({'n_individual_weights':0})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':1, 'compute_sepsavg':False})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':1})

    mpi = False
    try:
        from pycorr import mpi
        print('Has MPI')
    except ImportError:
        pass
    if mpi:
        list_options.append({'mpicomm': mpi.COMM_WORLD})

    #list_options.append({'weight_type':'inverse_bitwise','n_bitwise_weights':2})
    edges = np.linspace(1e-9, 100, 11)
    if mode == 'smu':
        edges = (edges, np.linspace(-1, 1, 21))
    elif mode == 'rppi':
        edges = (edges, np.linspace(0, 20, 21))
    elif mode == 'theta':
        edges = np.linspace(1e-5, 10, 11) # below 1e-5, self pairs are counted by Corrfunc

    for engine in list_engine:
        for options in list_options:
            options = options.copy()
            weights_one = options.pop('weights_one', [])
            n_individual_weights = options.get('n_individual_weights',1)
            n_bitwise_weights = options.get('n_bitwise_weights',0)
            data1, data2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=n_individual_weights, n_bitwise_weights=n_bitwise_weights, seed=42)
            randoms1, randoms2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=n_individual_weights, n_bitwise_weights=n_bitwise_weights, seed=43)
            shifted1, shifted2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=n_individual_weights, n_bitwise_weights=n_bitwise_weights, seed=44)
            autocorr = options.pop('autocorr', False)
            mpicomm = options.pop('mpicomm', None)
            with_randoms = options.pop('with_randoms', True)
            with_shifted = options.pop('with_shifted', False)
            options.setdefault('boxsize', None)
            los = options['los'] = options.get('los', 'x' if options['boxsize'] is not None else 'midpoint')
            options['position_type'] = 'xyz'
            npos = 3
            for label, catalog in zip(['D1','D2','R1','R2'], [data1, data2, randoms1, randoms2]):
                if label in weights_one:
                    catalog.append(np.ones_like(catalog[0]))

            subsampler = KMeansSubsampler(mode='theta', positions=data1[:npos], nsamples=5, position_type='xyz')
            data1.append(subsampler.label(data1[:npos]))
            data2.append(subsampler.label(data2[:npos]))
            randoms1.append(subsampler.label(randoms1[:npos]))
            randoms2.append(subsampler.label(randoms2[:npos]))
            shifted1.append(subsampler.label(shifted1[:npos]))
            shifted2.append(subsampler.label(shifted2[:npos]))

            def run_nojackknife(ii=None, **kwargs):
                data_positions1, data_weights1, data_samples1 = data1[:npos], data1[npos:-1], data1[-1]
                data_positions2, data_weights2, data_samples2 = data2[:npos], data2[npos:-1], data2[-1]
                randoms_positions1, randoms_weights1, randoms_samples1 = randoms1[:npos], randoms1[npos:-1], randoms1[-1]
                randoms_positions2, randoms_weights2, randoms_samples2 = randoms2[:npos], randoms2[npos:-1], randoms2[-1]
                shifted_positions1, shifted_weights1, shifted_samples1 = shifted1[:npos], shifted1[npos:-1], shifted1[-1]
                shifted_positions2, shifted_weights2, shifted_samples2 = shifted2[:npos], shifted2[npos:-1], shifted2[-1]
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

                return TwoPointCorrelationFunction(mode=mode, edges=edges, engine=engine, data_positions1=data_positions1, data_positions2=None if autocorr else data_positions2,
                                                   data_weights1=data_weights1, data_weights2=None if autocorr else data_weights2,
                                                   randoms_positions1=randoms_positions1 if with_randoms else None, randoms_positions2=None if autocorr else randoms_positions2,
                                                   randoms_weights1=randoms_weights1 if with_randoms else None, randoms_weights2=None if autocorr else randoms_weights2,
                                                   shifted_positions1=shifted_positions1 if with_shifted else None, shifted_positions2=None if autocorr else shifted_positions2,
                                                   shifted_weights1=shifted_weights1 if with_shifted else None, shifted_weights2=None if autocorr else shifted_weights2,
                                                   **options, **kwargs)

            def run_jackknife(pass_none=False, **kwargs):
                return TwoPointCorrelationFunction(mode=mode, edges=edges, engine=engine, data_positions1=None if pass_none else data1[:npos], data_positions2=None if pass_none or autocorr else data2[:npos],
                                                   data_weights1=None if pass_none else data1[npos:-1], data_weights2=None if pass_none or autocorr else data2[npos:-1],
                                                   randoms_positions1=randoms1[:npos] if with_randoms and not pass_none else None, randoms_positions2=None if pass_none or autocorr else randoms2[:npos],
                                                   randoms_weights1=randoms1[npos:-1] if with_randoms and not pass_none else None, randoms_weights2=None if pass_none or autocorr else randoms2[npos:-1],
                                                   shifted_positions1=shifted1[:npos] if with_shifted and not pass_none else None, shifted_positions2=None if pass_none or autocorr else shifted2[:npos],
                                                   shifted_weights1=shifted1[npos:-1] if with_shifted and not pass_none else None, shifted_weights2=None if pass_none or autocorr else shifted2[npos:-1],
                                                   data_samples1=None if pass_none else data1[-1], data_samples2=None if pass_none else data2[-1],
                                                   randoms_samples1=None if pass_none else randoms1[-1], randoms_samples2=None if pass_none else randoms2[-1],
                                                   shifted_samples1=None if pass_none else shifted1[-1], shifted_samples2=None if pass_none else shifted2[-1],
                                                   **options, **kwargs)

            def assert_allclose(res1, res2):
                tol = {'atol':1e-8, 'rtol':1e-6}
                assert np.allclose(res2.wcounts, res1.wcounts, **tol)
                assert np.allclose(res2.wnorm, res1.wnorm, **tol)

            def assert_allclose_estimators(res1, res2):
                mask = np.isfinite(res2.corr)
                assert np.allclose(res1.corr[mask], res2.corr[mask])
                assert np.allclose(res1.sep[mask], res2.sep[mask], equal_nan=True)

            estimator_nojackknife = run_nojackknife()
            estimator_jackknife = run_jackknife()
            assert_allclose_estimators(estimator_jackknife, estimator_nojackknife)
            if mode in ['theta', 'rp']:
                assert estimator_jackknife.cov().shape == (estimator_jackknife.shape[0],)*2

            nsplits = 10
            estimator_jackknife = JackknifeTwoPointEstimator.concatenate(*[run_jackknife(samples=samples) for samples in np.array_split(np.unique(data1[-1]), nsplits)])
            assert_allclose_estimators(estimator_jackknife, estimator_nojackknife)

            ii = data1[-1][0]
            estimator_nojackknife_ii = run_nojackknife(ii=ii)
            estimator_jackknife_ii = estimator_jackknife.realization(ii, correction=None)
            assert_allclose_estimators(estimator_jackknife_ii, estimator_nojackknife_ii)

            options_counts = options.copy()
            estimator = options_counts.pop('estimator', 'landyszalay')

            D1D2 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=data1[:npos], positions2=None if autocorr else data2[:npos],
                                   weights1=data1[npos:-1], weights2=None if autocorr else data2[npos:-1], **options_counts)

            assert_allclose(D1D2, estimator_jackknife.D1D2)
            if with_shifted:
                if estimator in ['landyszalay', 'natural']:
                    R1R2 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=randoms1[:npos], positions2=None if autocorr else randoms2[:npos],
                                           weights1=randoms1[npos:-1], weights2=None if autocorr else randoms2[npos:-1], **options_counts)
                    assert_allclose(R1R2, estimator_jackknife.R1R2)
                # for following computation
                randoms1 = shifted1
                randoms2 = shifted2
            if estimator in ['landyszalay', 'davispeebles']:
                D1R2 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=data1[:npos], positions2=randoms1[:npos] if autocorr else randoms2[:npos],
                                       weights1=data1[npos:-1], weights2=randoms1[npos:-1] if autocorr else randoms2[npos:-1], **options_counts)
                assert_allclose(D1R2, estimator_jackknife.D1S2)
                assert estimator_jackknife.with_reversed == ((not autocorr or los in ['firstpoint', 'endpoint']) and estimator in ['landyszalay'])
                if estimator_jackknife.with_reversed:
                    R1D2 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=randoms1[:npos], positions2=data1[:npos] if autocorr else data2[:npos],
                                           weights1=randoms1[npos:-1], weights2=data1[npos:-1] if autocorr else data2[npos:-1], **options_counts)
                    assert_allclose(R1D2, estimator_jackknife.S1D2)
                else:
                    assert_allclose(D1R2, estimator_jackknife.D1S2)
            if estimator in ['landyszalay', 'natural', 'weight'] and with_randoms:
                R1R2 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=randoms1[:npos], positions2=None if autocorr else randoms2[:npos],
                                       weights1=randoms1[npos:-1], weights2=None if autocorr else randoms2[npos:-1], **options_counts)
                assert_allclose(R1R2, estimator_jackknife.S1S2)
            if estimator in ['natural'] and not with_randoms:
                R1R2 = TwoPointCounter(mode=mode, edges=edges, engine='analytic', boxsize=estimator_jackknife.D1D2.boxsize,
                                       size1=estimator_jackknife.D1D2.size1, size2=None if estimator_jackknife.D1D2.autocorr else estimator_jackknife.D1D2.size2, los=options_counts['los'])
                assert_allclose(R1R2, estimator_jackknife.R1R2)

            if estimator_jackknife.D1D2.mode == 'smu':
                sep, xiell = project_to_multipoles(estimator_nojackknife, ells=(0,2,4))
                sep, xiell, cov = project_to_multipoles(estimator_jackknife, ells=(0,2,4))
                assert cov.shape == (sum([len(xi) for xi in xiell]),)*2
            if estimator_jackknife.D1D2.mode == 'rppi':

                def get_sepavg(estimator, sepmax):
                    mid = [(edges[:-1] + edges[1:])/2. for edges in estimator.edges]
                    if not estimator.D1D2.compute_sepavg:
                        return mid[0]
                    mask = mid[1] <= sepmax
                    sep = estimator.seps[0]
                    sep[np.isnan(sep)] = 0.
                    if getattr(estimator, 'R1R2', None) is not None:
                        wcounts = estimator.R1R2.wcounts
                    else:
                        wcounts = estimator.D1D2.wcounts
                    with np.errstate(divide='ignore', invalid='ignore'):
                        return np.sum(sep[:,mask]*wcounts[:,mask], axis=-1)/np.sum(wcounts[:,mask], axis=-1)

                pimax = 40
                sep, wp = project_to_wp(estimator_nojackknife, pimax=pimax)
                assert np.allclose(sep, get_sepavg(estimator_nojackknife, pimax), equal_nan=True)
                sep, wp, cov = project_to_wp(estimator_jackknife)
                sep, wp, cov = project_to_wp(estimator_jackknife, pimax=pimax)
                assert np.allclose(sep, get_sepavg(estimator_jackknife, pimax), equal_nan=True)
                assert cov.shape == (len(sep),)*2 == (len(wp),)*2

            with tempfile.TemporaryDirectory() as tmp_dir:
                fn = os.path.join(tmp_dir,'tmp.npy')
                for test in [estimator_nojackknife, estimator_jackknife]:
                    test.save(fn)
                    test2 = TwoPointEstimator.load(fn)
                    assert type(TwoPointCorrelationFunction.from_state(test2.__getstate__())) is type(test2)
                    test2 = TwoPointCorrelationFunction.load(fn)
                    assert test2.__class__ is test.__class__
                    assert test2.with_shifted is test.with_shifted
                    assert test2.with_reversed is test.with_reversed
                    test3 = test2.copy()
                    test3.rebin((2,5) if len(edges) == 2 else (2,))
                    assert test3.shape[0] == test2.shape[0]//2
                    test2 = test2[::2,::5] if len(edges) == 2 else test2[::2]
                    assert_allclose_estimators(test2, test3)
                    test3.select((0, 50.))
                    assert np.all((test3.sepavg(axis=0) <= 50.) | np.isnan(test3.sepavg(axis=0)))
                    if 'davispeebles' not in test.name:
                        test2 = run_jackknife(R1R2=test.R1R2)
                        assert_allclose_estimators(test2, test)

            if mpicomm is not None:
                test_mpi = run_jackknife(mpicomm=mpicomm, pass_none=mpicomm.rank != 0, mpiroot=0)
                assert_allclose_estimators(test_mpi, estimator_jackknife)

                data1 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in data1]
                data2 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in data2]
                randoms1 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in randoms1]
                randoms2 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in randoms2]
                test_mpi = run_nojackknife(mpicomm=mpicomm)
                assert_allclose_estimators(test_mpi, estimator_jackknife)


if __name__ == '__main__':

    setup_logging()

    test_multipoles()
    for mode in ['theta', 's', 'smu', 'rppi', 'rp']:
        test_estimator(mode=mode)
