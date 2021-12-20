import os
import tempfile

import numpy as np

from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, TwoPointCounter,\
                   JacknifeTwoPointEstimator, project_to_multipoles, project_to_wp, setup_logging


def generate_catalogs(size=100, boxsize=(1000,)*3, offset=(0,0,0), n_individual_weights=1, n_bitwise_weights=0, seed=42):
    rng = np.random.RandomState(seed=seed)
    toret = []
    for i in range(2):
        positions = [o + rng.uniform(0., 1., size)*b for o, b in zip(offset, boxsize)]
        weights = [rng.randint(0, 0xffffffff, size, dtype='i8') for i in range(n_bitwise_weights)]
        weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
        toret.append(positions+weights)
    return toret


def test_estimator(mode='s'):

    from pycorr import KMeansSubsampler

    list_engine = ['corrfunc']
    edges = np.linspace(1,100,10)
    size = 100
    boxsize = (1000,)*3
    list_options = []
    list_options.append({'estimator':'natural'})
    if mode not in ['theta', 'rp']:
        list_options.append({'estimator':'natural', 'boxsize':boxsize, 'with_randoms':False})
        list_options.append({'autocorr':True, 'estimator':'natural', 'boxsize':boxsize, 'with_randoms':False})
    list_options.append({'estimator':'davispeebles'})
    list_options.append({'estimator':'weight'})
    list_options.append({'with_shifted':True})
    list_options.append({'with_shifted':True, 'autocorr':True})
    list_options.append({'autocorr':True})
    list_options.append({'n_individual_weights':1, 'compute_sepavg':False})

    has_mpi = True
    try:
        import mpi4py
        import pmesh
    except ImportError:
        has_mpi = False
    if has_mpi:
        from pycorr import mpi
        print('Has MPI')
        list_options.append({'mpicomm': mpi.COMM_WORLD})

    #list_options.append({'weight_type':'inverse_bitwise','n_bitwise_weights':2})
    edges = np.linspace(1e-9,100,11)
    if mode == 'smu':
        edges = (edges, np.linspace(0,1,21))
    elif mode == 'rppi':
        edges = (edges, np.linspace(0,20,21))
    elif mode == 'theta':
        edges = np.linspace(1e-5,10,11) # below 1e-5, self pairs are counted by Corrfunc

    for engine in list_engine:
        for options in list_options:
            options = options.copy()
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
            options['los'] = 'z' if options['boxsize'] is not None else 'midpoint'
            options['position_type'] = 'xyz'
            npos = 3

            subsampler = KMeansSubsampler(mode='theta', positions=data1[:npos], nsamples=5, position_type='xyz')
            data1.append(subsampler.label(data1[:npos]))
            data2.append(subsampler.label(data2[:npos]))
            randoms1.append(subsampler.label(randoms1[:npos]))
            randoms2.append(subsampler.label(randoms2[:npos]))
            shifted1.append(subsampler.label(shifted1[:npos]))
            shifted2.append(subsampler.label(shifted2[:npos]))

            def run_nojacknife(ii=None, **kwargs):
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

            def run_jacknife(pass_none=False, **kwargs):
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
                #assert np.allclose(res1.sep[mask], res2.sep[mask])

            estimator_nojacknife = run_nojacknife()
            estimator_jacknife = run_jacknife()
            assert_allclose_estimators(estimator_jacknife, estimator_nojacknife)
            if mode in ['theta', 'rp']:
                assert estimator_jacknife.cov().shape == (estimator_jacknife.shape[0],)*2

            nsplits = 10
            estimator_jacknife = JacknifeTwoPointEstimator.concatenate(*[run_jacknife(samples=samples) for samples in np.array_split(np.unique(data1[-1]), nsplits)])
            assert_allclose_estimators(estimator_jacknife, estimator_nojacknife)

            ii = data1[-1][0]
            estimator_nojacknife_ii = run_nojacknife(ii=ii)
            estimator_jacknife_ii = estimator_jacknife.realization(ii, correction=None)
            assert_allclose_estimators(estimator_jacknife_ii, estimator_nojacknife_ii)

            options_counts = options.copy()
            estimator = options_counts.pop('estimator', 'landyszalay')

            D1D2 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=data1[:npos], positions2=None if autocorr else data2[:npos],
                                   weights1=data1[npos:-1], weights2=None if autocorr else data2[npos:-1], **options_counts)

            assert_allclose(D1D2, estimator_jacknife.D1D2)
            if with_shifted:
                if estimator in ['landyszalay', 'natural']:
                    R1R2 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=randoms1[:npos], positions2=None if autocorr else randoms2[:npos],
                                           weights1=randoms1[npos:-1], weights2=None if autocorr else randoms2[npos:-1], **options_counts)
                    assert_allclose(R1R2, estimator_jacknife.R1R2)
                # for following computation
                randoms1 = shifted1
                randoms2 = shifted2
            if estimator in ['landyszalay', 'davispeebles']:
                D1R2 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=data1[:npos], positions2=randoms1[:npos] if autocorr else randoms2[:npos],
                                       weights1=data1[npos:-1], weights2=randoms1[npos:-1] if autocorr else randoms2[npos:-1], **options_counts)
                assert_allclose(D1R2, estimator_jacknife.D1S2)
                if not autocorr and estimator in ['landyszalay']:
                    D2R1 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=randoms1[:npos], positions2=data2[:npos],
                                           weights1=randoms1[npos:-1], weights2=data2[npos:-1], **options_counts)
                    #D2R1 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=data2[:3], positions2=randoms1[:3],
                    #                       weights1=data2[3:], weights2=randoms1[3:], **options_counts)
                    assert_allclose(D2R1, estimator_jacknife.D2S1)
                else:
                    assert_allclose(D1R2, estimator_jacknife.D2S1)
            if estimator in ['landyszalay', 'natural', 'weight'] and with_randoms:
                R1R2 = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=randoms1[:npos], positions2=None if autocorr else randoms2[:npos],
                                       weights1=randoms1[npos:-1], weights2=None if autocorr else randoms2[npos:-1], **options_counts)
                assert_allclose(R1R2, estimator_jacknife.S1S2)
            if estimator in ['natural'] and not with_randoms:
                R1R2 = TwoPointCounter(mode=mode, edges=edges, engine='analytic', boxsize=estimator_jacknife.D1D2.boxsize,
                                       size1=estimator_jacknife.D1D2.size1, size2=None if estimator_jacknife.autocorr else estimator_jacknife.D1D2.size2, los=options_counts['los'])
                assert_allclose(R1R2, estimator_jacknife.R1R2)

            if estimator_jacknife.D1D2.mode == 'smu':
                sep, xiell = project_to_multipoles(estimator_nojacknife, ells=(0,2,4))
                sep, xiell, cov = project_to_multipoles(estimator_jacknife, ells=(0,2,4))
                assert cov.shape == (sum([len(xi) for xi in xiell]),)*2
            if estimator_jacknife.D1D2.mode == 'rppi':
                sep, wp = project_to_wp(estimator_nojacknife)
                sep, wp, cov = project_to_wp(estimator_jacknife)
                sep, wp, cov = project_to_wp(estimator_jacknife, pimax=40)
                assert cov.shape == (len(wp),)*2

            with tempfile.TemporaryDirectory() as tmp_dir:
                fn = os.path.join(tmp_dir,'tmp.npy')
                for test in [estimator_nojacknife, estimator_jacknife]:
                    test.save(fn)
                    test2 = TwoPointEstimator.load(fn)
                    assert test2.__class__ is test.__class__
                    assert test2.autocorr is test.autocorr
                    test2.rebin((2,5) if len(edges) == 2 else (2,))
                    test2 = run_jacknife(R1R2=test.R1R2)
                    assert_allclose_estimators(test2, test)

            if mpicomm is not None:
                test_mpi = run_jacknife(mpicomm=mpicomm, pass_none=mpicomm.rank != 0, mpiroot=0)
                assert_allclose_estimators(test_mpi, estimator_jacknife)

                data1 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in data1]
                data2 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in data2]
                randoms1 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in randoms1]
                randoms2 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in randoms2]
                test_mpi = run_nojacknife(mpicomm=mpicomm)
                assert_allclose_estimators(test_mpi, estimator_jacknife)


if __name__ == '__main__':

    setup_logging()
    for mode in ['theta','s','smu','rppi','rp']:
        test_estimator(mode=mode)
