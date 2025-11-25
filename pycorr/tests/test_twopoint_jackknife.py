import os
import tempfile

import numpy as np

from pycorr import BoxSubsampler, KMeansSubsampler, TwoPointCounter, JackknifeTwoPointCounter, utils, setup_logging


def generate_catalogs(size=200, boxsize=(1000,) * 3, offset=(1000, 0, 0), n_individual_weights=1, n_bitwise_weights=0, seed=42):
    rng = np.random.RandomState(seed=seed)
    toret = []
    for i in range(2):
        positions = [o + rng.uniform(0., 1., size) * b for o, b in zip(offset, boxsize)]
        weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(64 * n_bitwise_weights)], dtype=np.uint64)
        # weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(33)], dtype=np.uint64)
        # weights = [rng.randint(0, 0xffffffff, size, dtype=np.uint64) for i in range(n_bitwise_weights)]
        weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
        toret.append(positions + weights)
    return toret


def test_subsampler():

    import pytest
    mpi = False
    try:
        from pycorr import mpi
    except ImportError:
        pass

    boxsize_1d = 1000.
    boxsize = np.array([boxsize_1d] * 3)
    boxcenter = np.array([100., 0., 0.])
    catalog = generate_catalogs(size=1000, boxsize=boxsize, offset=boxcenter - boxsize / 2.)[0]
    positions = catalog[:3]
    positions_bak = np.array(positions, copy=True)
    nsamples_1d = 3
    nsamples = nsamples_1d ** 3
    subsampler = BoxSubsampler(boxsize=boxsize, boxcenter=boxcenter, nsamples=nsamples)
    assert np.allclose(subsampler.boxsize, boxsize)
    labels = subsampler.label(positions)
    assert np.max(labels) < nsamples
    assert np.allclose(positions, positions_bak)

    subsampler = BoxSubsampler(boxsize=boxsize, boxcenter=boxcenter, nsamples=nsamples)
    with pytest.raises(ValueError):
        subsampler.label(np.array(boxsize)[:, None])
    with pytest.raises(ValueError):
        subsampler.label(np.array(positions) + 2000.)
    subsampler = BoxSubsampler(boxsize=boxsize, boxcenter=boxcenter, nsamples=nsamples, wrap=True)
    subsampler.label(np.array(boxsize)[:, None])
    subsampler.label(np.array(positions) + 2000.)

    subsampler = BoxSubsampler(positions=positions, nsamples=nsamples)
    assert np.allclose(subsampler.boxsize, boxsize, rtol=1e-2)
    labels = subsampler.label(catalog[:3])
    assert np.max(labels) < nsamples

    # more accurate test for the labeling
    points_per_part = 5 # how many points to put into each part along each coordinates (so that each regions will contain points_per_part**3 points)
    coordinates_1d = np.linspace(0, boxsize_1d, 2 * points_per_part * nsamples_1d + 1)[1::2] # in the middle of the parts along each dimension (avoiding the very edges which can be ambiguous), with no shift to make things easier here
    labels_1d = np.repeat(np.arange(nsamples_1d), points_per_part) # obvious regions along each axis
    coordinates_alt = [np.ravel(_) for _ in np.meshgrid(coordinates_1d, coordinates_1d, coordinates_1d, indexing = 'ij')] # arrays of all 3D coordinate combinations taken from `coordinates_1d`
    labels_3d = [np.ravel(_) for _ in np.meshgrid(labels_1d, labels_1d, labels_1d, indexing = 'ij')] # arrays of all 3D index combinations taken from `labels_1d`, same order as the coordinates
    labels_alt = sum(_ * nsamples_1d ** i for (i, _) in enumerate(labels_3d[::-1])) # more explicit conversion to the multi-dimensional index
    assert np.max(labels_alt) == nsamples - 1
    subsampler2 = BoxSubsampler(boxsize=boxsize_1d, boxcenter=boxsize_1d/2, nsamples=nsamples) # subsampler for a strictly cubic box without shift
    assert np.array_equal(subsampler2.label(coordinates_alt, position_type = 'xyz'), labels_alt)

    if mpi:
        mpicomm = mpi.COMM_WORLD
        positions_mpi = positions
        if mpicomm.rank != 0:
            positions_mpi = [p[:0] for p in positions]

        def test_mpi(**kwargs):
            subsampler_mpi = BoxSubsampler(positions=positions, nsamples=nsamples, **kwargs)
            if mpicomm.rank == 0:
                labels_mpi = subsampler_mpi.label(catalog[:3])
                assert np.allclose(labels_mpi, labels)

        test_mpi(mpicomm=mpicomm, mpiroot=None)
        test_mpi(mpicomm=mpicomm, mpiroot=0)
        positions_mpi = [mpi.scatter(p, mpiroot=0, mpicomm=mpicomm) for p in positions]
        test_mpi(mpicomm=mpicomm, mpiroot=None)

    for nside in [None, 512]:

        nsamples = 100
        subsampler = KMeansSubsampler(mode='angular', positions=positions, nsamples=nsamples, nside=nside, random_state=42, position_type='xyz')
        labels = subsampler.label(positions)
        assert np.max(labels) < nsamples
        assert np.allclose(positions, positions_bak)

        if mpi:
            mpicomm = mpi.COMM_WORLD
            positions_mpi = positions
            if mpicomm.rank != 0:
                positions_mpi = [p[:0] for p in positions]

            def test_mpi(**kwargs):
                subsampler_mpi = KMeansSubsampler(mode='angular', positions=positions_mpi, nsamples=nsamples, nside=nside, random_state=42, position_type='xyz', **kwargs)
                labels_mpi = subsampler_mpi.label(catalog[:3])
                assert np.allclose(labels_mpi, labels)

            test_mpi(mpicomm=mpicomm, mpiroot=None)
            test_mpi(mpicomm=mpicomm, mpiroot=0)
            positions_mpi = [mpi.scatter(p, mpiroot=0, mpicomm=mpicomm) for p in positions]
            test_mpi(mpicomm=mpicomm, mpiroot=None)


def test_twopoint_counter(mode='s'):

    list_engine = ['corrfunc']
    size = 1000
    cboxsize = (500,) * 3

    ref_edges = np.linspace(0., 100., 41)
    if mode == 'theta':
        # ref_edges = np.linspace(1e-1, 10., 11) # below 1e-5 for float64 (1e-1 for float32), self pairs are counted by Corrfunc
        ref_edges = np.linspace(0., 10., 21)
    elif mode == 'smu':
        ref_edges = (ref_edges, np.linspace(-1., 1., 22))
    elif mode == 'rppi':
        ref_edges = (ref_edges, np.linspace(-40., 40., 55))

    from collections import namedtuple
    TwoPointWeight = namedtuple('TwoPointWeight', ['sep', 'weight'])
    twopoint_weights = TwoPointWeight(np.logspace(-4, 0, 40), np.linspace(4., 1., 40))

    list_options = []

    mpi = False
    try:
        from pycorr import mpi
        print('Has MPI')
    except ImportError:
        pass

    for autocorr in [False, True]:

        list_options.append({'autocorr': autocorr})
        # one-column of weights
        list_options.append({'autocorr': autocorr, 'weights_one': [1]})
        # position type
        for position_type in ['rdd', 'pos', 'xyz'] + (['rd'] if mode == 'theta' else []):
            list_options.append({'autocorr': autocorr, 'position_type': position_type})

        for dtype in ['f8']:  # in theta mode, lots of rounding errors!
            itemsize = np.dtype(dtype).itemsize
            for isa in ['fastest']:
                # binning
                edges = np.array([1, 8, 20, 42, 60])
                if mode == 'smu':
                    edges = (edges, np.linspace(-0.8, 0.8, 61))
                if mode == 'rppi':
                    edges = (edges, np.linspace(-90., 90., 181))

                list_options.append({'autocorr': autocorr, 'edges': edges, 'dtype': dtype, 'isa': isa})
                list_options.append({'autocorr': autocorr, 'compute_sepsavg': False, 'edges': edges, 'dtype': dtype, 'isa': isa})
                list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'bin_type': 'custom', 'dtype': dtype, 'isa': isa})
                # pip
                list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'dtype': dtype, 'isa': isa})
                list_options.append({'autocorr': autocorr, 'compute_sepsavg': False, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'dtype': dtype, 'isa': isa})
                list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 1, 'iip': 1, 'dtype': dtype, 'isa': isa})
                if not autocorr:
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 1, 'iip': 2, 'dtype': dtype, 'isa': isa})
                list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 1, 'bitwise_type': 'i4', 'iip': 1, 'dtype': dtype, 'isa': isa})
                list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'weight_attrs': {'nrealizations': 129, 'noffset': 3}, 'dtype': dtype, 'isa': isa})
                list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 2, 'weight_attrs': {'noffset': 0, 'default_value': 0.8}, 'dtype': dtype, 'isa': isa})
                # twopoint weights
                if itemsize > 4:
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'twopoint_weights': twopoint_weights, 'dtype': dtype, 'isa': isa})
                    list_options.append({'autocorr': autocorr, 'twopoint_weights': twopoint_weights, 'los': 'y', 'dtype': dtype, 'isa': isa})
                # boxsize
                if mode not in ['theta', 'rp']:
                    list_options.append({'autocorr': autocorr, 'boxsize': cboxsize, 'dtype': dtype, 'isa': isa})
                    list_options.append({'autocorr': autocorr, 'boxsize': cboxsize, 'dtype': dtype, 'isa': isa})
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'boxsize': cboxsize, 'los': 'x', 'dtype': dtype, 'isa': isa})
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'boxsize': cboxsize, 'los': 'y', 'dtype': dtype, 'isa': isa})
                # los
                list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'los': 'x', 'dtype': dtype, 'isa': isa})
                if mode in ['smu', 'rppi']:
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'los': 'firstpoint', 'edges': edges, 'dtype': dtype, 'isa': isa})
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'los': 'endpoint', 'edges': edges, 'dtype': dtype, 'isa': isa})
                    if itemsize > 4:
                        list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'los': 'endpoint', 'twopoint_weights': twopoint_weights, 'dtype': dtype, 'isa': isa})
                # mpi
                if mpi:
                    list_options.append({'autocorr': autocorr, 'mpicomm': mpi.COMM_WORLD, 'dtype': dtype, 'isa': isa})
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'mpicomm': mpi.COMM_WORLD, 'dtype': dtype, 'isa': isa})
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'twopoint_weights': twopoint_weights, 'mpicomm': mpi.COMM_WORLD, 'dtype': dtype, 'isa': isa})
                # labels
                list_options.append({'offset_label': 2})

    for engine in list_engine:
        for options in list_options:
            print(mode, options)
            options = options.copy()
            edges = options.pop('edges', ref_edges)
            weights_one = options.pop('weights_one', [])
            twopoint_weights = options.get('twopoint_weights', None)
            n_individual_weights = options.pop('n_individual_weights', 0)
            n_bitwise_weights = options.pop('n_bitwise_weights', 0)
            offset_label = options.pop('offset_label', 0)
            npos = 3
            data1, data2 = generate_catalogs(size, boxsize=cboxsize, n_individual_weights=n_individual_weights, n_bitwise_weights=n_bitwise_weights)

            subsampler = KMeansSubsampler(mode='angular', positions=data1[:npos], nsamples=5, nside=512, random_state=42, position_type='xyz')
            data1.append(subsampler.label(data1[:npos]) + offset_label)
            data2.append(subsampler.label(data2[:npos]) + offset_label)

            autocorr = options.pop('autocorr', False)
            boxsize = options.get('boxsize', None)
            los = options['los'] = options.get('los', 'x' if boxsize is not None else 'midpoint')
            bin_type = options.pop('bin_type', 'auto')
            mpicomm = options.pop('mpicomm', None)
            bitwise_type = options.pop('bitwise_type', None)
            iip = options.pop('iip', False)
            position_type = options.pop('position_type', 'xyz')
            dtype = options.pop('dtype', None)
            weight_attrs = options.get('weight_attrs', {}).copy()
            compute_sepavg = options.get('compute_sepsavg', True)
            if engine not in ['corrfunc']: compute_sepavg = False

            def setdefaultnone(di, key, value):
                if di.get(key, None) is None:
                    di[key] = value

            setdefaultnone(weight_attrs, 'nrealizations', n_bitwise_weights * 64 + 1)
            setdefaultnone(weight_attrs, 'noffset', 1)
            set_default_value = 'default_value' in weight_attrs
            setdefaultnone(weight_attrs, 'default_value', 0)
            if set_default_value:
                for w in data1[npos:npos + n_bitwise_weights] + data2[npos:npos + n_bitwise_weights]: w[:] = 0  # set to zero to make sure default_value is used

            def wiip(weights):
                denom = weight_attrs['noffset'] + utils.popcount(*weights)
                mask = denom == 0
                denom[mask] = 1.
                toret = weight_attrs['nrealizations'] / denom
                toret[mask] = weight_attrs['default_value']
                return toret

            def dataiip(data):
                return data[:npos] + [wiip(data[npos:npos + n_bitwise_weights])] + data[npos + n_bitwise_weights:]

            if iip == 1:
                data1 = dataiip(data1)
            elif iip == 2:
                data2 = dataiip(data2)
            if iip:
                n_bitwise_weights = 0
                weight_attrs['nrealizations'] = None

            itemsize = np.dtype('f8' if dtype is None else dtype).itemsize
            tol = {'atol': 1e-8, 'rtol': 1e-3} if itemsize <= 4 else {'atol': 1e-8, 'rtol': 1e-6}

            if bitwise_type is not None and n_bitwise_weights > 0:

                def update_bit_type(data):
                    return data[:npos] + utils.reformat_bitarrays(*data[npos:npos + n_bitwise_weights], dtype=bitwise_type) + data[npos + n_bitwise_weights:]

                data1 = update_bit_type(data1)
                data2 = update_bit_type(data2)

            if position_type in ['rd', 'rdd']:

                if position_type == 'rd': npos = 2

                def update_position_type(data):
                    rdd = list(utils.cartesian_to_sky(data[:3]))
                    if position_type == 'rdd':
                        return rdd + data[npos:]
                    if position_type == 'rd':
                        return rdd[:npos] + data[3:]
                    raise ValueError('Unknown position type {}'.format(position_type))

                data1 = update_position_type(data1)
                data2 = update_position_type(data2)

            def run_ref(ii=None, **kwargs):
                positions1, weights1, samples1 = data1[:npos], data1[npos:-1], data1[-1]
                positions2, weights2, samples2 = data2[:npos], data2[npos:-1], data2[-1]
                if ii is not None:
                    mask = samples1 == ii
                    positions1, weights1 = [position[~mask] for position in positions1], [weight[~mask] for weight in weights1]
                    mask = samples2 == ii
                    positions2, weights2 = [position[~mask] for position in positions2], [weight[~mask] for weight in weights2]
                if position_type == 'pos':
                    positions1 = np.array(positions1).T
                    positions2 = np.array(positions2).T
                return TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=positions1, positions2=None if autocorr else positions2,
                                       weights1=weights1, weights2=None if autocorr else weights2, position_type=position_type, bin_type=bin_type,
                                       dtype=dtype, **kwargs, **options)

            for label, catalog in zip([1, 2], [data1, data2]):
                if label in weights_one:
                    catalog.insert(-1, np.ones_like(catalog[0], dtype='f8'))

            def run(pass_none=False, pass_zero=False, reverse=False, **kwargs):
                tmpdata1, tmpdata2 = data1, data2
                if reverse and not autocorr:
                    tmpdata1, tmpdata2 = data2, data1
                positions1 = tmpdata1[:npos]
                positions2 = tmpdata2[:npos]
                weights1 = tmpdata1[npos:-1]
                weights2 = tmpdata2[npos:-1]
                samples1 = tmpdata1[-1]
                samples2 = tmpdata2[-1]

                def get_zero(arrays):
                    return [array[:0] for array in arrays]

                if pass_zero:
                    positions1 = get_zero(positions1)
                    positions2 = get_zero(positions2)
                    weights1 = get_zero(weights1)
                    weights2 = get_zero(weights2)
                    samples1 = samples1[:0]
                    samples2 = samples2[:0]

                if position_type == 'pos':
                    positions1 = np.array(positions1).T
                    positions2 = np.array(positions2).T
                positions1_bak = np.array(positions1, copy=True)
                positions2_bak = np.array(positions2, copy=True)
                if weights1: weights1_bak = np.array(weights1, copy=True)
                if weights2: weights2_bak = np.array(weights2, copy=True)
                samples1_bak = np.array(samples1, copy=True)
                samples2_bak = np.array(samples2, copy=True)
                toret = JackknifeTwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=None if pass_none else positions1, weights1=None if pass_none else weights1,
                                                 positions2=None if pass_none or autocorr else positions2, weights2=None if pass_none or autocorr else weights2,
                                                 samples1=None if pass_none else samples1, samples2=None if pass_none or autocorr else samples2,
                                                 position_type=position_type, bin_type=bin_type, dtype=dtype, **kwargs, **options)
                assert np.allclose(positions1, positions1_bak)
                assert np.allclose(positions2, positions2_bak)
                if weights1: assert np.allclose(weights1, weights1_bak)
                if weights2: assert np.allclose(weights2, weights2_bak)
                assert np.allclose(samples1, samples1_bak)
                assert np.allclose(samples2, samples2_bak)
                return toret

            def assert_allclose(res2, res1):
                assert np.allclose(res2.wcounts, res1.wcounts, **tol)
                assert np.allclose(res2.wnorm, res1.wnorm, **tol)
                if compute_sepavg:
                    assert np.allclose(res2.sep, res1.sep, **tol, equal_nan=True)
                assert res1.size1 == res2.size1
                assert res1.size2 == res2.size2

            ref = run_ref()
            test = run()
            assert_allclose(test, ref)

            nsplits = 10
            test = JackknifeTwoPointCounter.concatenate(*[run(samples=samples) for samples in np.array_split(np.unique(data1[-1]), nsplits)])
            assert_allclose(test, ref)

            ii = data1[-1][0]
            ref_ii = run_ref(ii=ii)
            test_ii = test.realization(ii, correction=None)
            assert_allclose(test_ii, ref_ii)

            test_zero = run(pass_zero=True)
            assert np.allclose(test_zero.wcounts, 0.)
            assert np.allclose(test_zero.wnorm, 0.)

            if engine == 'corrfunc':
                assert test.is_reversible == autocorr or (los not in ['firstpoint', 'endpoint'])
            if test.is_reversible:
                test_reversed = run(reverse=True)
                ref_reversed = test.reverse()
                assert np.allclose(test_reversed.wcounts, ref_reversed.wcounts, **tol)
                assert np.allclose(test_reversed.sep, ref_reversed.sep, **tol, equal_nan=True)
                for ii in ref_reversed.realizations:
                    assert np.allclose(test_reversed.realization(ii).wcounts, ref_reversed.realization(ii).wcounts, **tol)
                    assert np.allclose(test_reversed.realization(ii).sep, ref_reversed.realization(ii).sep, **tol, equal_nan=True)

            with tempfile.TemporaryDirectory() as tmp_dir:
                # tmp_dir = '_tests'
                fn = os.path.join(tmp_dir, 'tmp.npy')
                fn_txt = os.path.join(tmp_dir, 'tmp.txt')
                test.save(fn)
                test.save_txt(fn_txt)
                tmp = np.loadtxt(fn_txt, unpack=True)
                mids = np.meshgrid(*[test.sepavg(axis=axis, method='mid') for axis in range(test.ndim)], indexing='ij')
                seps = []
                for axis in range(test.ndim): seps += [mids[axis], test.seps[axis]]
                assert np.allclose([tt.reshape(test.shape) for tt in tmp], seps + [test.wcounts, utils._make_array_like(test.wnorm, test.wcounts)], equal_nan=True)
                for ii in test.realizations:
                    test.realization(ii).save_txt(fn_txt)
                    break
                test2 = JackknifeTwoPointCounter.load(fn)
                assert_allclose(test2, ref)
                test3 = test2.copy()
                test3.rebin((2, 3) if len(edges) == 2 else (2,))
                assert test3.shape[0] == test2.shape[0] // 2
                assert np.allclose(np.sum(test3.wcounts), np.sum(ref.wcounts))
                assert np.allclose(test2.wcounts, ref.wcounts)
                test2 = test2[::2, ::3] if len(edges) == 2 else test2[::2]
                assert test2.shape == test3.shape
                assert np.allclose(test2.wcounts, test3.wcounts, equal_nan=True)
                sepmax = 20.
                if mode == 'smu': sepmax = 0.8
                test3.select((0, sepmax))
                assert np.all((test3.sepavg(axis=0) <= sepmax) | np.isnan(test3.sepavg(axis=0)))

                def check_seps(test):
                    for iaxis in range(test.ndim):
                        if not test.compute_sepsavg[iaxis]:
                            mid = (test.edges[iaxis][:-1] + test.edges[iaxis][1:]) / 2.
                            if test.ndim == 2 and iaxis == 0: mid = mid[:, None]
                            assert np.allclose(test.seps[iaxis], mid)

                check_seps(test2)
                check_seps(test3)

                test2 = test + test
                assert np.allclose(test2.wcounts, 2. * test.wcounts, equal_nan=True)
                assert np.allclose(test2.normalized_wcounts(), test.normalized_wcounts(), equal_nan=True)
                test2 = test.concatenate_x(test[:test.shape[0] // 2], test[test.shape[0] // 2:])
                assert np.allclose(test2.wcounts, test.wcounts, equal_nan=True)

                test2 = test.normalize(wnorm=1.)
                assert np.allclose(test2.wnorm, 1.)
                test3 = test2 * 3
                assert np.allclose(test3.wcounts, 3. * test2.wcounts, equal_nan=True)
                assert np.allclose(test3.wnorm, 3.)

                if mode in ['smu', 'rppi'] and len(test.edges[1]) % 2 == 1:
                    test2 = test.wrap()
                    assert np.all(test2.edges[1] >= 0.)
                    assert np.all(test2.seps[1][~np.isnan(test2.seps[1])] >= 0.)
                    assert np.allclose(test2.wcounts.sum(axis=1), test.wcounts.sum(axis=1))

            if mpicomm is not None:

                test_mpi = run(mpicomm=mpicomm, pass_none=mpicomm.rank != 0, mpiroot=0, nprocs_per_real=2)
                assert_allclose(test_mpi, test)
                test_mpi = run(mpicomm=mpicomm, pass_zero=mpicomm.rank != 0, mpiroot=None, nprocs_per_real=2)
                assert_allclose(test_mpi, test)
                data1 = [mpi.scatter(d, mpiroot=0, mpicomm=mpicomm) for d in data1]
                data2 = [mpi.scatter(d, mpiroot=0, mpicomm=mpicomm) for d in data2]
                test_mpi = run(mpicomm=mpicomm)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_dir = '_tests'
                    utils.mkdir(tmp_dir)
                    mpicomm = test_mpi.mpicomm
                    fn = mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
                    fn_txt = mpicomm.bcast(os.path.join(tmp_dir, 'tmp.txt'), root=0)
                    test_mpi.save(fn)
                    test_mpi.save_txt(fn_txt)
                    mpicomm.Barrier()
                    test_mpi = JackknifeTwoPointCounter.load(fn)
                    fn = os.path.join(tmp_dir, 'tmp.npy')
                    test_mpi.save(fn)
                    import shutil
                    mpicomm.Barrier()
                    if mpicomm.rank == 0:
                        shutil.rmtree(tmp_dir)

                assert_allclose(test_mpi, test)


if __name__ == '__main__':

    setup_logging()

    test_subsampler()
    for mode in ['theta', 's', 'smu', 'rppi', 'rp']:
        test_twopoint_counter(mode=mode)
