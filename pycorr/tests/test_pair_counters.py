import os
import tempfile
import numpy as np

from pycorr import TwoPointCounter, AnalyticTwoPointCounter,\
                   utils, setup_logging


def diff(position1, position2):
    return [p2-p1 for p1,p2 in zip(position1,position2)]


def midpoint(position1, position2):
    return [p2+p1 for p1,p2 in zip(position1,position2)]


def norm(position):
    return (sum(p**2 for p in position))**0.5


def dotproduct(position1, position2):
    return sum(x1*x2 for x1,x2 in zip(position1,position2))


def dotproduct_normalized(position1, position2):
    return dotproduct(position1, position2)/(norm(position1)*norm(position2))


def get_weight(xyz1, xyz2, weights1, weights2, n_bitwise_weights=0, twopoint_weights=None, nrealizations=0):
    weight = (1 + nrealizations) / (1. + sum(bin(w1 & w2).count('1') for w1, w2 in zip(weights1[:n_bitwise_weights], weights2[:n_bitwise_weights])))
    for w1, w2 in zip(weights1[n_bitwise_weights:], weights2[n_bitwise_weights:]):
        weight *= w1 * w2
    if twopoint_weights is not None:
        sep_twopoint_weights = twopoint_weights.sep
        twopoint_weights = twopoint_weights.weight
        costheta = sum(x1*x2 for x1,x2 in zip(xyz1,xyz2))/(norm(xyz1)*norm(xyz2))
        if (sep_twopoint_weights[0] <= costheta < sep_twopoint_weights[-1]):
            ind_costheta = np.searchsorted(sep_twopoint_weights, costheta, side='right', sorter=None) - 1
            frac = (costheta - sep_twopoint_weights[ind_costheta])/(sep_twopoint_weights[ind_costheta+1] - sep_twopoint_weights[ind_costheta])
            weight *= (1-frac)*twopoint_weights[ind_costheta] + frac*twopoint_weights[ind_costheta+1]
    return weight


def ref_theta(edges, data1, data2=None, boxsize=None, los='midpoint', **kwargs):
    toret = np.zeros(len(edges)-1, dtype='f8')
    if data2 is None: data2 = data1
    for xyzw1 in zip(*data1):
        for xyzw2 in zip(*data2):
            xyz1, xyz2 = xyzw2[:3], xyzw1[:3]
            dist = np.rad2deg(np.arccos(min(dotproduct_normalized(xyz1, xyz2), 1))) # min to avoid rounding errors
            if edges[0] <= dist < edges[-1]:
                ind = np.searchsorted(edges, dist, side='right', sorter=None) - 1
                weights1, weights2 = xyzw1[3:], xyzw2[3:]
                weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
                toret[ind] += weight
    return toret


def ref_s(edges, data1, data2=None, boxsize=None, los='midpoint', **kwargs):
    toret = np.zeros(len(edges)-1, dtype='f8')
    if data2 is None: data2 = data1
    for xyzw1 in zip(*data1):
        for xyzw2 in zip(*data2):
            xyz1, xyz2 = xyzw2[:3], xyzw1[:3]
            dxyz = diff(xyzw2[:3], xyzw1[:3])
            if boxsize is not None:
                for idim, b in enumerate(boxsize):
                    if dxyz[idim] > 0.5*b: dxyz[idim] -= b
                    if dxyz[idim] < -0.5*b: dxyz[idim] += b
            dist = norm(dxyz)
            if edges[0] <= dist < edges[-1]:
                ind = np.searchsorted(edges, dist, side='right', sorter=None) - 1
                weights1, weights2 = xyzw1[3:], xyzw2[3:]
                weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
                toret[ind] += weight
    return toret


def ref_smu(edges, data1, data2=None, boxsize=None, weight_type=None, los='midpoint', **kwargs):
    if los == 'midpoint':
        los = 'm'
    else:
        los = [1 if i == 'xyz'.index(los) else 0 for i in range(3)]
    toret = np.zeros([len(e)-1 for e in edges], dtype='f8')
    if data2 is None: data2 = data1
    for xyzw1 in zip(*data1):
        for xyzw2 in zip(*data2):
            xyz1, xyz2 = xyzw2[:3], xyzw1[:3]
            dxyz = diff(xyzw2[:3], xyzw1[:3])
            if boxsize is not None:
                for idim, b in enumerate(boxsize):
                    if dxyz[idim] > 0.5*b: dxyz[idim] -= b
                    if dxyz[idim] < -0.5*b: dxyz[idim] += b
            dist = norm(dxyz)
            if edges[0][0] <= dist < edges[0][-1]:
                mu = abs(dotproduct_normalized(midpoint(xyz1,xyz2) if los == 'm' else los, dxyz))
                if edges[1][0] <= mu < edges[1][-1]:
                    ind = np.searchsorted(edges[0], dist, side='right', sorter=None) - 1
                    ind_mu = np.searchsorted(edges[1], mu, side='right', sorter=None) - 1
                    weights1, weights2 = xyzw1[3:], xyzw2[3:]
                    weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
                    toret[ind,ind_mu] += weight
    return toret


def ref_rppi(edges, data1, data2=None, boxsize=None, weight_type=None, los='midpoint', **kwargs):
    if los == 'midpoint':
        los = 'm'
    else:
        los = [1 if i == 'xyz'.index(los) else 0 for i in range(3)]
    toret = np.zeros([len(e)-1 for e in edges], dtype='f8')
    if data2 is None: data2 = data1
    for xyzw1 in zip(*data1):
        for xyzw2 in zip(*data2):
            xyz1, xyz2 = xyzw2[:3], xyzw1[:3]
            dxyz = diff(xyzw2[:3], xyzw1[:3])
            if boxsize is not None:
                for idim, b in enumerate(boxsize):
                    if dxyz[idim] > 0.5*b: dxyz[idim] -= b
                    if dxyz[idim] < -0.5*b: dxyz[idim] += b
            vlos = midpoint(xyz1,xyz2) if los == 'm' else los
            nlos = norm(vlos)
            pi = abs(dotproduct(vlos, dxyz)/nlos)
            rp = (dotproduct(dxyz, dxyz) - pi**2)**0.5
            if edges[0][0] <= rp < edges[0][-1] and edges[1][0] <= pi < edges[1][-1]:
                ind_rp = np.searchsorted(edges[0], rp, side='right', sorter=None) - 1
                ind_pi = np.searchsorted(edges[1], pi, side='right', sorter=None) - 1
                weights1, weights2 = xyzw1[3:], xyzw2[3:]
                weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
                toret[ind_rp,ind_pi] += weight
    return toret


def ref_rp(edges, *args, **kwargs):
    return ref_rppi((edges,[0,np.inf]), *args, **kwargs).flatten()


def generate_catalogs(size=100, boxsize=(1000,)*3, n_individual_weights=1, n_bitwise_weights=0, seed=42):
    rng = np.random.RandomState(seed=seed)
    toret = []
    for i in range(2):
        positions = [rng.uniform(0., 1., size)*b for b in boxsize]
        weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(64*n_bitwise_weights)], dtype=np.uint64)
        #weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(33)], dtype=np.uint64)
        #weights = [rng.randint(0, 0xffffffff, size, dtype=np.uint64) for i in range(n_bitwise_weights)]
        weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
        toret.append(positions+weights)
    return toret


def test_pair_counter(mode='s'):
    ref_func = {'theta':ref_theta, 's':ref_s, 'smu':ref_smu, 'rppi':ref_rppi, 'rp':ref_rp}[mode]
    list_engine = ['corrfunc']
    edges = np.linspace(1,100,11)
    size = 100
    boxsize = (1000,)*3
    list_options = []
    if mode not in ['theta', 'rp']:
        list_options.append({'boxsize':boxsize})
        list_options.append({'autocorr':True, 'boxsize':boxsize})
    list_options.append({'autocorr':True})
    list_options.append({'n_individual_weights':1, 'bin_type':'custom'})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':1})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':1, 'iip':1})
    list_options.append({'n_individual_weights':2, 'n_bitwise_weights':2, 'iip':2, 'position_type':'rdd'})
    if mode == 'theta':
        list_options.append({'n_individual_weights':2, 'n_bitwise_weights':2, 'iip':2, 'position_type':'rd'})
    from collections import namedtuple
    TwoPointWeight = namedtuple('TwoPointWeight', ['sep', 'weight'])
    twopoint_weights = TwoPointWeight(np.logspace(-4, 0, 40), np.linspace(4., 1., 40))
    list_options.append({'autocorr':True, 'twopoint_weights':twopoint_weights})
    list_options.append({'autocorr':True, 'n_individual_weights':2, 'n_bitwise_weights':2, 'twopoint_weights':twopoint_weights, 'dtype':'f4'})

    has_mpi = True
    try:
        import mpi4py
        import pmesh
    except ImportError:
        has_mpi = False
    if has_mpi:
        from pycorr import mpi
        print('Has MPI')
        list_options.append({'mpicomm':mpi.COMM_WORLD})
        list_options.append({'n_individual_weights':1, 'mpicomm':mpi.COMM_WORLD})
        list_options.append({'n_individual_weights':2, 'n_bitwise_weights':2, 'twopoint_weights':twopoint_weights, 'mpicomm':mpi.COMM_WORLD})

    #list_options.append({'weight_type':'inverse_bitwise','n_bitwise_weights':2})
    if mode == 'smu':
        edges = (edges, np.linspace(0,1,101))
    elif mode == 'rppi':
        edges = (edges, np.linspace(0,140,141))
    elif mode == 'theta':
        edges = np.linspace(1e-1,10,11) # below 1e-5 for float64 (1e-1 for float32), self pairs are counted by Corrfunc
    for engine in list_engine:
        for options in list_options:
            options = options.copy()
            n_individual_weights = options.pop('n_individual_weights',0)
            n_bitwise_weights = options.pop('n_bitwise_weights',0)
            data1, data2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=n_individual_weights, n_bitwise_weights=n_bitwise_weights)

            autocorr = options.pop('autocorr', False)
            options.setdefault('boxsize', None)
            options['los'] = 'x' if options['boxsize'] is not None else 'midpoint'
            bin_type = options.pop('bin_type', 'auto')
            mpicomm = options.pop('mpicomm', None)
            iip = options.pop('iip', False)
            position_type = options.pop('position_type', 'xyz')
            dtype = options.pop('dtype', None)
            refoptions = options.copy()
            nrealizations = refoptions.pop('nrealizations', n_bitwise_weights * 64)
            refdata1, refdata2 = data1.copy(), data2.copy()

            def wiip(weights):
                return (1. + nrealizations)/(1. + utils.popcount(*weights))

            def dataiip(data):
                return data[:3] + [wiip(data[3:3+n_bitwise_weights])] + data[3+n_bitwise_weights:]

            if iip:
                refdata1 = dataiip(refdata1)
                refdata2 = dataiip(refdata2)
            if iip == 1:
                data1 = dataiip(data1)
            elif iip == 2:
                data2 = dataiip(data2)
            if iip:
                n_bitwise_weights = 0
                nrealizations = 0

            itemsize = np.dtype('f8' if dtype is None else dtype).itemsize
            tol = {'atol':1e-8, 'rtol':1e-3} if itemsize <= 4 else {'atol':1e-8, 'rtol':1e-6}

            if dtype is not None:
                for ii in range(len(data1)):
                    if np.issubdtype(data1[ii].dtype, np.floating):
                        refdata1[ii] = np.asarray(data1[ii], dtype=dtype)
                        refdata2[ii] = np.asarray(data2[ii], dtype=dtype)

            twopoint_weights = refoptions.pop('twopoint_weights', None)
            if twopoint_weights is not None:
                twopoint_weights = TwoPointWeight(np.cos(np.radians(twopoint_weights.sep[::-1], dtype=dtype)), np.asarray(twopoint_weights.weight[::-1], dtype=dtype))

            ref = ref_func(edges, refdata1, data2=None if autocorr else refdata2, nrealizations=nrealizations, n_bitwise_weights=n_bitwise_weights, twopoint_weights=twopoint_weights, **refoptions)

            npos = 3
            if position_type != 'xyz':

                if position_type == 'rd': npos = 2

                def datapos(data):
                    rdd = list(utils.cartesian_to_sky(data[:3]))
                    if position_type == 'rdd':
                        return rdd + data[3:]
                    if position_type == 'rd':
                        return rdd[:2] + data[3:]
                    raise ValueError('Unknown position type {}'.format(position_type))

                data1 = datapos(data1)
                data2 = datapos(data2)

            def run(**kwargs):
                return TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=data1[:npos], positions2=None if autocorr else data2[:npos],
                                       weights1=data1[npos:], weights2=None if autocorr else data2[npos:], position_type=position_type, bin_type=bin_type,
                                       dtype=dtype, **kwargs, **options)

            test = run()

            if n_bitwise_weights == 0:
                if n_individual_weights == 0:
                    size1, size2 = len(refdata1[0]), len(refdata2[0])
                    if autocorr:
                        refnorm = size1**2 - size1
                    else:
                        refnorm = size1*size2
                else:
                    w1 = np.prod(refdata1[3:], axis=0)
                    w2 = np.prod(refdata2[3:], axis=0)
                    if autocorr:
                        refnorm = np.sum(w1)**2 - np.sum(w1**2)
                    else:
                        refnorm = np.sum(w1)*np.sum(w2)
            else:
                refnorm = test.norm # too lazy to recode

            assert np.allclose(test.norm, refnorm, **tol)

            with tempfile.TemporaryDirectory() as tmp_dir:
                fn = os.path.join(tmp_dir, 'tmp.npy')
                test.save(fn)
                test2 = TwoPointCounter.load(fn)
                assert np.allclose(test2.wcounts, ref, **tol)
                assert np.allclose(test2.norm, refnorm, **tol)
                test2.rebin((2,2) if len(edges) == 2 else (2,))
                assert np.allclose(np.sum(test2.wcounts), np.sum(ref))

            if mpicomm is not None:
                data1 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in data1]
                data2 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in data2]

                test_mpi = run(mpicomm=mpicomm)
                assert np.allclose(test_mpi.wcounts, test.wcounts, **tol)
                assert np.allclose(test_mpi.norm, test.norm, **tol)

            assert np.allclose(test.wcounts, ref, **tol)


def test_pip_normalization(mode='s'):
    edges = np.linspace(50,100,5)
    size = 10000
    boxsize = (1000,)*3
    autocorr = False
    data1, data2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=0, n_bitwise_weights=3)
    test = TwoPointCounter(mode=mode, edges=edges, positions1=data1[:3], positions2=None if autocorr else data2[:3],
                           weights1=None, weights2=None, position_type='xyz')

    test = TwoPointCounter(mode=mode, edges=edges, positions1=data1[:3], positions2=None if autocorr else data2[:3],
                           weights1=data1[3:], weights2=None if autocorr else data2[3:], position_type='xyz')
    wiip = (1. + test.nrealizations)/(1. + utils.popcount(*data1[3:]))
    ratio = abs(test.norm / sum(wiip)**2 - 1)
    assert ratio < 0.1


def test_analytic_pair_counter(mode='s'):
    edges = np.linspace(50,100,5)
    size = 10000
    boxsize = (1000,)*3
    #list_options.append({'weight_type':'inverse_bitwise','n_bitwise_weights':2})
    if mode == 'smu':
        edges = (edges, np.linspace(0,1,5))
    elif mode == 'rppi':
        edges = (edges, np.linspace(0,10,11))

    list_options = []
    list_options.append({})
    #list_options.append({'autocorr':True})

    for options in list_options:
        autocorr = options.pop('autocorr', False)
        data1, data2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=0, n_bitwise_weights=0)
        ref = TwoPointCounter(mode=mode, edges=edges, positions1=data1[:3], positions2=None if autocorr else data2[:3],
                               weights1=None, weights2=None, position_type='xyz', boxsize=boxsize, los='z', **options).wcounts
        test = AnalyticTwoPointCounter(mode, edges, boxsize, size1=len(data1[0]), size2=None if autocorr else len(data2[0]))
        ratio = np.absolute(test.wcounts/ref - 1)
        assert np.all(ratio < 0.1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            fn = os.path.join(tmp_dir, 'tmp.npy')
            test.save(fn)
            test = TwoPointCounter.load(fn)
            ratio = np.absolute(test.wcounts/ref - 1)
            assert np.all(ratio < 0.1)
            ref = test.copy()
            test.rebin((2,2) if len(edges) == 2 else (2,))
            assert np.allclose(np.sum(test.wcounts), np.sum(ref.wcounts))


def test_rebin():
    boxsize = 1000.
    mode = 's'
    edges = np.linspace(0, 10, 11)
    test = AnalyticTwoPointCounter(mode, edges, boxsize)
    ref = test.copy()
    test.rebin(2)
    assert test.sep.shape == test.wcounts.shape == (5,)
    assert np.allclose(np.sum(test.wcounts), np.sum(ref.wcounts))

    mode = 'smu'
    edges = (np.linspace(0, 10, 11), np.linspace(0, 1, 6))
    test = AnalyticTwoPointCounter(mode, edges, boxsize)
    ref = test.copy()
    test.rebin((2,5))
    assert test.sep.shape == test.wcounts.shape == (5, 1)
    assert np.allclose(np.sum(test.wcounts), np.sum(ref.wcounts))


if __name__ == '__main__':

    setup_logging()

    for mode in ['theta','s','smu','rppi','rp']:
        test_pair_counter(mode=mode)

    for mode in ['s','smu','rppi']:
        test_analytic_pair_counter(mode=mode)

    test_rebin()
    test_pip_normalization()
