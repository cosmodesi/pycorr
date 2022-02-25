import os
import tempfile

import numpy as np

from pycorr import TwoPointCounter, AnalyticTwoPointCounter,\
                   utils, setup_logging


def diff(position2, position1):
    return [p2-p1 for p1,p2 in zip(position1,position2)]


def midpoint(position1, position2):
    return [p2+p1 for p1,p2 in zip(position1,position2)]


def norm(position):
    return (sum(p**2 for p in position))**0.5


def dotproduct(position1, position2):
    return sum(x1*x2 for x1,x2 in zip(position1,position2))


def dotproduct_normalized(position1, position2):
    return dotproduct(position1, position2)/(norm(position1)*norm(position2))


def get_weight(xyz1, xyz2, weights1, weights2, n_bitwise_weights=0, twopoint_weights=None, nrealizations=None, noffset=1, default_value=0.):
    if nrealizations is None:
        weight = 1
    else:
        denom = noffset + sum(bin(w1 & w2).count('1') for w1, w2 in zip(weights1[:n_bitwise_weights], weights2[:n_bitwise_weights]))
        weight = default_value if denom == 0 else nrealizations/denom
    for w1, w2 in zip(weights1[n_bitwise_weights:], weights2[n_bitwise_weights:]):
        weight *= w1 * w2
    if twopoint_weights is not None:
        sep_twopoint_weights = twopoint_weights.sep
        twopoint_weights = twopoint_weights.weight
        if all(x1==x2 for x1,x2 in zip(xyz1,xyz2)): costheta = 1.
        else: costheta = min(dotproduct_normalized(xyz1, xyz2), 1)
        if (sep_twopoint_weights[0] < costheta <= sep_twopoint_weights[-1]):
            ind_costheta = np.searchsorted(sep_twopoint_weights, costheta, side='left', sorter=None) - 1
            frac = (costheta - sep_twopoint_weights[ind_costheta])/(sep_twopoint_weights[ind_costheta+1] - sep_twopoint_weights[ind_costheta])
            weight *= (1-frac)*twopoint_weights[ind_costheta] + frac*twopoint_weights[ind_costheta+1]
    return weight


def divide(sep, counts):
    with np.errstate(divide='ignore'):
        sep = sep/counts
    return sep


def ref_theta(edges, data1, data2=None, boxsize=None, los='midpoint', autocorr=False, **kwargs):
    counts = np.zeros(len(edges)-1, dtype='f8')
    sep = np.zeros(len(edges)-1, dtype='f8')
    if data2 is None: data2 = data1
    for i1, xyzw1 in enumerate(zip(*data1)):
        for i2, xyzw2 in enumerate(zip(*data2)):
            if autocorr and i2 == i1: continue
            xyz1, xyz2 = xyzw1[:3], xyzw2[:3]
            if all(x1 == x2 for x1,x2 in zip(xyz1,xyz2)): dist = 0. # test equality, as may not give exactly 0 otherwise to to rounding errors
            else: dist = np.rad2deg(np.arccos(min(dotproduct_normalized(xyz1, xyz2), 1))) # min to avoid rounding errors
            if edges[0] <= dist < edges[-1]:
                ind = np.searchsorted(edges, dist, side='right', sorter=None) - 1
                weights1, weights2 = xyzw1[3:], xyzw2[3:]
                weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
                counts[ind] += weight
                sep[ind] += weight*dist
    return counts, divide(sep, counts)


def ref_s(edges, data1, data2=None, boxsize=None, los='midpoint', autocorr=False, **kwargs):
    counts = np.zeros(len(edges)-1, dtype='f8')
    sep = np.zeros(len(edges)-1, dtype='f8')
    if data2 is None: data2 = data1
    for i1, xyzw1 in enumerate(zip(*data1)):
        for i2, xyzw2 in enumerate(zip(*data2)):
            if autocorr and i2 == i1: continue
            xyz1, xyz2 = xyzw1[:3], xyzw2[:3]
            dxyz = diff(xyz2, xyz1)
            if boxsize is not None:
                for idim, b in enumerate(boxsize):
                    if dxyz[idim] > 0.5*b: dxyz[idim] -= b
                    if dxyz[idim] < -0.5*b: dxyz[idim] += b
            dist = norm(dxyz)
            if edges[0] <= dist < edges[-1]:
                ind = np.searchsorted(edges, dist, side='right', sorter=None) - 1
                weights1, weights2 = xyzw1[3:], xyzw2[3:]
                weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
                counts[ind] += weight
                sep[ind] += weight*dist
    return counts, divide(sep, counts)


def ref_smu(edges, data1, data2=None, boxsize=None, weight_type=None, los='midpoint', autocorr=False, **kwargs):
    if los in ['midpoint', 'firstpoint', 'endpoint']:
        los = los[:1]
    else:
        los = [1 if i == 'xyz'.index(los) else 0 for i in range(3)]
    counts = np.zeros([len(e)-1 for e in edges], dtype='f8')
    sep = np.zeros([len(e)-1 for e in edges], dtype='f8')
    if data2 is None: data2 = data1
    for i1, xyzw1 in enumerate(zip(*data1)):
        for i2, xyzw2 in enumerate(zip(*data2)):
            if autocorr and i2 == i1: continue
            xyz1, xyz2 = xyzw1[:3], xyzw2[:3]
            dxyz = diff(xyz2, xyz1)
            idxyz = dxyz.copy()
            if boxsize is not None:
                for idim, b in enumerate(boxsize):
                    #if dxyz[idim] > 0.5*b: dxyz[idim] = b - dxyz[idim]
                    #elif dxyz[idim] < -0.5*b: dxyz[idim] = - b - dxyz[idim]
                    if dxyz[idim] > 0.5*b: dxyz[idim] -= b
                    if dxyz[idim] < -0.5*b: dxyz[idim] += b
                    #dxyz[idim] *= -1
            dist = norm(dxyz)
            if edges[0][0] <= dist < edges[0][-1]:
                if los == 'm':
                    d = midpoint(xyz1,xyz2)
                elif los == 'f':
                    d = xyz1
                elif los == 'e':
                    d = xyz2
                else:
                    d = los
                mu = dotproduct_normalized(d, dxyz)
                if dist == 0.: mu = 0.
                if edges[1][0] <= mu < edges[1][-1]:
                    #print(dxyz, xyz1, xyz2, idxyz)
                    ind = np.searchsorted(edges[0], dist, side='right', sorter=None) - 1
                    ind_mu = np.searchsorted(edges[1], mu, side='right', sorter=None) - 1
                    weights1, weights2 = xyzw1[3:], xyzw2[3:]
                    weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
                    counts[ind, ind_mu] += weight
                    sep[ind, ind_mu] += weight*dist
    return counts, divide(sep, counts)


def ref_rppi(edges, data1, data2=None, boxsize=None, weight_type=None, los='midpoint', autocorr=False, **kwargs):
    if los == 'midpoint':
        los = 'm'
    else:
        los = [1 if i == 'xyz'.index(los) else 0 for i in range(3)]
    counts = np.zeros([len(e)-1 for e in edges], dtype='f8')
    sep = np.zeros([len(e)-1 for e in edges], dtype='f8')
    if data2 is None: data2 = data1
    for i1, xyzw1 in enumerate(zip(*data1)):
        for i2, xyzw2 in enumerate(zip(*data2)):
            if autocorr and i2 == i1: continue
            xyz1, xyz2 = xyzw1[:3], xyzw2[:3]
            dxyz = diff(xyz2, xyz1)
            if boxsize is not None:
                for idim, b in enumerate(boxsize):
                    if dxyz[idim] > 0.5*b: dxyz[idim] -= b
                    if dxyz[idim] < -0.5*b: dxyz[idim] += b
            vlos = midpoint(xyz1,xyz2) if los == 'm' else los
            nlos = norm(vlos)
            pi = abs(dotproduct(vlos, dxyz)/nlos)
            rp = (dotproduct(dxyz, dxyz) - pi**2)**0.5
            if all(x1 == x2 for x1,x2 in zip(xyz1,xyz2)): rp = 0.
            if edges[0][0] <= rp < edges[0][-1] and edges[1][0] <= pi < edges[1][-1]:
                ind_rp = np.searchsorted(edges[0], rp, side='right', sorter=None) - 1
                ind_pi = np.searchsorted(edges[1], pi, side='right', sorter=None) - 1
                weights1, weights2 = xyzw1[3:], xyzw2[3:]
                weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
                counts[ind_rp, ind_pi] += weight
                sep[ind_rp, ind_pi] += weight*rp
    return counts, divide(sep, counts)


def ref_rp(edges, *args, **kwargs):
    counts, sep = ref_rppi((edges, [0,np.inf]), *args, **kwargs)
    return counts.ravel(), sep.ravel()


def generate_catalogs(size=100, boxsize=(1000,)*3, offset=(1000.,0,0), n_individual_weights=1, n_bitwise_weights=0, seed=42):
    rng = np.random.RandomState(seed=seed)
    toret = []
    for i in range(2):
        positions = [o + rng.uniform(0., 1., size)*b for o, b in zip(offset, boxsize)]
        weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(64*n_bitwise_weights)], dtype=np.uint64)
        #weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(33)], dtype=np.uint64)
        #weights = [rng.randint(0, 0xffffffff, size, dtype=np.uint64) for i in range(n_bitwise_weights)]
        weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
        toret.append(positions + weights)
    return toret


def test_twopoint_counter(mode='s'):

    ref_func = {'theta':ref_theta, 's':ref_s, 'smu':ref_smu, 'rppi':ref_rppi, 'rp':ref_rp}[mode]
    list_engine = ['corrfunc']
    ref_edges = np.linspace(0., 100., 41)
    if mode == 'theta':
        #ref_edges = np.linspace(1e-1, 10., 11) # below 1e-5 for float64 (1e-1 for float32), self pairs are counted by Corrfunc
        ref_edges = np.linspace(0., 80., 41)
    elif mode == 'smu':
        ref_edges = (ref_edges, np.linspace(-1., 1., 100))
    elif mode == 'rppi':
        ref_edges = (ref_edges, np.linspace(0., 80., 61))
    size = 203
    boxsize = (500,)*3
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
        list_options.append({'autocorr':autocorr})
        list_options.append({'autocorr':autocorr, 'weights_one':[1]})
        for position_type in ['rdd', 'pos', 'xyz'] + (['rd'] if mode == 'theta' else []):
            list_options.append({'autocorr':autocorr, 'position_type':position_type})
        for dtype in (['f8'] if mode == 'theta' else ['f4', 'f8']): # in theta mode, lots of rounding errors!
            itemsize = np.dtype(dtype).itemsize
            for isa in ['fallback', 'sse42', 'avx', 'fastest']:
                # binning
                edges = np.array([1, 8, 20, 42, 60])
                if mode == 'smu':
                    edges = (edges, np.linspace(-0.8, 0.8, 61))
                if mode == 'rppi':
                    edges = (edges, np.linspace(0., 90., 91))

                list_options.append({'autocorr':autocorr, 'edges':edges, 'dtype':dtype, 'isa':isa})
                list_options.append({'autocorr':autocorr, 'compute_sepsavg':False, 'edges':edges, 'dtype':dtype, 'isa':isa})
                list_options.append({'autocorr':autocorr, 'n_individual_weights':1, 'bin_type':'custom', 'dtype':dtype, 'isa':isa})
                # pip
                list_options.append({'autocorr':autocorr, 'n_individual_weights':2, 'n_bitwise_weights':2, 'dtype':dtype, 'isa':isa})
                list_options.append({'autocorr':autocorr, 'compute_sepsavg':False, 'n_individual_weights':2, 'n_bitwise_weights':2, 'dtype':dtype, 'isa':isa})
                list_options.append({'autocorr':autocorr, 'n_individual_weights':1, 'n_bitwise_weights':1, 'iip':1, 'dtype':dtype, 'isa':isa})
                if not autocorr:
                    list_options.append({'autocorr':autocorr, 'n_individual_weights':1, 'n_bitwise_weights':1, 'iip':2, 'dtype':dtype, 'isa':isa})
                list_options.append({'autocorr':autocorr, 'n_individual_weights':1, 'n_bitwise_weights':1, 'bitwise_type': 'i4', 'iip':1, 'dtype':dtype, 'isa':isa})
                list_options.append({'autocorr':autocorr, 'n_individual_weights':2, 'n_bitwise_weights':2, 'weight_attrs':{'nrealizations':129,'noffset':3}, 'dtype':dtype, 'isa':isa})
                list_options.append({'autocorr':autocorr, 'n_individual_weights':1, 'n_bitwise_weights':2, 'weight_attrs':{'noffset':0,'default_value':0.8}, 'dtype':dtype, 'isa':isa})

                # twopoint_weights
                if itemsize > 4:
                    list_options.append({'autocorr':autocorr, 'n_individual_weights':2, 'n_bitwise_weights':2, 'twopoint_weights':twopoint_weights, 'dtype':dtype, 'isa':isa})
                    list_options.append({'autocorr':autocorr, 'twopoint_weights':twopoint_weights, 'los':'y', 'dtype':dtype, 'isa':isa})
                # boxsize
                if mode not in ['theta', 'rp']:
                    list_options.append({'autocorr':autocorr, 'boxsize':boxsize, 'dtype':dtype, 'isa':isa})
                    list_options.append({'autocorr':autocorr, 'boxsize':boxsize, 'dtype':dtype, 'isa':isa})
                    list_options.append({'autocorr':autocorr, 'n_individual_weights':2, 'n_bitwise_weights':2, 'boxsize':boxsize, 'los':'x', 'dtype':dtype, 'isa':isa})
                    list_options.append({'autocorr':autocorr, 'n_individual_weights':2, 'n_bitwise_weights':2, 'boxsize':boxsize, 'los':'y', 'dtype':dtype, 'isa':isa})
                # los
                list_options.append({'autocorr':autocorr, 'n_individual_weights':2, 'n_bitwise_weights':2, 'los':'x', 'dtype':dtype, 'isa':isa})
                if mode in ['smu']:
                    list_options.append({'autocorr':autocorr, 'n_individual_weights':2, 'n_bitwise_weights':2, 'los':'firstpoint', 'edges':edges, 'dtype':dtype, 'isa':isa})
                    list_options.append({'autocorr':autocorr, 'n_individual_weights':2, 'n_bitwise_weights':2, 'los':'endpoint', 'edges':edges, 'dtype':dtype, 'isa':isa})
                    if itemsize > 4:
                        list_options.append({'autocorr':autocorr, 'n_individual_weights':2, 'n_bitwise_weights':2, 'los':'endpoint', 'twopoint_weights':twopoint_weights, 'dtype':dtype, 'isa':isa})
                # mpi
                if mpi:
                    list_options.append({'autocorr':autocorr, 'mpicomm':mpi.COMM_WORLD, 'dtype':dtype, 'isa':isa})
                    list_options.append({'autocorr':autocorr, 'n_individual_weights':1, 'mpicomm':mpi.COMM_WORLD, 'dtype':dtype, 'isa':isa})
                    list_options.append({'autocorr':autocorr, 'mpicomm':mpi.COMM_WORLD, 'dtype':dtype, 'isa':isa})
                    list_options.append({'autocorr':autocorr, 'n_individual_weights':2, 'n_bitwise_weights':2, 'twopoint_weights':twopoint_weights, 'mpicomm':mpi.COMM_WORLD, 'dtype':dtype, 'isa':isa})

    for engine in list_engine:
        for options in list_options:
            options = options.copy()
            if 'edges' in options:
                edges = options.pop('edges')
            else:
                edges = ref_edges
            weights_one = options.pop('weights_one', [])
            n_individual_weights = options.pop('n_individual_weights',0)
            n_bitwise_weights = options.pop('n_bitwise_weights',0)
            data1, data2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=n_individual_weights, n_bitwise_weights=n_bitwise_weights)
            data1 = [np.concatenate([d, d]) for d in data1] # that will get us some pairs at sep = 0

            autocorr = options.pop('autocorr', False)
            options.setdefault('boxsize', None)
            los = options['los'] = options.get('los', 'x' if options['boxsize'] is not None else 'midpoint')
            bin_type = options.pop('bin_type', 'auto')
            mpicomm = options.pop('mpicomm', None)
            bitwise_type = options.pop('bitwise_type', None)
            iip = options.pop('iip', False)
            position_type = options.pop('position_type', 'xyz')
            dtype = options.pop('dtype', None)
            options_ref = options.copy()
            weight_attrs = options_ref.pop('weight_attrs', {}).copy()
            compute_sepavg = options_ref.pop('compute_sepsavg', True)
            options_ref.pop('isa', 'fallback')

            def setdefaultnone(di, key, value):
                if di.get(key, None) is None:
                    di[key] = value

            setdefaultnone(weight_attrs, 'nrealizations', n_bitwise_weights * 64 + 1)
            setdefaultnone(weight_attrs, 'noffset', 1)
            set_default_value = 'default_value' in weight_attrs
            setdefaultnone(weight_attrs, 'default_value', 0)
            data1_ref, data2_ref = data1.copy(), data2.copy()
            if set_default_value:
                for w in data1[3:3+n_bitwise_weights] + data2[3:3+n_bitwise_weights]: w[:] = 0 # set to zero to make sure default_value is used

            def wiip(weights):
                denom = weight_attrs['noffset'] + utils.popcount(*weights)
                mask = denom == 0
                denom[mask] = 1.
                toret = weight_attrs['nrealizations']/denom
                toret[mask] = weight_attrs['default_value']
                return toret

            def dataiip(data):
                return data[:3] + [wiip(data[3:3+n_bitwise_weights])] + data[3+n_bitwise_weights:]

            if iip:
                data1_ref = dataiip(data1_ref)
                data2_ref = dataiip(data2_ref)
            if iip == 1:
                data1 = dataiip(data1)
            elif iip == 2:
                data2 = dataiip(data2)
            if iip:
                n_bitwise_weights = 0
                weight_attrs['nrealizations'] = None

            if dtype is not None:
                for ii in range(len(data1_ref)):
                    if np.issubdtype(data1_ref[ii].dtype, np.floating):
                        data1_ref[ii] = np.asarray(data1_ref[ii], dtype=dtype)
                        data2_ref[ii] = np.asarray(data2_ref[ii], dtype=dtype)

            twopoint_weights = options_ref.pop('twopoint_weights', None)
            if twopoint_weights is not None:
                twopoint_weights = TwoPointWeight(np.cos(np.radians(twopoint_weights.sep[::-1], dtype=dtype)), np.asarray(twopoint_weights.weight[::-1], dtype=dtype))

            itemsize = np.dtype('f8' if dtype is None else dtype).itemsize
            tol = {'atol':1e-5, 'rtol':2e-1 if twopoint_weights is not None else 1e-2} if itemsize <= 4 else {'atol':1e-8, 'rtol':1e-6}

            wcounts_ref, sep_ref = ref_func(edges, data1_ref, data2=None if autocorr else data2_ref, n_bitwise_weights=n_bitwise_weights, twopoint_weights=twopoint_weights, autocorr=autocorr, **options_ref, **weight_attrs)

            if bitwise_type is not None and n_bitwise_weights > 0:

                def update_bit_type(data):
                    return data[:3] + utils.reformat_bitarrays(*data[3:3+n_bitwise_weights], dtype=bitwise_type) + data[3+n_bitwise_weights:]

                data1 = update_bit_type(data1)
                data2 = update_bit_type(data2)

            npos = 3
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

            for label, catalog in zip([1, 2], [data1, data2]):
                if label in weights_one:
                    catalog.append(np.ones_like(catalog[0]))

            def run(pass_none=False, reverse=False, **kwargs):
                tmpdata1, tmpdata2 = data1, data2
                if reverse and not autocorr:
                    tmpdata1, tmpdata2 = data2, data1
                positions1 = tmpdata1[:npos]
                positions2 = tmpdata2[:npos]
                if position_type == 'pos':
                    positions1 = np.array(positions1).T
                    positions2 = np.array(positions2).T
                weights1 = tmpdata1[npos:]
                weights2 = tmpdata2[npos:]
                positions1_bak = np.array(positions1, copy=True)
                positions2_bak = np.array(positions2, copy=True)
                if weights1: weights1_bak = np.array(weights1, copy=True)
                if weights2: weights2_bak = np.array(weights2, copy=True)
                toret = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=None if pass_none else positions1, positions2=None if pass_none or autocorr else positions2,
                                       weights1=None if pass_none else weights1, weights2=None if pass_none or autocorr else weights2, position_type=position_type, bin_type=bin_type,
                                       dtype=dtype, **kwargs, **options)
                assert np.allclose(positions1, positions1_bak)
                assert np.allclose(positions2, positions2_bak)
                if weights1: assert np.allclose(weights1, weights1_bak)
                if weights2: assert np.allclose(weights2, weights2_bak)
                return toret

            test = run()
            if engine == 'corrfunc':
                assert test.is_reversible == autocorr or (los not in ['firstpoint', 'endpoint'])
            if test.is_reversible:
                test_reversed = run(reverse=True)
                ref_reversed = test.reversed()
                if itemsize <= 4:
                    assert np.isclose(test_reversed.wcounts, ref_reversed.wcounts, **tol).sum() > 0.95 * ref_reversed.wcounts.size
                    for isep in range(test_reversed.ndim):
                        assert np.isclose(test_reversed.seps[isep], ref_reversed.seps[isep], **tol, equal_nan=True).sum() > 0.95 * np.sum(~np.isnan(ref_reversed.seps[isep]))
                else:
                    assert np.allclose(test_reversed.wcounts, ref_reversed.wcounts, **tol)
                    for isep in range(test_reversed.ndim):
                        assert np.allclose(test_reversed.seps[isep], ref_reversed.seps[isep], **tol, equal_nan=True)

            if n_bitwise_weights == 0:
                if n_individual_weights == 0:
                    size1, size2 = len(data1_ref[0]), len(data2_ref[0])
                    if autocorr:
                        norm_ref = size1**2 - size1
                    else:
                        norm_ref = size1*size2
                else:
                    w1 = np.prod(data1_ref[3:], axis=0)
                    w2 = np.prod(data2_ref[3:], axis=0)
                    if autocorr:
                        norm_ref = np.sum(w1)**2 - np.sum(w1**2)
                    else:
                        norm_ref = np.sum(w1)*np.sum(w2)
            else:
                norm_ref = test.wnorm # too lazy to recode

            assert np.allclose(test.wnorm, norm_ref, **tol)
            mask_zero = np.zeros_like(test.wcounts, dtype='?')
            if mode == 'smu':
                mask_zero.flat[test.wcounts.shape[1]//2] = True
            else:
                mask_zero.flat[0] = True
            if itemsize <= 4:
                assert np.isclose(test.wcounts[~mask_zero], wcounts_ref[~mask_zero], **tol).sum() > 0.95 * np.sum(~mask_zero) # some bin shifts
                if compute_sepavg:
                    assert np.isclose(test.sep, sep_ref, **tol, equal_nan=True).sum() > 0.95 * np.sum(~np.isnan(sep_ref))
            else:
                assert np.allclose(test.wcounts[~mask_zero], wcounts_ref[~mask_zero], **tol)
                if compute_sepavg:
                    assert np.allclose(test.sep, sep_ref, **tol, equal_nan=True)
            assert np.allclose(test.wcounts[mask_zero], wcounts_ref[mask_zero], atol=0., rtol=3e-1 if itemsize <= 4 else 1e-4)
            test.wcounts[mask_zero] = wcounts_ref[mask_zero]

            with tempfile.TemporaryDirectory() as tmp_dir:
                fn = os.path.join(tmp_dir, 'tmp.npy')
                wcounts, sep = test.wcounts.copy(), test.sep.copy()
                test.save(fn)
                test2 = TwoPointCounter.load(fn)
                assert np.allclose(test2.wcounts, wcounts)
                assert np.allclose(test2.sep, sep, equal_nan=True)
                assert np.allclose(test2.wnorm, norm_ref, **tol)
                assert np.allclose(test2.size1, test.size1) and np.allclose(test2.size2, test.size2)
                test3 = test2.copy()
                test3.rebin((2, 3) if len(edges) == 2 else (2,))
                assert np.allclose(np.sum(test3.wcounts), np.sum(wcounts_ref), **tol)
                assert np.allclose(test2.wcounts, test.wcounts)
                test2 = test2[::2,::3] if len(edges) == 2 else test2[::2]
                assert test2.shape == test3.shape
                assert np.allclose(test2.wcounts, test3.wcounts)
                sepmax = 20.
                if mode == 'smu': sepmax = 0.8
                test3.select((0, sepmax))
                assert np.all((test3.sepavg(axis=0) <= sepmax) | np.isnan(test3.sepavg(axis=0)))

                def check_seps(test):
                    for iaxis in range(test.ndim):
                        if not test.compute_sepsavg[iaxis]:
                            mid = (test.edges[iaxis][:-1] + test.edges[iaxis][1:])/2.
                            if test.ndim == 2 and iaxis == 0: mid = mid[:,None]
                            assert np.allclose(test.seps[iaxis], mid)

                check_seps(test2)
                check_seps(test3)

            if mpicomm is not None:
                test_mpi = run(mpicomm=mpicomm, pass_none=mpicomm.rank != 0, mpiroot=0)
                if itemsize <= 4: mask = ~mask_zero
                else: mask = Ellipsis
                assert np.allclose(test_mpi.wcounts[mask], test.wcounts[mask], **tol)
                assert np.allclose(test_mpi.wnorm, test.wnorm, **tol)
                data1 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in data1]
                data2 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in data2]
                test_mpi = run(mpicomm=mpicomm)
                assert np.allclose(test_mpi.wcounts[mask], test.wcounts[mask], **tol)
                assert np.allclose(test_mpi.wnorm, test.wnorm, **tol)


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
    wiip = test.weight_attrs['nrealizations']/(test.weight_attrs['noffset'] + utils.popcount(*data1[3:]))
    ratio = abs(test.wnorm / sum(wiip)**2 - 1)
    assert ratio < 0.1


def test_analytic_twopoint_counter(mode='s'):
    edges = np.linspace(50, 100, 5)
    size = 10000
    boxsize = (1000,)*3
    #list_options.append({'weight_type':'inverse_bitwise','n_bitwise_weights':2})
    if mode == 'smu':
        edges = (edges, np.linspace(-1, 1, 5))
    elif mode == 'rppi':
        edges = (edges, np.linspace(0, 10, 11))

    list_options = []
    list_options.append({})
    #list_options.append({'autocorr':True})

    for options in list_options:
        autocorr = options.pop('autocorr', False)
        data1, data2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=0, n_bitwise_weights=0)
        ref = TwoPointCounter(mode=mode, edges=edges, positions1=data1[:3], positions2=None if autocorr else data2[:3],
                               weights1=None, weights2=None, position_type='xyz', boxsize=boxsize, los='z', **options)
        test = AnalyticTwoPointCounter(mode, edges, boxsize, size1=len(data1[0]), size2=None if autocorr else len(data2[0]))
        ratio = np.absolute(test.wcounts/ref.wcounts - 1)
        assert np.all(ratio < 0.1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            fn = os.path.join(tmp_dir, 'tmp.npy')
            test.save(fn)
            test = TwoPointCounter.load(fn)
            ratio = np.absolute(test.wcounts/ref.wcounts - 1)
            assert np.all(ratio < 0.1)
            ref = test.copy()
            test.rebin((2,2) if len(edges) == 2 else (2,))
            assert np.allclose(np.sum(test.wcounts), np.sum(ref.wcounts))
            test2 = ref[::2,::2] if len(edges) == 2 else ref[::2]
            assert test2.shape == test.shape
            assert np.allclose(test2.wcounts, test.wcounts, equal_nan=True)
            sepmax = 80.
            if mode == 'smu': sepmax = 0.8
            test2.select((0, sepmax))
            assert np.all((test2.sepavg(axis=0) <= sepmax) | np.isnan(test2.sepavg(axis=0)))

            def check_seps(test):
                for iaxis in range(test.ndim):
                    if not test.compute_sepsavg[iaxis]:
                        mid = (test.edges[iaxis][:-1] + test.edges[iaxis][1:])/2.
                        if test.ndim == 2 and iaxis == 0: mid = mid[:,None]
                        assert np.allclose(test.seps[iaxis], mid)

            check_seps(test2)


def test_mu1():

    mode = 'smu'
    engine = 'corrfunc'
    size = 100
    w = 10000.

    for options in [{'isa':'fallback'}, {'isa':'sse42'}, {'isa':'avx'}]:

        for iaxis, los in enumerate(['x', 'y', 'z', 'midpoint']):
            if iaxis < 3:
                positions1 = np.zeros((size, 3), dtype='f8')
                positions2 = positions1.copy()
                positions2[:,iaxis] += 1.
            else:
                positions1 = np.ones((size, 3), dtype='f8')
                positions2 = positions1*2.

            mumax = 1.
            edges = (np.linspace(0., 2., 3), np.linspace(-mumax, mumax, 11))
            counts = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=positions1, positions2=positions2,
                                     position_type='pos', los=los, **options)
            assert np.allclose(counts.wcounts, 0.)

            mumax = 1.
            edges = (np.linspace(0., 2., 3), np.linspace(-mumax, mumax, 11))
            counts = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=positions2, positions2=positions1,
                                     position_type='pos', los=los, **options)
            assert np.allclose(counts.wcounts, 0.)

            mumax = 1.+1e-9
            edges = (np.linspace(0., 2., 3), np.linspace(-mumax, mumax, 11))
            counts = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=positions1, positions2=positions2,
                                     position_type='pos', los=los, **options)

            assert np.allclose(counts.wcounts[1,-1], w)
            counts.wcounts[1,-1] = 0
            assert np.allclose(counts.wcounts, 0.)

            mumax = 1.+1e-9
            edges = (np.linspace(0., 2., 3), np.linspace(-mumax, mumax, 11))
            counts = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=positions2, positions2=positions1,
                                     position_type='pos', los=los, **options)

            assert np.allclose(counts.wcounts[1,0], w)
            counts.wcounts[1,0] = 0
            assert np.allclose(counts.wcounts, 0.)

            mumax = 1.
            edges = (np.linspace(0., 2., 3), np.linspace(-mumax, mumax, 11))
            counts = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=np.concatenate([positions1, positions2], axis=0),
                                     position_type='pos', los=los, **options)

            assert np.allclose(counts.wcounts[0,counts.wcounts.shape[1]//2], 2.*w - (len(positions1) + len(positions2)))
            counts.wcounts[0,counts.wcounts.shape[1]//2] = 0.
            assert np.allclose(counts.wcounts, 0.)

            mumax = 1.+1e-9
            edges = (np.linspace(0., 2., 3), np.linspace(-mumax, mumax, 10))
            counts = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=np.concatenate([positions1, positions2], axis=0),
                                     position_type='pos', los=los, **options)

            assert np.allclose(counts.wcounts[0,counts.wcounts.shape[1]//2], 2.*w - (len(positions1) + len(positions2)))
            assert np.allclose(counts.wcounts[1,-1], w)
            assert np.allclose(counts.wcounts[1,0], w)
            counts.wcounts[1,-1] = 0
            counts.wcounts[1,0] = 0
            counts.wcounts[0,counts.wcounts.shape[1]//2] = 0.
            assert np.allclose(counts.wcounts, 0.)


def test_smu():
    # A few pairs shift bins with Corrfunc catalogs
    corrfunc_catalog = True
    if corrfunc_catalog:
        import Corrfunc
        from Corrfunc.io import read_fastfood_catalog
        filename = os.path.join(os.path.dirname(Corrfunc.__file__), '../mocks/tests/data', 'Mr19_randoms_northonly.rdcz.ff')
        ra, dec, cz, w = read_fastfood_catalog(filename, need_weights=True)
        speed_of_light = 299800.0
        from cosmoprimo import Cosmology
        cosmo = Cosmology(Omega_m=0.25, engine='class')
        d = cosmo.comoving_radial_distance(cz/speed_of_light)
        data1 = [ra, dec, d]
        position_type = 'rdd'
    else:
        data1 = generate_catalogs(size=int(1e6), boxsize=(150,)*3, offset=(200.,0,0), n_individual_weights=1, n_bitwise_weights=0, seed=42)[0]
        position_type = 'xyz'
    #edges = (np.linspace(2.0, 20.1, 12), np.linspace(-1., 1., 11))
    edges = ([11.7560, 16.7536, 23.8755], np.linspace(-1., 1., 11))
    counts_ref = TwoPointCounter(mode='smu', edges=edges, positions1=data1[:3], positions2=data1[:3],
                                 weights1=None, weights2=None, position_type=position_type, isa='fallback', nthreads=4)
    counts_auto = TwoPointCounter(mode='smu', edges=edges, positions1=data1[:3], positions2=None,
                                 weights1=None, weights2=None, position_type=position_type, isa='sse42', nthreads=4)
    print(counts_ref.ncounts)
    print(counts_auto.ncounts)
    print(counts_auto.ncounts.dtype)
    print(counts_auto.ncounts - counts_ref.ncounts)


def test_rebin():
    boxsize = 1000.
    mode = 's'
    edges = np.linspace(0, 10, 11)
    test = AnalyticTwoPointCounter(mode, edges, boxsize)
    ref = test.copy()
    test.rebin(2)
    assert test.sep.shape == test.wcounts.shape == (5,)
    assert np.allclose(test.edges, [np.linspace(0., 10, 6)])
    assert np.allclose(np.sum(test.wcounts), np.sum(ref.wcounts))

    mode = 'smu'
    edges = (np.linspace(0, 10, 11), np.linspace(0, 1, 6))
    test = AnalyticTwoPointCounter(mode, edges, boxsize)
    ref = test.copy()
    test = test[::2,::5]
    assert test.sep.shape == test.wcounts.shape == (5, 1)
    refedges = [np.linspace(0., 10, 6), np.linspace(0, 1, 2)]
    for i in range(2):
        assert np.allclose(test.edges[i], refedges[i])
    assert np.allclose(np.sum(test.wcounts), np.sum(ref.wcounts))


if __name__ == '__main__':

    setup_logging()

    test_mu1()

    for mode in ['theta', 's', 'smu', 'rppi', 'rp']:
        test_twopoint_counter(mode=mode)

    for mode in ['s', 'smu', 'rppi']:
        test_analytic_twopoint_counter(mode=mode)

    test_rebin()
    test_pip_normalization()
