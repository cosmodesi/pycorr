import os
import time
import tempfile

import pytest
import numpy as np

from pycorr import TwoPointCounter, AnalyticTwoPointCounter, utils, setup_logging
from pycorr.twopoint_counter import TwoPointCounterError, BaseTwoPointCounter, get_inverse_probability_weight, normalization


def diff(position2, position1):
    return [p2 - p1 for p1, p2 in zip(position1, position2)]


def midpoint(position1, position2):
    return [p2 + p1 for p1, p2 in zip(position1, position2)]


def norm(position):
    return (sum(p**2 for p in position))**0.5


def dotproduct(position1, position2):
    return sum(x1 * x2 for x1, x2 in zip(position1, position2))


def dotproduct_normalized(position1, position2):
    return dotproduct(position1, position2) / (norm(position1) * norm(position2))


def get_weight(xyz1, xyz2, weights1, weights2, n_bitwise_weights=0, twopoint_weights=None, nrealizations=None, noffset=1, default_value=0., correction=None):
    if nrealizations is None:
        weight = 1
    else:
        denom = noffset + sum(bin(w1 & w2).count('1') for w1, w2 in zip(weights1[:n_bitwise_weights], weights2[:n_bitwise_weights]))
        if denom == 0:
            weight = default_value
        else:
            weight = nrealizations / denom
            if correction is not None:
                c = tuple(sum(bin(w).count('1') for w in weights[:n_bitwise_weights]) for weights in [weights1, weights2])
                weight /= correction[c]
    for w1, w2 in zip(weights1[n_bitwise_weights:], weights2[n_bitwise_weights:]):
        weight *= w1 * w2
    if twopoint_weights is not None:
        sep_twopoint_weights = twopoint_weights.sep
        twopoint_weights = twopoint_weights.weight
        if all(x1 == x2 for x1, x2 in zip(xyz1, xyz2)): costheta = 1.
        else: costheta = min(dotproduct_normalized(xyz1, xyz2), 1)
        if (sep_twopoint_weights[0] < costheta <= sep_twopoint_weights[-1]):
            ind_costheta = np.searchsorted(sep_twopoint_weights, costheta, side='left', sorter=None) - 1
            frac = (costheta - sep_twopoint_weights[ind_costheta]) / (sep_twopoint_weights[ind_costheta + 1] - sep_twopoint_weights[ind_costheta])
            weight *= (1 - frac) * twopoint_weights[ind_costheta] + frac * twopoint_weights[ind_costheta + 1]
    return weight


def divide(sep, counts):
    with np.errstate(divide='ignore'):
        sep = sep / counts
    return sep


def ref_theta(edges, data1, data2=None, boxsize=None, los='midpoint', autocorr=False, selection_attrs=None, **kwargs):
    counts = np.zeros(len(edges) - 1, dtype='f8')
    sep = np.zeros(len(edges) - 1, dtype='f8')
    if data2 is None: data2 = data1
    for i1, xyzw1 in enumerate(zip(*data1)):
        for i2, xyzw2 in enumerate(zip(*data2)):
            if autocorr and i2 == i1: continue
            xyz1, xyz2 = xyzw1[:3], xyzw2[:3]
            if all(x1 == x2 for x1, x2 in zip(xyz1, xyz2)): dist = 0.  # test equality, as may not give exactly 0 otherwise to to rounding errors
            else: dist = np.rad2deg(np.arccos(min(dotproduct_normalized(xyz1, xyz2), 1)))  # min to avoid rounding errors
            if edges[0] <= dist < edges[-1]:
                ind = np.searchsorted(edges, dist, side='right', sorter=None) - 1
                weights1, weights2 = xyzw1[3:], xyzw2[3:]
                weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
                counts[ind] += weight
                sep[ind] += weight * dist
    return counts, divide(sep, counts)


def wrap(data, boxsize=None):
    if boxsize is None:
        return data
    toret = [[], [], []]
    for xyzw in zip(*data):
        for i in range(3):
            toret[i].append(xyzw[i] % boxsize[i])
    return toret + data[3:]


def ref_s(edges, data1, data2=None, boxsize=None, los='midpoint', autocorr=False, selection_attrs=None, **kwargs):
    counts = np.zeros(len(edges) - 1, dtype='f8')
    sep = np.zeros(len(edges) - 1, dtype='f8')
    if data2 is None: data2 = data1
    data1, data2 = wrap(data1, boxsize=boxsize), wrap(data2, boxsize=boxsize)
    for i1, xyzw1 in enumerate(zip(*data1)):
        for i2, xyzw2 in enumerate(zip(*data2)):
            if autocorr and i2 == i1: continue
            xyz1, xyz2 = xyzw1[:3], xyzw2[:3]
            dxyz = diff(xyz2, xyz1)
            if boxsize is not None:
                for idim, b in enumerate(boxsize):
                    if dxyz[idim] > 0.5 * b: dxyz[idim] -= b
                    if dxyz[idim] < -0.5 * b: dxyz[idim] += b
            dist = norm(dxyz)
            if edges[0] <= dist < edges[-1]:
                ind = np.searchsorted(edges, dist, side='right', sorter=None) - 1
                weights1, weights2 = xyzw1[3:], xyzw2[3:]
                weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
                counts[ind] += weight
                sep[ind] += weight * dist
    return counts, divide(sep, counts)


def ref_smu(edges, data1, data2=None, boxsize=None, weight_type=None, los='midpoint', autocorr=False, selection_attrs=None, **kwargs):
    if los in ['midpoint', 'firstpoint', 'endpoint']:
        los = los[:1]
    else:
        los = [1 if i == 'xyz'.index(los) else 0 for i in range(3)]
    counts = np.zeros([len(e) - 1 for e in edges], dtype='f8')
    sep = np.zeros([len(e) - 1 for e in edges], dtype='f8')
    if data2 is None: data2 = data1
    data1, data2 = wrap(data1, boxsize=boxsize), wrap(data2, boxsize=boxsize)
    selection_attrs = dict(selection_attrs or {})
    rp_limits = selection_attrs.get('rp', None)
    theta_limits = selection_attrs.get('theta', None)
    if theta_limits is not None:
        costheta_limits = np.cos(np.deg2rad(theta_limits)[::-1])
    for i1, xyzw1 in enumerate(zip(*data1)):
        for i2, xyzw2 in enumerate(zip(*data2)):
            if autocorr and i2 == i1: continue
            xyz1, xyz2 = xyzw1[:3], xyzw2[:3]
            dxyz = diff(xyz2, xyz1)
            if boxsize is not None:
                for idim, b in enumerate(boxsize):
                    # if dxyz[idim] > 0.5*b: dxyz[idim] = b - dxyz[idim]
                    # elif dxyz[idim] < -0.5*b: dxyz[idim] = - b - dxyz[idim]
                    if dxyz[idim] > 0.5 * b: dxyz[idim] -= b
                    if dxyz[idim] < -0.5 * b: dxyz[idim] += b
                    # dxyz[idim] *= -1
            dist = norm(dxyz)
            if edges[0][0] <= dist < edges[0][-1]:
                if los == 'm':
                    d = midpoint(xyz1, xyz2)
                elif los == 'f':
                    d = xyz1
                elif los == 'e':
                    d = xyz2
                else:
                    d = los
                mu = dotproduct_normalized(d, dxyz)
                if dist == 0.: mu = 0.
                if rp_limits is not None:
                    rp2 = (1. - mu**2) * dist**2
                    if rp2 < rp_limits[0]**2 or rp2 >= rp_limits[1]**2: continue
                if theta_limits is not None:
                    if all(x1 == x2 for x1, x2 in zip(xyz1, xyz2)): costheta = 1.
                    else: costheta = min(dotproduct_normalized(xyz1, xyz2), 1)
                    if costheta <= costheta_limits[0] or costheta > costheta_limits[1]: continue
                if edges[1][0] < mu < edges[1][-1]:
                    # print(dxyz, xyz1, xyz2, idxyz)
                    ind = np.searchsorted(edges[0], dist, side='right', sorter=None) - 1
                    ind_mu = np.searchsorted(edges[1], mu, side='right', sorter=None) - 1
                    weights1, weights2 = xyzw1[3:], xyzw2[3:]
                    weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
                    counts[ind, ind_mu] += weight
                    sep[ind, ind_mu] += weight * dist
    return counts, divide(sep, counts)


def ref_rppi(edges, data1, data2=None, boxsize=None, weight_type=None, los='midpoint', autocorr=False, selection_attrs=None, **kwargs):
    if los in ['midpoint', 'firstpoint', 'endpoint']:
        los = los[:1]
    else:
        los = [1 if i == 'xyz'.index(los) else 0 for i in range(3)]
    counts = np.zeros([len(e) - 1 for e in edges], dtype='f8')
    sep = np.zeros([len(e) - 1 for e in edges], dtype='f8')
    if data2 is None: data2 = data1
    data1, data2 = wrap(data1, boxsize=boxsize), wrap(data2, boxsize=boxsize)
    for i1, xyzw1 in enumerate(zip(*data1)):
        for i2, xyzw2 in enumerate(zip(*data2)):
            if autocorr and i2 == i1: continue
            xyz1, xyz2 = xyzw1[:3], xyzw2[:3]
            dxyz = diff(xyz2, xyz1)
            if boxsize is not None:
                for idim, b in enumerate(boxsize):
                    if dxyz[idim] > 0.5 * b: dxyz[idim] -= b
                    if dxyz[idim] < -0.5 * b: dxyz[idim] += b
            if los == 'm':
                d = midpoint(xyz1, xyz2)
            elif los == 'f':
                d = xyz1
            elif los == 'e':
                d = xyz2
            else:
                d = los
            nd = norm(d)
            pi = dotproduct(d, dxyz) / nd
            rp = (dotproduct(dxyz, dxyz) - pi**2)**0.5
            if all(x1 == x2 for x1, x2 in zip(xyz1, xyz2)): pi = rp = 0.
            if edges[0][0] <= rp < edges[0][-1] and edges[1][0] < pi < edges[1][-1]:
                ind_rp = np.searchsorted(edges[0], rp, side='right', sorter=None) - 1
                ind_pi = np.searchsorted(edges[1], pi, side='right', sorter=None) - 1
                weights1, weights2 = xyzw1[3:], xyzw2[3:]
                weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
                counts[ind_rp, ind_pi] += weight
                sep[ind_rp, ind_pi] += weight * rp
    return counts, divide(sep, counts)


def ref_rp(edges, *args, **kwargs):
    counts, sep = ref_rppi((edges, [-np.inf, np.inf]), *args, **kwargs)
    return counts.ravel(), sep.ravel()


def generate_catalogs(size=100, boxsize=(1000,) * 3, offset=(1000., 0, 0), n_individual_weights=1, n_bitwise_weights=0, seed=42):
    rng = np.random.RandomState(seed=seed)
    toret = []
    for i in range(2):
        positions = [o + rng.uniform(0., 1., size) * b for o, b in zip(offset, boxsize)]
        weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(64 * n_bitwise_weights)], dtype=np.uint64)
        # weights = [rng.randint(0, 0xffffffff, size, dtype=np.uint64) for i in range(n_bitwise_weights)]
        weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
        toret.append(positions + weights)
    return toret


def test_twopoint_counter(mode='s'):

    ref_func = {'theta': ref_theta, 's': ref_s, 'smu': ref_smu, 'rppi': ref_rppi, 'rp': ref_rp}[mode]
    list_engine = ['corrfunc']
    ref_edges = np.linspace(0., 100., 41)
    #ref_edges = np.linspace(0., 100., 11)
    if mode == 'theta':
        ref_edges = np.linspace(1e-1, 10., 11)  # below 1e-5 for float64 (1e-1 for float32), self-pairs are counted by Corrfunc
        # ref_edges = np.linspace(0., 80., 41)
    elif mode == 'smu':
        ref_edges = (ref_edges, np.linspace(-1., 1., 61))
    elif mode == 'rppi':
        ref_edges = (ref_edges, np.linspace(-80., 80., 61))
        #ref_edges = (ref_edges, np.linspace(-80., 80., 11))
    size = 300
    cboxsize = (300,) * 3
    from collections import namedtuple
    TwoPointWeight = namedtuple('TwoPointWeight', ['sep', 'weight'])
    twopoint_weights = TwoPointWeight(np.logspace(-4, 0, 40), np.linspace(4., 1., 40))
    twopoint_weights2 = TwoPointWeight(np.linspace(1., 10., 40), np.linspace(4., 1., 40))

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
        for dtype in (['f8'] if mode == 'theta' else ['f4', 'f8'][1:]):  # in theta mode, lots of rounding errors!
            itemsize = np.dtype(dtype).itemsize
            for isa in ['fallback', 'sse42', 'avx', 'fastest']:
                # binning
                edges = np.array([1, 8, 20, 42, 60])
                if mode == 'smu':
                    edges = (edges, np.linspace(-0.8, 0.8, 100))
                if mode == 'rppi':
                    edges = (edges, np.linspace(-90., 90., 100))

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
                nthreads = int(os.getenv('OMP_NUM_THREADS', '4'))
                list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 2, 'weight_attrs': {'noffset': 0, 'default_value': 0.8}, 'dtype': dtype, 'isa': isa, 'nthreads': nthreads})
                list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 2, 'weight_attrs': {'normalization': 'counter'}, 'dtype': dtype, 'isa': isa, 'nthreads': nthreads})
                list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'n_bitwise_weights': 0, 'weight_attrs': {'normalization': 'counter'}, 'dtype': dtype, 'isa': isa, 'nthreads': nthreads})

                # twopoint weights
                if itemsize > 4:
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'twopoint_weights': twopoint_weights, 'dtype': dtype, 'isa': isa})
                    list_options.append({'autocorr': autocorr, 'twopoint_weights': twopoint_weights, 'los': 'y', 'dtype': dtype, 'isa': isa, 'nthreads': nthreads, 'mesh_refine_factors': (3, 3) if mode == 'theta' else (3, 3, 3)})

                # boxsize
                if mode not in ['theta']:
                    for boxsize in [cboxsize, (201., 300., 300.)]:
                        list_options.append({'autocorr': autocorr, 'boxsize': boxsize, 'dtype': dtype, 'isa': isa})
                        list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'boxsize': boxsize, 'los': 'x', 'dtype': dtype, 'isa': isa})
                        list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'boxsize': boxsize, 'los': 'z', 'dtype': dtype, 'isa': isa})

                # los
                if mode in ['smu', 'rppi']:
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'los': 'firstpoint', 'edges': edges, 'dtype': dtype, 'isa': isa})
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'los': 'endpoint', 'edges': edges, 'dtype': dtype, 'isa': isa})
                    if itemsize > 4:
                        list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'los': 'firstpoint', 'twopoint_weights': twopoint_weights2, 'dtype': dtype, 'isa': isa})
                        list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'los': 'endpoint', 'twopoint_weights': twopoint_weights, 'dtype': dtype, 'isa': isa})

                # selection
                if mode == 'smu':
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'dtype': dtype, 'isa': isa, 'los': 'firstpoint', 'selection_attrs': {'rp': (0., 10.)}})
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'dtype': dtype, 'isa': isa, 'los': 'endpoint', 'selection_attrs': {'rp': (10., np.inf)}})
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'dtype': dtype, 'isa': isa, 'los': 'midpoint', 'selection_attrs': {'rp': (40., 100.)}})
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'boxsize': cboxsize, 'los': 'y', 'dtype': dtype, 'isa': isa, 'selection_attrs': {'rp': (0., 10.)}})
                    if dtype != 'f4':
                        list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'dtype': dtype, 'isa': isa, 'los': 'firstpoint', 'selection_attrs': {'theta': (0., 5.)}})
                        list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'dtype': dtype, 'isa': isa, 'los': 'endpoint', 'selection_attrs': {'theta': (10., np.inf)}})
                        list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'dtype': dtype, 'isa': isa, 'los': 'midpoint', 'selection_attrs': {'theta': (5., 100.)}})
                        list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'los': 'y', 'dtype': dtype, 'isa': isa, 'selection_attrs': {'theta': (0., 5.)}})
                        list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'los': 'z', 'dtype': dtype, 'isa': isa, 'selection_attrs': {'theta': (0., 5.)}})

                # mpi
                if mpi:
                    list_options.append({'autocorr': autocorr, 'mpicomm': mpi.COMM_WORLD, 'dtype': dtype, 'isa': isa})
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 1, 'mpicomm': mpi.COMM_WORLD, 'dtype': dtype, 'isa': isa})
                    list_options.append({'autocorr': autocorr, 'n_individual_weights': 2, 'n_bitwise_weights': 2, 'twopoint_weights': twopoint_weights, 'mpicomm': mpi.COMM_WORLD, 'dtype': dtype, 'isa': isa})

    for engine in list_engine:
        for options in list_options:
            print(mode, options)
            options = options.copy()
            nthreads = options.pop('nthreads', None)
            edges = options.pop('edges', ref_edges)
            weights_one = options.pop('weights_one', [])
            n_individual_weights = options.pop('n_individual_weights', 0)
            n_bitwise_weights = options.pop('n_bitwise_weights', 0)
            data1, data2 = generate_catalogs(size, boxsize=cboxsize, n_individual_weights=n_individual_weights, n_bitwise_weights=n_bitwise_weights)
            data1 = [np.concatenate([d, d]) for d in data1]  # that will get us some pairs at sep = 0

            autocorr = options.pop('autocorr', False)
            boxsize = options.get('boxsize', None)
            los = options['los'] = options.get('los', 'x' if boxsize is not None else 'midpoint')
            bin_type = options.pop('bin_type', 'auto')
            mpicomm = options.pop('mpicomm', None)
            bitwise_type = options.pop('bitwise_type', None)
            iip = options.pop('iip', False)
            position_type = options.pop('position_type', 'xyz')
            dtype = options.pop('dtype', None)
            options_ref = options.copy()
            weight_attrs = options_ref.pop('weight_attrs', {}).copy()
            selection_attrs = options_ref.pop('selection_attrs', {}).copy()
            compute_sepavg = options_ref.pop('compute_sepsavg', True)
            if engine not in ['corrfunc']: compute_sepavg = False  # only corrfunc supports that for now
            options_ref.pop('isa', 'fallback')
            options_ref.pop('mesh_refine_factors', None)

            def setdefaultnone(di, key, value):
                if di.get(key, None) is None:
                    di[key] = value

            setdefaultnone(weight_attrs, 'nrealizations', n_bitwise_weights * 64 + 1)
            setdefaultnone(weight_attrs, 'noffset', 1)
            set_default_value = 'default_value' in weight_attrs
            setdefaultnone(weight_attrs, 'default_value', 0)
            data1_ref, data2_ref = data1.copy(), data2.copy()
            if set_default_value:
                for w in data1[3: 3 + n_bitwise_weights] + data2[3: 3 + n_bitwise_weights]: w[:] = 0  # set to zero to make sure default_value is used

            def wiip(weights):
                denom = weight_attrs['noffset'] + utils.popcount(*weights)
                mask = denom == 0
                denom[mask] = 1.
                toret = weight_attrs['nrealizations'] / denom
                toret[mask] = weight_attrs['default_value']
                return toret

            def dataiip(data):
                return data[:3] + [wiip(data[3: 3 + n_bitwise_weights])] + data[3 + n_bitwise_weights:]

            if n_bitwise_weights == 0:
                weight_attrs['nrealizations'] = None
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

            if n_bitwise_weights and weight_attrs.get('normalization', None) == 'counter':
                nalways = weight_attrs.get('nalways', 0)
                noffset = weight_attrs.get('noffset', 1)
                nrealizations = weight_attrs['nrealizations']
                noffset = weight_attrs['noffset']
                joint = utils.joint_occurences(nrealizations, noffset=noffset + nalways, default_value=weight_attrs['default_value'])
                correction = np.ones((n_bitwise_weights * 64 + 1,) * 2, dtype='f8')
                for c1 in range(nalways, min(nrealizations - noffset, n_bitwise_weights * 64) + 1):
                    for c2 in range(nalways, min(nrealizations - noffset, n_bitwise_weights * 64) + 1):
                        correction[c1][c2] = joint[c1 - nalways][c2 - nalways] if c2 <= c1 else joint[c2 - nalways][c1 - nalways]
                        correction[c1][c2] /= (nrealizations / (noffset + c1) * nrealizations / (noffset + c2))
                weight_attrs['correction'] = correction
            weight_attrs.pop('normalization', None)

            wcounts_ref, sep_ref = ref_func(edges, data1_ref, data2=data2_ref if not autocorr else None, n_bitwise_weights=n_bitwise_weights, twopoint_weights=twopoint_weights, autocorr=autocorr, selection_attrs=selection_attrs, **options_ref, **weight_attrs)

            itemsize = np.dtype('f8' if dtype is None else dtype).itemsize
            tol = {'atol': 1e-5, 'rtol': 2e-1 if twopoint_weights is not None else 1e-2} if itemsize <= 4 else {'atol': 1e-8, 'rtol': 1e-6}

            if bitwise_type is not None and n_bitwise_weights > 0:

                def update_bit_type(data):
                    return data[:3] + utils.reformat_bitarrays(*data[3: 3 + n_bitwise_weights], dtype=bitwise_type) + data[3 + n_bitwise_weights:]

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

            def run(pass_none=False, pass_zero=False, reverse=False, single_weight=False, **kwargs):
                tmpdata1, tmpdata2 = data1, data2
                if reverse and not autocorr:
                    tmpdata1, tmpdata2 = data2, data1
                positions1 = tmpdata1[:npos]
                positions2 = tmpdata2[:npos]
                weights1 = tmpdata1[npos:]
                weights2 = tmpdata2[npos:]
                has_weights = (bool(weights1), bool(weights2))
                if single_weight:
                    weights1, weights2 = weights1[0], weights2[0]

                def get_zero(arrays):
                    if isinstance(arrays, list):
                        return [array[:0] for array in arrays]
                    return arrays[:0]

                if pass_zero:
                    positions1 = get_zero(positions1)
                    positions2 = get_zero(positions2)
                    weights1 = get_zero(weights1)
                    weights2 = get_zero(weights2)

                if position_type == 'pos':
                    positions1 = np.array(positions1).T
                    positions2 = np.array(positions2).T
                else:
                    positions1 = tuple(positions1)  # to test tuple
                    positions2 = list(positions2)

                positions1_bak = np.array(positions1, copy=True)
                positions2_bak = np.array(positions2, copy=True)
                if has_weights[0]: weights1_bak = np.array(weights1, copy=True)
                if has_weights[1]: weights2_bak = np.array(weights2, copy=True)
                toret = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=None if pass_none else positions1, positions2=None if pass_none or autocorr else positions2,
                                        weights1=None if pass_none else weights1, weights2=None if pass_none or autocorr else weights2, position_type=position_type, bin_type=bin_type,
                                        dtype=dtype, nthreads=nthreads, **{**options, **kwargs})
                assert np.allclose(positions1, positions1_bak)
                assert np.allclose(positions2, positions2_bak)
                if has_weights[0]: assert np.allclose(weights1, weights1_bak)
                if has_weights[1]: assert np.allclose(weights2, weights2_bak)
                return toret

            def run_normalization(pass_none=False, pass_zero=False, reverse=False, single_weight=False, weight_attrs=None, **kwargs):
                tmpdata1, tmpdata2 = data1, data2
                if reverse and not autocorr:
                    tmpdata1, tmpdata2 = data2, data1
                weights1 = tmpdata1[npos:]
                weights2 = tmpdata2[npos:]
                has_weights = (bool(weights1), bool(weights2))
                if single_weight:
                    weights1, weights2 = weights1[0], weights2[0]
                size1, size2 = len(data1[0]), len(data2[0])

                def get_zero(arrays):
                    if isinstance(arrays, list):
                        return [array[:0] for array in arrays]
                    return arrays[:0]

                if pass_zero:
                    size1 = size2 = 0
                    weights1 = get_zero(weights1)
                    weights2 = get_zero(weights2)

                # print(weights1, weights2, has_weights, size1, size2)

                if has_weights[0]: weights1_bak = np.array(weights1, copy=True)
                else: weights1 = size1
                if has_weights[1]: weights2_bak = np.array(weights2, copy=True)
                else: weights2 = size2
                toret = normalization(weights1=None if pass_none else weights1, weights2=None if pass_none or autocorr else weights2,
                                      dtype=dtype, nthreads=nthreads, weight_attrs=options.get('weight_attrs', {}), **kwargs) #weight_attrs={**(weight_attrs or {}), **options.get('weight_attrs', {})}, **kwargs)
                if has_weights[0]: assert np.allclose(weights1, weights1_bak)
                if has_weights[1]: assert np.allclose(weights2, weights2_bak)
                return toret

            if mode == 'theta':
                with pytest.raises(TwoPointCounterError):
                    run(boxsize=1000.)

            test = run()

            if engine == 'corrfunc':
                assert test.is_reversible == autocorr or (los not in ['firstpoint', 'endpoint'])
            if test.is_reversible:
                test_reversed = run(reverse=True, single_weight=len(data1) == len(data2) == 4)
                ref_reversed = test.reverse()
                if itemsize <= 4:
                    assert np.isclose(test_reversed.wcounts, ref_reversed.wcounts, **tol).sum() > 0.95 * ref_reversed.wcounts.size
                    for isep in range(test_reversed.ndim):
                        assert np.isclose(test_reversed.seps[isep], ref_reversed.seps[isep], **tol, equal_nan=True).sum() > 0.95 * np.sum(~np.isnan(ref_reversed.seps[isep]))
                else:
                    #print(test_reversed.wcounts, ref_reversed.wcounts, test.wcounts)
                    assert np.allclose(test_reversed.wcounts, ref_reversed.wcounts, **tol)
                    for isep in range(test_reversed.ndim):
                        assert np.allclose(test_reversed.seps[isep], ref_reversed.seps[isep], **tol, equal_nan=True)

            if n_bitwise_weights == 0:
                if n_individual_weights == 0:
                    size1, size2 = len(data1_ref[0]), len(data2_ref[0])
                    if autocorr:
                        norm_ref = size1**2 - size1
                    else:
                        norm_ref = size1 * size2
                else:
                    w1 = np.prod(data1_ref[3:], axis=0)
                    w2 = np.prod(data2_ref[3:], axis=0)
                    if autocorr:
                        norm_ref = np.sum(w1)**2 - np.sum(w1**2)
                    else:
                        norm_ref = np.sum(w1) * np.sum(w2)
                    norm_ref = np.asarray(norm_ref).astype(dtype)
            else:
                norm_ref = test.wnorm  # too lazy to recode

            test_zero = run(pass_zero=True)
            assert np.allclose(test_zero.wcounts, 0.)
            assert np.allclose(test_zero.wnorm, 0.)

            assert np.allclose(test.wnorm, norm_ref, **tol)

            for wattrs in [None, {'normalization': 'brute_force'}, {'normalization': 'brute_force_npy'}]:
                assert np.allclose(run_normalization(pass_zero=True, weight_attrs=wattrs), 0.)
                #print(run_normalization(weight_attrs=wattrs) / norm_ref - 1., wattrs)
                assert np.allclose(run_normalization(weight_attrs=wattrs), norm_ref, rtol=1e-1 if wattrs is not None else 1e-10)
            assert np.allclose(run_normalization(weight_attrs={'normalization': 'brute_force'}), run_normalization(weight_attrs={'normalization': 'brute_force_npy'}),
                               rtol=1e-4 if itemsize <= 4 else 1e-9)

            mask_zero = np.zeros_like(test.wcounts, dtype='?')
            if mode in ['smu', 'rppi']:
                mask_zero[0, test.wcounts.shape[1] // 2] = True
                if los == 'endpoint':
                    mask_zero[0, (test.wcounts.shape[1] - 1) // 2] = True
            else:
                mask_zero.flat[0] = True
            if itemsize <= 4:
                assert np.isclose(test.wcounts[~mask_zero], wcounts_ref[~mask_zero], **tol).sum() > 0.95 * np.sum(~mask_zero)  # some bin shifts
                if compute_sepavg:
                    assert np.isclose(test.sep, sep_ref, **tol, equal_nan=True).sum() > 0.95 * np.sum(~np.isnan(sep_ref))
            else:
                #print(test.wcounts)
                #print(wcounts_ref)
                #print(test.wcounts[mask_zero] - wcounts_ref[mask_zero])
                #print(test.wcounts[~mask_zero], wcounts_ref[~mask_zero])
                #diff = np.abs(test.wcounts[~mask_zero] - wcounts_ref[~mask_zero])
                #print(diff.sum(), diff.sum() / np.sum(wcounts_ref[~mask_zero] != 0.))
                assert np.allclose(test.wcounts[~mask_zero], wcounts_ref[~mask_zero], **tol)
                #print(test.sep - sep_ref)
                if compute_sepavg:
                    assert np.allclose(test.sep[~mask_zero], sep_ref[~mask_zero], **tol, equal_nan=True)
            #print(np.abs(test.wcounts[mask_zero] - wcounts_ref[mask_zero]) / wcounts_ref[mask_zero], test.wcounts[mask_zero], wcounts_ref[mask_zero])
            #assert np.allclose(test.wcounts[mask_zero], wcounts_ref[mask_zero], atol=0., rtol=5e-1 if itemsize <= 4 else 1e-4)
            if itemsize > 4:
                assert np.allclose(test.wcounts[mask_zero].sum(), wcounts_ref[mask_zero].sum(), atol=0., rtol=1e-4)
            test.wcounts[mask_zero] = wcounts_ref[mask_zero]

            with tempfile.TemporaryDirectory() as tmp_dir:
                fn = os.path.join(tmp_dir, 'tmp.npy')
                fn_txt = os.path.join(tmp_dir, 'tmp.txt')
                wcounts, sep = test.wcounts.copy(), test.sep.copy()
                test.save_txt(fn_txt)
                tmp = np.loadtxt(fn_txt, unpack=True)

                def mid(edges):
                    return (edges[:-1] + edges[1:]) / 2.

                for axis in range(test.ndim): assert np.allclose(test.sepavg(axis=axis, method='mid'), mid(test.edges[axis]))
                mids = np.meshgrid(*[mid(test.edges[axis]) for axis in range(test.ndim)], indexing='ij')
                seps = []
                for axis in range(test.ndim): seps += [mids[axis], test.seps[axis]]
                assert np.allclose([tt.reshape(test.shape) for tt in tmp], seps + [wcounts, utils._make_array_like(test.wnorm, test.wcounts)], equal_nan=True)
                test.save(fn)
                test2 = TwoPointCounter.load(fn)
                assert np.allclose(test2.wcounts, wcounts)
                assert np.allclose(test2.sep, sep, equal_nan=True)
                assert np.allclose(test2.wnorm, norm_ref, **tol)
                assert np.allclose(test2.size1, test.size1) and np.allclose(test2.size2, test.size2)
                if twopoint_weights is None:
                    assert test2.cos_twopoint_weights is None
                else:
                    assert np.allclose(test2.cos_twopoint_weights.sep, test.cos_twopoint_weights.sep, equal_nan=True)
                    assert np.allclose(test2.cos_twopoint_weights.weight, test.cos_twopoint_weights.weight, equal_nan=True)
                test3 = test2.copy()
                test3.rebin((2, 3) if len(edges) == 2 else (2,))
                assert np.allclose(np.sum(test3.wcounts), np.sum(wcounts_ref), **tol)
                assert np.allclose(test2.wcounts, test.wcounts)
                test2 = test2[::2, ::3] if len(edges) == 2 else test2[::2]
                assert test2.shape == test3.shape
                assert np.allclose(test2.wcounts, test3.wcounts)
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

                if itemsize <= 4: mask = ~mask_zero
                else: mask = Ellipsis

                test_mpi = run(mpicomm=mpicomm, pass_zero=mpicomm.rank > 0, mpiroot=None)
                assert np.allclose(test_mpi.wcounts[mask], test.wcounts[mask], **tol)
                assert np.allclose(test_mpi.wnorm, test.wnorm, **tol)

                for wattrs in [None, {'normalization': 'brute_force'}, {'normalization': 'brute_force_npy'}]:
                    assert np.allclose(run_normalization(mpicomm=mpicomm, pass_zero=mpicomm.rank > 0, mpiroot=None, weight_attrs=wattrs), norm_ref,
                                       rtol=1e-1 if wattrs is not None else 1e-10)

                #print(run_normalization(mpicomm=mpicomm, pass_zero=mpicomm.rank > 0, mpiroot=None, weight_attrs={'normalization': 'brute_force'}) / run_normalization(mpicomm=mpicomm, pass_zero=mpicomm.rank > 0, mpiroot=None, weight_attrs={'normalization': 'brute_force_npy'}))
                assert np.allclose(run_normalization(mpicomm=mpicomm, pass_zero=mpicomm.rank > 0, mpiroot=None, weight_attrs={'normalization': 'brute_force'}),
                                   run_normalization(mpicomm=mpicomm, pass_zero=mpicomm.rank > 0, mpiroot=None, weight_attrs={'normalization': 'brute_force_npy'}),
                                   rtol=1e-4 if itemsize <= 4 else 1e-9)

                test_mpi = run(mpicomm=mpicomm, pass_none=mpicomm.rank > 0, mpiroot=0)
                assert np.allclose(test_mpi.wcounts[mask], test.wcounts[mask], **tol)
                assert np.allclose(test_mpi.wnorm, test.wnorm, **tol)
                assert np.allclose(run_normalization(mpicomm=mpicomm, pass_zero=mpicomm.rank > 0, mpiroot=0), test_mpi.wnorm, **tol)
                data1 = [mpi.scatter(d, mpiroot=0, mpicomm=mpicomm) for d in data1]
                data2 = [mpi.scatter(d, mpiroot=0, mpicomm=mpicomm) for d in data2]
                test_mpi = run(mpicomm=mpicomm)
                assert np.allclose(run_normalization(mpicomm=mpicomm), test_mpi.wnorm, **tol)
                test_zero = run(mpicomm=mpicomm, pass_zero=True, mpiroot=None)
                assert np.allclose(run_normalization(mpicomm=mpicomm, pass_zero=True, mpiroot=None), test_zero.wnorm, **tol)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    fn = test_mpi.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
                    fn_txt = test_mpi.mpicomm.bcast(os.path.join(tmp_dir, 'tmp.txt'), root=0)
                    test_mpi.save(fn)
                    test_mpi.save_txt(fn_txt)
                    test_mpi.mpicomm.Barrier()
                    test_mpi = TwoPointCounter.load(fn)
                    fn = os.path.join(tmp_dir, 'tmp.npy')
                    test_mpi.save(fn)

                assert np.allclose(test_mpi.wcounts[mask], test.wcounts[mask], **tol)
                assert np.allclose(test_mpi.wnorm, test.wnorm, **tol)


def test_gpu(mode='smu'):

    ref_func = {'theta': ref_theta, 's': ref_s, 'smu': ref_smu, 'rppi': ref_rppi, 'rp': ref_rp}[mode]
    list_engine = ['corrfunc']
    ref_edges = np.linspace(0., 100., 41)
    #ref_edges = np.linspace(0., 100., 11)
    if mode == 'theta':
        ref_edges = np.linspace(1e-1, 10., 11)  # below 1e-5 for float64 (1e-1 for float32), self-pairs are counted by Corrfunc
        # ref_edges = np.linspace(0., 80., 41)
    elif mode == 'smu':
        ref_edges = (ref_edges, np.linspace(-1., 1., 61))
    elif mode == 'rppi':
        ref_edges = (ref_edges, np.linspace(-80., 80., 61))
        #ref_edges = (ref_edges, np.linspace(-80., 80., 11))
    csize = int(1e5)
    cboxsize = (500,) * 3
    edges = ref_edges

    from collections import namedtuple
    TwoPointWeight = namedtuple('TwoPointWeight', ['sep', 'weight'])
    twopoint_weights = TwoPointWeight(np.logspace(-4, 0, 40), np.linspace(4., 1., 40))
    #twopoint_weights = None

    mpi = False
    try:
        from pycorr import mpi
        print('Has MPI')
    except ImportError:
        pass

    list_options = []
    for autocorr in [False, True]:
        for dtype in ['f4', 'f8']:
            for los in ['midpoint', 'firstpoint', 'x', 'z']:
                list_options.append({'autocorr': autocorr, 'los': los, 'dtype': dtype})
                list_options.append({'autocorr': autocorr, 'los': los, 'n_individual_weights': 1, 'dtype': dtype})
                list_options.append({'autocorr': autocorr, 'los': los, 'n_individual_weights': 1, 'size': 0, 'dtype': dtype})
                list_options.append({'autocorr': autocorr, 'los': los, 'selection_attrs': {'rp': (0., 20.)}, 'n_individual_weights': 1, 'n_bitwise_weights': 2, 'twopoint_weights': twopoint_weights, 'dtype': dtype})
                list_options.append({'autocorr': autocorr, 'los': los, 'selection_attrs': {'theta': (0., 5.)}, 'n_individual_weights': 1, 'n_bitwise_weights': 2, 'twopoint_weights': twopoint_weights, 'dtype': dtype})
                list_options.append({'autocorr': autocorr, 'los': los, 'n_individual_weights': 1, 'n_bitwise_weights': 2, 'twopoint_weights': twopoint_weights, 'weight_attrs': {'normalization': 'counter'}, 'dtype': dtype})

    for options in list_options:
        print(options)
        options = options.copy()
        dtype = options.get('dtype', None)
        itemsize = np.dtype('f8' if dtype is None else dtype).itemsize
        size = options.pop('size', csize)
        autocorr = options.pop('autocorr', False)
        n_individual_weights = options.pop('n_individual_weights', 0)
        n_bitwise_weights = options.pop('n_bitwise_weights', 0)
        data1, data2 = generate_catalogs(size, boxsize=cboxsize, n_individual_weights=n_individual_weights, n_bitwise_weights=n_bitwise_weights)
        twopoint_weights = options.pop('twopoint_weights', None)

        kwargs = dict(mode=mode, edges=edges, engine='corrfunc', positions1=data1[:3], positions2=None if autocorr else data2[:3],
                      weights1=data1[3:], weights2=None if autocorr else data2[3:], position_type='xyz', verbose=False, twopoint_weights=twopoint_weights, **options)
        if mpi:
            kwargs.update(mpicomm=mpi.COMM_WORLD, mpiroot=0)

        #TwoPointCounter(nthreads=128, **kwargs)  # imports, to remove them from time count
        t0 = time.time()
        cpu = TwoPointCounter(nthreads=128, **kwargs)
        dt_cpu, t0 = time.time() - t0, time.time()
        gpu = TwoPointCounter(**kwargs, gpu=True, nthreads=4)
        dt_gpu = time.time() - t0
        print('autocorr is {}, GPU time is {:.4f} vs CPU time {:4f}'.format(autocorr, dt_gpu, dt_cpu))
        #print(gpu.wcounts - cpu.wcounts)
        tol = {'atol': 1e-5, 'rtol': 2e-1 if twopoint_weights is not None else 1e-2} if itemsize <= 4 else {'atol': 1e-8, 'rtol': 1e-6}
        assert np.allclose(gpu.wcounts, cpu.wcounts, **tol)
        assert np.allclose(gpu.seps[0], cpu.seps[0], equal_nan=True, **tol)

        if size:
            with pytest.raises(NotImplementedError):
                kw = {**kwargs, 'mode': 'rppi'}
                kw.pop('selection_attrs', None)
                TwoPointCounter(**kw, gpu=True)
        '''
        from pycorr import TwoPointCorrelationFunction
        TwoPointCorrelationFunction(mode=mode, edges=edges, engine='corrfunc',
                                    data_positions1=data1[:3], data_positions2=None if autocorr else data2[:3],
                                    data_weights1=data1[3:], data_weights2=None if autocorr else data2[3:],
                                    randoms_positions1=data1[:3], randoms_positions2=None if autocorr else data2[:3],
                                    randoms_weights1=data1[3:], randoms_weights2=None if autocorr else data2[3:],
                                    position_type='xyz', gpu=True, nthreads=4, verbose=False, **options)
        '''


class FakeTwoPointCounter(BaseTwoPointCounter):

    def run(self):
        pass


def format_pip_reference():
    ref_dir = 'reference_pip'
    for mode in ['gal_w_uncorrelated', 'gal_w_correlated']:
        fn = os.path.join(ref_dir, 'Arnaud_test_{}.dat'.format(mode))
        nbits = 62
        dtype = [(axis, np.float64) for axis in 'xyz']
        dtype += [('indweight', np.float64), ('bitweight0', np.int32), ('bitweight1', np.int32)]
        dtype += [('bit{:d}'.format(ibit), np.bool_) for ibit in range(nbits)]
        tmp = []
        with open(fn, 'r') as file:
            for line in file:
                if line.strip().startswith('#'): continue
                line = [el.strip() for el in line.split(' ')]
                line = [el for el in line if el]
                if line[3] == 'F':
                    continue
                for ii in range(4, len(line)):
                    if ii == 4:
                        line[ii] = float(line[ii])
                    elif ii in [5, 6]:
                        line[ii] = int(line[ii])
                    else:
                        line[ii] = line[ii] == 'T'
                tmp.append(tuple(line[:3] + line[4:]))
        tmp = np.array(tmp, dtype=dtype)
        np.save(os.path.join(ref_dir, '{}.npy'.format(mode)), tmp)
    fn = os.path.join(ref_dir, 'Arnaud_test_ran.dat')
    tmp = np.loadtxt(fn, dtype=[(axis, np.float64) for axis in 'xyz'])
    np.save(os.path.join(ref_dir, 'ran.npy'), tmp)


def test_pip_normalization():

    mpicomm = None
    try:
        from pycorr import mpi
        print('Has MPI')
    except ImportError:
        pass
    else:
        mpicomm = mpi.COMM_WORLD

    edges = np.linspace(50, 100, 5)
    size = 10000
    boxsize = (1000,) * 3
    autocorr = False
    cdata1, cdata2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=0, n_bitwise_weights=3)
    data1, data2 = cdata1, cdata2
    if mpicomm is not None:
        data1 = [mpi.scatter(d, mpicomm=mpicomm, mpiroot=0) for d in cdata1]
        data2 = [mpi.scatter(d, mpicomm=mpicomm, mpiroot=0) for d in cdata2]
    test = TwoPointCounter(mode='s', edges=edges, positions1=data1[:3], positions2=None if autocorr else data2[:3],
                           weights1=data1[3:], weights2=None if autocorr else data2[3:], position_type='xyz', mpicomm=mpicomm, mpiroot=None)
    wiip = test.weight_attrs['nrealizations'] / (test.weight_attrs['noffset'] + utils.popcount(*cdata1[3:]))
    ratio = abs(test.wnorm / sum(wiip)**2 - 1)
    assert ratio < 0.1

    # format_pip_reference()
    ref_dir = 'reference_pip'
    truths = {}
    truths['uncorrelated'] = {'wiip': 1879818236162.4490, 'wpip': 1447036254738.5381, 'wpip_brute_force': 1447030955971.1658}
    truths['correlated'] = {'wiip': 1809802777533.6121, 'wpip': 599059643422.05042, 'wpip_brute_force': 599055116656.72998}
    truths['noweights'] = {'wiip': 75112458874.431274, 'wpip': 57853152992.695000, 'wpip_brute_force': 57852904486.340004}
    for mode, truth in truths.items():
        no_indweight = mode == 'noweights'
        if no_indweight: mode = 'correlated'
        fn = os.path.join(ref_dir, 'gal_w_{}.npy'.format(mode))
        tmp = np.load(fn)
        nbits = len([name for name in tmp.dtype.names if name.startswith('bit') and 'weight' not in name])
        for bitweights in [[tmp['bitweight0'], tmp['bitweight1']], utils.pack_bitarrays(*[tmp['bit{:d}'.format(ibit)] for ibit in range(nbits)])]:
            weights = [tmp['indweight']] + bitweights
            if mpicomm is not None:
                weights = [mpi.scatter(d, mpicomm=mpicomm, mpiroot=0) for d in weights]
            weights_iip = [weights[0], get_inverse_probability_weight(weights[1:], noffset=1, nrealizations=1 + nbits, default_value=0.)]
            if no_indweight:
                weights = weights[1:]
                weights_iip = weights_iip[1:]

            positions = [np.zeros(len(weights[0]), dtype='f8')] * 3
            edges = np.linspace(50, 100, 5)
            kwargs = dict(position_type='xyz', nthreads=4, dtype='f8', mpicomm=mpicomm, mpiroot=None)

            if 'wiip' in truth:
                counter = FakeTwoPointCounter(mode='s', edges=edges, positions1=positions, weights1=weights_iip, weight_attrs={'nrealizations': 1 + nbits}, **kwargs)
                assert np.allclose(counter.wnorm / 2., truth['wiip'], rtol=1e-12)

            if 'wpip' in truth:
                counter = FakeTwoPointCounter(mode='s', edges=edges, positions1=positions, weights1=weights, weight_attrs={'nrealizations': 1 + nbits}, **kwargs)
                assert np.allclose(counter.wnorm / 2., truth['wpip'], rtol=1e-12)

            if 'wpip_brute_force' in truth:
                counter = FakeTwoPointCounter(mode='s', edges=edges, positions1=positions, weights1=weights, weight_attrs={'nrealizations': 1 + nbits, 'normalization': 'brute_force'}, **kwargs)
                assert np.allclose(counter.wnorm / 2., truth['wpip_brute_force'], rtol=1e-8 if no_indweight else 1e-12)
                counter_npy = FakeTwoPointCounter(mode='s', edges=edges, positions1=positions, weights1=weights, weight_attrs={'nrealizations': 1 + nbits, 'normalization': 'brute_force_npy'}, **kwargs)
                # print(no_indweight, counter_npy.wnorm / counter.wnorm - 1., counter_npy.wnorm / 2. / truth['wpip_brute_force'] - 1., counter.wnorm / 2. / truth['wpip_brute_force'] - 1.)
                assert np.allclose(counter_npy.wnorm, counter.wnorm, rtol=1e-8 if no_indweight else 1e-12)


def test_pip_counts():

    # format_pip_reference()
    from pycorr import TwoPointCorrelationFunction
    ref_dir = 'reference_pip'
    for mode in ['uncorrelated', 'correlated', 'noweights']:
        ref_fn = os.path.join(ref_dir, '4Arnaud_{}.txt'.format(mode))
        no_indweight = mode == 'noweights'
        if no_indweight: mode = 'correlated'
        fn = os.path.join(ref_dir, 'gal_w_{}.npy'.format(mode))
        tmp = np.load(fn)
        nbits = len([name for name in tmp.dtype.names if name.startswith('bit') and 'weight' not in name])
        data_weights = [tmp['indweight']] + utils.pack_bitarrays(*[tmp['bit{:d}'.format(ibit)] for ibit in range(nbits)])
        data_weights_iip = [data_weights[0], get_inverse_probability_weight(data_weights[1:], noffset=1, nrealizations=1 + nbits)]
        if no_indweight:
            data_weights = data_weights[1:]
            data_weights_iip = data_weights_iip[1:]
        data_positions = [tmp[axis] for axis in 'xyz']
        fn = os.path.join(ref_dir, 'ran.npy')
        tmp = np.load(fn)
        randoms_positions = [tmp[axis] for axis in 'xyz']
        edges = np.linspace(0., 100., 101)
        result = TwoPointCorrelationFunction(mode='s', edges=edges, data_positions1=data_positions, randoms_positions1=randoms_positions, los='midpoint',
                                             data_weights1=data_weights, weight_attrs={'nrealizations': 1 + nbits, 'normalization': 'brute_force'}, position_type='xyz', nthreads=4)
        xi, DD, RR, DR = [XX[:result.shape[0]] for XX in np.loadtxt(ref_fn, unpack=True, usecols=range(1, 5))]
        tol = {'rtol': 1e-9, 'atol': 0.}
        assert np.allclose(result.D1D2.wcounts / 2., DD, **tol)
        assert np.allclose(result.D1R2.wcounts, DR, **tol)
        assert np.allclose(result.R1R2.wcounts / 2., RR, **tol)
        assert np.allclose(result.corr, xi, rtol=1e-9, atol=1e-7)


def test_pip_counts_correction():

    # format_pip_reference()
    from pycorr import TwoPointCorrelationFunction
    ref_dir = 'reference_pip'
    ref_fn = os.path.join(ref_dir, 'pctest_arnaud2_xi_lin.dat')
    mode = 'correlated'
    no_indweight = False
    fn = os.path.join(ref_dir, 'gal_w_{}.npy'.format(mode))
    tmp = np.load(fn)
    nbits = len([name for name in tmp.dtype.names if name.startswith('bit') and 'weight' not in name])
    data_weights = [tmp['indweight']] + utils.pack_bitarrays(*[tmp['bit{:d}'.format(ibit)] for ibit in range(nbits)])
    data_weights_iip = [data_weights[0], get_inverse_probability_weight(data_weights[1:], noffset=1, nrealizations=1 + nbits)]
    if no_indweight:
        data_weights = data_weights[1:]
        data_weights_iip = data_weights_iip[1:]
    data_positions = [tmp[axis] for axis in 'xyz']
    fn = os.path.join(ref_dir, 'ran.npy')
    tmp = np.load(fn)
    randoms_positions = [tmp[axis] for axis in 'xyz']
    edges = np.linspace(0., 100., 101)
    ref = None
    for isa in ['fallback', 'sse42', 'avx']:
        result = TwoPointCorrelationFunction(mode='s', edges=edges, data_positions1=data_positions, randoms_positions1=randoms_positions, los='midpoint',
                                             data_weights1=data_weights, weight_attrs={'nrealizations': 1 + nbits, 'normalization': 'counter'},
                                             isa=isa, position_type='xyz', nthreads=4)
        result2 = TwoPointCorrelationFunction(mode='s', edges=edges, data_positions1=data_positions, randoms_positions1=randoms_positions, los='midpoint',
                                              data_weights1=data_weights, weight_attrs={'nrealizations': 1 + nbits, 'normalization': None},
                                              isa=isa, position_type='xyz', nthreads=4)
        result3 = TwoPointCorrelationFunction(mode='s', edges=edges, data_positions1=data_positions, randoms_positions1=randoms_positions, los='midpoint',
                                              data_weights1=data_weights_iip, weight_attrs={'nrealizations': 1 + nbits, 'normalization': None},
                                              isa=isa, position_type='xyz', nthreads=4)
        if ref is not None:
            assert np.allclose(result.D1D2.wcounts, ref)
        else:
            ref = result.D1D2.wcounts
        xi, DD, RR, DR = [XX[:result.shape[0]] for XX in np.loadtxt(ref_fn, unpack=True, usecols=range(1, 5))]
        tol = {'rtol': 1e-9, 'atol': 0.}
        print(result.D1D2.wcounts / 2. / DD)
        print(result.D1D2.normalized_wcounts() / result2.D1D2.normalized_wcounts())
        print(result.D1D2.normalized_wcounts() / result3.D1D2.normalized_wcounts())
        #print(result.D1R2.wcounts / DR)
        assert np.allclose(result.D1D2.wcounts / 2., DD, **tol)
        assert np.allclose(result.D1R2.wcounts, DR, **tol)
        assert np.allclose(result.R1R2.wcounts / 2., RR, **tol)
        assert np.allclose(result.corr, xi, rtol=1e-9, atol=1e-7)


def test_analytic_twopoint_counter(mode='s'):
    edges = np.linspace(50, 100, 5)
    size = 40000
    boxsize = (1000,) * 3
    if mode == 'smu':
        edges = (edges, np.linspace(-1, 1, 5))
    elif mode == 'rppi':
        edges = (edges, np.linspace(-10, 10, 11))

    list_options = []
    list_options.append({})
    list_options.append({'autocorr': True})
    if mode in ['smu']:
        list_options.append({'autocorr': True, 'selection_attrs': {'rp': (70., 90.)}})

    for options in list_options:
        autocorr = options.pop('autocorr', False)
        data1, data2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=0, n_bitwise_weights=0)
        ref = TwoPointCounter(mode=mode, edges=edges, positions1=data1[:3], positions2=None if autocorr else data2[:3],
                              weights1=None, weights2=None, position_type='xyz', boxsize=boxsize, los='z', **options)
        test = AnalyticTwoPointCounter(mode, edges, boxsize, size1=len(data1[0]), size2=None if autocorr else len(data2[0]), selection_attrs=options.get('selection_attrs', {}))
        diff = np.abs(test.wcounts - ref.wcounts)
        assert np.all(diff <= 3. * np.sqrt(ref.wcounts))
        if mode == 'smu' and (edges[1][0], edges[1][-1]) == (-1, 1):
            test2 = AnalyticTwoPointCounter('s', edges[:1], boxsize, size1=len(data1[0]), size2=None if autocorr else len(data2[0]), selection_attrs=options.get('selection_attrs', {}))
            assert np.allclose(test2.wcounts, test.wcounts.sum(axis=-1))

        with tempfile.TemporaryDirectory() as tmp_dir:
            fn = os.path.join(tmp_dir, 'tmp.npy')
            bak = np.copy(test.wcounts)
            test.save(fn)
            test = TwoPointCounter.load(fn)
            assert np.allclose(test.wcounts, bak)
            ref = test.copy()
            test.rebin((2, 2) if len(edges) == 2 else (2,))
            assert np.allclose(np.sum(test.wcounts), np.sum(ref.wcounts))
            test2 = ref[::2, ::2] if len(edges) == 2 else ref[::2]
            assert test2.shape == test.shape
            assert np.allclose(test2.wcounts, test.wcounts, equal_nan=True)
            sepmax = 80.
            if mode == 'smu': sepmax = 0.8
            test2.select((0, sepmax))
            assert np.all((test2.sepavg(axis=0) <= sepmax) | np.isnan(test2.sepavg(axis=0)))

            def check_seps(test):
                for iaxis in range(test.ndim):
                    if not test.compute_sepsavg[iaxis]:
                        mid = (test.edges[iaxis][:-1] + test.edges[iaxis][1:]) / 2.
                        if test.ndim == 2 and iaxis == 0: mid = mid[:, None]
                        assert np.allclose(test.seps[iaxis], mid)

            check_seps(test2)


def test_mu1():

    mode = 'smu'
    engine = 'corrfunc'
    size = 100
    w = 10000.

    for options in [{'isa': 'fallback'}, {'isa': 'sse42'}, {'isa': 'avx'}]:

        for iaxis, los in enumerate(['x', 'y', 'z', 'midpoint']):
            if iaxis < 3:
                positions1 = np.zeros((size, 3), dtype='f8')
                positions2 = positions1.copy()
                positions2[:, iaxis] += 1.
            else:
                positions1 = np.ones((size, 3), dtype='f8')
                positions2 = positions1 * 2.

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

            mumax = 1. + 1e-9
            edges = (np.linspace(0., 2., 3), np.linspace(-mumax, mumax, 11))
            counts = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=positions1, positions2=positions2,
                                     position_type='pos', los=los, **options)

            assert np.allclose(counts.wcounts[1, -1], w)
            counts.wcounts[1, -1] = 0
            assert np.allclose(counts.wcounts, 0.)

            mumax = 1. + 1e-9
            edges = (np.linspace(0., 2., 3), np.linspace(-mumax, mumax, 11))
            counts = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=positions2, positions2=positions1,
                                     position_type='pos', los=los, **options)

            assert np.allclose(counts.wcounts[1, 0], w)
            counts.wcounts[1, 0] = 0
            assert np.allclose(counts.wcounts, 0.)

            mumax = 1.
            edges = (np.linspace(0., 2., 3), np.linspace(-mumax, mumax, 11))
            counts = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=np.concatenate([positions1, positions2], axis=0),
                                     position_type='pos', los=los, **options)

            assert np.allclose(counts.wcounts[0, counts.wcounts.shape[1] // 2], 2. * w - (len(positions1) + len(positions2)))
            counts.wcounts[0, counts.wcounts.shape[1] // 2] = 0.
            assert np.allclose(counts.wcounts, 0.)

            mumax = 1. + 1e-9
            edges = (np.linspace(0., 2., 3), np.linspace(-mumax, mumax, 10))
            counts = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=np.concatenate([positions1, positions2], axis=0),
                                     position_type='pos', los=los, **options)

            assert np.allclose(counts.wcounts[0, counts.wcounts.shape[1] // 2], 2. * w - (len(positions1) + len(positions2)))
            assert np.allclose(counts.wcounts[1, -1], w)
            assert np.allclose(counts.wcounts[1, 0], w)
            counts.wcounts[1, -1] = 0
            counts.wcounts[1, 0] = 0
            counts.wcounts[0, counts.wcounts.shape[1] // 2] = 0.
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
        d = cosmo.comoving_radial_distance(cz / speed_of_light)
        data1 = [ra, dec, d]
        position_type = 'rdd'
    else:
        data1 = generate_catalogs(size=int(1e6), boxsize=(150,) * 3, offset=(200., 0, 0), n_individual_weights=1, n_bitwise_weights=0, seed=42)[0]
        position_type = 'xyz'
    # edges = (np.linspace(2.0, 20.1, 12), np.linspace(-1., 1., 11))
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
    edges = (np.linspace(0, 10, 11), np.linspace(-1, 1, 6))
    test = AnalyticTwoPointCounter(mode, edges, boxsize)
    ref = test.copy()
    test = test[::2, ::5]
    assert test.sep.shape == test.wcounts.shape == (5, 1)
    refedges = [np.linspace(0., 10, 6), np.linspace(-1, 1, 2)]
    for i in range(2):
        assert np.allclose(test.edges[i], refedges[i])
    assert np.allclose(np.sum(test.wcounts), np.sum(ref.wcounts))


if __name__ == '__main__':

    setup_logging()

    #test_gpu()
    test_mu1()

    for mode in ['theta', 's', 'smu', 'rppi', 'rp']:
        test_twopoint_counter(mode=mode)

    for mode in ['s', 'smu', 'rppi']:
        test_analytic_twopoint_counter(mode=mode)

    test_rebin()
    test_pip_normalization()
    test_pip_counts()
    test_pip_counts_correction()
