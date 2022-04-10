import os

import numpy as np

from pycorr import TwoPointCorrelationFunction, utils, setup_logging
from pycorr.twopoint_estimator import BaseTwoPointEstimator


def save_catalogs(filename, size=1000, boxsize=(100,) * 3, offset=(1000, 0, 0), n_bitwise_weights=2, seed=42, parent_fn=None):
    rng = np.random.RandomState(seed=seed)
    parent_size = int(size * 1.2)
    for icat in range(2):
        positions = [o + rng.uniform(-0.5, 0.5, parent_size) * b for b, o in zip(boxsize, offset)]
        if n_bitwise_weights:
            bitwise_weights = [utils.pack_bitarrays(*[rng.randint(0, 2, parent_size) for i in range(31)], np.ones(parent_size, dtype='?'), dtype=np.uint32)[0]]
            bitwise_weights += [utils.pack_bitarrays(*[rng.randint(0, 2, parent_size) for i in range(32)], dtype=np.uint32)[0] for iw in range(n_bitwise_weights - 1)]
        else:
            bitwise_weights = []
        individual_weights = rng.uniform(0.5, 1., parent_size)
        mask = rng.choice(parent_size, size, replace=False)
        if parent_fn is not None:
            utils.mkdir(os.path.dirname(parent_fn))
            template = ' '.join(['{:.7g}'] * 3) + '\n'
            with open(parent_fn.format(icat + 1), 'w') as file:
                header = '#x y z\n'
                file.write(header)
                for ii in range(parent_size):
                    line = template.format(*[p[ii] for p in positions])
                    file.write(line)
        positions = [p[mask] for p in positions]
        bitwise_weights = [w[mask] for w in bitwise_weights]
        individual_weights = individual_weights[mask]
        template = '{:.7g} ' * 3 + '{:d} ' * n_bitwise_weights + '{:.7g}\n'
        utils.mkdir(os.path.dirname(filename))
        with open(filename.format(icat + 1), 'w') as file:
            header = '#x y z '
            for iw in range(n_bitwise_weights): header += 'bitwise_weight{:d}(uint32,{:d}realizations) '.format(iw + 1, 32 * n_bitwise_weights)
            header += 'individual_weight\n'
            file.write(header)
            for ii in range(size):
                line = template.format(*[p[ii] for p in positions], *[w[ii] for w in bitwise_weights], individual_weights[ii])
                file.write(line)


def save_angular_upweights(filename, size=41):
    sep = np.logspace(-4, 0, size)
    wp = np.linspace(4., 1., size)
    template = '{:.7g} {:.7g}\n'
    utils.mkdir(os.path.dirname(filename))
    with open(filename, 'w') as file:
        header = '#theta[deg] weight\n'
        file.write(header)
        for ii in range(size):
            line = template.format(sep[ii], wp[ii])
            file.write(line)


def save_readme(filename, catalog_dirname='catalogs', results_dirname='twopoint', angular_upweights_dirname='angular_upweights', nrealizations=62):
    tmp = 'Reference test to compare pair counters.\n'\
          'Data and randoms catalogs are in {0}/.\n'\
          'These catalogs are purely random; not clustering is expected -- hence flat correlation function.\n'\
          'Bitwise weights are only provided for data, not randoms; and these weights are turned into IIP weights for D1R2, R1D2.\n'\
          'The last to MSB of each bitwise weight has been set to 1, to ensure non zero-probability pairs.\n'\
          'The total number of realizations is {3:d}, such that each PIP weight is given by {3:d}/popcount(w1 & w2), with w1 and w2 bitwise weights of particles 1 and 2.\n'\
          'Pair counts and correlation function estimations (Landy-Szalay) are saved in {1}_theta/, {1}_s/, {1}_smu/, {1}_rppi/,\n'\
          'without weights (*_no_weights), with individual weights (*_individual_weights), with PIP weights (*_bitwise_weights),\n'\
          'with both (*_individual_bitwise_weights), and with angular weights provided in {0}/custom_angular_upweights.txt (*_individual_bitwise_angular_upweights).\n'\
          'Angular weights are linearly interpolated (in terms of costheta) in the theta range, set to 1 outside. Those are only applied to D1D2, D1R2, D2R1.\n'\
          'Separations along the first dimension (e.g. s, rp) are computed as the (unweighted) mean separation in each bin.\n'\
          'Separations along the second dimension (e.g. mu, pi) are computed as the bin centers.\n'\
          'Pair count normalizations are provided on top of the pair count files, #norm = .... .\n'\
          'For bitwise weights use the zero-truncated estimator, in which case the current realization is included in the bitwise weights (and in the number of realizations = {3:d}).\n'\
          'Angular weights, this time calculated from parent, data and randoms catalogs in catalogs/ are given in {2}/.\n'.format(catalog_dirname, results_dirname, angular_upweights_dirname, nrealizations)
    utils.mkdir(os.path.dirname(filename))
    with open(filename, 'w') as file:
        file.write(tmp)


def save_result(result, filename, header=''):
    sep = [sep.flatten() for sep in result.seps]
    edges = [result.edges[0][:-1], result.edges[0][1:]]
    template = '{:.7g} ' * 3 + '{:.12g}\n'
    if result.ndim > 1:
        edges_low = np.meshgrid(result.edges[0][:-1], result.edges[1][:-1], indexing='ij')
        edges_high = np.meshgrid(result.edges[0][1:], result.edges[1][1:], indexing='ij')
        edges = [edges_low[0].flatten(), edges_high[0].flatten(), edges_low[1].flatten(), edges_high[1].flatten()]
        template = '{:.7g} ' * 6 + '{:.12g}\n'
    header = '#{}\n'.format(header)
    header += '#'
    if result.mode == 'theta':
        header += 'theta_low[deg] theta_high[deg] theta[deg]'
    elif result.mode == 's':
        header += 's_low s_high s'
    elif result.mode == 'smu':
        header += 's_low s_high mu_low mu_high s mu'
    elif result.mode == 'rppi':
        header += 'rp_low rp_high pi_low pi_high rp pi'
    isestimator = isinstance(result, BaseTwoPointEstimator)
    if isestimator:
        header += ' corr\n'
        counts = result.corr.flatten()
    else:
        header = '#norm = {}\n'.format(result.wnorm) + header + ' wcounts\n'
        counts = result.wcounts.flatten()
    utils.mkdir(os.path.dirname(filename))
    with open(filename, 'w') as file:
        file.write(header)
        for ii in range(len(sep[0])):
            line = template.format(*[e[ii] for e in edges], *[s[ii] for s in sep], counts[ii])
            file.write(line)


def assert_allclose(result, filename):
    isestimator = isinstance(result, BaseTwoPointEstimator)
    ref = np.loadtxt(filename, usecols=-1)
    if isestimator:
        mask = ~np.isnan(ref)
        assert np.allclose(result.corr.flatten()[mask], ref[mask], atol=1e-9, rtol=1e-6)
    else:
        with open(filename, 'r') as file:
            import re
            for line in file:
                wnorm = float(re.match('#norm = (.*)', line).group(1))
                break
            assert np.allclose(result.wnorm, wnorm, atol=0, rtol=1e-10)
        assert np.allclose(result.wcounts.flatten(), ref, atol=0, rtol=1e-8)


def load_catalog(catalog_fn):
    data = None
    fmt = None
    with open(catalog_fn, 'r') as file:
        for line in file:
            if line.startswith('#'): continue
            line = line.split()
            ncols = len(line)
            if data is None:
                data = [[] for icol in range(ncols)]
                if ncols == 4: fmt = [float] * 4
                else: fmt = [float] * 3 + [int] * (ncols - 4) + [float]
            for icol in range(ncols): data[icol].append(fmt[icol](line[icol]))
    for icol in range(ncols): data[icol] = np.array(data[icol])
    return data


def save_reference(base_dir):
    catalog_dirname = 'catalogs'
    results_dirname = 'twopoint'
    angular_upweights_dirname = 'angular_upweights'
    catalog_dir = os.path.join(base_dir, catalog_dirname)
    results_dir = os.path.join(base_dir, results_dirname)
    angular_upweights_dir = os.path.join(base_dir, angular_upweights_dirname)
    estimator_fn = os.path.join('{}_{{}}'.format(results_dir), 'correlation_function_{}.txt')
    counts_fn = os.path.join('{}_{{}}'.format(results_dir), 'counts_{}.txt')
    estimated_angular_weight_fn = os.path.join(angular_upweights_dir, 'weights_{}.txt')
    data_fn = os.path.join(catalog_dir, 'data_{:d}.txt')
    parent_fn = os.path.join(catalog_dir, 'parent_data_{:d}.txt')
    randoms_fn = os.path.join(catalog_dir, 'randoms_{:d}.txt')
    angular_upweights_fn = os.path.join(base_dir, 'custom_angular_upweights.txt')
    readme_fn = os.path.join(base_dir, 'README')

    n_bitwise_weights = 2
    nrealizations = 32 * n_bitwise_weights
    save_catalogs(data_fn, size=1000, n_bitwise_weights=n_bitwise_weights, seed=42, parent_fn=parent_fn)
    save_catalogs(randoms_fn, size=4000, n_bitwise_weights=0, seed=84)
    save_angular_upweights(angular_upweights_fn, size=41)
    save_readme(readme_fn, catalog_dirname=catalog_dirname, results_dirname=results_dirname, angular_upweights_dirname=angular_upweights_dirname, nrealizations=nrealizations)

    data1 = load_catalog(data_fn.format(1))
    data2 = load_catalog(data_fn.format(2))
    parent1 = load_catalog(parent_fn.format(1))
    parent2 = load_catalog(parent_fn.format(2))
    randoms1 = load_catalog(randoms_fn.format(1))
    randoms2 = load_catalog(randoms_fn.format(2))
    angular_upweights = np.loadtxt(angular_upweights_fn, unpack=True)

    mode_edges = {}
    mode_edges['theta'] = 'np.logspace(-2, 0, 31)'
    mode_edges['s'] = 'np.logspace(-0.5, 1, 31)'
    mode_edges['smu'] = '(np.linspace(1, 41, 41), np.linspace(-1, 1, 21))'
    mode_edges['rppi'] = '(np.linspace(1, 41, 41), np.linspace(0, 20, 21))'
    weight_attrs = {'nrealizations': nrealizations, 'noffset': 0, 'default_value': 0., 'nalways': 1}

    for mode, name_edges in mode_edges.items():
        edges = eval(name_edges, {'np': np})
        header = 'edges = {}'.format(name_edges)
        for weight in ['no_weights', 'individual_weights', 'bitwise_weights', 'individual_bitwise_weights', 'individual_bitwise_angular_upweights']:
            data_weights1, data_weights2, randoms_weights1, randoms_weights2 = None, None, None, None
            if weight == 'individual_weights':
                data_weights1, data_weights2, randoms_weights1, randoms_weights2 = data1[-1:], data2[-1:], randoms1[3:], randoms2[3:]
            if weight == 'bitwise_weights':
                data_weights1, data_weights2, randoms_weights1, randoms_weights2 = data1[3:-1], data2[3:-1], np.ones_like(randoms1[0]), np.ones_like(randoms2[0])
            if 'individual_bitwise' in weight:
                data_weights1, data_weights2, randoms_weights1, randoms_weights2 = data1[3:], data2[3:], randoms1[3:], randoms2[3:]
            kwargs = {'weight_attrs': weight_attrs, 'position_type': 'xyz', 'compute_sepsavg': True}
            if 'angular' in weight:
                kwargs['D1D2_twopoint_weights'] = kwargs['D1R2_twopoint_weights'] = kwargs['R1D2_twopoint_weights'] = angular_upweights
            result = TwoPointCorrelationFunction(mode, edges, data_positions1=data1[:3], data_weights1=data_weights1,
                                                 data_positions2=data2[:3], data_weights2=data_weights2,
                                                 randoms_positions1=randoms1[:3], randoms_weights1=randoms_weights1,
                                                 randoms_positions2=randoms2[:3], randoms_weights2=randoms_weights2, **kwargs)
            save_result(result, estimator_fn.format(mode, 'cross_{}'.format(weight)), header=header)
            for pc in result.requires(with_reversed=True, join=''):
                if pc == 'R1R2' and 'bitwise' in weight: continue
                save_result(getattr(result, pc), counts_fn.format(mode, '{}_{}'.format(pc, weight)), header=header)

            result = TwoPointCorrelationFunction(mode, edges, data_positions1=data1[:3], data_weights1=data_weights1,
                                                 randoms_positions1=randoms1[:3], randoms_weights1=randoms_weights1, **kwargs)
            save_result(result, estimator_fn.format(mode, 'auto1_{}'.format(weight)), header=header)
            for pc in result.requires(with_reversed=False, join=''):
                if pc == 'R1R2' and 'bitwise' in weight: continue
                save_result(getattr(result, pc), counts_fn.format(mode, '{}_{}'.format(pc.replace('2', '1'), weight)), header=header)

    mode, name_edges = 'theta', 'np.logspace(-2.5, 0, 31)'
    edges = eval(name_edges, {'np': np})
    header = 'edges = {}'.format(name_edges)
    data_weights1, data_weights2 = data1[3:-1], data2[3:-1]
    randoms_weights1, randoms_weights2 = np.ones_like(randoms1[0]), np.ones_like(randoms2[0])
    parent_weights1, parent_weights2 = np.ones_like(parent1[0]), np.ones_like(parent2[0])
    kwargs = {'weight_attrs': weight_attrs, 'position_type': 'xyz', 'compute_sepsavg': True}
    # D1_parentD2_parent/D1D2_pip
    result = TwoPointCorrelationFunction(mode, edges, data_positions1=data1[:3], data_weights1=data_weights1,
                                         data_positions2=data2[:3], data_weights2=data_weights2,
                                         randoms_positions1=parent1[:3], randoms_weights1=parent_weights1,
                                         randoms_positions2=parent2[:3], randoms_weights2=parent_weights2,
                                         estimator='weight', **kwargs)
    save_result(result, estimated_angular_weight_fn.format('for_D1D2_bitwise_weights'), header=header)

    # D1_parentR2/D1_iipR2
    result = TwoPointCorrelationFunction(mode, edges, data_positions1=data1[:3], data_weights1=data_weights1,
                                         data_positions2=randoms2[:3], data_weights2=randoms_weights2,
                                         randoms_positions1=parent1[:3], randoms_weights1=parent_weights1,
                                         randoms_positions2=randoms2[:3], randoms_weights2=randoms_weights2,
                                         estimator='weight', **kwargs)
    save_result(result, estimated_angular_weight_fn.format('for_D1R2_bitwise_weights'), header=header)


def test_reference(base_dir):
    catalog_dirname = 'catalogs'
    results_dirname = 'twopoint'
    angular_upweights_dirname = 'angular_upweights'
    catalog_dir = os.path.join(base_dir, catalog_dirname)
    results_dir = os.path.join(base_dir, results_dirname)
    angular_upweights_dir = os.path.join(base_dir, angular_upweights_dirname)
    estimator_fn = os.path.join('{}_{{}}'.format(results_dir), 'correlation_function_{}.txt')
    counts_fn = os.path.join('{}_{{}}'.format(results_dir), 'counts_{}.txt')
    estimated_angular_weight_fn = os.path.join(angular_upweights_dir, 'weights_{}.txt')
    data_fn = os.path.join(catalog_dir, 'data_{:d}.txt')
    parent_fn = os.path.join(catalog_dir, 'parent_data_{:d}.txt')
    randoms_fn = os.path.join(catalog_dir, 'randoms_{:d}.txt')
    angular_upweights_fn = os.path.join(base_dir, 'custom_angular_upweights.txt')

    data1 = load_catalog(data_fn.format(1))
    data2 = load_catalog(data_fn.format(2))
    parent1 = load_catalog(parent_fn.format(1))
    parent2 = load_catalog(parent_fn.format(2))
    randoms1 = load_catalog(randoms_fn.format(1))
    randoms2 = load_catalog(randoms_fn.format(2))
    angular_upweights = np.loadtxt(angular_upweights_fn, unpack=True)

    mode_edges = {}
    mode_edges['theta'] = 'np.logspace(-2, 0, 31)'
    mode_edges['s'] = 'np.logspace(-0.5, 1, 31)'
    mode_edges['smu'] = '(np.linspace(1, 41, 41), np.linspace(-1, 1, 21))'
    mode_edges['rppi'] = '(np.linspace(1, 41, 41), np.linspace(0, 20, 21))'
    weight_attrs = {'nrealizations': 64, 'noffset': 0, 'default_value': 0., 'nalways': 1, 'isa': 'fallback'}
    nthreads = int(os.getenv('OMP_NUM_THREADS', '4'))

    for mode, name_edges in mode_edges.items():
        edges = eval(name_edges, {'np': np})
        for weight in ['no_weights', 'individual_weights', 'bitwise_weights', 'individual_bitwise_weights', 'individual_bitwise_angular_upweights']:
            data_weights1, data_weights2, randoms_weights1, randoms_weights2 = None, None, None, None
            if weight == 'individual_weights':
                data_weights1, data_weights2, randoms_weights1, randoms_weights2 = data1[-1:], data2[-1:], randoms1[3:], randoms2[3:]
            if weight == 'bitwise_weights':
                data_weights1, data_weights2, randoms_weights1, randoms_weights2 = data1[3:-1], data2[3:-1], np.ones_like(randoms1[0]), np.ones_like(randoms2[0])
            if 'individual_bitwise' in weight:
                data_weights1, data_weights2, randoms_weights1, randoms_weights2 = data1[3:], data2[3:], randoms1[3:], randoms2[3:]
            kwargs = {'weight_attrs': weight_attrs, 'position_type': 'xyz', 'compute_sepsavg': True, 'nthreads': nthreads}
            if 'angular' in weight:
                kwargs['D1D2_twopoint_weights'] = kwargs['D1R2_twopoint_weights'] = kwargs['R1D2_twopoint_weights'] = angular_upweights
            result = TwoPointCorrelationFunction(mode, edges, data_positions1=data1[:3], data_weights1=data_weights1,
                                                 data_positions2=data2[:3], data_weights2=data_weights2,
                                                 randoms_positions1=randoms1[:3], randoms_weights1=randoms_weights1,
                                                 randoms_positions2=randoms2[:3], randoms_weights2=randoms_weights2, **kwargs)

            for pc in result.requires(with_reversed=True, join=''):
                if pc == 'R1R2' and 'bitwise' in weight: continue
                assert_allclose(getattr(result, pc), counts_fn.format(mode, '{}_{}'.format(pc, weight)))
            assert_allclose(result, estimator_fn.format(mode, 'cross_{}'.format(weight)))

            result = TwoPointCorrelationFunction(mode, edges, data_positions1=data1[:3], data_weights1=data_weights1,
                                                 randoms_positions1=randoms1[:3], randoms_weights1=randoms_weights1, **kwargs)

            for pc in result.requires(with_reversed=False, join=''):
                if pc == 'R1R2' and 'bitwise' in weight: continue
                assert_allclose(getattr(result, pc), counts_fn.format(mode, '{}_{}'.format(pc.replace('2', '1'), weight)))
            assert_allclose(result, estimator_fn.format(mode, 'auto1_{}'.format(weight)))

    mode, name_edges = 'theta', 'np.logspace(-2.5, 0, 31)'
    edges = eval(name_edges, {'np': np})
    data_weights1, data_weights2 = data1[3:-1], data2[3:-1]
    randoms_weights1, randoms_weights2 = np.ones_like(randoms1[0]), np.ones_like(randoms2[0])
    parent_weights1, parent_weights2 = np.ones_like(parent1[0]), np.ones_like(parent2[0])
    kwargs = {'weight_attrs': weight_attrs, 'position_type': 'xyz', 'compute_sepsavg': True, 'nthreads': nthreads}
    # D1_parentD2_parent/D1D2_pip
    result = TwoPointCorrelationFunction(mode, edges, data_positions1=data1[:3], data_weights1=data_weights1,
                                         data_positions2=data2[:3], data_weights2=data_weights2,
                                         randoms_positions1=parent1[:3], randoms_weights1=parent_weights1,
                                         randoms_positions2=parent2[:3], randoms_weights2=parent_weights2,
                                         estimator='weight', **kwargs)
    assert_allclose(result, estimated_angular_weight_fn.format('for_D1D2_bitwise_weights'))

    # D1_parentR2/D1_iipR2
    result = TwoPointCorrelationFunction(mode, edges, data_positions1=data1[:3], data_weights1=data_weights1,
                                         data_positions2=randoms2[:3], data_weights2=randoms_weights2,
                                         randoms_positions1=parent1[:3], randoms_weights1=parent_weights1,
                                         randoms_positions2=randoms2[:3], randoms_weights2=randoms_weights2,
                                         estimator='weight', **kwargs)
    assert_allclose(result, estimated_angular_weight_fn.format('for_D1R2_bitwise_weights'))


if __name__ == '__main__':

    base_dir = os.path.join(os.path.dirname(__file__), 'reference')

    setup_logging()
    save_reference(base_dir)
    test_reference(base_dir)
