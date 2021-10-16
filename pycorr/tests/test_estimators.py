import os
import tempfile
import numpy as np

from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, project_to_multipoles,\
                    project_to_wp, setup_logging


def generate_catalogs(size=100, boxsize=(1000,)*3, n_individual_weights=1, n_bitwise_weights=0, seed=42):
    rng = np.random.RandomState(seed=seed)
    toret = []
    for i in range(2):
        positions = [rng.uniform(0., 1., size)*b for b in boxsize]
        weights = [rng.randint(0, 0xffffffff, size, dtype='i8') for i in range(n_bitwise_weights)]
        weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
        toret.append(positions+weights)
    return toret


def test_estimator(mode='s'):
    list_engine = ['corrfunc']
    edges = np.linspace(1,100,10)
    size = 100
    boxsize = (1000,)*3
    list_options = []
    if mode not in ['theta', 'rp']:
        list_options.append({'estimator':'natural','boxsize':boxsize})
    list_options.append({'estimator':'weight'})
    list_options.append({'autocorr':True})
    list_options.append({'n_individual_weights':1})
    if mode not in ['theta', 'rp']:
        list_options.append({'estimator':'natural','boxsize':boxsize})
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
        edges = (edges, np.linspace(0,1,101))
    elif mode == 'rppi':
        edges = (edges, np.linspace(0,100,101))
    elif mode == 'theta':
        edges = np.linspace(1e-5,10,11) # below 1e-5, self pairs are counted by Corrfunc
    for engine in list_engine:
        for options in list_options:
            options = options.copy()
            data1, randoms1 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=options.get('n_individual_weights',1), n_bitwise_weights=options.get('n_bitwise_weights',0))
            data2, randoms2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=options.get('n_individual_weights',1), n_bitwise_weights=options.get('n_bitwise_weights',0))
            autocorr = options.pop('autocorr', False)
            mpicomm = options.pop('mpicomm', None)
            options.setdefault('boxsize', None)
            options['los'] = 'z' if options['boxsize'] is not None else 'midpoint'

            def run(**kwargs):
                return TwoPointCorrelationFunction(mode=mode, edges=edges, engine=engine, data_positions1=data1[:3], data_positions2=None if autocorr else data2[:3],
                                                   data_weights1=data1[3:], data_weights2=None if autocorr else data2[3:],
                                                   randoms_positions1=randoms1[:3], randoms_positions2=None if autocorr else randoms2[:3],
                                                   randoms_weights1=randoms1[3:], randoms_weights2=None if autocorr else randoms2[3:],
                                                   position_type='xyz', **options, **kwargs)

            test = run()

            if test.D1D2.mode == 'smu':
                sep, xiell = project_to_multipoles(test, ells=(0,2,4))
            if test.D1D2.mode == 'rppi':
                sep, wp = project_to_wp(test)
                sep, wp = project_to_wp(test, pimax=40)
            with tempfile.TemporaryDirectory() as tmp_dir:
                fn = os.path.join(tmp_dir,'tmp.npy')
                test.save(fn)
                test2 = TwoPointEstimator.load(fn)
                assert test2.__class__ is test.__class__
                assert test2.autocorr is test.autocorr
                test2.rebin((2,5) if len(edges) == 2 else (2,))
                test2 = run(R1R2=test.R1R2)
                mask = np.isfinite(test2.corr) & np.isfinite(test.corr)
                assert np.allclose(test2.corr[mask], test.corr[mask])

            if mpicomm is not None:
                data1 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in data1]
                data2 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in data2]
                randoms1 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in randoms1]
                randoms2 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in randoms2]
                test_mpi = run(mpicomm=mpicomm)
                mask = np.isfinite(test.corr)
                assert np.allclose(test_mpi.corr[mask], test.corr[mask])
                assert np.allclose(test_mpi.sep[mask], test.sep[mask])


if __name__ == '__main__':

    setup_logging()
    for mode in ['theta','s','smu','rppi','rp']:
        test_estimator(mode=mode)
