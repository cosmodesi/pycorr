import numpy as np

from pycorr import TwoPointCorrelationFunction


def generate_catalogs(size=100, boxsize=(1000,)*3, n_individual_weights=1, n_bitwise_weights=0, seed=42):
    rng = np.random.RandomState(seed=seed)
    toret = []
    for i in range(2):
        positions = [rng.uniform(0., 1., size)*b for b in boxsize]
        weights = [rng.randint(0, 0xffffffff, size, dtype='i8') for i in range(n_bitwise_weights)]
        weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
        toret.append(positions+weights)
    return toret


def test(mode='s'):
    list_engine = ['corrfunc']
    list_options = []
    list_options.append({'autocorr':True,'weight_type':None})
    list_options.append({'weight_type':'pair_product'})
    #list_options.append({'weight_type':'inverse_bitwise','n_bitwise_weights':2})
    edges = np.linspace(1e-9,100,10)
    boxsize = (1000,)*3
    if mode == 'smu':
        edges = (edges, np.linspace(0,1,100))
    elif mode == 'rppi':
        edges = (edges, np.linspace(0,100,101))
    elif mode == 'theta':
        edges = np.linspace(1e-5,10,10) # below 1e-5, self pairs are counted by Corrfunc
    for engine in list_engine:
        for options in list_options:
            options = options.copy()
            data1, randoms1 = generate_catalogs(boxsize=boxsize, n_individual_weights=options.get('n_individual_weights',1), n_bitwise_weights=options.get('n_bitwise_weights',0))
            data2, randoms2 = generate_catalogs(boxsize=boxsize, n_individual_weights=options.get('n_individual_weights',1), n_bitwise_weights=options.get('n_bitwise_weights',0))
            autocorr = options.pop('autocorr',False)
            options['boxsize'] = boxsize if options.pop('periodic',False) else None
            options['los'] = 'z' if options['boxsize'] is not None else 'midpoint'
            test = TwoPointCorrelationFunction(mode=mode, edges=edges, engine=engine, data_positions1=data1[:3], data_positions2=None if autocorr else data2[:3],
                                               data_weights1=data1[3], data_weights2=None if autocorr else data2[3],
                                               randoms_positions1=randoms1[:3], randoms_positions2=None if autocorr else randoms2[:3],
                                               randoms_weights1=randoms1[3], randoms_weights2=None if autocorr else randoms2[3],
                                               position_type='xyz', **options)


if __name__ == '__main__':

    for mode in ['theta','s','smu','rppi','rp']:
        test(mode=mode)
