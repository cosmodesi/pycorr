import numpy as np

from pycorr import TwoPointCounter


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


def get_weight(weights1, weights2, weight_type=1, n_bitwise_weights=0, costheta_weight=None, costheta_min=0, costheta_max=None):
    weight = 1.
    if weight_type == 1:
        weight = 1. / (1. + sum(bin(w1 & w2).count('1') for w1, w2 in zip(weights1[:n_bitwise_weights], weights2[:n_bitwise_weights])))
        for w1, w2 in zip(weights1[n_bitwise_weights:], weights2[n_bitwise_weights:]):
            weight *= w1 * w2
        if costheta_weight is not None:
            costheta = dotproduct_normalized(xyz1,xyz2)
            findex = (costheta - costheta_min)/(costheta_max - costheta_min) * (len(costheta_weight) - 1)
            index = int(findex)
            if 0 <= index < len(costheta_weight) - 1:
                frac = findex - index
                weight *= (1-frac)*costheta_weight[index] + frac*costheta_weight[index+1]
    elif weight_type == 0:
        weight = weights1[0] * weights2[0]
    return weight


def ref_theta(edges, data1, data2=None, boxsize=None, weight_type=None, los='midpoint', **kwargs):
    weight_type = {None:-1, 'pair_product':0, 'inverse_bitwise':1}[weight_type]
    toret = np.zeros(len(edges)-1, dtype='f8')
    if data2 is None: data2 = data1
    for xyzw1 in zip(*data1):
        for xyzw2 in zip(*data2):
            xyz1, xyz2 = xyzw2[:3], xyzw1[:3]
            dist = np.rad2deg(np.arccos(min(dotproduct_normalized(xyz1, xyz2),1))) # min to avoid rounding errors
            if edges[0] <= dist < edges[-1]:
                ind = np.searchsorted(edges, dist, side='right', sorter=None) - 1
                weights1, weights2 = xyzw1[3:], xyzw2[3:]
                weight = get_weight(weights1, weights2, weight_type=weight_type, **kwargs)
                toret[ind] += weight
    return toret


def ref_s(edges, data1, data2=None, boxsize=None, weight_type=None, los='midpoint', **kwargs):
    weight_type = {None:-1, 'pair_product':0, 'inverse_bitwise':1}[weight_type]
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
                weight = get_weight(weights1, weights2, weight_type=weight_type, **kwargs)
                toret[ind] += weight
    return toret


def ref_smu(edges, data1, data2=None, boxsize=None, weight_type=None, los='midpoint', **kwargs):
    weight_type = {None:-1, 'pair_product':0, 'inverse_bitwise':1}[weight_type]
    if los == 'midpoint':
        los = 'm'
    else:
        los = [0 if i == 'xyz'.index(los) else 1 for i in range(3)]
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
                    weight = get_weight(weights1, weights2, weight_type=weight_type, **kwargs)
                    toret[ind,ind_mu] += weight
    return toret


def ref_rppi(edges, data1, data2=None, boxsize=None, weight_type=None, los='midpoint', **kwargs):
    weight_type = {None:-1, 'pair_product':0, 'inverse_bitwise':1}[weight_type]
    if los == 'midpoint':
        los = 'm'
    else:
        los = [0 if i == 'xyz'.index(los) else 1 for i in range(3)]
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
            vlos /= norm(vlos)
            pi = abs(dotproduct(vlos, dxyz))
            rp = (dotproduct(dxyz, dxyz) - pi**2)**0.5
            if edges[0][0] <= rp < edges[0][-1] and edges[1][0] <= pi < edges[1][-1]:
                ind_rp = np.searchsorted(edges[0], rp, side='right', sorter=None) - 1
                ind_pi = np.searchsorted(edges[1], pi, side='right', sorter=None) - 1
                weights1, weights2 = xyzw1[3:], xyzw2[3:]
                weight = get_weight(weights1, weights2, weight_type=weight_type, **kwargs)
                toret[ind_rp,ind_pi] += weight
    return toret


def ref_rp(edges, *args, **kwargs):
    return ref_rppi((edges,[0,np.inf]), *args, **kwargs).flatten()


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
    ref_func = {'theta':ref_theta, 's':ref_s, 'smu':ref_smu, 'rppi':ref_rppi, 'rp':ref_rp}[mode]
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
            data1, data2 = generate_catalogs(boxsize=boxsize, n_individual_weights=options.get('n_individual_weights',1), n_bitwise_weights=options.get('n_bitwise_weights',0))
            autocorr = options.pop('autocorr',False)
            options['boxsize'] = boxsize if options.pop('periodic',False) else None
            options['los'] = 'z' if options['boxsize'] is not None else 'midpoint'
            ref = ref_func(edges, data1, data2=None if autocorr else data2, **options)
            test = TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=data1[:3], positions2=None if autocorr else data2[:3],
                                   weights1=data1[3], weights2=None if autocorr else data2[3], position_type='xyz', **options).wcounts
            assert np.allclose(test, ref)


if __name__ == '__main__':

    for mode in ['theta','s','smu','rppi','rp']:
        test(mode=mode)
