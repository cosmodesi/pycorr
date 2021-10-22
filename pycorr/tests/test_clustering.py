import os

import numpy as np
from matplotlib import pyplot as plt
import fitsio

from pycorr import TwoPointCorrelationFunction, project_to_multipoles,\
                    project_to_wp, setup_logging


catalog_dir = os.path.join(os.path.dirname(__file__), '_catalogs')
data_fn = os.path.join(catalog_dir, 'data.fits')
randoms_fn = os.path.join(catalog_dir, 'randoms.fits')
bias = 2.0


def mkdir(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        pass


def save_lognormal_catalogs(data_fn, randoms_fn, seed=42):
    from nbodykit.lab import cosmology, LogNormalCatalog, UniformCatalog
    from nbodykit.transform import CartesianToSky
    from nbodykit.utils import GatherArray

    def save_fits(cat, fn):
        array = np.empty(cat.size,dtype=[(col,cat[col].dtype,cat[col].shape[1:]) for col in cat.columns])
        for col in cat.columns: array[col] = cat[col].compute()
        array = GatherArray(array,comm=cat.comm)
        if cat.comm.rank == 0:
            fitsio.write(fn,array,clobber=True)

    redshift = 0.7
    cosmo = cosmology.Planck15.match(Omega0_m=0.3)
    Plin = cosmology.LinearPower(cosmo,redshift,transfer='CLASS')
    nbar = 1e-4
    BoxSize = 800
    catalog = LogNormalCatalog(Plin=Plin, nbar=nbar, BoxSize=BoxSize, Nmesh=256, bias=bias, seed=seed)
    #print(redshift,cosmo.scale_independent_growth_rate(redshift),cosmo.comoving_distance(redshift))

    offset = cosmo.comoving_distance(redshift) - BoxSize/2.
    offset = np.array([offset,0,0])
    catalog['Position'] += offset
    distance = np.sum(catalog['Position']**2,axis=-1)**0.5
    los = catalog['Position']/distance[:,None]
    catalog['Position'] += (catalog['VelocityOffset']*los).sum(axis=-1)[:,None]*los
    #mask = (catalog['Position'] >= offset) & (catalog['Position'] < offset + BoxSize)
    #catalog = catalog[np.all(mask,axis=-1)]
    catalog['NZ'] = nbar*np.ones(catalog.size,dtype='f8')
    catalog['Weight'] = np.ones(catalog.size,dtype='f8')
    catalog['RA'],catalog['DEC'],catalog['Z'] = CartesianToSky(catalog['Position'],cosmo)
    save_fits(catalog,data_fn)

    catalog = UniformCatalog(BoxSize=BoxSize, nbar=nbar*5, seed=seed)
    catalog['Position'] += offset
    catalog['Weight'] = np.ones(catalog.size,dtype='f8')
    catalog['NZ'] = nbar*np.ones(catalog.size,dtype='f8')
    catalog['RA'],catalog['DEC'],catalog['Z'] = CartesianToSky(catalog['Position'],cosmo)
    save_fits(catalog,randoms_fn)


def setup():
    mkdir(catalog_dir)
    save_lognormal_catalogs(data_fn, randoms_fn, seed=42)


def compute_correlation_function():

    data = fitsio.read(data_fn)
    randoms = fitsio.read(randoms_fn)

    def get_positions_weights(catalog):
        return catalog['Position'].T, None #catalog['Weight']

    data_positions, data_weights = get_positions_weights(data)
    randoms_positions, randoms_weights = get_positions_weights(randoms)

    def run(mode, edges, ells=(0,2,4)):
        corrmode = mode
        if mode == 'multi':
            corrmode = 'smu'
        if mode == 'wp':
            corrmode = 'rppi'
        if corrmode == 'smu':
            edges = (edges, np.linspace(0, 1, 101))
        if corrmode == 'rppi':
            edges = (edges, np.linspace(0, 40, 41))
        result = TwoPointCorrelationFunction(corrmode, edges, data_positions1=data_positions, data_weights1=data_weights,
                                         randoms_positions1=randoms_positions, randoms_weights1=randoms_weights,
                                         engine='corrfunc', position_type='xyz', estimator='landyszalay', nthreads=4)
        if mode == 'multi':
            return project_to_multipoles(result, ells=ells)
        if mode == 'wp':
            return project_to_wp(result)
        return result.sep[0], result.corr

    fig, lax = plt.subplots(ncols=3, nrows=1, sharex=False, sharey=False, figsize=(18,7))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    lax = lax.flatten()
    for ax in lax: ax.grid()
    ax = lax[0]
    sep, xi = run('theta', edges=np.linspace(1e-3, 4, 11))
    ax.plot(sep, sep*xi)
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel(r'$\theta w(\theta)$')
    ax = lax[1]
    ells = (0,2,4)
    sep, xi = run('multi', edges=np.linspace(1e-3, 140, 41), ells=ells)
    for ill,ell in enumerate(ells):
        ax.plot(sep, sep**2*xi[ill], label='$\ell = {:d}$'.format(ell))
    ax.set_xlabel('$s$ [$\mathrm{Mpc}/h$]')
    ax.set_ylabel(r'$s^{2}\xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
    ax.legend()
    ax = lax[2]
    sep, wp = run('wp', edges=np.logspace(-1.5, 1.8, 21))
    ax.plot(sep, sep*wp)
    ax.set_xscale('log')
    ax.set_xlabel('$r_{p}$ [$\mathrm{Mpc}/h$]')
    ax.set_ylabel('$r_{p} w_{p}$ [$(\mathrm{{Mpc}}/h)^{{2}}$]')
    plt.show()


if __name__ == '__main__':

    setup_logging()
    setup()
    compute_correlation_function()
