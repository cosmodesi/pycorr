import os

import numpy as np
from matplotlib import pyplot as plt

from cosmoprimo import PowerToCorrelation
from cosmoprimo.fiducial import DESI
from mockfactory import EulerianLinearMock
from mockfactory.make_survey import RandomBoxCatalog
from pycorr import TwoPointCorrelationFunction, setup_logging


catalog_dir = os.path.join(os.path.dirname(__file__), '_catalogs')
data_fn = os.path.join(catalog_dir, 'data.fits')
randoms_fn = os.path.join(catalog_dir, 'randoms.fits')

cosmo = DESI()
z = 1.
pklin = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
f = cosmo.get_fourier().sigma8_z(z=z, of='theta_cb') / cosmo.get_fourier().sigma8_z(z=z, of='delta_cb')
bias = 1.5
boxsize = 500.
boxcenter = np.array([600., 0., 0.])
nbar = 1e-3


def pk_kaiser_model(k, ell=0):
    pk = bias ** 2 * pklin(k)
    beta = f / bias
    if ell == 0: return (1. + 2. / 3. * beta + 1. / 5. * beta**2) * pk + 1. / nbar
    if ell == 2: return (4. / 3. * beta + 4. / 7. * beta**2) * pk
    if ell == 4: return 8. / 35 * beta**2 * pk
    return np.zeros_like(k)


def xi_kaiser_model(s, ell=0):
    k = np.logspace(-4, 2, 1000)
    slog, xiell = PowerToCorrelation(k, ell=ell)(pk_kaiser_model(k, ell))
    return np.interp(s, slog, xiell)


def test_clustering(seed=42):

    nmesh = 128
    mock = EulerianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias)
    mock.set_rsd(f=f, los=None)

    data = RandomBoxCatalog(nbar=nbar, boxsize=boxsize, boxcenter=boxcenter, seed=seed)
    randoms = RandomBoxCatalog(nbar=2 * nbar, boxsize=boxsize, boxcenter=boxcenter, seed=seed)
    data['Weight'] = mock.readout(data['Position'], field='delta', resampler='tsc', compensate=True) + 1.

    def run(mode, edges, ells=(0, 2, 4)):
        corrmode = mode
        if mode == 'multi':
            corrmode = 'smu'
        if mode == 'wp':
            corrmode = 'rppi'
        if corrmode == 'smu':
            edges = (edges, np.linspace(-1, 1, 101))
        if corrmode == 'rppi':
            edges = (edges, np.linspace(0, 40, 41))
        result = TwoPointCorrelationFunction(corrmode, edges, data_positions1=data['Position'], data_weights1=data['Weight'],
                                             randoms_positions1=randoms['Position'], engine='corrfunc', position_type='pos',
                                             estimator='landyszalay', nthreads=4, mpicomm=data.mpicomm)
        if mode == 'multi':
            return result(ells=ells, return_sep=True)
        if mode == 'wp':
            return result(pimax=40, return_sep=True)
        return result.sep, result.corr

    fig, lax = plt.subplots(ncols=3, nrows=1, sharex=False, sharey=False, figsize=(18, 7))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    lax = lax.flatten()
    for ax in lax: ax.grid()
    ax = lax[0]
    sep, xi = run('theta', edges=np.linspace(1e-3, 4, 11))
    ax.plot(sep, sep * xi)
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel(r'$\theta w(\theta)$')
    ax = lax[1]
    ells = (0, 2, 4)
    sep, xi = run('multi', edges=np.linspace(10, 80, 41), ells=ells)
    for ill, ell in enumerate(ells):
        xi_theory = xi_kaiser_model(sep, ell=ell)
        ax.plot(sep, sep**2 * xi_theory, label='theory' if ill == 0 else None, color='C{:d}'.format(ill), linestyle=':')
        ax.plot(sep, sep**2 * xi[ill], label=r'$\ell = {:d}$'.format(ell), color='C{:d}'.format(ill))
    ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
    ax.set_ylabel(r'$s^{2}\xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
    ax.legend()
    ax = lax[2]
    sep, wp = run('wp', edges=np.logspace(-1.5, 1.8, 21))
    ax.plot(sep, sep * wp)
    ax.set_xscale('log')
    ax.set_xlabel(r'$r_{p}$ [$\mathrm{Mpc}/h$]')
    ax.set_ylabel(r'$r_{p} w_{p}$ [$(\mathrm{{Mpc}}/h)^{{2}}$]')
    plt.show()


if __name__ == '__main__':

    setup_logging()

    test_clustering()
