"""Implement Corrfunc pair counter engine."""

import numpy as np
from Corrfunc import theory, mocks

from .pair_counter import BaseTwoPointCounter, PairCounterError
from . import utils


class CorrfuncTwoPointCounter(BaseTwoPointCounter):

    """Extend :class:`BaseTwoPointCounter` for Corrfunc pair counting code."""

    def run(self):
        """Compute the pair counts and set :attr:`wcounts` and :attr:`sep`."""

        output_weightavg = self.weights1 is not None

        def boxsize():
            if self.periodic:
                toret = self.boxsize[0]
                if not np.all(self.boxsize == toret):
                    raise PairCounterError('Corrfunc does not support non-cubic box')
                return toret
            return None

        def check_los():
            if self.los != 'midpoint':
                raise PairCounterError('Corrfunc only supports midpoint line-of-sight')
            return self.los

        def check_mu():
            edges = self.edges[1]
            if edges[0] != 0:
                raise PairCounterError('Corrfunc only supports mu starting at 0')
            lin = np.linspace(edges[0],edges[-1],len(edges))
            if not np.allclose(edges,lin):
                raise PairCounterError('Corrfunc only supports linear mu binning')

        def check_pi():
            edges = self.edges[1]
            if edges[0] != 0:
                raise PairCounterError('Corrfunc only supports pi starting at 0')
            lin = np.linspace(edges[0],edges[-1],int(edges[-1])+1)
            if len(lin) != len(edges) or not np.allclose(edges,lin):
                raise PairCounterError('Corrfunc only supports linear pi binning, with n = int(pimax) bins')

        def rotated_positions():
            # rotating coordinates to put los along z
            def rotate(positions):
                toret = list(positions)
                if self.los == 'x':
                    # rotation around -y: x' = z and z' = -x
                    toret[0] = positions[2]
                    toret[2] = -positions[0]
                elif self.los == 'y':
                    # rotation around x: y' = z and z' = -y
                    toret[1] = positions[2]
                    toret[2] = -positions[1]
                return toret

            positions1 = rotate(self.positions1)
            positions2 = [None]*3
            if not self.autocorr:
                positions2 = rotate(self.positions2)
            return positions1, positions2

        def sky_positions():
            positions1 = utils.cartesian_to_sky(self.positions1)
            positions2 = [None]*3
            if not self.autocorr:
                positions2 = utils.cartesian_to_sky(self.positions2)
            return positions1, positions2

        weight_type = None
        weights1, weights2 = self.weights1, self.weights2
        if self.n_bitwise_weights:
            weight_type = 'inverse_bitwise'
            dtype = {4:np.int32, 8:np.int64}[self.dtype.itemsize]

            def reformat_bitweights(weights):
                return utils.reformat_bitarrays(*weights[:self.n_bitwise_weights], dtype=dtype) + weights[self.n_bitwise_weights:]

            weights2 = weights1 = reformat_bitweights(self.weights1)
            if not self.autocorr:
                weights2 = reformat_bitweights(self.weights2)
        elif self.weights1 is not None:
            weight_type = 'pair_product'

        pair_weights, sep_pair_weights = None, None
        if self.twopoint_weights is not None:
            weight_type = 'inverse_bitwise'
            pair_weights = self.twopoint_weights.weight
            sep_pair_weights = self.twopoint_weights.sep

        kwargs = {'weights1': weights1, 'weights2': weights2,
                  'bin_type': self.bin_type, 'weight_type': weight_type,
                  'pair_weights': pair_weights, 'sep_pair_weights':sep_pair_weights,
                  'verbose': False,
                  'isa': 'fallback'} # to be set to 'fastest' when bitwise weights included in all kernels

        positions2 = self.positions2
        if self.autocorr:
            positions2 = [None]*3

        if self.mode == 'theta':
            if self.periodic:
                raise PairCounterError('Corrfunc does not provide periodic boundary conditions for the angular correlation function')
            result = mocks.DDtheta_mocks(self.autocorr, nthreads=self.nthreads, binfile=self.edges[0],
                                         RA1=self.positions1[0], DEC1=self.positions1[1], RA2=positions2[0], DEC2=positions2[1],
                                         output_thetaavg=self.output_sepavg, fast_acos=self.attrs.get('fast_acos',False), **kwargs)
            key_sep = 'thetaavg'

        elif self.mode == 's':
            result = theory.DD(self.autocorr, nthreads=self.nthreads, binfile=self.edges[0],
                               X1=self.positions1[0], Y1=self.positions1[1], Z1=self.positions1[2],
                               X2=positions2[0], Y2=positions2[1], Z2=positions2[2],
                               periodic=self.periodic, boxsize=boxsize(),
                               output_ravg=self.output_sepavg, **kwargs)

            key_sep = 'ravg'

        elif self.mode == 'smu':
            check_mu()
            if self.los in ['x','y','z']:
                positions1, positions2 = rotated_positions()
                result = theory.DDsmu(self.autocorr, nthreads=self.nthreads,
                                      binfile=self.edges[0], mu_max=self.edges[1].max(), nmu_bins=len(self.edges[1]) - 1,
                                      X1=positions1[0], Y1=positions1[1], Z1=positions1[2],
                                      X2=positions2[0], Y2=positions2[1], Z2=positions2[2],
                                      periodic=self.periodic, boxsize=boxsize(),
                                      output_savg=self.output_sepavg, **kwargs)
            else:
                check_los()
                positions1, positions2 = sky_positions()
                result = mocks.DDsmu_mocks(self.autocorr, cosmology=1, nthreads=self.nthreads,
                                           mu_max=self.edges[1].max(), nmu_bins=len(self.edges[1]) - 1, binfile=self.edges[0],
                                           RA1=positions1[1], DEC1=positions1[2], CZ1=positions1[0],
                                           RA2=positions2[1], DEC2=positions2[2], CZ2=positions2[0],
                                           is_comoving_dist=True,
                                           output_savg=self.output_sepavg, **kwargs)

            key_sep = 'savg'

        elif self.mode == 'rppi':
            check_pi()
            if self.los in ['x','y','z']:
                positions1, positions2 = rotated_positions()
                result = theory.DDrppi(self.autocorr, nthreads=self.nthreads,
                                       binfile=self.edges[0], pimax=self.edges[1].max(),
                                       X1=positions1[0], Y1=positions1[1], Z1=positions1[2],
                                       X2=positions2[0], Y2=positions2[1], Z2=positions2[2],
                                       periodic=self.periodic, boxsize=boxsize(),
                                       output_rpavg=self.output_sepavg, **kwargs)
            else:
                check_los()
                positions1, positions2 = sky_positions()
                result = mocks.DDrppi_mocks(self.autocorr, cosmology=1, nthreads=self.nthreads,
                                            pimax=self.edges[1].max(), binfile=self.edges[0],
                                            RA1=positions1[1], DEC1=positions1[2], CZ1=positions1[0],
                                            RA2=positions2[1], DEC2=positions2[2], CZ2=positions2[0],
                                            is_comoving_dist=True,
                                            output_rpavg=self.output_sepavg, **kwargs)

            key_sep = 'rpavg'

        elif self.mode == 'rp':
            key_sep = 'rpavg'
            if self.los in ['x','y','z']:
                raise PairCounterError('Corrfunc does not provide (cross-) xi(rp) for periodic boundary conditions')
                positions1, positions2 = rotated_positions()
                boxsize = boxsize()
                pimax = boxsize + 1. # los axis is z
                result = theory.DDrppi(self.autocorr, nthreads=self.nthreads,
                                       binfile=self.edges[0], pimax=pimax,
                                       X1=positions1[0], Y1=positions1[1], Z1=positions1[2],
                                       X2=positions2[0], Y2=positions2[1], Z2=positions2[2],
                                       periodic=self.periodic, boxsize=boxsize,
                                       output_rpavg=self.output_sepavg, **kwargs)
            else:
                check_los()
                positions1, positions2 = sky_positions()
                # \pi = \hat{\ell} \cdot (r_{1} - r_{2}) < r_{1} + r_{2}
                #if self.autocorr:
                #    pimax = 2*positions1[0].max()
                #else:
                #    pimax = 2*max(positions1[0].max(),positions2[0].max()
                # local calculation, since integrated over pi
                # \pi = \hat{\ell} \cdot (r_{1} - r_{2}) < | r_{1} - r_{2} | < boxsize
                if self.autocorr:
                    boxsize = [p.max() - p.min() for p in self.positions1]
                else:
                    boxsize = [max(p1.max(), p2.max()) - min(p1.min(), p2.min()) for p1, p2 in zip(self.positions1, self.positions2)]
                pimax = sum(p**2 for p in boxsize)**0.5
                result = mocks.DDrppi_mocks(self.autocorr, cosmology=1, nthreads=self.nthreads,
                                            pimax=pimax, binfile=self.edges[0],
                                            RA1=positions1[1], DEC1=positions1[2], CZ1=positions1[0],
                                            RA2=positions2[1], DEC2=positions2[2], CZ2=positions2[0],
                                            is_comoving_dist=True,
                                            output_rpavg=self.output_sepavg, **kwargs)

            # sum over pi to keep only rp
            result = {key:result[key] for key in ['npairs','weightavg',key_sep]}
            result[key_sep].shape = result['weightavg'].shape = result['npairs'].shape = self.shape + (-1,)
            npairs = result['npairs']
            result['weightavg'][npairs == 0] = 0.
            result['npairs'] = np.sum(result['npairs'], axis=-1)
            sumnpairs = np.maximum(result['npairs'], 1e-9) # just to avoid division by 0
            result[key_sep] = np.sum(result[key_sep]*npairs, axis=-1)/sumnpairs
            result['weightavg'] = np.sum(result['weightavg']*npairs, axis=-1)/sumnpairs

        else:
            raise PairCounterError('Corrfunc does not support mode {}'.format(self.mode))

        self.npairs = result['npairs']
        self.wcounts = self.npairs*(result['weightavg'] if output_weightavg else 1)\
                       *(self.nrealizations + 1 if self.n_bitwise_weights else 1)
        self.wcounts[self.npairs == 0] = 0.
        self.sep = result[key_sep]
        self.sep.shape = self.wcounts.shape = self.npairs.shape = self.shape
        #self.wcounts.shape = self.shape[:1] + (1400,)
        #self.wcounts = self.wcounts.sum(axis=-1)

        if self.with_mpi:
            self.wcounts = self.mpicomm.allreduce(self.wcounts)
            npairs = np.maximum(self.mpicomm.allreduce(self.npairs), 1e-9) # just to avoid division by 0
            self.sep = self.mpicomm.allreduce(self.sep * self.npairs)/npairs
            self.npairs = npairs
