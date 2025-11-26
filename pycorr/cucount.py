"""Implement cucount two-point counter engine."""

import warnings

import numpy as np
from cucount.numpy import count2, Particles, BinAttrs, WeightAttrs, SelectionAttrs, MeshAttrs

from .twopoint_counter import BaseTwoPointCounter, TwoPointCounterError
from . import utils


class CucountTwoPointCounter(BaseTwoPointCounter):

    """Extend :class:`BaseTwoPointCounter` for cucount two-point counting code."""

    name = 'cucount'

    def run(self):
        """Compute the two-point counts and set :attr:`wcounts` and :attr:`sep`."""

        assert self.dtype.itemsize == 8, 'cucount only supports float64 data'
        if self.ndim == 2:
            self.compute_sepsavg[1] = False
        if self.compute_sepsavg[0]:
            warnings.warn('cucount does not provide separation averages; setting compute_sepsavg[0] = False')
        self.compute_sepsavg = [False] * len(self.compute_sepsavg)

        (positions1, weights1), (positions2, weights2) = self._mpi_decompose()

        autocorr = self.autocorr and not self.with_mpi

        bitwise = None
        if self.n_bitwise_weights:
            dtype = {4: np.int32, 8: np.int64}[self.dtype.itemsize]

            def reformat_bitweights(weights):
                # Individual weights are stored first, then bitwise weights
                return weights[self.n_bitwise_weights:] + utils.reformat_bitarrays(*weights[:self.n_bitwise_weights], dtype=dtype)

            weights1 = reformat_bitweights(weights1)
            if not autocorr:
                weights2 = reformat_bitweights(weights2)
            bitwise = {name: self.weight_attrs[name] for name in ['noffset', 'default_value', 'nrealizations']}
            correction = self.weight_attrs.get('correction', None)
            bitwise['p_correction_nbits'] = correction if correction is not None else False

        # Prepare WeightAttrs
        angular = None
        if self.cos_twopoint_weights is not None:
            angular = dict(sep=self.twopoint_weights.sep, weight=self.twopoint_weights.weight)

        wattrs = WeightAttrs(bitwise=bitwise, angular=angular)

        zero_indices = tuple(np.digitize(0, edges, right=False) - 1 for edges in self.edges)
        with_auto_pairs = all((zero_index >= 0) and (zero_index < len(edges) - 1) for zero_index, edges in zip(zero_indices, self.edges))
        # Prepare SelectionAttrs
        allowed_selection = ['theta']
        if not set(self.selection_attrs).issubset(set(allowed_selection)):
            raise NotImplementedError('selection only available for theta')
        if self.selection_attrs:
            with_auto_pairs &= all(limits[0] <= 0. for limits in self.selection_attrs.values())
        sattrs = SelectionAttrs(**self.selection_attrs)

        # Prepare BinAttrs
        battrs = {}
        if self.mode == 'theta':
            battrs['theta'] = self.edges[0]
        elif self.mode == 's':
            battrs['s'] = self.edges[0]
        elif self.mode == 'smu':
            battrs['s'], battrs['mu'] = self.edges
        elif self.mode == 'rppi':
            battrs['rp'], battrs['pi'] = self.edges
        elif self.mode == 'rp':
            battrs['rp'] = self.edges[0]
        else:
            raise TwoPointCounterError('cucount does not support mode {}'.format(self.mode))
        for name in ['mu', 'rp', 'pi']:  # requires los_type
            if name in battrs:
                battrs[name] = (battrs[name], self.los_type)
        battrs = BinAttrs(**battrs)

        attrs = dict(self.attrs)
        meshsize = attrs.pop('meshsize', None)
        refine = attrs.pop('refine', 1.)
        if attrs:
            warnings.warn('These arguments are not read: {}'.format(attrs))

        if positions2 is None:
            positions2, weights2 = positions1, weights1

        if len(positions1[0]) and (autocorr or len(positions2[0])):
            # Prepare Particles
            particles = [Particles(positions=positions, weights=weights, positions_type='rd' if self.mode == 'theta' else 'xyz') for positions, weights in
                        [(positions1, weights1), (positions2, weights2)]]
            mattrs = dict(meshsize=meshsize, sattrs=sattrs, battrs=battrs, refine=refine)
            if self.periodic:
                if self.mode == 'theta':
                    raise TwoPointCounterError('cucount does not provide periodic boundary conditions for the angular correlation function')
                mattrs.update(boxsize=self.boxsize, periodic=True)

            mattrs = MeshAttrs(*particles, **mattrs)
            self.wcounts = count2(*particles, mattrs=mattrs, wattrs=wattrs, battrs=battrs, sattrs=sattrs, nthreads=self.nthreads)['weight']

        if self.with_mpi:
            self.wcounts = self.mpicomm.allreduce(self.wcounts)

        if with_auto_pairs:  # remove auto-pairs
            autocounts = self._sum_auto_weights()
            self.wcounts[zero_indices] -= autocounts
