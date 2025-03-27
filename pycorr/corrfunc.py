"""Implement Corrfunc two-point counter engine."""

import numpy as np
from Corrfunc import theory, mocks

from .twopoint_counter import BaseTwoPointCounter, TwoPointCounterError
from . import utils
from .utils import _get_box


class CorrfuncTwoPointCounter(BaseTwoPointCounter):

    """Extend :class:`BaseTwoPointCounter` for Corrfunc two-point counting code."""

    name = 'corrfunc'

    def run(self):
        """Compute the two-point counts and set :attr:`wcounts` and :attr:`sep`."""
        if self.ndim == 2:
            self.compute_sepsavg[1] = False

        (dpositions1, dweights1), (dpositions2, dweights2) = self._mpi_decompose()

        los_type = self.los_type
        if self.los_type == 'endpoint':
            los_type = 'firstpoint'
            if not self.autocorr:
                dpositions1, dpositions2 = dpositions2, dpositions1
                dweights1, dweights2 = dweights2, dweights1

        # if self.mode in ['rppi']:
        #     if self.los_type not in ['x', 'y', 'z', 'midpoint']:
        #         raise TwoPointCounterError('Corrfunc only supports x / y / z / midpoint line-of-sight for mode {}'.format(self.mode))

        if self.mode in ['smu', 'rppi']:
            axis = {'smu': 'mu', 'rppi': 'pi'}[self.mode]
            edges = self.edges[1]
            if edges[0] != - edges[-1]:
                raise TwoPointCounterError('Corrfunc only supports symmetric binning: {0}min = -{0}max'.format(axis))
            lin = np.linspace(-edges[-1], edges[-1], len(edges))
            if not np.allclose(edges, lin):
                raise TwoPointCounterError('Corrfunc only supports linear {} binning'.format(axis))

        autocorr = self.autocorr and not self.with_mpi

        def _get_boxsize():
            if self.periodic:
                return tuple(self.boxsize)
            return None

        def get_rotated_positions():
            boxsize = _get_boxsize()
            # Rotating coordinates to put los along z

            def rotate(positions):
                toret = list(positions)
                if self.los_type == 'x':
                    # rotation around -y: x' = -z and z' = x
                    toret[0] = - positions[2]
                    toret[2] = positions[0]
                elif self.los_type == 'y':
                    # rotation around x: y' = -z and z' = y
                    toret[1] = - positions[2]
                    toret[2] = positions[1]
                return toret

            positions1 = rotate(dpositions1)
            positions2 = [None] * 3
            if not autocorr:
                positions2 = rotate(dpositions2)
            if boxsize is not None:
                boxsize = np.abs(rotate(boxsize))
            return positions1, positions2, boxsize

        weight_type = None
        output_weightavg = False
        weights1, weights2 = dweights1 if dweights1 else None, dweights2 if dweights2 else None
        weight_attrs = None

        if self.n_bitwise_weights:
            output_weightavg = True
            weight_type = 'inverse_bitwise'
            dtype = {4: np.int32, 8: np.int64}[self.dtype.itemsize]

            def reformat_bitweights(weights):
                return utils.reformat_bitarrays(*weights[:self.n_bitwise_weights], dtype=dtype) + weights[self.n_bitwise_weights:]

            weights1 = reformat_bitweights(dweights1)
            if not autocorr:
                weights2 = reformat_bitweights(dweights2)
            weight_attrs = {'noffset': self.weight_attrs['noffset'], 'default_value': self.weight_attrs['default_value'] / self.weight_attrs['nrealizations']}
            correction = self.weight_attrs.get('correction', None)
            if correction is not None: weight_attrs['correction'] = correction

        elif weights1:
            output_weightavg = True
            weight_type = 'pair_product'

        pair_weights, sep_pair_weights = None, None
        if self.cos_twopoint_weights is not None:
            output_weightavg = True
            weight_type = 'inverse_bitwise'
            pair_weights = self.cos_twopoint_weights.weight
            sep_pair_weights = self.cos_twopoint_weights.sep
        attrs = dict(self.attrs)

        kwargs = {'weights1': weights1, 'weights2': weights2,
                  'bin_type': self.bin_type, 'weight_type': weight_type,
                  'pair_weights': pair_weights, 'sep_pair_weights': sep_pair_weights,
                  'attrs_pair_weights': weight_attrs, 'verbose': attrs.pop('verbose', False),
                  'isa': attrs.pop('isa', 'fastest')}  # to be set to 'fastest' when bitwise weights included in all kernels

        if attrs.pop('gpu', False):
            kwargs['gpu'] = True

        refine_factors = attrs.pop('mesh_refine_factors', None)
        if refine_factors is not None:
            nfactors = 3 - int(self.mode == 'theta')
            if not utils.is_sequence(refine_factors):
                refine_factors = [refine_factors] * nfactors
            if len(refine_factors) != nfactors:
                raise TwoPointCounterError('Provide {:d} refine factors for mode {} (found {:d})'.format(nfactors, self.mode, len(refine_factors)))
            refine_factors = [int(factor) for factor in refine_factors]
            if self.mode == 'theta':
                bases = ['ra', 'dec']
            else:
                bases = ['xbin', 'ybin', 'zbin']
            for base, factor in zip(bases, refine_factors): kwargs[base + '_refine_factor'] = int(factor)

        with_auto_pairs = self.edges[0][0] <= 0.

        if self.selection_attrs:
            if self.mode != 'smu':
                raise NotImplementedError('selection only available for mode smu')
            kwargs['attrs_selection'] = self.selection_attrs
            with_auto_pairs &= all(limits[0] <= 0. for limits in self.selection_attrs.values())

        positions1 = dpositions1
        positions2 = dpositions2
        if autocorr:
            positions2 = [None] * 3

        if self.mode == 'theta':
            fast_acos = attrs.pop('fast_acos', False)

        if attrs:
            import warnings
            warnings.warn('These arguments are not read: {}'.format(attrs))

        def call_corrfunc(method, *args, **kwargs):
            try:
                return method(*args, **kwargs)
            except TypeError as exc:
                if kwargs['gpu']:
                    raise NotImplementedError('GPU kernel not implemented for kernel {}'.format(method)) from exc
                raise TwoPointCounterError('Please reinstall relevant Corrfunc branch (including PIP weights):\n\
                                            > pip uninstall Corrfunc\n\
                                            > pip install git+https://github.com/cosmodesi/Corrfunc@desi\n') from exc

        if len(positions1[0]) and (autocorr or len(positions2[0])):

            if self.mode == 'theta':
                if self.periodic:
                    raise TwoPointCounterError('Corrfunc does not provide periodic boundary conditions for the angular correlation function')
                result = call_corrfunc(mocks.DDtheta_mocks, autocorr, nthreads=self.nthreads, binfile=self.edges[0],
                                       RA1=positions1[0], DEC1=positions1[1], RA2=positions2[0], DEC2=positions2[1],
                                       output_thetaavg=self.compute_sepavg, fast_acos=fast_acos, **kwargs)
                key_sep = 'thetaavg'

            elif self.mode == 's':
                result = call_corrfunc(theory.DD, autocorr, nthreads=self.nthreads, binfile=self.edges[0],
                                       X1=positions1[0], Y1=positions1[1], Z1=positions1[2],
                                       X2=positions2[0], Y2=positions2[1], Z2=positions2[2],
                                       periodic=self.periodic, boxsize=_get_boxsize(),
                                       output_ravg=self.compute_sepavg, **kwargs)

                key_sep = 'ravg'

            elif self.mode == 'smu':
                if self.los_type in ['x', 'y', 'z']:
                    positions1, positions2, boxsize = get_rotated_positions()
                    result = call_corrfunc(theory.DDsmu, autocorr, nthreads=self.nthreads,
                                           binfile=self.edges[0], mumax=self.edges[1][-1], nmubins=len(self.edges[1]) - 1,
                                           X1=positions1[0], Y1=positions1[1], Z1=positions1[2],
                                           X2=positions2[0], Y2=positions2[1], Z2=positions2[2],
                                           periodic=self.periodic, boxsize=boxsize,
                                           output_savg=self.compute_sepavg, **kwargs)
                else:
                    result = call_corrfunc(mocks.DDsmu_mocks, autocorr, nthreads=self.nthreads,
                                           binfile=self.edges[0], mumax=self.edges[1][-1], nmubins=len(self.edges[1]) - 1,
                                           X1=positions1[0], Y1=positions1[1], Z1=positions1[2],
                                           X2=positions2[0], Y2=positions2[1], Z2=positions2[2],
                                           output_savg=self.compute_sepavg, los_type=los_type, **kwargs)

                key_sep = 'savg'

            elif self.mode == 'rppi':
                if self.los_type in ['x', 'y', 'z']:
                    positions1, positions2, boxsize = get_rotated_positions()
                    result = call_corrfunc(theory.DDrppi, autocorr, nthreads=self.nthreads,
                                           binfile=self.edges[0], pimax=self.edges[1][-1], npibins=len(self.edges[1]) - 1,
                                           X1=positions1[0], Y1=positions1[1], Z1=positions1[2],
                                           X2=positions2[0], Y2=positions2[1], Z2=positions2[2],
                                           periodic=self.periodic, boxsize=boxsize,
                                           output_rpavg=self.compute_sepavg, **kwargs)
                else:
                    result = call_corrfunc(mocks.DDrppi_mocks, autocorr, nthreads=self.nthreads,
                                           binfile=self.edges[0], pimax=self.edges[1][-1], npibins=len(self.edges[1]) - 1,
                                           X1=positions1[0], Y1=positions1[1], Z1=positions1[2],
                                           X2=positions2[0], Y2=positions2[1], Z2=positions2[2],
                                           output_rpavg=self.compute_sepavg, los_type=los_type, **kwargs)

                key_sep = 'rpavg'

            elif self.mode == 'rp':
                key_sep = 'rpavg'

                def _get_surveysize(*positions):
                    posmin, posmax = _get_box(*positions)
                    return posmax - posmin

                if self.los_type in ['x', 'y', 'z']:
                    positions1, positions2, boxsize = get_rotated_positions()
                    if self.periodic:
                        pimax = boxsize[-1] / 2.  # los axis is z
                    else:
                        if autocorr:
                            boxsize = _get_surveysize(positions1)
                        else:
                            boxsize = _get_surveysize(positions1, positions2)
                        pimax = boxsize[-1] + 1.  # los axis is z
                    result = call_corrfunc(theory.DDrppi, autocorr, nthreads=self.nthreads,
                                           binfile=self.edges[0], pimax=pimax, npibins=1,
                                           X1=positions1[0], Y1=positions1[1], Z1=positions1[2],
                                           X2=positions2[0], Y2=positions2[1], Z2=positions2[2],
                                           periodic=self.periodic, boxsize=boxsize,
                                           output_rpavg=self.compute_sepavg, **kwargs)
                else:
                    # \pi = \hat{\ell} \cdot (r_{1} - r_{2}) < r_{1} + r_{2}
                    # local calculation, since integrated over pi
                    # \pi = \hat{\ell} \cdot (r_{1} - r_{2}) < | r_{1} - r_{2} | < boxsize
                    if autocorr:
                        boxsize = _get_surveysize(positions1)
                    else:
                        boxsize = _get_surveysize(positions1, positions2)
                    pimax = sum(p**2 for p in boxsize)**0.5 + 1.
                    result = call_corrfunc(mocks.DDrppi_mocks, autocorr, nthreads=self.nthreads,
                                           binfile=self.edges[0], pimax=pimax, npibins=1,
                                           X1=positions1[0], Y1=positions1[1], Z1=positions1[2],
                                           X2=positions2[0], Y2=positions2[1], Z2=positions2[2],
                                           output_rpavg=self.compute_sepavg, los_type=los_type, **kwargs)

                # sum over pi to keep only rp
                result = {key: result[key] for key in ['npairs', 'weightavg', key_sep]}
                result[key_sep].shape = result['weightavg'].shape = result['npairs'].shape = self.shape + (-1,)
                wpairs = result['npairs'] * (result['weightavg'] if output_weightavg else 1.)
                result['npairs'] = np.sum(result['npairs'], axis=-1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    result['weightavg'] = np.sum(wpairs, axis=-1) / result['npairs']
                    result[key_sep] = np.sum(result[key_sep] * wpairs, axis=-1) / (result['npairs'] * result['weightavg'])
                result['weightavg'][result['npairs'] == 0] = 0.
                result[key_sep][result['npairs'] == 0] = 0.

            else:

                raise TwoPointCounterError('Corrfunc does not support mode {}'.format(self.mode))

            self.ncounts = result['npairs']
            self.wcounts = self.ncounts * (result['weightavg'] if output_weightavg else 1.)\
                           * (self.weight_attrs['nrealizations'] if self.n_bitwise_weights else 1.)
            self.wcounts[self.ncounts == 0] = 0.
            if self.compute_sepavg:
                self.sep = result[key_sep]
            self.sep.shape = self.wcounts.shape = self.ncounts.shape = self.shape

        self.wcounts = self.wcounts.astype('f8')
        self.ncounts = self.ncounts.astype('i8')

        if self.with_mpi:
            wcounts = self.wcounts
            self.wcounts = self.mpicomm.allreduce(self.wcounts)
            self.ncounts = self.mpicomm.allreduce(self.ncounts)
            if self.compute_sepavg:
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.sep = self.mpicomm.allreduce(self.sep * wcounts) / self.wcounts

        if with_auto_pairs:  # remove auto-pairs
            index_zero = 0
            if self.mode in ['smu', 'rppi']: index_zero = self.shape[1] // 2  # mu = 0 bin
            self.ncounts.flat[index_zero] -= self.size1
            autocounts = self._sum_auto_weights()
            if self.compute_sepavg:
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.sep.flat[index_zero] *= self.wcounts.flat[index_zero] / (self.wcounts.flat[index_zero] - autocounts)
            self.wcounts.flat[index_zero] -= autocounts

        self.wcounts[self.ncounts == 0] = 0.  # as above may create uncertainty
        if self.compute_sepavg:
            self.sep[self.ncounts == 0] = np.nan

        if self.mode in ['smu', 'rppi'] and self.los_type == 'endpoint':
            # endpoint is 1 <-> 2 with firstpoint, and counts reversed
            for name in ['wcounts', 'ncounts']:
                setattr(self, name, getattr(self, name)[:, ::-1])
            self.sep = self.sep[:, ::-1]
