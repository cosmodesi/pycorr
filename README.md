# pycorr

**pycorr** is a wrapper for correlation function estimation, designed to handle different two-point counter engines (currently only Corrfunc).
It currently supports:

  - theta (angular), s, s-mu, rp-pi binning schemes
  - analytical two-point counts with periodic boundary conditions
  - inverse bitwise weights (in any integer format) and (angular) upweighting
  - MPI parallelization (further requires mpi4py and pmesh)
  - jackknife estimate of the correlation function covariance matrix

A typical auto-correlation function estimation is as simple as:
```
import numpy as np
from pycorr import TwoPointCorrelationFunction

edges = (np.linspace(1, 101, 51), np.linspace(0, 50, 51))
# pass e.g. mpicomm = MPI.COMM_WORLD if input positions and weights are MPI-scattered
result = TwoPointCorrelationFunction('rppi', edges, data_positions1=data_positions1, data_weights1=data_weights1,
                                     randoms_positions1=randoms_positions1, randoms_weights1=randoms_weights1,
                                     engine='corrfunc', nthreads=4)
# separation array in result.sep
# correlation function in result.corr
```

Example notebooks presenting most use cases are provided in directory nb/.

## Documentation

Documentation is hosted on Read the Docs, [pycorr docs](https://py2pcf.readthedocs.io/).

## Requirements

Strict requirements are:

  - numpy
  - scipy

To use Corrfunc as two-point-counting engine (only engine linked so far):

  - git+https://github.com/adematti/Corrfunc@desi

To run with MPI:

  - mpi4py
  - pmesh

## Installation

See [pycorr docs](https://py2pcf.readthedocs.io/en/latest/user/building.html).

## License

**pycorr** is free software distributed under a BSD3 license. For details see the [LICENSE](https://github.com/cosmodesi/pycorr/blob/main/LICENSE).

## Credits

- Lehman Garrison and Manodeep Sinha for advice when implementing linear binning, and PIP and angular weights into Corrfunc.
- Davide Bianchi for cross-checks of two-point counts with PIP weights.
- Svyatoslav Trusov for script to compute jackknife covariance estimates based on https://arxiv.org/pdf/2109.07071.pdf: https://github.com/theonefromnowhere/JK_pycorr/blob/main/CF_JK_ST_conf.py.
- Enrique Paillas & Seshadri Nadathur for suggestions about reconstructed 2pcf measurements 
