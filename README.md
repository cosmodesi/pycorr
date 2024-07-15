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

Example notebooks presenting most use cases are provided in directory [nb/](https://github.com/cosmodesi/pycorr/tree/main/nb).

## Documentation

Documentation is hosted on Read the Docs, [pycorr docs](https://py2pcf.readthedocs.io/).
As mentioned above, you may find more practical guidance in [the example notebooks directory](https://github.com/cosmodesi/pycorr/tree/main/nb).

## Requirements

Strict requirements are:

  - numpy
  - scipy

To use Corrfunc as two-point-counting engine (only engine linked so far):

  - https://github.com/adematti/Corrfunc (desi branch)

To run with MPI:

  - mpi4py
  - pmesh

## Installation

See [pycorr docs](https://py2pcf.readthedocs.io/en/latest/user/building.html).

## License

**pycorr** is free software distributed under a BSD3 license. For details see the [LICENSE](https://github.com/cosmodesi/pycorr/blob/main/LICENSE).

## Acknowledgments

- Lehman Garrison and Manodeep Sinha for advice when implementing linear binning, and PIP and angular weights into Corrfunc.
- Davide Bianchi for cross-checks of two-point counts with PIP weights.
- Svyatoslav Trusov for script to compute jackknife covariance estimates based on https://arxiv.org/pdf/2109.07071.pdf: https://github.com/theonefromnowhere/JK_pycorr/blob/main/CF_JK_ST_conf.py.
- Enrique Paillas and Seshadri Nadathur for suggestions about reconstructed 2pcf measurements
- Craig Warner for GPU-izing Corrfunc 'smu' counts
- Craig Warner, James Lasker, Misha Rashkovetskyi and Edmond Chaussidon for spotting typos / bug reports

# Citations

If you use ``pycorr`` with the ``Corrfunc`` engine (default one) for research, please cite the MNRAS ``Corrfunc`` code papers with the following
bibtex entries:

```
@ARTICLE{2020MNRAS.491.3022S,
    author = {{Sinha}, Manodeep and {Garrison}, Lehman H.},
    title = "{CORRFUNC - a suite of blazing fast correlation functions on
    the CPU}",
    journal = {\mnras},
    keywords = {methods: numerical, galaxies: general, galaxies:
    haloes, dark matter, large-scale structure of Universe, cosmology:
    theory},
    year = "2020",
    month = "Jan",
    volume = {491},
    number = {2},
    pages = {3022-3041},
    doi = {10.1093/mnras/stz3157},
    adsurl =
    {https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.3022S},
    adsnote = {Provided by the SAO/NASA
    Astrophysics Data System}
}


@InProceedings{10.1007/978-981-13-7729-7_1,
    author="Sinha, Manodeep and Garrison, Lehman",
    editor="Majumdar, Amit and Arora, Ritu",
    title="CORRFUNC: Blazing Fast Correlation Functions with AVX512F SIMD Intrinsics",
    booktitle="Software Challenges to Exascale Computing",
    year="2019",
    publisher="Springer Singapore",
    address="Singapore",
    pages="3--20",
    isbn="978-981-13-7729-7",
    url={https://doi.org/10.1007/978-981-13-7729-7_1}
}
```