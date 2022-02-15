.. _user-building:

Building
========

Requirements
------------
Strict requirements are:

  - numpy
  - scipy

To use Corrfunc as two-point counter engine (only engine linked so far):

  - git+https://github.com/adematti/Corrfunc@desi

To perform MPI parallelization:

  - mpi4py
  - pmesh

pip
---
To install **pycorr** alone (without two-point counter engine, so fairly useless), simply run::

  python -m pip install git+https://github.com/cosmodesi/pycorr

Corrfunc is currently the only two-point counter engine implemented. We currently use a branch of Corrfunc,
located `here <https://github.com/adematti/Corrfunc/tree/desi>`_.
Uninstall previous Corrfunc version (if any)::

  pip uninstall Corrfunc

then::

  python -m pip install git+https://github.com/cosmodesi/pycorr#egg=pycorr[corrfunc]

or if **pycorr** is already installed::

  python -m pip install git+https://github.com/adematti/Corrfunc@desi

To further perform MPI parallelization or use scikit-learn's KMeans algorithm to split catalog into subsamples for jackknife estimates::

  python -m pip install git+https://github.com/cosmodesi/pycorr#egg=pycorr[mpi,jackknife,corrfunc]

git
---
First::

  git clone https://github.com/cosmodesi/pycorr.git

To install the code::

  python setup.py install --user

Or in development mode (any change to Python code will take place immediately)::

  python setup.py develop --user
