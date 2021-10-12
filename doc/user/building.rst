.. _user-building:

Building
========

Requirements
------------
Strict requirements are:

  - numpy
  - scipy

To use Corrfunc as pair-counting engine (only engine linked so far):

  - git+https://github.com/adematti/Corrfunc@pipweights

To perform MPI parallelization:

  - mpi4py
  - pmesh

pip
---
To install **pycorr**, simply run::

  python -m pip install git+https://github.com/adematti/pycorr

To also install Corrfunc::

  python -m pip install git+https://github.com/adematti/pycorr#egg=pycorr[corrfunc]

To perform MPI parallelization::

  python -m pip install git+https://github.com/adematti/pycorr#egg=pycorr[mpi]

git
---
First::

  git clone https://github.com/adematti/pycorr.git

To install the code::

  python setup.py install --user

Or in development mode (any change to Python code will take place immediately)::

  python setup.py develop --user
