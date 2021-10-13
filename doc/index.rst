.. title:: pycorr docs

**************************************
Welcome to pycorr's documentation!
**************************************

.. toctree::
  :maxdepth: 1
  :caption: User documentation

  user/building
  api/api

.. toctree::
  :maxdepth: 1
  :caption: Developer documentation

  developer/documentation
  developer/tests
  developer/contributing
  developer/changes

.. toctree::
  :hidden:

************
Introduction
************

**pycorr** is a wrapper for correlation function estimation, designed to handle different pair counter engines (currently only Corrfunc).
It currently supports:

  - theta (angular), s, s-mu, rp-pi binning schemes
  - analytical pair counts with periodic boundary conditions
  - inverse bitwise weights (in any integer format) and (angular) upweighting
  - MPI parallelization (further requires mpi4py and pmesh)

A typicall auto-correlation function estimation is as simple as:

.. code-block:: python

  import numpy as np
  from pycorr import TwoPointCorrelationFunction

  edges = (np.linspace(1, 100, 51), np.linspace(0, 50, 51))
  # pass e.g. mpicomm = MPI.COMM_WORLD if input positions and weights are MPI-scattered
  result = TwoPointCorrelationFunction('rppi', edges, data_positions1=data_positions1, data_weights1=data_weights1,
                                       randoms_positions1=randoms_positions1, randoms_weights1=randoms_weights1,
                                       engine='corrfunc', nthreads=4)
  # separation array in result.sep
  # correlation function in result.corr

Example notebooks are provided in :root:`pycorr/nb`.

**************
Code structure
**************

The code structure is the following:

  - pair_counter.py implements the base pair counter class
  - estimator.py implements correlation function estimators based on pair counts
  - correlation_function.py implements high-level interface for correlation function estimation
  - utils.py implements various utilities
  - a module for each pair-counting engine


Changelog
=========

* :doc:`developer/changes`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
