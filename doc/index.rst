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

**pycorr** is a Python package to compute correlation function with various pair-counting engines.


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
