.. _user-building:

Building
========

Requirements
------------
Strict requirements are:

  - numpy
  - scipy
  - Corrfunc

pip
---
To install **pycorr**, simply run::

  python -m pip install git+https://github.com/adematti/pycorr

git
---
First::

  git clone https://github.com/adematti/pycorr.git

To install the code::

  python setup.py install --user

Or in development mode (any change to Python code will take place immediately)::

  python setup.py develop --user
