import os
import sys
from setuptools import setup


package_basename = 'pycorr'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), package_basename))
import _version
version = _version.__version__


setup(name=package_basename,
      version=version,
      author='cosmodesi',
      author_email='',
      description='Estimation of correlation functions',
      license='BSD3',
      url='http://github.com/cosmodesi/pycorr',
      install_requires=['numpy', 'scipy'],
      extras_require={'mpi':['mpi4py', 'pmesh'], 'jackknife':['scikit-learn', 'healpy'], 'corrfunc':['Corrfunc @ git+https://github.com/adematti/Corrfunc@desi']},
      packages=['pycorr']
)
