import os
import sys
import sysconfig
from setuptools import setup, Extension

import numpy as np

# base directory of package
package_basedir = os.path.abspath(os.path.dirname(__file__))
package_basename = 'pycorr'

sys.path.insert(0, os.path.join(package_basedir, package_basename))
import _version
version = _version.__version__


def find_compiler():
    compiler = os.getenv('CC', None)
    if compiler is None:
        compiler = sysconfig.get_config_vars().get('CC', None)
    return compiler


def get_flags():
    compiler = find_compiler()
    os.environ.setdefault('CC', compiler)
    if compiler == 'clang':
        flags = ['-Xclang', '-fopenmp', '-lomp']
    elif compiler in ['cc', 'icc']:
        flags = ['-fopenmp', '-lgomp', '-limf', '-liomp5']
    else:
        flags = ['-fopenmp', '-lgomp']
    flags += ['-DUSE_OMP']
    return flags


if __name__ == '__main__':

    flags = get_flags()
    setup(name=package_basename,
          version=version,
          author='cosmodesi',
          author_email='',
          description='Estimation of correlation functions',
          license='BSD3',
          url='http://github.com/cosmodesi/pycorr',
          install_requires=['numpy', 'scipy'],
          extras_require={'mpi': ['mpi4py', 'pmesh'], 'jackknife': ['scikit-learn', 'healpy'], 'corrfunc': ['Corrfunc @ git+https://github.com/adematti/Corrfunc@desi']},
          ext_modules=[Extension(f'{package_basename}._utils', [f'{package_basename}/_utils.pyx'],
                       depends=[f'{package_basename}/_utils_imp.h', f'{package_basename}/_utils_generics.h'],
                       libraries=['m'],
                       include_dirs=['./', np.get_include()],
                       extra_compile_args=flags,
                       extra_link_args=flags)],
          packages=[package_basename])
