from setuptools import setup


setup(name='pycorr',
      version='1.0.0',
      author='cosmodesi',
      author_email='',
      description='Estimation of correlation functions',
      license='BSD3',
      url='http://github.com/cosmodesi/pycorr',
      install_requires=['numpy', 'scipy'],
      extras_require={'mpi':['mpi4py','pmesh'], 'corrfunc':['Corrfunc @ git+https://github.com/adematti/Corrfunc@desi'], 'jacknife':['scikit-learn']},
      packages=['pycorr']
)
