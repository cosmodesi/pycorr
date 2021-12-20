from setuptools import setup


setup(name='pycorr',
      version='0.0.1',
      author='cosmodesi',
      author_email='',
      description='Estimation of correlation functions',
      license='BSD3',
      url='http://github.com/cosmodesi/pycorr',
      install_requires=['numpy', 'scipy'],
      extras_require={'mpi':['mpi4py','pmesh'], 'corrfunc':['Corrfunc @ git+https://github.com/adematti/Corrfunc@pipweights'], 'jacknife':['scikit-learn']},
      packages=['pycorr']
)
