from setuptools import setup


setup(name='pycorr',
      version='0.0.1',
      author='Arnaud de Mattia',
      author_email='',
      description='Estimation of correlation functions',
      license='GPL3',
      url='http://github.com/adematti/pycorr',
      install_requires=['numpy', 'scipy', 'Corrfunc'],
      packages=['pycorr']
)
