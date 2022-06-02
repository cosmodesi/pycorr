import os
import sys
import sysconfig
import subprocess
import shutil
from distutils.command.build import build
from distutils.command.clean import clean
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.develop import develop
from setuptools import setup

# base directory of package
package_basedir = os.path.abspath(os.path.dirname(__file__))
package_basename = 'pycorr'

sys.path.insert(0, os.path.join(package_basedir, package_basename))
import _version
import utils
version = _version.__version__
lib_dir = utils.lib_dir
src_dir = os.path.join(package_basedir, 'src')


def find_compiler():
    compiler = os.getenv('CC', None)
    if compiler is None:
        compiler = sysconfig.get_config_vars().get('CC', None)
    return compiler


class custom_build(build):

    def run(self):
        super(custom_build, self).run()

        # lib_dir = os.path.join(os.path.abspath(self.build_lib),'pyrecon','lib')
        os.environ.setdefault('LIBDIR', lib_dir)
        library_dir = sysconfig.get_config_var('LIBDIR')

        compiler = find_compiler()
        os.environ.setdefault('CC', compiler)
        if compiler == 'clang':
            flags = '-Xclang -fopenmp -L{} -lomp'.format(library_dir)
        elif compiler in ['cc', 'icc']:
            flags = '-fopenmp -L{} -lgomp -limf -liomp5'.format(library_dir)
        else:
            flags = '-fopenmp -L{} -lgomp'.format(library_dir)
        os.environ.setdefault('OMPFLAG', flags)

        def compile():
            subprocess.call('make', shell=True, cwd=src_dir)

        self.execute(compile, [], 'Compiling')
        new_lib_dir = os.path.join(os.path.abspath(self.build_lib), package_basename, 'lib')
        shutil.rmtree(new_lib_dir, ignore_errors=True)
        shutil.copytree(lib_dir, new_lib_dir)


class custom_bdist_egg(bdist_egg):

    def run(self):
        self.run_command('build')
        super(custom_bdist_egg, self).run()


class custom_develop(develop):

    def run(self):
        self.run_command('build')
        super(custom_develop, self).run()


class custom_clean(clean):

    def run(self):
        # run the built-in clean
        super(custom_clean, self).run()
        # remove the recon products
        shutil.rmtree(lib_dir, ignore_errors=True)


if __name__ == '__main__':

    setup(name=package_basename,
          version=version,
          author='cosmodesi',
          author_email='',
          description='Estimation of correlation functions',
          license='BSD3',
          url='http://github.com/cosmodesi/pycorr',
          install_requires=['numpy', 'scipy'],
          extras_require={'mpi': ['mpi4py', 'pmesh'], 'jackknife': ['scikit-learn', 'healpy'], 'corrfunc': ['Corrfunc @ git+https://github.com/adematti/Corrfunc@desi']},
          cmdclass={
              'build': custom_build,
              'develop': custom_develop,
              'bdist_egg': custom_bdist_egg,
              'clean': custom_clean
          },
          packages=[package_basename])
