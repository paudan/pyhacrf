from setuptools import setup, Extension
import numpy
import numpy.distutils

try:
    from Cython.Build import cythonize
    use_cython = True
except ImportError:
    use_cython = False

if use_cython:
    ext_modules = cythonize([Extension('pyhacrf.adjacent',
                                       ['pyhacrf/adjacent.pyx'])])
else:
    ext_modules = [Extension('pyhacrf.algorithms',
                             ['pyhacrf/algorithms.c'],
                             extra_compile_args = ["-ffast-math", "-O4"],
                             **numpy.distutils.misc_util.get_info('npymath')),
                   Extension('pyhacrf.adjacent',
                             ['pyhacrf/adjacent.c'],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args = ["-ffast-math", "-O4"])]


def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='pyhacrf-datamade',
    version='0.2.3',
    packages=['pyhacrf'],
    install_requires=["numpy>=1.10.4; python_version<'3.6'",
                      "numpy>=1.12.1; python_version=='3.6'",
                      "numpy>=1.15.0; python_version=='3.7'",
                      'PyLBFGS>=0.1.3'],
    ext_modules=ext_modules,
    url='https://github.com/datamade/pyhacrf',
    author='Dirko Coetsee',
    author_email='dpcoetsee@gmail.com',
    maintainer='Forest Gregg',
    maintainer_email='fgregg@gmail.com',
    description='Hidden alignment conditional random field, a discriminative string edit distance',
    long_description=readme(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        ],
    )
