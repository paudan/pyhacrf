from setuptools import setup, Extension
import numpy

# Get the long description from the relevant file
with open('README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='pyhacrf-datamade',
    version='0.2.1',
    packages=['pyhacrf'],
    install_requires=['numpy>=1.9', 'PyLBFGS>=0.1.3'],
    ext_modules=[Extension('pyhacrf.algorithms',
                           ['pyhacrf/algorithms.c'],
                           include_dirs=numpy.get_include(),
                           extra_compile_args = ["-ffast-math", "-O4"]),
                 Extension('pyhacrf.adjacent',
                           ['pyhacrf/adjacent.c'],
                           include_diers=numpy.get_include(),
                           extra_compile_args = ["-ffast-math", "-O4"])],
    url='https://github.com/datamade/pyhacrf',
    author='Dirko Coetsee',
    author_email='dpcoetsee@gmail.com',
    maintainer='Forest Gregg',
    maintiner_email='fgregg@gmail.com',
    description='Hidden alignment conditional random field, a discriminative string edit distance',
    long_description=long_description,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        ],
    )
