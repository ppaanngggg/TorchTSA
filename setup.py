from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(
    name='TorchTSA',
    version='0.0.1',
    author='hantian.pang',
    packages=[
        'TorchTSA',
        'TorchTSA/model',
        'TorchTSA/simulate',
        'TorchTSA/utils',
    ],
    ext_modules=cythonize([
        'TorchTSA/utils/recursions.pyx',
    ]),
    include_dirs=[
        numpy.get_include()
    ],
    install_requires=[
        'numpy', 'scipy',
    ]
)
