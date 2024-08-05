#!/usr/bin/env python
"""
setup.py
    Setup definitions for distutils package creation
    Based on setup.py from https://github.com/KerstinKaspar/pypulseq-cest
"""

from setuptools import setup, Extension
import numpy
from pathlib import Path

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# defining paths to the desired files
eigen_path = Path('Eigen')
np_path = Path(numpy_include)
np_path_np = np_path / 'numpy'


BMCSimulator_module = Extension(name='_BMCSimulator',
                                sources=['BMCSimulator.i', 'BMCSim.cpp','BMCSimulator.cpp', 'SimulationParameters.cpp',
                                         'ExternalSequence.cpp'],
                                include_dirs=[eigen_path, np_path, np_path_np],
                                extra_compile_args=["-O2"],
                                swig_opts=['-c++'],
                                language='c++'
                                )

setup(name='BMCSimulator',
      version='0.2',
      author="N. Vladimirov",
      author_email='nikitav@mail.tau.ac.il',
      description="Python package to use the C++ code for pulseq-CEST simulations.",
      ext_modules=[BMCSimulator_module],
      py_modules=["BMCSimulator"],
      )



