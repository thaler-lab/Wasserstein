import os
import platform
import re
import subprocess
import sys

from setuptools import setup
from setuptools.extension import Extension

import numpy as np

with open(os.path.join('wasserstein', '__init__.py'), 'r') as f:
    __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)

cxxflags = ['-fopenmp', '-std=c++14']
ldflags = ['-fopenmp']
libs = []
if platform.system() == 'Darwin':
    cxxflags.insert(0, '-Xpreprocessor')
    del ldflags[0]
    libs = ['omp']

wasserstein = Extension('wasserstein._wasserstein',
                        sources=['wasserstein/wasserstein.cpp'],
                        include_dirs=[np.get_include(), '.'],
                        extra_compile_args=cxxflags,
                        extra_link_args=ldflags,
                        libraries=libs)

if sys.argv[1] == 'swig':
    command = 'swig -python -c++ -fastproxy -py3 -w511 -keyword -o wasserstein/wasserstein.cpp swig/wasserstein.i'
    print(command)
    subprocess.run(command.split())
    sys.exit()

setup(
    ext_modules=[wasserstein],
    version=__version__
)