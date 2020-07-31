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

omp_compile_args = ['-fopenmp']
omp_link_args = ['-fopenmp']
extra_objects = []
if platform.system() == 'Darwin':
    omp_compile_args.insert(0, '-Xpreprocessor')
    omp_link_args = ['-lomp']
    #extra_objects = ['/usr/local/lib/libomp.a']

wasserstein = Extension('wasserstein._wasserstein',
                        sources=['wasserstein/wasserstein.cpp'],
                        include_dirs=[np.get_include(), '.'],
                        extra_compile_args=['-std=c++14'] + omp_compile_args,
                        extra_objects=extra_objects,
                        extra_link_args=omp_link_args
                        )

if sys.argv[1] == 'swig':
    command = 'swig -python -c++ -fastproxy -py3 -w511 -keyword -o wasserstein/wasserstein.cpp swig/wasserstein.i'
    print(command)
    subprocess.run(command.split())
    sys.exit()

setup(
    ext_modules=[wasserstein],
    version=__version__
)