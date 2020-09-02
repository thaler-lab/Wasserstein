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

cxxflags = ['-fopenmp', '-std=c++14', '-Iwasserstein']
ldflags = ['-fopenmp']
libs = []
if platform.system() == 'Darwin':
    cxxflags.insert(0, '-Xpreprocessor')
    del ldflags[0]
    libs = ['omp']
elif platform.system() == 'Windows':
    cxxflags[0] = ldflags[0] = '/openmp'
    cxxflags[1] = '/std:c++14'
    del ldflags[0]

wasserstein = Extension('wasserstein._wasserstein',
                        sources=[os.path.join('wasserstein', 'wasserstein.cpp')],
                        include_dirs=[np.get_include(), '.'],
                        extra_compile_args=cxxflags,
                        extra_link_args=ldflags,
                        libraries=libs)

if sys.argv[1] == 'swig':
    opts = '-fastproxy -w511 -keyword'
    if len(sys.argv) >= 3 and sys.argv[2] == '-py3':
        opts += ' -py3'
    command = 'swig -python -c++ {} -Iwasserstein -o wasserstein/wasserstein.cpp swig/wasserstein.i'.format(opts)
    print(command)
    subprocess.run(command.split())

else:
    setup(
        ext_modules=[wasserstein],
        version=__version__
    )