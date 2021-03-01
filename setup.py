#------------------------------------------------------------------------
# This file is part of Wasserstein, a C++ library with a Python wrapper
# that computes the Wasserstein/EMD distance. If you use it for academic
# research, please cite or acknowledge the following works:
#
#   - Komiske, Metodiev, Thaler (2019) arXiv:1902.02346
#       https://doi.org/10.1103/PhysRevLett.123.041801
#   - Komiske, Metodiev, Thaler (2020) arXiv:2004.04159
#       https://doi.org/10.1007/JHEP07%282020%29006
#   - Boneel, van de Panne, Paris, Heidrich (2011)
#       https://doi.org/10.1145/2070781.2024192
#   - LEMON graph library https://lemon.cs.elte.hu/trac/lemon
#
# Copyright (C) 2019-2021 Patrick T. Komiske III
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#------------------------------------------------------------------------

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
ldflags = []
libs = []

# MaxOS
if platform.system() == 'Darwin':
    cxxflags.insert(0, '-Xpreprocessor')
    libs.append('omp')

# Linux
elif platform.system() == 'Linux':
    ldflags.append('-fopenmp')

# Windows
elif platform.system() == 'Windows':
    cxxflags = ['/openmp', '/std:c++14']
    ldflags = ['/openmp']

if sys.argv[1] == 'swig':
    opts = '-fastproxy -w511 -keyword'
    if len(sys.argv) >= 3 and sys.argv[2] == '-py3':
        opts += ' -py3'
    command = 'swig -python -c++ {} -o wasserstein/wasserstein.cpp wasserstein/swig/wasserstein.i'.format(opts)
    print(command)
    subprocess.run(command.split())

else:
    wasserstein = Extension('wasserstein._wasserstein',
                            sources=[os.path.join('wasserstein', 'wasserstein.cpp')],
                            include_dirs=[np.get_include(), '.'],
                            extra_compile_args=cxxflags,
                            extra_link_args=ldflags,
                            libraries=libs)

    setup(
        ext_modules=[wasserstein],
        version=__version__
    )