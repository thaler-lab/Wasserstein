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
# Copyright (C) 2019-2022 Patrick T. Komiske III
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

import fileinput
import os
import platform
import re
import subprocess
import sys

# generate swig wrapper
if sys.argv[1] == 'swig':
    command = ('swig -python -c++ -fastproxy -w511 -keyword -py3 '
               '-o wasserstein/wasserstein.cpp wasserstein/swig/wasserstein.i')
    print(command)
    subprocess.run(command.split())

    # patch up output
    with fileinput.input('wasserstein/wasserstein.cpp', inplace=True) as f:
        for line in f:
            if 'define' in line and 'SWIG_name' in line:
                sys.stdout.write('#define SWIG_name "#WASSERSTEIN_SWIG_NAME"')
            else:
                sys.stdout.write(line)

# compile python package
else:

    import numpy as np
    from setuptools import setup
    from setuptools.extension import Extension

    with open(os.path.join('wasserstein', '__init__.py'), 'r') as f:
        __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)

    macros = [('SWIG_TYPE_TABLE', 'wasserstein'),
              ('PyInit__wasserstein', 'PyInit__wasserstein_omp'),
              ('WASSERSTEIN_SWIG_NAME', '_wasserstein_omp')]
    includes = [np.get_include(), '.']
    cxxflags = ['-std=c++14', '-ffast-math',  '-g0']
    omp_cxxflags = ['-fopenmp']
    libs = []

    # MacOS
    if platform.system() == 'Darwin':
        omp_cxxflags.insert(0, '-Xpreprocessor')
        libs = ['omp']
        ldflags = ['-Wl,-rpath,/usr/local/lib']

    # Linux
    elif platform.system() == 'Linux':
        ldflags = ['-fopenmp']

    # Windows
    elif platform.system() == 'Windows':
        omp_cxxflags = ldflags = ['/openmp']
        cxxflags = ['/std:c++14', '/fp:fast']

    else:
        raise RuntimeError('{} not supported'.format(platform.system()))

    # wasserstein library with openmp
    exts = [Extension('wasserstein._wasserstein_omp',
                      sources=[os.path.join('wasserstein', 'wasserstein.cpp')],
                      define_macros=macros,
                      include_dirs=includes,
                      extra_compile_args=cxxflags + omp_cxxflags,
                      extra_link_args=ldflags,
                      libraries=libs)]

    # wasserstein library without openmp
    if platform.system() == 'Darwin':
        noomp_fpath = 'wasserstein/wasserstein_noomp.cpp'
        if not os.path.exists(noomp_fpath):
            os.symlink('wasserstein.cpp', noomp_fpath)
        exts.append(Extension('wasserstein._wasserstein_noomp',
                              sources=[noomp_fpath],
                              define_macros=[(x[0], x[1].replace('omp', 'noomp')) for x in macros],
                              include_dirs=includes,
                              extra_compile_args=cxxflags))

    setup(ext_modules=exts, version=__version__)
