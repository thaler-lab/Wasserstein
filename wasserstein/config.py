#------------------------------------------------------------------------
# This file is part of Wasserstein, a C++ library with a Python wrapper
# that computes the Wasserstein/EMD distance. If you use it for academic
# research, please cite or acknowledge the following works:
#
#   - Komiske, Metodiev, Thaler (2019) arXiv:1902.02346
#       https://doi.org/10.1103/PhysRevLett.123.041801
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

import platform
import warnings

__all__ = ['with_openmp', 'without_openmp', 'openmp']

# use openmp by default
_CAN_SET_OPENMP = _USE_WITH_OPENMP = True

def with_openmp():
    global _USE_WITH_OPENMP
    if _CAN_SET_OPENMP:
        _USE_WITH_OPENMP = True
    elif not _USE_WITH_OPENMP:
        raise RuntimeError('already loaded without OpenMP, cannot change now')

def without_openmp():
    if platform.system() != 'Darwin':
        #raise RuntimeError('cannot opt-out of OpenMP on {}'.format(platform.system()))
        warnings.warn('cannot opt-out of OpenMP on {}'.format(platform.system()))

    global _USE_WITH_OPENMP
    if _CAN_SET_OPENMP:
        _USE_WITH_OPENMP = False
    elif _USE_WITH_OPENMP:
        raise RuntimeError('already loaded with OpenMP, cannot change now')

def openmp():
    return _USE_WITH_OPENMP
