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

import wasserstein.config

if wasserstein.config.openmp():
	from ._wasserstein_omp import *
else:
	from ._wasserstein_noomp import *

# check that openmp selection is consistent
if wasserstein.config.openmp() != cvar.COMPILED_WITH_OPENMP:
	raise ImportError('OpenMP mismatch in compiled C-extension')

# can no longer change openmp preferences
wasserstein.config._CAN_SET_OPENMP = False
