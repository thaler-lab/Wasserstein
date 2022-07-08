r"""

$$\      $$\                                                               $$\               $$\           
$$ | $\  $$ |                                                              $$ |              \__|          
$$ |$$$\ $$ | $$$$$$\   $$$$$$$\  $$$$$$$\  $$$$$$\   $$$$$$\   $$$$$$$\ $$$$$$\    $$$$$$\  $$\ $$$$$$$\  
$$ $$ $$\$$ | \____$$\ $$  _____|$$  _____|$$  __$$\ $$  __$$\ $$  _____|\_$$  _|  $$  __$$\ $$ |$$  __$$\ 
$$$$  _$$$$ | $$$$$$$ |\$$$$$$\  \$$$$$$\  $$$$$$$$ |$$ |  \__|\$$$$$$\    $$ |    $$$$$$$$ |$$ |$$ |  $$ |
$$$  / \$$$ |$$  __$$ | \____$$\  \____$$\ $$   ____|$$ |       \____$$\   $$ |$$\ $$   ____|$$ |$$ |  $$ |
$$  /   \$$ |\$$$$$$$ |$$$$$$$  |$$$$$$$  |\$$$$$$$\ $$ |      $$$$$$$  |  \$$$$  |\$$$$$$$\ $$ |$$ |  $$ |
\__/     \__| \_______|\_______/ \_______/  \_______|\__|      \_______/    \____/  \_______|\__|\__|  \__|

------------------------------------------------------------------------
 This file is part of Wasserstein, a C++ library with a Python wrapper
 that computes the Wasserstein/EMD distance. If you use it for academic
 research, please cite or acknowledge the following works:

   - Komiske, Metodiev, Thaler (2019) arXiv:1902.02346
       https://doi.org/10.1103/PhysRevLett.123.041801
   - Boneel, van de Panne, Paris, Heidrich (2011)
       https://doi.org/10.1145/2070781.2024192
   - LEMON graph library https://lemon.cs.elte.hu/trac/lemon

 Copyright (C) 2019-2022 Patrick T. Komiske III
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
------------------------------------------------------------------------
"""

# basic package info
__author__  = 'Patrick T. Komiske III'
__email__   = 'pkomiske@gmail.com'
__license__ = 'GPLv3'
__version__ = '1.1.0'

from . import config
from .config import *

# these are all attributes of wasserstein submodule that should be top-level visible
__all__ = [

    # primary functionality with variable dtype
    'EMD', 'EMDYPhi',
    'PairwiseEMD', 'PairwiseEMDYPhi',
    'CorrelationDimension',

    # EMDStatus enum constants
    'EMDStatus_Success',
    'EMDStatus_Empty',
    'EMDStatus_SupplyMismatch',
    'EMDStatus_Unbounded',
    'EMDStatus_MaxIterReached',
    'EMDStatus_Infeasible',

    # ExtraParticle enum constants
    'ExtraParticle_Neither',
    'ExtraParticle_Zero',
    'ExtraParticle_One',

    # EMDPairsStorage enum constants
    'EMDPairsStorage_Full',
    'EMDPairsStorage_FullSymmetric',
    'EMDPairsStorage_FlattenedSymmetric',
    'EMDPairsStorage_External',

    # other functions
    'check_emd_status',

    # classes with fixed dtype
    'EMDFloat64', 'EMDFloat32',
    'EMDYPhiFloat64', 'EMDYPhiFloat32',
    'PairwiseEMDFloat64', 'PairwiseEMDFloat32',
    'PairwiseEMDYPhiFloat64', 'PairwiseEMDYPhiFloat32',
    'ExternalEMDHandlerFloat64', 'ExternalEMDHandlerFloat32',
    'Histogram1DHandlerLogFloat64', 'Histogram1DHandlerLogFloat32',
    'Histogram1DHandlerFloat64', 'Histogram1DHandlerFloat32',
    'CorrelationDimensionFloat64', 'CorrelationDimensionFloat32',
] + config.__all__

# enables lazy importing of wasserstein submodule
def __getattr__(name):
    if name in __all__:
        from . import wasserstein
        attr = getattr(wasserstein, name)
        globals()[name] = attr
        return attr

    # handle specially grabbing the submodule
    elif name == 'wasserstein':
        import importlib
        wasserstein = importlib.import_module('.wasserstein', __name__)
        globals()['wasserstein'] = wasserstein
        return wasserstein

    raise AttributeError(f'module `{__name__}` has no attribute `{name}`')

# properly lists everything available in this module
def __dir__():
    return __all__ + [
        'wasserstein',
        'config',
        '__author__',
        '__email__',
        '__license__',
        '__version__',
        '__doc__',
        '__name__',
        '__package__'
    ]
