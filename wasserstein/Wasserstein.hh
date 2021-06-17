//------------------------------------------------------------------------
// This file is part of Wasserstein, a C++ library with a Python wrapper
// that computes the Wasserstein/EMD distance. If you use it for academic
// research, please cite or acknowledge the following works:
//
//   - Komiske, Metodiev, Thaler (2019) arXiv:1902.02346
//       https://doi.org/10.1103/PhysRevLett.123.041801
//   - Komiske, Metodiev, Thaler (2020) arXiv:2004.04159
//       https://doi.org/10.1007/JHEP07%282020%29006
//   - Boneel, van de Panne, Paris, Heidrich (2011)
//       https://doi.org/10.1145/2070781.2024192
//   - LEMON graph library https://lemon.cs.elte.hu/trac/lemon
//
// Copyright (C) 2019-2021 Patrick T. Komiske III
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//------------------------------------------------------------------------

/* __          __      _____ _____ ______ _____   _____ _______ ______ _____ _   _ 
 * \ \        / /\    / ____/ ____|  ____|  __ \ / ____|__   __|  ____|_   _| \ | |
 *  \ \  /\  / /  \  | (___| (___ | |__  | |__) | (___    | |  | |__    | | |  \| |
 *   \ \/  \/ / /\ \  \___ \\___ \|  __| |  _  / \___ \   | |  |  __|   | | | . ` |
 *    \  /\  / ____ \ ____) |___) | |____| | \ \ ____) |  | |  | |____ _| |_| |\  |
 *     \/  \/_/    \_\_____/_____/|______|_|  \_\_____/   |_|  |______|_____|_| \_|
 */

#ifndef WASSERSTEIN_HH
#define WASSERSTEIN_HH

#include "internal/CenterWeightedCentroid.hh"
#include "internal/CorrelationDimension.hh"
#include "internal/EMD.hh"
#include "internal/Event.hh"
#include "internal/NetworkSimplex.hh"
#include "internal/PairwiseDistance.hh"
#include "internal/PairwiseEMD.hh"


BEGIN_WASSERSTEIN_NAMESPACE

#ifdef DECLARE_WASSERSTEIN_TEMPLATES
  WASSERSTEIN_TEMPLATES
#endif

// EMD using double precision
template<template<typename> class Event = DefaultEvent,
         template<typename> class PairwiseDistance = DefaultPairwiseDistance>
using EMDFloat64 = EMD<double, Event, PairwiseDistance>;

// EMD using single precision
template<template<typename> class Event = DefaultEvent,
         template<typename> class PairwiseDistance = DefaultPairwiseDistance>
using EMDFloat32 = EMD<float, Event, PairwiseDistance>;

END_WASSERSTEIN_NAMESPACE

#endif // WASSERSTEIN_HH
