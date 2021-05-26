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

/*  ______ __  __ _____  _    _ _______ _____ _       _____ 
 * |  ____|  \/  |  __ \| |  | |__   __|_   _| |     / ____|
 * | |__  | \  / | |  | | |  | |  | |    | | | |    | (___  
 * |  __| | |\/| | |  | | |  | |  | |    | | | |     \___ \
 * | |____| |  | | |__| | |__| |  | |   _| |_| |____ ____) |
 * |______|_|  |_|_____/ \____/   |_|  |_____|______|_____/
 */

#ifndef WASSERSTEIN_EMDUTILS_HH
#define WASSERSTEIN_EMDUTILS_HH

// C++ standard library
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

// namespace macros
#ifndef BEGIN_EMD_NAMESPACE
#define EMDNAMESPACE emd
#define BEGIN_EMD_NAMESPACE namespace EMDNAMESPACE {
#define END_EMD_NAMESPACE }
#endif

// fastjet macros
#ifdef __FASTJET_PSEUDOJET_HH__
#define WASSERSTEIN_FASTJET
#endif

// parse default types
#ifndef WASSERSTEIN_DEFAULT_VALUE_TYPE
#define WASSERSTEIN_DEFAULT_VALUE_TYPE double
#endif

#ifndef WASSERSTEIN_INDEX_TYPE
#define WASSERSTEIN_INDEX_TYPE std::ptrdiff_t
#endif


BEGIN_EMD_NAMESPACE

using default_value_type = WASSERSTEIN_DEFAULT_VALUE_TYPE;
using index_type = WASSERSTEIN_INDEX_TYPE;


////////////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////////////

const double PI = 3.14159265358979323846;
const double TWOPI = 2*PI;


////////////////////////////////////////////////////////////////////////////////
// Enums
////////////////////////////////////////////////////////////////////////////////

enum class EMDStatus : char { 
  Success = 0,
  Empty = 1,
  SupplyMismatch = 2,
  Unbounded = 3,
  MaxIterReached = 4,
  Infeasible = 5
};
enum class ExtraParticle : char { Neither = -1, Zero = 0, One = 1 };
enum class EMDPairsStorage : char { Full = 0, FullSymmetric = 1, FlattenedSymmetric = 2, External = 3 };


////////////////////////////////////////////////////////////////////////////////
// Base classes
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
class EMDBase;

template<class WeightCollection, class ParticleCollection>
struct EventBase;

template <class PairwiseDistance, class ParticleCollection, typename Value>
class PairwiseDistanceBase;


////////////////////////////////////////////////////////////////////////////////
// NetworkSimplex
////////////////////////////////////////////////////////////////////////////////

template<typename Value, typename Arc, typename Node, typename Bool>
class NetworkSimplex;

template<typename Value>
using DefaultNetworkSimplex = NetworkSimplex<Value, index_type, int, char>;


////////////////////////////////////////////////////////////////////////////////
// EuclideanParticle classes
////////////////////////////////////////////////////////////////////////////////

template<unsigned N, typename Value = default_value_type>
struct EuclideanParticleND;

template<typename Value = default_value_type>
struct EuclideanParticle2D;

template<typename Value = default_value_type>
struct EuclideanParticle3D;


////////////////////////////////////////////////////////////////////////////////
// PairwiseDistance classes
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
class DefaultPairwiseDistance;

template<typename Value>
class EuclideanArrayDistance;

template<typename Value>
class YPhiArrayDistance;

template<class Particle>
class EuclideanParticleDistance;

template<typename Value>
class YPhiParticleDistance;

template<typename Value = default_value_type>
using EuclideanDistance2D = EuclideanParticleDistance<EuclideanParticle2D<Value>>;

template<typename Value = default_value_type>
using EuclideanDistance3D = EuclideanParticleDistance<EuclideanParticle3D<Value>>;


////////////////////////////////////////////////////////////////////////////////
// [Weight/Particle]Collection classes
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
struct ArrayWeightCollection;

template<typename Value>
struct ArrayParticleCollection;

template<typename Value>
struct Array2ParticleCollection;


////////////////////////////////////////////////////////////////////////////////
// Event classes
////////////////////////////////////////////////////////////////////////////////

// Event using Value vectors to hold weights and particles
template<typename Value = default_value_type>
struct VectorEvent;

template<typename Value>
using DefaultEvent = VectorEvent<Value>;

// Event containing weights and particles as smart contiguous arrays
template<typename Value, template<typename> class ParticleCollection>
struct ArrayEvent;

template<typename Value>
using DefaultArrayEvent = ArrayEvent<Value, ArrayParticleCollection>;

template<typename Value>
using DefaultArray2Event = ArrayEvent<Value, Array2ParticleCollection>;

// Event composed of EuclideanParticle
template<class Particle>
struct EuclideanParticleEvent;

template<typename Value = default_value_type>
using EuclideanEvent2D = EuclideanParticleEvent<EuclideanParticle2D<Value>>;

template<typename Value = default_value_type>
using EuclideanEvent3D = EuclideanParticleEvent<EuclideanParticle3D<Value>>;

struct FastJetEventBase;
struct FastJetParticleWeight;

template<class ParticleWeight>
struct FastJetEvent;


////////////////////////////////////////////////////////////////////////////////
// ExternalEMDHandler classes
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
class ExternalEMDHandler;

template<class Transform, typename Value>
class Histogram1DHandler;

template<typename Value = default_value_type>
class CorrelationDimension;


////////////////////////////////////////////////////////////////////////////////
// Preprocessor classes
////////////////////////////////////////////////////////////////////////////////

template<class EMD>
class Preprocessor;

template<class EMD>
class CenterWeightedCentroid;


////////////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////////////

// function that raises appropriate error from a status code
inline void check_emd_status(EMDStatus status) {
  if (status != EMDStatus::Success)
    switch (status) {
      case EMDStatus::Empty:
        throw std::runtime_error("EMDStatus - Empty");
        break;
      case EMDStatus::SupplyMismatch:
        throw std::runtime_error("EMDStatus - SupplyMismatch, consider increasing epsilon_large_factor");
        break;
      case EMDStatus::Unbounded:
        throw std::runtime_error("EMDStatus - Unbounded");
        break;
      case EMDStatus::MaxIterReached:
        throw std::runtime_error("EMDStatus - MaxIterReached, consider increasing n_iter_max");
        break;
      case EMDStatus::Infeasible:
        throw std::runtime_error("EMDStatus - Infeasible");
        break;
      default:
        throw std::runtime_error("EMDStatus - Unknown");
    }
}

// helps with freeing memory
template<typename T>
void free_vector(std::vector<T> & vec) {
  std::vector<T>().swap(vec);
}


////////////////////////////////////////////////////////////////////////////////
// Preprocessor - base class for preprocessing operations
////////////////////////////////////////////////////////////////////////////////

// base class for preprocessing events
template<class EMD>
class Preprocessor {
public:

  typedef typename EMD::Event Event;

  virtual ~Preprocessor() {}

  // returns description
  virtual std::string description() const { return "Preprocessor"; };

  // call this preprocessor on event
  virtual Event & operator()(Event & event) const { return event; };

}; // Preprocessor

END_EMD_NAMESPACE

#endif // WASSERSTEIN_EMDUTILS_HH
