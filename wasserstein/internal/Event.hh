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

#ifndef WASSERSTEIN_EVENT_HH
#define WASSERSTEIN_EVENT_HH

// C++ standard library
#include <cassert>
#include <cmath>
#include <tuple>

#include "EMDUtils.hh"

BEGIN_EMD_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// EventBase - "events" constitute a weighted collection of "particles"
////////////////////////////////////////////////////////////////////////////////

template<class ParticleCollection, class WeightCollection>
struct EventBase {
  typedef typename WeightCollection::value_type Value;

  // constructor from particle collection only
  EventBase(const ParticleCollection & particles) :
    particles_(particles), total_weight_(0), has_weights_(false)
  {}

  // constructor from particle collection and weight collection
  EventBase(const ParticleCollection & particles, const WeightCollection & weights) :
    particles_(particles), weights_(weights), total_weight_(0), has_weights_(true)
  {}

  EventBase() {}
  virtual ~EventBase() {}

  // access particles
  ParticleCollection & particles() { return particles_; }
  const ParticleCollection & particles() const { return particles_; }

  // access weights
  WeightCollection & weights() { return weights_; }
  const WeightCollection & weights() const { return weights_; }
  Value & total_weight() { return total_weight_; }
  const Value & total_weight() const { return total_weight_; }
  bool has_weights() const { return has_weights_; }

  // overloaded by FastJetEvent
  void ensure_weights() { assert(has_weights_); }

  // normalize weights and total
  void normalize_weights() {
    if (!has_weights_)
      throw std::logic_error("Weights must be set prior to calling normalize_weights.");

    // normalize each weight
    Value norm_total(0);
    for (Value & w : weights_)
      norm_total += (w /= total_weight_);
    total_weight_ = norm_total;
  }

protected:

  ParticleCollection particles_;
  WeightCollection weights_;
  Value total_weight_;
  bool has_weights_;

}; // EventBase

////////////////////////////////////////////////////////////////////////////////
// ArrayEvent - an event where the weights and particle are contiguous arrays
////////////////////////////////////////////////////////////////////////////////

template<typename V = double>
struct ArrayEvent : public EventBase<ArrayParticleCollection<V>, ArrayWeightCollection<V>> {
  typedef ArrayParticleCollection<V> ParticleCollection;
  typedef ArrayWeightCollection<V> WeightCollection;
  typedef EventBase<ParticleCollection, WeightCollection> Base;

  static_assert(std::is_floating_point<V>::value, "ArrayEvent template parameter must be floating point.");

  ArrayEvent(V * particle_array, V * weight_array, int size, int stride) :
    EventBase<ParticleCollection, WeightCollection>(ParticleCollection(particle_array, size, stride),
                                                    WeightCollection(weight_array, size))
  {
    // set total weight
    for (int i = 0; i < size; i++)
      this->total_weight_ += weight_array[i];
  }
  ArrayEvent(const std::tuple<V*, V*, int, int> & tup) :
    ArrayEvent(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup), std::get<3>(tup))
  {}
  ArrayEvent() {}

  // ensure that we don't modify original array
  void normalize_weights() {
    this->weights_.copy();
    Base::normalize_weights();
  }

  static std::string name() {
    std::ostringstream oss;
    oss << "ArrayEvent<" << sizeof(V) << "-byte float>";
    return oss.str();
  }

}; // ArrayEvent

////////////////////////////////////////////////////////////////////////////////
// GenericEvent - an event constructed from a vector of "particles"
////////////////////////////////////////////////////////////////////////////////

template<class P>
struct GenericEvent : public EventBase<std::vector<P>, std::vector<typename P::Value>> {
  typedef std::vector<P> ParticleCollection;
  typedef std::vector<typename P::Value> WeightCollection;

  GenericEvent() {}
  GenericEvent(const ParticleCollection & particles) :
    EventBase<ParticleCollection, WeightCollection>(particles)
  {
    // weights are assumed to be contained in the particles as public properties
    this->weights_.reserve(particles.size());
    for (const P & particle : particles) {
      this->total_weight_ += particle.weight;
      this->weights_.push_back(particle.weight);
    }
  }

  static std::string name() {
    std::ostringstream oss;
    oss << "GenericEvent<" << P::name() << '>';
    return oss.str();
  }

}; // GenericEvent

////////////////////////////////////////////////////////////////////////////////
// EuclideanEvent - consists of double-precision euclidean particles
////////////////////////////////////////////////////////////////////////////////

using EuclideanEvent2D = GenericEvent<EuclideanParticle2D<>>;
using EuclideanEvent3D = GenericEvent<EuclideanParticle3D<>>;
template<unsigned N>
using EuclideanEventND = GenericEvent<EuclideanParticleND<N>>;

END_EMD_NAMESPACE

#endif // WASSERSTEIN_EVENT_HH
