//------------------------------------------------------------------------
// This file is part of Wasserstein, a C++ library with a Python wrapper
// that computes the Wasserstein/EMD distance. If you use it for academic
// research, please cite or acknowledge the following works:
//
//   - Komiske, Metodiev, Thaler (2019) arXiv:1902.02346
//       https://doi.org/10.1103/PhysRevLett.123.041801
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

#ifndef EVENTGEOMETRY_EVENT_HH
#define EVENTGEOMETRY_EVENT_HH

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "wasserstein/internal/EMDUtils.hh"

#ifdef __FASTJET_PSEUDOJET_HH__
FASTJET_BEGIN_NAMESPACE
namespace contrib {
#else
namespace emd {
#endif

// base class for "events", associates weights with a particle collection
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
      throw std::runtime_error("Weights must be set prior to calling normalize_weights.");

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

class FastJetEventBase {};

#ifdef __FASTJET_PSEUDOJET_HH__

// base class to use for checking types later
struct FastJetParticleWeight {};

// use pT as weight, most typical choice for hadronic colliders
struct TransverseMomentum : FastJetParticleWeight {
  static const char * name() { return "TransverseMomentum"; }
  static double weight(const PseudoJet & pj) { return pj.pt(); }
};

// use ET as weight, typical for hadronic colliders if mass is relevant
struct TransverseEnergy : FastJetParticleWeight {
  static const char * name() { return "TransverseEnergy"; }
  static double weight(const PseudoJet & pj) { return pj.Et(); }
};

// use |p3| as weight, typical of e+e- colliders treating pjs as massless
struct Momentum : FastJetParticleWeight {
  static const char * name() { return "Momentum"; }
  static double weight(const PseudoJet & pj) { return pj.modp(); }
};

// use E as weight, typical of e+e- colliders
struct Energy : FastJetParticleWeight {
  static const char * name() { return "Energy"; }
  static double weight(const PseudoJet & pj) { return pj.E(); }
};

// FastJetEvent class
template<class PW>
struct FastJetEvent : public EventBase<std::vector<PseudoJet>, std::vector<double>>,
                      public FastJetEventBase {
  typedef PW ParticleWeight;
  typedef std::vector<PseudoJet> ParticleCollection;
  typedef std::vector<double> WeightCollection;

  // constructor from PseudoJet, possibly with constituents
  FastJetEvent(const PseudoJet & pj) :
    EventBase<ParticleCollection, WeightCollection>(pj.has_constituents() ? pj.constituents() : ParticleCollection{pj}),
    axis_(pj)
  {}

  // constructor from vector of PseudoJets
  FastJetEvent(const ParticleCollection & pjs) :
    EventBase<ParticleCollection, WeightCollection>(pjs)
  {}

  FastJetEvent() {}

  // name of event
  static std::string name() {
    std::ostringstream oss;
    oss << "FastJetEvent<" << ParticleWeight::name() << '>';
    return oss.str();
  }

  // determine weights
  void ensure_weights() {
    if (!has_weights_) {
      weights_.reserve(particles_.size());
      for (const PseudoJet & pj : particles_) {
        weights_.push_back(ParticleWeight::weight(pj));
        total_weight_ += weights_.back();
      }
      has_weights_ = true;
    }
  }

  // access/set PseudoJet
  PseudoJet & axis() { return axis_; }

private:

  // hold original PseudoJet if given one
  PseudoJet axis_;

}; // FastJetEvent

#endif // __FASTJET_PSEUDOJET_HH__

// generic event contains a vector of particles
template<class P>
struct GenericEvent : public EventBase<std::vector<P>, std::vector<typename P::Value>> {
  typedef std::vector<P> ParticleCollection;
  typedef std::vector<typename P::Value> WeightCollection;

  GenericEvent(const ParticleCollection & particles) :
    EventBase<ParticleCollection, WeightCollection>(particles)
  {
    // weights are assumed to be contained in the particles as public properties
    this->weights_.reserve(particles.size());
    for (const P & particle : particles) {
      this->total_weight_ += particle.weight;
      this->weights_.push_back(particle.weight);
    }
    this->has_weights_ = true;
  }

  GenericEvent() {}

  static std::string name() {
    std::ostringstream oss;
    oss << "GenericEvent<" << P::name() << '>';
    return oss.str();
  }
}; // GenericEvent

// define some double-precision euclidean events
using EuclideanEvent2D = GenericEvent<EuclideanParticle2D<>>;
using EuclideanEvent3D = GenericEvent<EuclideanParticle3D<>>;
template<unsigned int N>
using EuclideanEventND = GenericEvent<EuclideanParticleND<N>>;

// event that wraps plain arrays of particles and weights
template<typename V = double>
struct ArrayEvent : public EventBase<ArrayParticleCollection<V>, ArrayWeightCollection<V>> {
  typedef ArrayParticleCollection<V> ParticleCollection;
  typedef ArrayWeightCollection<V> WeightCollection;

  static_assert(std::is_floating_point<V>::value, "ArrayEvent template parameter must be floating point.");

  ArrayEvent(V * particle_array, V * weight_array, int size, int stride) :
    EventBase<ParticleCollection, WeightCollection>(ParticleCollection(particle_array, size, stride),
                                                    WeightCollection(weight_array, size))
  {
    // set total weight
    for (int i = 0; i < size; i++)
      this->total_weight_ += weight_array[i];
    this->has_weights_ = true;
  }

  ArrayEvent(const std::tuple<V*, V*, int, int> & tup) :
    ArrayEvent(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup), std::get<3>(tup))
  {}

  ArrayEvent() {}

  static std::string name() {
    std::ostringstream oss;
    oss << "ArrayEvent<" << sizeof(V) << "-byte float>";
    return oss.str();
  }

}; // ArrayEvent

#ifdef __FASTJET_PSEUDOJET_HH__
} // namespace contrib
FASTJET_END_NAMESPACE
#else
} // namespace emd
#endif

#endif // EVENTGEOMETRY_EVENT_HH
