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

/*  ________      ________ _   _ _______ 
 * |  ____\ \    / /  ____| \ | |__   __|
 * | |__   \ \  / /| |__  |  \| |  | |   
 * |  __|   \ \/ / |  __| | . ` |  | |   
 * | |____   \  /  | |____| |\  |  | |   
 * |______|   \/   |______|_| \_|  |_|   
 */

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
// GenericEvent - an event constructed from a vector of "particles"
////////////////////////////////////////////////////////////////////////////////

template<class P>
struct GenericEvent : public EventBase<std::vector<P>, std::vector<typename P::Value>> {
  typedef P Particle;
  typedef std::vector<Particle> ParticleCollection;
  typedef std::vector<typename Particle::Value> WeightCollection;

  GenericEvent() {}
  GenericEvent(const ParticleCollection & particles) :
    EventBase<ParticleCollection, WeightCollection>(particles)
  {
    // weights are assumed to be contained in the particles as public methods
    this->weights_.reserve(particles.size());
    for (const P & particle : particles) {
      this->total_weight_ += particle.weight();
      this->weights_.push_back(particle.weight());
    }
    this->has_weights_ = true;
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

////////////////////////////////////////////////////////////////////////////////
// ArrayWeightCollection - implements a "smart" 1D array from a plain array
////////////////////////////////////////////////////////////////////////////////

template<typename V>
struct ArrayWeightCollection {
  typedef V value_type;

  // contructor, int is used for compatibility with SWIG's numpy.i
  ArrayWeightCollection(V * array, int size) : array_(array), size_(size), delete_(false) {}

  ~ArrayWeightCollection() {
    if (delete_)
      delete[] array_;
  }

  int size() const { return size_; }
  V * begin() { return array_; }
  V * end() { return array_ + size_; }
  const V * begin() const { return array_; }
  const V * end() const { return array_ + size_; }

  // copy internal memory
  void copy() {
    if (delete_)
      throw std::runtime_error("Should not call copy twice on an ArrayWeightCollection");
    delete_ = true;

    // get new chunk of memory
    //V * new_array((V*) malloc(nbytes));
    V * new_array(new V[size()]);

    // copy old array into new one
    memcpy(new_array, array_, size_t(size())*sizeof(V));
    array_ = new_array;
  }

private:
  V * array_;
  int size_;
  bool delete_;

}; // ArrayWeightCollection

////////////////////////////////////////////////////////////////////////////////
// ArrayParticleCollection - implements a "smart" 2D array from a plain array
////////////////////////////////////////////////////////////////////////////////

template<typename V>
struct ArrayParticleCollection {

  template<typename T>
  class templated_iterator {
    T * ptr_;
    int stride_;

  public:
    templated_iterator(T * ptr, int stride) : ptr_(ptr), stride_(stride) {}
    templated_iterator<T> & operator++() {
      ptr_ += stride_;
      return *this;
    }
    T * operator*() const { return ptr_; }
    bool operator!=(const templated_iterator & other) const { return ptr_ != other.ptr_; }
    int stride() const { return stride_; }
  };

  using const_iterator = templated_iterator<const V>;
  using value_type = const_iterator;

  // contructor, int is used for compatibility with SWIG's numpy.i
  ArrayParticleCollection(V * array, int size, int stride) :
    array_(array), size_(size), stride_(stride)
  {}

  int size() const { return size_; }
  int stride() const { return stride_; }
  const_iterator begin() const { return const_iterator(array_, stride_); }
  const_iterator end() const { return const_iterator(array_ + size_*stride_, stride_); }

private:
  V * array_;
  int size_, stride_;

}; // ArrayParticleCollection

////////////////////////////////////////////////////////////////////////////////
// ArrayEvent - an event where the weights and particle are contiguous arrays
////////////////////////////////////////////////////////////////////////////////

template<typename V = double>
struct ArrayEvent : public EventBase<ArrayParticleCollection<V>, ArrayWeightCollection<V>> {
  typedef ArrayParticleCollection<V> ParticleCollection;
  typedef ArrayWeightCollection<V> WeightCollection;
  typedef EventBase<ParticleCollection, WeightCollection> Base;

  static_assert(std::is_floating_point<V>::value, "ArrayEvent template parameter must be floating point.");

  // full constructor
  ArrayEvent(V * particle_array, V * weight_array, int size, int stride) :
    Base(ParticleCollection(particle_array, size, stride),
         WeightCollection(weight_array, size))
  {
    // set total weight
    for (int i = 0; i < size; i++)
      this->total_weight_ += weight_array[i];
  }

  // constructor from single argument (for use with Python)
  ArrayEvent(const std::tuple<V*, V*, int, int> & tup) :
    ArrayEvent(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup), std::get<3>(tup))
  {}

  // default constructor
  ArrayEvent() : ArrayEvent(nullptr, nullptr, 0, 0) {}

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
// VectorEvent - an event where the weights and particle are vectors
////////////////////////////////////////////////////////////////////////////////

template<typename V = double>
struct VectorEvent : public EventBase<std::vector<V>, std::vector<V>> {
  typedef std::vector<V> ParticleCollection;
  typedef std::vector<V> WeightCollection;
  typedef EventBase<ParticleCollection, WeightCollection> Base;

  static_assert(std::is_floating_point<V>::value, "VectorEvent template parameter must be floating point.");

  // constructor with a single argument
  VectorEvent(const std::pair<ParticleCollection, WeightCollection> && proto_event) :
    VectorEvent(proto_event.first, proto_event.second)
  {}

  // constructor from vectors of particles and weights
  VectorEvent(const ParticleCollection & particles, const WeightCollection & weights) :
    Base(particles, weights)
  {
    if (particles.size() % weights.size() != 0)
      throw std::invalid_argument("particles.size() must be cleanly divisible by weights.size()");

    // set total weight
    for (V weight : this->weights_)
      this->total_weight_ += weight;
  }

  // constructor from single vector of weights (for use with external dists)
  VectorEvent(const WeightCollection & weights) :
    VectorEvent(ParticleCollection(), weights)
  {}

  // default constructor
  VectorEvent() {}

  static std::string name() {
    std::ostringstream oss;
    oss << "VectorEvent<" << sizeof(V) << "-byte float>";
    return oss.str();
  }

}; // VectorEvent

END_EMD_NAMESPACE

#endif // WASSERSTEIN_EVENT_HH
