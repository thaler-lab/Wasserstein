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

  typedef typename WeightCollection::value_type value_type;

  // constructor from particle collection only
  EventBase(const ParticleCollection & particles, value_type event_weight = 1) :
    particles_(particles), total_weight_(0), event_weight_(event_weight), has_weights_(false)
  {}

  // constructor from particle collection and weight collection
  EventBase(const ParticleCollection & particles,
            const WeightCollection & weights,
            value_type event_weight = 1) :
    EventBase(particles, event_weight)
  {
    weights_ = weights;
    has_weights_ = true;
  }

  EventBase() {}
  virtual ~EventBase() {}

  // access event_weight
  value_type event_weight() const { return event_weight_; }

  // access particles
  ParticleCollection & particles() { return particles_; }
  const ParticleCollection & particles() const { return particles_; }

  // access weights
  WeightCollection & weights() { return weights_; }
  const WeightCollection & weights() const { return weights_; }
  value_type & total_weight() { return total_weight_; }
  const value_type & total_weight() const { return total_weight_; }
  bool has_weights() const { return has_weights_; }

  // overloaded by FastJetEvent
  void ensure_weights() { assert(has_weights_); }

  // normalize weights and total
  void normalize_weights() {
    if (!has_weights_)
      throw std::logic_error("Weights must be set prior to calling normalize_weights.");

    // normalize each weight
    value_type norm_total(0);
    for (value_type & w : weights_)
      norm_total += (w /= total_weight_);
    total_weight_ = norm_total;
  }

protected:

  ParticleCollection particles_;
  WeightCollection weights_;
  value_type total_weight_, event_weight_;
  bool has_weights_;

}; // EventBase


////////////////////////////////////////////////////////////////////////////////
// GenericEvent - an event constructed from a vector of "particles"
////////////////////////////////////////////////////////////////////////////////

template<class _Particle>
struct GenericEvent : public EventBase<std::vector<_Particle>,
                                       std::vector<typename _Particle::value_type>> {

  typedef _Particle Particle;
  typedef typename Particle::value_type value_type;
  typedef std::vector<Particle> ParticleCollection;
  typedef std::vector<value_type> WeightCollection;

  GenericEvent() {}
  GenericEvent(const ParticleCollection & particles, value_type event_weight = 1) :
    EventBase<ParticleCollection, WeightCollection>(particles, event_weight)
  {
    // weights are assumed to be contained in the particles as public methods
    this->weights_.reserve(particles.size());
    for (const Particle & particle : particles) {
      this->total_weight_ += particle.weight();
      this->weights_.push_back(particle.weight());
    }
    this->has_weights_ = true;
  }

  static std::string name() {
    std::ostringstream oss;
    oss << "GenericEvent<" << Particle::name() << '>';
    return oss.str();
  }

}; // GenericEvent


////////////////////////////////////////////////////////////////////////////////
// EuclideanEvent - consists of double-precision euclidean particles
////////////////////////////////////////////////////////////////////////////////

using EuclideanEvent2D = GenericEvent<EuclideanParticle2D<default_value_type>>;
using EuclideanEvent3D = GenericEvent<EuclideanParticle3D<default_value_type>>;
template<unsigned N>
using EuclideanEventND = GenericEvent<EuclideanParticleND<N, default_value_type>>;


////////////////////////////////////////////////////////////////////////////////
// ArrayWeightCollection - implements a "smart" 1D array from a plain array
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
struct ArrayWeightCollection {

  typedef Value value_type;

  // contructor, int is used for compatibility with SWIG's numpy.i
  ArrayWeightCollection(value_type * array, index_type size) : array_(array), size_(size), delete_(false) {}
  ArrayWeightCollection() : ArrayWeightCollection(nullptr, 0) {}

  ~ArrayWeightCollection() {
    if (delete_)
      delete[] array_;
  }

  index_type size() const { return size_; }
  value_type * begin() { return array_; }
  value_type * end() { return array_ + size_; }
  const value_type * begin() const { return array_; }
  const value_type * end() const { return array_ + size_; }

  // copy internal memory
  void copy() {
    if (delete_)
      throw std::runtime_error("Should not call copy twice on an ArrayWeightCollection");
    delete_ = true;

    // get new chunk of memory
    //V * new_array((V*) malloc(nbytes));
    value_type * new_array(new value_type[size()]);

    // copy old array into new one
    memcpy(new_array, array_, std::size_t(size())*sizeof(value_type));
    array_ = new_array;
  }

private:
  value_type * array_;
  index_type size_;
  bool delete_;

}; // ArrayWeightCollection


////////////////////////////////////////////////////////////////////////////////
// ArrayParticleCollectionBase - implements a "smart" 2D array from a plain array
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
struct ArrayParticleCollection {
protected:

  Value * array_;
  index_type size_, stride_;

  Value * array() const { return array_; }
  index_type stride() const { return stride_; }

  template<typename T>
  class templated_iterator {
  protected:
    T * ptr_;
    index_type stride_;

  public:
    templated_iterator(T * ptr, index_type stride) : ptr_(ptr), stride_(stride) {}
    templated_iterator<T> & operator++() {
      ptr_ += stride_;
      return *this;
    }
    T * operator*() const { return ptr_; }
    bool operator!=(const templated_iterator & other) const { return ptr_ != other.ptr_; }
    index_type stride() const { return stride_; }
  };

public:

  ArrayParticleCollection() : ArrayParticleCollection(nullptr, 0, 0) {}
  ArrayParticleCollection(Value * array, index_type size, index_type stride) :
    array_(array), size_(size), stride_(stride)
  {}

  index_type size() const { return size_; }
  static index_type stride_static() { return -1; }

  using const_iterator = templated_iterator<const Value>;
  typedef const_iterator value_type;

  const_iterator begin() const { return const_iterator(array(), stride()); }
  const_iterator end() const { return const_iterator(array() + size()*stride(), stride()); }

}; // ArrayParticleCollection


////////////////////////////////////////////////////////////////////////////////
// Array2ParticleCollection - implements a "smart" 2D array from a plain array
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
struct Array2ParticleCollection : public ArrayParticleCollection<Value> {
private:

  template<typename T>
  class templated_iterator : public ArrayParticleCollection<Value>::template templated_iterator<T> {
  public:
    templated_iterator(T * ptr) : ArrayParticleCollection<Value>::template templated_iterator<T>(ptr, 2) {}
    templated_iterator<T> & operator++() {
      this->ptr_ += 2;
      return *this;
    }
    index_type stride() const { return 2; }
  };

public:

  using ArrayParticleCollection<Value>::ArrayParticleCollection;
  using const_iterator = templated_iterator<const Value>;
  typedef const_iterator value_type;

  const_iterator begin() const { return const_iterator(this->array()); }
  const_iterator end() const { return const_iterator(this->array() + 2*this->size()); }
  static index_type stride_static() { return 2; }

}; // Array2ParticleCollection


////////////////////////////////////////////////////////////////////////////////
// ArrayEvent - an event where the weights and particle are contiguous arrays
////////////////////////////////////////////////////////////////////////////////

template<typename Value, template<typename> class _ParticleCollection>
struct ArrayEvent : public EventBase<_ParticleCollection<Value>, ArrayWeightCollection<Value>> {

  typedef Value value_type;
  typedef _ParticleCollection<Value> ParticleCollection;
  
  typedef ArrayWeightCollection<value_type> WeightCollection;
  typedef EventBase<ParticleCollection, WeightCollection> Base;

  // full constructor
  ArrayEvent(value_type * particle_array, value_type * weight_array,
             index_type size, index_type stride,
             value_type event_weight = 1) :
    Base(ParticleCollection(particle_array, size, stride),
         WeightCollection(weight_array, size),
         event_weight)
  {
    // set total weight
    for (index_type i = 0; i < size; i++)
      this->total_weight_ += weight_array[i];
  }

  // constructor from single argument (for use with Python)
  ArrayEvent(const std::tuple<value_type*, value_type*, index_type, index_type> & tup,
             value_type event_weight = 1) :
    ArrayEvent(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup), std::get<3>(tup), event_weight)
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
    oss << "ArrayEvent<" << sizeof(value_type) << "-byte float, stride ";
    if (std::is_same<ParticleCollection, ArrayParticleCollection<value_type>>::value)
      oss << "variable>";
    else
      oss << ParticleCollection::stride_static() << ">";

    return oss.str();
  }

}; // ArrayEvent

template<typename V>
using DefaultArrayEvent = ArrayEvent<V, ArrayParticleCollection>;

template<typename V>
using DefaultArray2Event = ArrayEvent<V, Array2ParticleCollection>;


////////////////////////////////////////////////////////////////////////////////
// VectorEvent - an event where the weights and particle are vectors
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
struct VectorEvent : public EventBase<std::vector<Value>, std::vector<Value>> {

  typedef Value value_type;
  typedef std::vector<value_type> ParticleCollection;
  typedef std::vector<value_type> WeightCollection;
  typedef EventBase<ParticleCollection, WeightCollection> Base;

  // constructor with a single argument
  VectorEvent(const std::pair<ParticleCollection, WeightCollection> && proto_event, value_type event_weight = 1) :
    VectorEvent(proto_event.first, proto_event.second, event_weight)
  {}

  // constructor from vectors of particles and weights
  VectorEvent(const ParticleCollection & particles, const WeightCollection & weights, value_type event_weight = 1) :
    Base(particles, weights, event_weight)
  {
    if (particles.size() % weights.size() != 0)
      throw std::invalid_argument("particles.size() must be cleanly divisible by weights.size()");

    // set total weight
    for (value_type weight : this->weights_)
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
    oss << "VectorEvent<" << sizeof(value_type) << "-byte float>";
    return oss.str();
  }

}; // VectorEvent

END_EMD_NAMESPACE

#endif // WASSERSTEIN_EVENT_HH
