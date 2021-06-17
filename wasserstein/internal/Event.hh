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
#include <algorithm>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "EMDUtils.hh"
#include "EuclideanParticle.hh"


BEGIN_WASSERSTEIN_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// EventBase - "events" constitute a weighted collection of "particles"
////////////////////////////////////////////////////////////////////////////////

template<class WeightCollection, class ParticleCollection>
struct EventBase {

  typedef typename WeightCollection::value_type value_type;

  // constructor from particle collection only
  EventBase(const ParticleCollection & particles, value_type event_weight = 1) :
    particles_(particles), event_weight_(event_weight), total_weight_(0), has_weights_(false)
  {}

  // constructor from particle collection and weight collection
  EventBase(const WeightCollection & weights,
            const ParticleCollection & particles,
            value_type event_weight = 1) :
    EventBase(particles, event_weight)
  {
    weights_ = weights;
    has_weights_ = true;
  }

  // default constructor
  EventBase() {}

  // access event_weight
  value_type event_weight() const { return event_weight_; }

  // access particles
  ParticleCollection & particles() { return particles_; }
  const ParticleCollection & particles() const { return particles_; }
  index_type dimension() const { throw std::logic_error("shouldn't get here"); }

  // access particle weights
  WeightCollection & weights() { return weights_; }
  const WeightCollection & weights() const { return weights_; }
  value_type & total_weight() { return total_weight_; }
  const value_type & total_weight() const { return total_weight_; }
  bool has_weights() const { return has_weights_; }
  void ensure_weights() {
    if (!has_weights())
      throw std::logic_error("must have weights here");
  }

  // normalize weights and total
  void normalize_weights() {
    if (!has_weights())
      throw std::logic_error("Weights must be set prior to calling normalize_weights.");

    // normalize each weight
    value_type norm_total(0);
    for (value_type & w : weights_)
      norm_total += (w /= total_weight_);
    total_weight_ = norm_total;
  }

protected:

  // don't ever expect to access event via pointer to base class
  ~EventBase() = default;

  ParticleCollection particles_;
  WeightCollection weights_;
  value_type event_weight_, total_weight_;
  bool has_weights_;

}; // EventBase


////////////////////////////////////////////////////////////////////////////////
// EuclideanParticleEvent - an event constructed from a vector of "particles"
////////////////////////////////////////////////////////////////////////////////

template<class _Particle>
struct EuclideanParticleEvent : public EventBase<std::vector<typename _Particle::value_type>,
                                                 std::vector<_Particle>> {

  typedef _Particle Particle;
  typedef typename Particle::value_type value_type;
  typedef std::vector<Particle> ParticleCollection;
  typedef std::vector<value_type> WeightCollection;

  EuclideanParticleEvent() {}
  EuclideanParticleEvent(const ParticleCollection & particles, value_type event_weight = 1) :
    EventBase<WeightCollection, ParticleCollection>(particles, event_weight)
  {
    // weights are assumed to be contained in the particles as public methods
    this->weights().reserve(particles.size());
    for (const Particle & particle : particles) {
      this->total_weight() += particle.weight();
      this->weights().push_back(particle.weight());
    }
    this->has_weights_ = true;
  }

  index_type dimension() const {
    return this->particles().size() == 0 ? -1 : this->particles()[0].dimension();
  }

  static std::string name() {
    std::ostringstream oss;
    oss << "EuclideanParticleEvent<" << Particle::name() << '>';
    return oss.str();
  }

}; // EuclideanParticleEvent


////////////////////////////////////////////////////////////////////////////////
// ArrayWeightCollection - implements a "smart" 1D array from a plain array
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
struct ArrayWeightCollection {

  typedef Value value_type;

  // contructor
  ArrayWeightCollection(Value * array, index_type size) :
    array_(array), size_(size), delete_array_on_destruction_(false)
  {}

  // default constructor
  ArrayWeightCollection() : ArrayWeightCollection(nullptr, 0) {}

  // destructor checks for freeing any memory we may have allocated
  ~ArrayWeightCollection() {
    if (delete_array_on_destruction_)
      delete[] array_;
  }

  index_type size() const { return size_; }
  Value * begin() { return array_; }
  Value * end() { return begin() + size(); }
  const Value * begin() const { return array_; }
  const Value * end() const { return begin() + size(); }

  const Value & operator[](index_type i) const { return array_[i]; }
  Value & operator[](index_type i) { return array_[i]; }

  // copy internal memory, used to avoid affecting arrays if we norm the weights
  void copy() {

    if (delete_array_on_destruction_)
      throw std::runtime_error("Should not call copy twice on an ArrayWeightCollection");
    delete_array_on_destruction_ = true;

    // get new chunk of memory
    Value * new_array(new Value[size()]);

    // copy old array into new one
    std::copy(begin(), end(), new_array);
    //memcpy(new_array, array_, std::size_t(size())*sizeof(Value));
    array_ = new_array;
  }

private:

  Value * array_;
  index_type size_;
  bool delete_array_on_destruction_;

}; // ArrayWeightCollection


////////////////////////////////////////////////////////////////////////////////
// ArrayParticleCollection - implements a "smart" 2D array from a plain array
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
struct ArrayParticleCollection {
protected:

  Value * array_;
  index_type size_, stride_;

  Value * array() const { return array_; }

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

  ArrayParticleCollection() : ArrayParticleCollection(nullptr, 0, -1) {}
  ArrayParticleCollection(Value * array, index_type size, index_type stride) :
    array_(array), size_(size), stride_(stride)
  {}

  index_type size() const { return size_; }
  index_type stride() const { return stride_; }
  index_type dimension() const { return stride(); }
  static index_type expected_stride() { return -1; }

  using const_iterator = templated_iterator<const Value>;
  using iterator = templated_iterator<Value>;
  using value_type = const_iterator;

  const_iterator begin() const { return const_iterator(array(), stride()); }
  const_iterator end() const { return const_iterator(array() + size()*stride(), stride()); }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }
  iterator begin() { return iterator(array(), stride()); }
  iterator end() { return iterator(array() + size()*stride(), stride()); }

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
  using iterator = templated_iterator<Value>;
  using value_type = const_iterator;

  Array2ParticleCollection(Value * array, index_type size, index_type stride) :
    ArrayParticleCollection<Value>(array, size, stride)
  {
    if (this->stride() != 2)
      throw std::invalid_argument("expected particles to have 2 dimensions");
  }

  const_iterator begin() const { return const_iterator(this->array()); }
  const_iterator end() const { return const_iterator(this->array() + 2*this->size()); }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }
  iterator begin() { return iterator(this->array()); }
  iterator end() { return iterator(this->array() + 2*this->size()); }
  static index_type expected_stride() { return 2; }

}; // Array2ParticleCollection


////////////////////////////////////////////////////////////////////////////////
// ArrayEvent - an event where the weights and particle are contiguous arrays
////////////////////////////////////////////////////////////////////////////////

template<typename Value, template<typename> class _ParticleCollection>
struct ArrayEvent : public EventBase<ArrayWeightCollection<Value>, _ParticleCollection<Value>> {

  typedef Value value_type;
  typedef _ParticleCollection<Value> ParticleCollection;
  typedef ArrayWeightCollection<Value> WeightCollection;
  typedef EventBase<WeightCollection, ParticleCollection> Base;

  // full constructor
  ArrayEvent(Value * weight_array, Value * particle_array,
             index_type size, index_type stride,
             Value event_weight = 1) :
    Base(WeightCollection(weight_array, size),
         ParticleCollection(particle_array, size, stride),
         event_weight)
  {
    // set total weight
    for (index_type i = 0; i < size; i++)
      this->total_weight() += weight_array[i];
  }

  // constructor from single argument (for use with Python)
  ArrayEvent(const std::tuple<Value*, Value*, index_type, index_type> & tup,
             Value event_weight = 1) :
    ArrayEvent(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup), std::get<3>(tup), event_weight)
  {}

  // default constructor
  ArrayEvent() : ArrayEvent(nullptr, nullptr, 0, 0) {}

  // ensure that we don't modify original array
  void normalize_weights() {
    this->weights().copy();
    Base::normalize_weights();
  }

  index_type dimension() const {
    return this->particles().dimension();
  }

  static std::string name() {
    index_type stride(ParticleCollection::expected_stride());

    std::ostringstream oss;
    oss << "ArrayEvent<" << sizeof(Value) << "-byte float, ";
    if (stride < 0)
      oss << "variable particle dimension>";
    else
      oss << stride << "-dimensional particles>";

    return oss.str();
  }

}; // ArrayEvent


////////////////////////////////////////////////////////////////////////////////
// VectorEvent - an event where the weights and particle are vectors
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
struct VectorEvent : public EventBase<std::vector<Value>, std::vector<Value>> {

  typedef Value value_type;
  typedef std::vector<Value> ParticleCollection;
  typedef std::vector<Value> WeightCollection;
  typedef EventBase<WeightCollection, ParticleCollection> Base;

  // constructor with a single argument
  VectorEvent(const std::pair<ParticleCollection, WeightCollection> && proto_event, Value event_weight = 1) :
    VectorEvent(proto_event.first, proto_event.second, event_weight)
  {}

  // constructor from vectors of particles and weights
  VectorEvent(const ParticleCollection & particles, const WeightCollection & weights, Value event_weight = 1) :
    Base(weights, particles, event_weight)
  {
    if (particles.size() % weights.size() != 0)
      throw std::invalid_argument("particles.size() must be cleanly divisible by weights.size()");

    // set total weight
    for (Value weight : this->weights_)
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
    oss << "VectorEvent<" << sizeof(Value) << "-byte float>";
    return oss.str();
  }

}; // VectorEvent

END_WASSERSTEIN_NAMESPACE

#endif // WASSERSTEIN_EVENT_HH
