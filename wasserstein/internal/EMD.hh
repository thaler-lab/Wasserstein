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

/*  ______ __  __ _____  
 * |  ____|  \/  |  __ \
 * | |__  | \  / | |  | |
 * |  __| | |\/| | |  | |
 * | |____| |  | | |__| |
 * |______|_|  |_|_____/
 */

#ifndef WASSERSTEIN_EMD_HH
#define WASSERSTEIN_EMD_HH

// C++ standard library
#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Wasserstein headers (required for EMD functionality)
#include "EMDBase.hh"
#include "ExternalEMDHandler.hh"


BEGIN_WASSERSTEIN_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// EMD - Computes the Earth/Energy Mover's Distance between two "events" which
//       contain weights and "particles", between which a pairwise distance
//       can be evaluated
////////////////////////////////////////////////////////////////////////////////

template<typename Value,
         template<typename> class _Event = DefaultEvent,
         template<typename> class _PairwiseDistance = DefaultPairwiseDistance,
         template<typename> class _NetworkSimplex = DefaultNetworkSimplex>
class EMD : public EMDBase<Value> {
public:

  // typedefs from template parameters
  typedef Value value_type;
  typedef _PairwiseDistance<Value> PairwiseDistance;
  typedef _NetworkSimplex<Value> NetworkSimplex;
  typedef typename std::conditional<std::is_base_of<FastJetParticleWeight, _Event<Value>>::value,
                                    FastJetEvent<_Event<Value>>,
                                    _Event<Value>>::type Event;

  // this is used with fastjet and harmless otherwise
  #ifdef WASSERSTEIN_FASTJET
    typedef typename Event::ParticleWeight ParticleWeight;
  #endif

  // event-dependent typedefs
  typedef typename Event::WeightCollection WeightCollection;
  typedef typename Event::ParticleCollection ParticleCollection;

  // typedef base class and self
  typedef EMDBase<Value> Base;
  typedef EMD<Value, _Event, _PairwiseDistance, _NetworkSimplex> Self;
  
  // gives PairwiseEMD access to private members
  template<class T, typename V>
  friend class PairwiseEMD;

  // check that value_type has been consistently defined 
  static_assert(std::is_same<Value, typename Event::value_type>::value,
                "WeightCollection and NetworkSimplex should have the same value_type.");
  static_assert(std::is_same<Value, typename PairwiseDistance::value_type>::value,
                "PairwiseDistance and NetworkSimplex should have the same value_type.");
  static_assert(std::is_same<typename ParticleCollection::value_type,
                             typename PairwiseDistance::Particle>::value,
                "ParticleCollection and PairwiseDistance should have the same Particle type.");

  // check for consistent template arguments
  static_assert(std::is_base_of<EventBase<WeightCollection, ParticleCollection>, Event>::value,
                "First  EMD template parameter should be derived from EventBase<...>.");
  static_assert(std::is_base_of<PairwiseDistanceBase<PairwiseDistance, ParticleCollection, Value>,
                                PairwiseDistance>::value,
                "Second EMD template parameter should be derived from PairwiseDistanceBase<...>.");
  static_assert(std::is_base_of<WASSERSTEIN_NAMESPACE::NetworkSimplex<
                                                       typename NetworkSimplex::Value,
                                                       typename NetworkSimplex::Arc, 
                                                       typename NetworkSimplex::Node,
                                                       typename NetworkSimplex::Bool>,
                                NetworkSimplex>::value,
                "This EMD template parameter should be derived from NetworkSimplex<...>.");

  // constructor with entirely default arguments
  EMD(Value R = 1, Value beta = 1, bool norm = false,
       bool do_timing = false, bool external_dists = false,
       std::size_t n_iter_max = 100000,
       Value epsilon_large_factor = 1000,
       Value epsilon_small_factor = 1) :

    // base class initialization
    Base(norm, do_timing, external_dists),

    // initialize contained objects
    pairwise_distance_(R, beta),
    network_simplex_(n_iter_max, epsilon_large_factor, epsilon_small_factor)
  {
    // setup units correctly (only relevant here if norm = true)
    this->scale_ = 1;

    // automatically set external dists in the default case
    this->set_external_dists(std::is_same<PairwiseDistance,
                                          DefaultPairwiseDistance<Value>>::value);
  }

  virtual ~EMD() = default;

  // these avoid needing this-> with common functions
  #ifndef SWIG_PREPROCESSOR
    using Base::norm;
    using Base::external_dists;
    using Base::n0;
    using Base::n1;
    using Base::weightdiff;
    using Base::scale;
  #endif

  // access underlying network simplex and pairwise distance objects
  const NetworkSimplex & network_simplex() const { return network_simplex_; }
  const PairwiseDistance & pairwise_distance() const { return pairwise_distance_; }

  // return a description of this object
  std::string description(bool write_preprocessors = true) const {
    std::ostringstream oss;
    oss << std::boolalpha;
    oss << "EMD" << '\n'
        << "  " << Event::name() << '\n'
        << "    norm - " << norm() << '\n'
        << "    external_dists - " << external_dists() << '\n'
        << '\n'
        << pairwise_distance().description()
        << network_simplex().description();

    if (write_preprocessors)
      output_preprocessors(oss);

    return oss.str();
  }

  // add preprocessor to internal list
  template<template<class> class Preproc, typename... Args>
  Self & preprocess(Args && ... args) {
    preprocessors_.emplace_back(new Preproc<Self>(std::forward<Args>(args)...));
    return *this;
  }

  // runs computation from anything that an Event can be constructed from
  // this includes preprocessing the events
  template<class ProtoEvent0, class ProtoEvent1>
  Value operator()(const ProtoEvent0 & pev0, const ProtoEvent1 & pev1) {
    Event ev0(pev0), ev1(pev1);
    check_emd_status(compute(preprocess(ev0), preprocess(ev1)));
    return this->emd();
  }

  // runs the computation on two events without any preprocessing
  // returns the status enum value from the network simplex solver:
  //   - EMDStatus::Success = 0
  //   - EMDStatus::Empty = 1
  //   - SupplyMismatch = 2
  //   - Unbounded = 3
  //   - MaxIterReached = 4
  //   - Infeasible = 5
  EMDStatus compute(const Event & ev0, const Event & ev1) {

    const WeightCollection & ws0(ev0.weights()), & ws1(ev1.weights());

    // check for timing request
    if (this->do_timing())
      this->start_timing();

    // grab number of particles
    this->n0_ = ws0.size();
    this->n1_ = ws1.size();

    // handle adding fictitious particle
    this->weightdiff_ = ev1.total_weight() - ev0.total_weight();

    // for norm or already equal or custom distance, don't add particle
    if (norm() || external_dists() || weightdiff() == 0) {
      this->extra_ = ExtraParticle::Neither;
      weights().resize(n0() + n1() + 1); // + 1 is to match what network simplex will do anyway
      std::copy(ws1.begin(), ws1.end(), std::copy(ws0.begin(), ws0.end(), weights().begin()));
    }

    // total weights unequal, add extra particle to event0 as it has less total weight
    else if (weightdiff() > 0) {
      this->extra_ = ExtraParticle::Zero;
      this->n0_++;
      weights().resize(n0() + n1() + 1); // +1 is to match what network simplex will do anyway

      // put weight diff after ws0
      auto it(std::copy(ws0.begin(), ws0.end(), weights().begin()));
      *it = weightdiff();
      std::copy(ws1.begin(), ws1.end(), ++it);
    }

    // total weights unequal, add extra particle to event1 as it has less total weight
    else {
      this->extra_ = ExtraParticle::One;
      this->n1_++;
      weights().resize(n0() + n1() + 1); // +1 is to match what network simplex will do anyway
      *std::copy(ws1.begin(),
                 ws1.end(),
                 std::copy(ws0.begin(),
                           ws0.end(),
                           weights().begin())) = -weightdiff();
    }

    // if not norm, prepare to scale each weight by the max total
    if (!norm()) {
      this->scale_ = std::max(ev0.total_weight(), ev1.total_weight());
      for (Value & w : weights()) w /= scale();
    }

    // store distances in network simplex if not externally provided
    if (!external_dists())
      pairwise_distance_.fill_distances(ev0.particles(), ev1.particles(),
                                        ground_dists(), this->extra());

    // run the EarthMoversDistance at this point
    this->status_ = network_simplex_.compute(n0(), n1());
    this->emd_ = network_simplex_.total_cost();

    // account for weight scale if not normed
    if (this->status() == EMDStatus::Success && !norm())
      this->emd_ *= scale();

    // end timing and get duration
    if (this->do_timing())
      this->store_duration();

    // return status
    return this->status();
  }

  // access ground dists in network simplex directly
  std::vector<Value> & ground_dists() { return network_simplex_.dists(); }
  const std::vector<Value> & ground_dists() const { return network_simplex_.dists(); }

// these functions should be private since Python will access them via the base class
#ifdef SWIG
private:
#endif

  // access/set R and beta parameters
  Value R() const { return pairwise_distance_.R(); }
  Value beta() const { return pairwise_distance_.beta(); }
  void set_R(Value R) { pairwise_distance_.set_R(R); }
  void set_beta(Value beta) { pairwise_distance_.set_beta(beta); }

  // set network simplex parameters
  void set_network_simplex_params(std::size_t n_iter_max=100000,
                                  Value epsilon_large_factor=1000,
                                  Value epsilon_small_factor=1) {
    network_simplex_.set_params(n_iter_max, epsilon_large_factor, epsilon_small_factor);
  }

  // free all dynamic memory help by this object
  void clear() {
    preprocessors_.clear();
    network_simplex_.free();
  }

  // access dists
  std::vector<Value> dists() const {
    return std::vector<Value>(network_simplex().dists().begin(),
                              network_simplex().dists().begin() + n0()*n1());
  }

  // returns all flows 
  std::vector<Value> flows() const {

    // copy flows in the valid range
    std::vector<Value> unscaled_flows(network_simplex().flows().begin(), 
                                      network_simplex().flows().begin() + n0()*n1());
    // unscale all values
    for (Value & f: unscaled_flows)
      f *= scale();

    return unscaled_flows;
  }

  // emd flow values between particle i in event0 and particle j in event1
  Value flow(index_type i, index_type j) const {

    // allow for negative indices
    if (i < 0) i += n0();
    if (j < 0) j += n1();

    // check for improper indexing
    if (i >= n0() || j >= n1() || i < 0 || j < 0)
      throw std::out_of_range("EMD::flow - Indices out of range");

    return flow(i*n1() + j);
  }

  // "raw" access to EMD flow
  Value flow(std::size_t ind) const {
    return network_simplex_.flows()[ind] * scale(); 
  }

  // access number of iterations of the network simplex solver
  std::size_t n_iter() const { return network_simplex().n_iter(); }

  // access node potentials of network simplex solver
  std::pair<std::vector<Value>, std::vector<Value>> node_potentials() const {
    std::pair<std::vector<Value>, std::vector<Value>> nps;
    nps.first.resize(n0());
    nps.second.resize(n1());

    std::copy(network_simplex().potentials().begin(),
              network_simplex().potentials().begin() + n0(),
              nps.first.begin());
    std::copy(network_simplex().potentials().begin() + n0(),
              network_simplex().potentials().begin() + n0() + n1(),
              nps.second.begin());

    return nps;
  }

private:

  // set weights of network simplex
  std::vector<Value> & weights() { return network_simplex_.weights(); }

  // access raw flows
  const std::vector<Value> & raw_flows() const {
    return network_simplex().flows();
  }

  // applies preprocessors to an event
  Event & preprocess(Event & event) const {

    // run preprocessing
    for (const auto & preproc : preprocessors_)
      (*preproc)(event);

    // ensure that we have weights
    event.ensure_weights();

    // perform normalization
    if (norm())
      event.normalize_weights();

    return event;
  }

  // output description of preprocessors contained in this object
  // does nothing if there are no preprocessors
  void output_preprocessors(std::ostream & oss) const {
    if(preprocessors_.size()) {
      oss << "\n  Preprocessors:\n";
      for (const auto & preproc : preprocessors_)
        oss << "    - " << preproc->description() << '\n';  
    }
  }

#ifdef WASSERSTEIN_SERIALIZATION
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & boost::serialization::base_object<Base>(*this)
       & pairwise_distance_ & network_simplex_;
  }
#endif

  /////////////////////
  // class data members
  /////////////////////

  // helper objects
  PairwiseDistance pairwise_distance_;
  NetworkSimplex network_simplex_;

  // preprocessor objects
  std::vector<std::shared_ptr<Preprocessor<Self>>> preprocessors_;

}; // EMD

END_WASSERSTEIN_NAMESPACE

#endif // WASSERSTEIN_EMD_HH
