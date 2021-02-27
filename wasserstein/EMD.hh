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
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>

// OpenMP for multithreading
#ifdef _OPENMP
#include <omp.h>
#endif

// Wasserstein headers
#include "internal/Event.hh"
#include "internal/NetworkSimplex.hh"
#include "internal/PairwiseDistance.hh"

BEGIN_EMD_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// EMD - Computes the Earth/Energy Mover's Distance between two "events" which
//       contain weights and "particles", between which a pairwise distance
//       can be evaluated
////////////////////////////////////////////////////////////////////////////////

template<class E, class PD = DefaultPairwiseDistance<>, class NetworkSimplex = lemon::NetworkSimplex<>>
#ifdef SWIG
class EMD : public EMDBase<Value> {
#else
class EMD : public EMDBase<typename NetworkSimplex::Value> {
#endif
public:

  // allow passing FastJetParticleWeight as first template parameter
  #ifdef WASSERSTEIN_FASTJET
  typedef E ParticleWeight;
  typedef typename std::conditional<std::is_base_of<FastJetParticleWeight, E>::value, FastJetEvent<E>, E>::type Event;
  #else
  typedef E Event;
  #endif

  // use Value from base class if not in swig
  #ifndef SWIG
  using Value = typename EMDBase<typename NetworkSimplex::Value>::Value;
  using ValueVector = typename EMDBase<typename NetworkSimplex::Value>::ValueVector;
  #endif

  typedef PD PairwiseDistance;
  typedef typename Event::ParticleCollection ParticleCollection;
  typedef typename Event::WeightCollection WeightCollection;
  typedef EMD<E, PairwiseDistance, NetworkSimplex> Self;
  
  // gives PairwiseEMD access to private members
  template<class T>
  friend class PairwiseEMD;

  // check that Value has been consistently defined 
  static_assert(std::is_same<Value, typename WeightCollection::value_type>::value,
                "WeightCollection and NetworkSimplex should have the same Value type.");
  static_assert(std::is_same<Value, typename PairwiseDistance::Value>::value,
                "PairwiseDistance and NetworkSimplex should have the same Value type.");
  static_assert(std::is_same<typename ParticleCollection::value_type,
                             typename PairwiseDistance::Particle>::value,
                "ParticleCollection and PairwiseDistance should have the same Particle type.");

  // check for consistent template arguments
  static_assert(std::is_base_of<EventBase<ParticleCollection, WeightCollection>, Event>::value,
                "First  EMD template parameter should be derived from EventBase<...>.");
  static_assert(std::is_base_of<PairwiseDistanceBase<PairwiseDistance, ParticleCollection, Value>,
                                PairwiseDistance>::value,
                "Second EMD template parameter should be derived from PairwiseDistanceBase<...>.");
  static_assert(std::is_base_of<lemon::NetworkSimplex<typename NetworkSimplex::Node, 
                                                      typename NetworkSimplex::Arc,
                                                      Value, typename NetworkSimplex::Bool>,
                                NetworkSimplex>::value,
                "Second EMD template parameter should be derived from PairwiseDistanceBase<...>.");

public:

  // constructor with entirely default arguments
  EMD(Value R = 1, Value beta = 1, bool norm = false,
      bool do_timing = false, bool external_dists = false,
      unsigned n_iter_max = 100000,
      Value epsilon_large_factor = 10000,
      Value epsilon_small_factor = 1) :

    // base class initialization
    EMDBase<Value>(norm, do_timing, external_dists),

    // initialize contained objects
    pairwise_distance_(R, beta),
    network_simplex_(n_iter_max, epsilon_large_factor, epsilon_small_factor)
  {
    // setup units correctly (only relevant here if norm = true)
    this->scale_ = 1;

    // automatically set external dists in the default case
    this->set_external_dists(std::is_same<PairwiseDistance, DefaultPairwiseDistance<Value>>::value);
  }

  // virtual destructor
  virtual ~EMD() {}

  // access/set R and beta parameters
  Value R() const { return pairwise_distance_.R(); }
  Value beta() const { return pairwise_distance_.beta(); }
  void set_R(Value R) { pairwise_distance_.set_R(R); }
  void set_beta(Value beta) { pairwise_distance_.set_beta(beta); }

  // these avoid needing this-> everywhere
  #ifndef SWIG_PREPROCESSOR
    using EMDBase<Value>::norm;
    using EMDBase<Value>::external_dists;
    using EMDBase<Value>::n0;
    using EMDBase<Value>::n1;
    using EMDBase<Value>::extra;
    using EMDBase<Value>::weightdiff;
    using EMDBase<Value>::scale;
    using EMDBase<Value>::emd;
    using EMDBase<Value>::status;
    using EMDBase<Value>::do_timing;
  #endif

  // set network simplex parameters
  void set_network_simplex_params(unsigned n_iter_max=100000,
                                  Value epsilon_large_factor=10000,
                                  Value epsilon_small_factor=1) {
    network_simplex_.set_params(n_iter_max, epsilon_large_factor, epsilon_small_factor);
  }

  // access underlying network simplex and pairwise distance objects
  const NetworkSimplex & network_simplex() const { return network_simplex_; }
  const PairwiseDistance & pairwise_distance() const { return pairwise_distance_; }

  // return a description of this object
  std::string description(bool write_preprocessors = true) const {
    std::ostringstream oss;
    oss << "EMD" << '\n'
        << "  " << Event::name() << '\n'
        << "    norm - " << (norm() ? "true" : "false") << '\n'
        << '\n'
        << pairwise_distance_.description()
        << network_simplex_.description();

    if (write_preprocessors)
      output_preprocessors(oss);

    return oss.str();
  }

  // free all dynamic memory help by this object
  void clear() {
    preprocessors_.clear();
    network_simplex_.free();
  }

  // add preprocessor to internal list
  template<template<class> class P, typename... Args>
  EMD & preprocess(Args && ... args) {
    preprocessors_.emplace_back(new P<Self>(std::forward<Args>(args)...));
    return *this;
  }

  // runs computation from anything that an Event can be constructed from
  // this includes preprocessing the events
  template<class ProtoEvent0, class ProtoEvent1>
  Value operator()(const ProtoEvent0 & pev0, const ProtoEvent1 & pev1) {
    Event ev0(pev0), ev1(pev1);
    check_emd_status(compute(preprocess(ev0), preprocess(ev1)));
    return emd();
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
    if (do_timing()) this->start_timing();

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
      *std::copy(ws1.begin(), ws1.end(), std::copy(ws0.begin(), ws0.end(), weights().begin())) = -weightdiff();
    }

    // if not norm, prepare to scale each weight by the max total
    if (!norm()) {
      this->scale_ = std::max(ev0.total_weight(), ev1.total_weight());
      for (Value & w : weights()) w /= scale();
    }

    // store distances in network simplex if not externally provided
    if (!external_dists())
      pairwise_distance_.fill_distances(ev0.particles(), ev1.particles(), ground_dists(), extra());

    // run the EarthMoversDistance at this point
    this->status_ = network_simplex_.compute(n0(), n1());
    this->emd_ = network_simplex_.total_cost();

    // account for weight scale if not normed
    if (status() == EMDStatus::Success && !norm())
      this->emd_ *= scale();

    // end timing and get duration
    if (do_timing())
      this->store_duration();

    // return status
    return status();
  }

  // access dists
  ValueVector dists() const {
    return ValueVector(network_simplex_.dists().begin(),
                       network_simplex_.dists().begin() + n0()*n1());
  }

  // returns all flows 
  ValueVector flows() const {

    // copy flows in the valid range
    ValueVector unscaled_flows(network_simplex_.flows().begin(), 
                               network_simplex_.flows().begin() + n0()*n1());
    // unscale all values
    for (Value & f: unscaled_flows)
      f *= scale();

    return unscaled_flows;
  }

  // emd flow values between particle i in event0 and particle j in event1
  Value flow(long long i, long long j) const {

    // allow for negative indices
    if (i < 0) i += n0();
    if (j < 0) j += n1();

    // check for improper indexing
    if (size_t(i) >= n0() || size_t(j) >= n1() || i < 0 || j < 0)
      throw std::out_of_range("EMD::flow - Indices out of range");

    return flow(i*n1() + j);
  }

  // "raw" access to EMD flow
  Value flow(std::size_t ind) const {
    return network_simplex_.flows()[ind] * scale(); 
  }

  // access ground dists in network simplex directly
  ValueVector & ground_dists() { return network_simplex_.dists(); }
  const ValueVector & ground_dists() const { return network_simplex_.dists(); }

private:

  // set weights of network simplex
  ValueVector & weights() { return network_simplex_.weights(); }

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
  void output_preprocessors(std::ostream & oss) const {
  #ifndef SWIG_WASSERSTEIN
    oss << "\n  Preprocessors:\n";
    for (const auto & preproc : preprocessors_)
      oss << "    - " << preproc->description() << '\n';
  #endif
  }

  /////////////////////
  // class data members
  /////////////////////

  // helper objects
  PairwiseDistance pairwise_distance_;
  NetworkSimplex network_simplex_;

  // preprocessor objects
  std::vector<std::shared_ptr<Preprocessor<Self>>> preprocessors_;

}; // EMD

////////////////////////////////////////////////////////////////////////////////
// PairwiseEMD - Facilitates computing EMDs between all event-pairs in a set
////////////////////////////////////////////////////////////////////////////////

template<class EMD>
class PairwiseEMD {
public:

  #ifndef SWIG
  typedef typename EMD::Value Value;
  typedef std::vector<Value> ValueVector;
  #endif

  typedef typename EMD::Event Event;
  typedef std::vector<Event> EventVector;

private:

  // records how the emd pairs are being stored
  enum EMDPairsStorage : char { Full, FullSymmetric, FlattenedSymmetric, External };

  // needs to be before emd_objs_ in order to determine actual number of threads
  int num_threads_, omp_dynamic_chunksize_;

  // EMD objects
  std::vector<EMD> emd_objs_;

  // variables local to this class
  long long print_every_;
  ExternalEMDHandler * handler_;
  unsigned verbose_;
  bool store_sym_emds_flattened_, throw_on_error_, request_mode_;
  std::ostream * print_stream_;
  std::ostringstream oss_;

  // vectors of events and EnergyMoversDistances
  EventVector events_;
  ValueVector emds_, full_emds_;
  std::vector<std::string> error_messages_;

  // info about stored events
  size_t nevA_, nevB_, emd_counter_, num_emds_, num_emds_width_;
  EMDPairsStorage emd_storage_;
  bool two_event_sets_;

public:

  // contructor that initializes the EMD object, uses the same default arguments
  PairwiseEMD(Value R = 1, Value beta = 1, bool norm = false,
              int num_threads = -1,
              long long print_every = -10,
              unsigned verbose = 1,
              bool store_sym_emds_flattened = true,
              bool throw_on_error = false,
              unsigned n_iter_max = 100000,
              Value epsilon_large_factor = 10000,
              Value epsilon_small_factor = 1,
              std::ostream & os = std::cout) :
    num_threads_(determine_num_threads(num_threads)),
    emd_objs_(num_threads_, EMD(R, beta, norm, false, false,
                                n_iter_max, epsilon_large_factor, epsilon_small_factor))
  {
    setup(print_every, verbose, store_sym_emds_flattened, throw_on_error, &os);
  }

// avoid overloading constructor so swig can handle keyword arguments
#ifndef SWIG

  // contructor uses existing EMD instance
  PairwiseEMD(const EMD & emd,
              int num_threads = -1,
              long long print_every = -10,
              unsigned verbose = 1,
              bool store_sym_emds_flattened = true,
              bool throw_on_error = false,
              std::ostream & os = std::cout) :
    num_threads_(determine_num_threads(num_threads)),
    emd_objs_(num_threads_, emd)
  {
    if (emd.external_dists())
      throw std::invalid_argument("Cannot use PairwiseEMD with external distances");

    setup(print_every, verbose, store_sym_emds_flattened, throw_on_error, &os);
  }

#endif // SWIG

  // virtual destructor
  virtual ~PairwiseEMD() {}

  // add preprocessor to internal list
  template<template<class> class P, typename... Args>
  PairwiseEMD & preprocess(Args && ... args) {
    for (EMD & emd_obj : emd_objs_)
      emd_obj.template preprocess<P>(std::forward<Args>(args)...);
    return *this;
  }

  // get/set R
  Value R() const { return emd_objs_[0].R(); }
  void set_R(Value R) {
    for (EMD & emd_obj : emd_objs_) emd_obj.set_R(R);
  }

  // get/set beta
  Value beta() const { return emd_objs_[0].beta(); }
  void set_beta(Value beta) {
    for (EMD & emd_obj : emd_objs_) emd_obj.set_beta(beta);
  }

  // get/set norm
  bool norm() const { return emd_objs_[0].norm(); }
  void set_norm(bool norm) {
    for (EMD & emd_obj : emd_objs_) emd_obj.set_norm(norm);
  }

  // set network simplex parameters
  void set_network_simplex_params(unsigned n_iter_max=100000,
                                  Value epsilon_large_factor=10000,
                                  Value epsilon_small_factor=1) {
    for (EMD & emd_obj : emd_objs_)
      emd_obj.set_network_simplex_params(n_iter_max, epsilon_large_factor, epsilon_small_factor);
  }

  // externally set the number of EMD evaluations that will be spooled to each OpenMP thread at a time
  void set_omp_dynamic_chunksize(int chunksize) {
    omp_dynamic_chunksize_ = std::abs(chunksize);
  }
  int omp_dynamic_chunksize() const {
    return omp_dynamic_chunksize_;
  }

  // set a handler to process EMDs on the fly instead of storing them
  void set_external_emd_handler(ExternalEMDHandler & handler) {
    handler_ = &handler;
  }
  bool external_handler() const {
    return handler_ != nullptr;
  }

  // turn on or off request mode, where nothing is stored or handled but
  // EMD distances can be queried and computed on the fly
  void set_request_mode(bool mode) {
    request_mode_ = mode;
  }
  bool request_mode() const {
    return request_mode_;
  }

  // return a description of this object as a string
  std::string description(bool write_preprocessors = true) const {
    std::ostringstream oss;
    oss << "Pairwise" << emd_objs_[0].description(false) << '\n'
        << "  num_threads - " << num_threads_ << '\n'
        << "  print_every - ";

    // handle print_every logic
    if (print_every_ > 0) oss << print_every_;
    else
      oss << "auto, " << std::abs(print_every_) << " total chunks";

    oss << '\n'
        << "  store_sym_emds_flattened - " << (store_sym_emds_flattened_ ? "true\n" : "false\n")
        << "  throw_on_error - " << (throw_on_error_ ? "true\n" : "false\n")
        << '\n'
        << (handler_ ? handler_->description() : "  Pairwise EMD distance matrix stored internally\n");
        
    if (write_preprocessors)
      emd_objs_[0].output_preprocessors(oss);

    return oss.str();
  }

  // clears internal storage
  void clear(bool free_memory = true) {

    events_.clear();
    emds_.clear();
    full_emds_.clear();
    error_messages_.clear();

    emd_storage_ = External;
    nevA_ = nevB_ = emd_counter_ = num_emds_ = 0;

    // start clock for overall timing
    emd_objs_[0].start_timing();

    if (free_memory) {
      handler_ = nullptr;
      //EventVector().swap(events_);
      //ValueVector().swap(emds_);
      //ValueVector().swap(full_emds_);
      //std::vector<std::string>().swap(error_messages_);
      free_vector(events_);
      free_vector(emds_);
      free_vector(full_emds_);
      free_vector(error_messages_);
      for (EMD & emd_obj : emd_objs_)
        emd_obj.clear();
    }
  }

  // compute EMDs between all pairs of proto events, including preprocessing
  template<class ProtoEvent>
  void operator()(const std::vector<ProtoEvent> & proto_events) {
    init(proto_events.size());
    store_proto_events(proto_events);
    compute();
  }

  // compute EMDs between two sets of proto events, including preprocessing
  template<class ProtoEventA, class ProtoEventB>
  void operator()(const std::vector<ProtoEventA> & proto_eventsA,
                  const std::vector<ProtoEventB> & proto_eventsB) {
    init(proto_eventsA.size(), proto_eventsB.size());
    store_proto_events(proto_eventsA);
    store_proto_events(proto_eventsB);
    compute();
  }

  // compute pairs among same set of events (no preprocessing)
  void compute(const EventVector & events) {
    init(events.size());
    events_ = events;
    compute();
  }

  // compute pairs between different sets of events
  void compute(const EventVector & eventsA, const EventVector & eventsB) {
    init(eventsA.size(), eventsB.size());
    events_.reserve(nevA_ + nevB_);
    events_.insert(events_.end(), eventsA.begin(), eventsA.end());
    events_.insert(events_.end(), eventsB.begin(), eventsB.end());
    compute();
  }

  // access all emds as a matrix flattened into a vector
  const ValueVector & emds(bool raw = false) {

    // return raw emds if requested
    if (raw) return emds_;

    // check for having no emds stored
    if (emd_storage_ == External)
      throw std::logic_error("No EMDs stored");

    // check if we need to construct a new full matrix from a flattened symmetric one
    if (emd_storage_ == FlattenedSymmetric) {

      // allocate a new vector for holding the full emds
      full_emds_.resize(nevA_*nevB_);

      // zeros on the diagonal
      for (size_t i = 0; i < nevA_; i++)
        full_emds_[i*i] = 0;

      // fill out matrix
      for (size_t i = 0; i < nevA_; i++)
        for (size_t j = 0; j < i; j++)
          full_emds_[i*nevB_ + j] = full_emds_[j*nevB_ + i] = emds_[index_symmetric(i, j)];

      return full_emds_;
    }

    // full emds stored
    else return emds_;
  }

  // access a specific emd
  Value emd(long long i, long long j, int thread = 0) {

    // allow for negative indices
    if (i < 0) i += nevA_;
    if (j < 0) j += nevB_;

    // check for improper indexing
    if (size_t(i) >= nevA_ || size_t(j) >= nevB_ || i < 0 || j < 0) {
      std::ostringstream message("PairwiseEMD::emd - Accessing emd value at (", std::ios_base::ate);
      message << i << ", " << j << ") exceeds allowed range";
      throw std::out_of_range(message.str());
    }

    // calculate EMD if in request mode
    if (request_mode()) {

      if (thread >= num_threads_)
        throw std::out_of_range("invalid thread index");

      // run and check for failure
      check_emd_status(emd_objs_[thread].compute(events_[i], events_[two_event_sets_ ? nevA_ + j : j]));
      if (handler_) (*handler_)(emd_objs_[thread].emd());
      return emd_objs_[thread].emd();
    }

    // check for External handling, in which case we don't have any emds stored
    if (emd_storage_ == External)
      throw std::logic_error("EMD requested but external handler provided, so no EMDs stored");

    // index into emd vector
    if (emd_storage_ == FlattenedSymmetric)
      return (i == j ? 0 : emds_[i > j ? index_symmetric(i, j) : index_symmetric(j, i)]);
    else return emds_[i*nevB_ + j];
  }

  // error reporting
  bool errored() const { return error_messages_.size() > 0; }
  const std::vector<std::string> & error_messages() const { return error_messages_; }
  /*void report_errors(std::ostream & os = std::cerr) const {
    for (const std::string & err : error_messages())
      os << err << '\n';
    os << std::flush;
  }*/

  // number of unique emds computed
  size_t num_emds() const { return num_emds_; }

  // access events
  size_t nevA() const { return nevA_; }
  size_t nevB() const { return nevB_; }
  const EventVector & events() const { return events_; }

  // access timing information
  double duration() const { return emd_objs_[0].duration(); }

// wasserstein needs access to these functions in order to use CustomArrayDistance
#ifndef SWIG_WASSERSTEIN
private:
#endif

  // access modifiable events
  EventVector & events() { return events_; }

  // preprocesses the last event added
  void preprocess_back_event() {
    emd_objs_[0].preprocess(events_.back());
  }

  // init self pairs
  void init(size_t nev) {

    if (!request_mode())
      clear(false);

    nevA_ = nevB_ = nev;
    two_event_sets_ = false;

    // resize emds
    num_emds_ = nevA_*(nevA_ - 1)/2;
    if (!external_handler() && !request_mode()) {
      emd_storage_ = (store_sym_emds_flattened_ ? FlattenedSymmetric : FullSymmetric);
      emds_.resize(emd_storage_ == FullSymmetric ? nevA_*nevB_ : num_emds_);
    }

    // reserve space for events
    events_.reserve(nevA_);
  }

  // init pairs
  void init(size_t nevA, size_t nevB) {

    if (!request_mode())
      clear(false);

    nevA_ = nevA;
    nevB_ = nevB;
    two_event_sets_ = true;

    // resize emds
    num_emds_ = nevA_*nevB_;
    if (!external_handler() && !request_mode()) {
      emd_storage_ = Full;
      emds_.resize(num_emds_);  
    }

    // reserve space for events
    events_.reserve(nevA_ + nevB_);
  }

  void compute() {

    // check that we're not in request mode
    if (request_mode())
      throw std::runtime_error("cannot compute pairwise EMDs in request mode");

    num_emds_width_ = std::to_string(num_emds_).size();

    long long print_every(print_every_);
    if (print_every < 0) {
      print_every = num_emds_/std::abs(print_every_);
      if (print_every == 0 || num_emds_ % std::abs(print_every_) != 0)
        print_every++;
    }

    if (verbose_) {
      oss_.str("Finished preprocessing ");
      oss_ << events_.size() << " events in "
           << std::setprecision(4) << emd_objs_[0].store_duration() << 's';
      *print_stream_ << oss_.str() << std::endl;
    }

    int omp_for_dynamic_chunksize(omp_dynamic_chunksize_);
    if (omp_for_dynamic_chunksize < print_every/num_threads_)
      omp_for_dynamic_chunksize = print_every/num_threads_;

    // iterate over emd pairs
    std::mutex failure_mutex;
    size_t begin(0);
    while (emd_counter_ < num_emds_ && !(throw_on_error_ && error_messages().size())) {
      emd_counter_ += size_t(print_every);
      if (emd_counter_ > num_emds_) emd_counter_ = num_emds_;

      #pragma omp parallel num_threads(num_threads_) default(shared)
      {
        // get thread id
        int thread(0);
        #ifdef _OPENMP
        thread = omp_get_thread_num();
        #endif

        // grab EMD object for this thread
        EMD & emd_obj(emd_objs_[thread]);

        // parallelize loop over EMDs
        #pragma omp for schedule(dynamic, omp_for_dynamic_chunksize)
        for (long long k = begin; k < (long long) emd_counter_; k++) {

          size_t i(k/nevB_), j(k%nevB_);
          if (two_event_sets_) {

            // run and check for failure
            EMDStatus status(emd_obj.compute(events_[i], events_[nevA_ + j]));
            if (status != EMDStatus::Success) {
              std::lock_guard<std::mutex> failure_lock(failure_mutex);
              record_failure(status, i, j);
            }
            if (handler_) (*handler_)(emd_obj.emd());
            else emds_[k] = emd_obj.emd(); 
          }
          else {

            // this properly sets indexing for symmetric case
            if (j >= ++i) {
              i = nevA_ - i;
              j = nevA_ - j - 1;
            }

            // run and check for failure
            EMDStatus status(emd_obj.compute(events_[i], events_[j]));
            if (status != EMDStatus::Success) {
              std::lock_guard<std::mutex> failure_lock(failure_mutex);
              record_failure(status, i, j);
            }

            // store emd value
            if (emd_storage_ == FlattenedSymmetric)
              emds_[index_symmetric(i, j)] = emd_obj.emd();
            else if (emd_storage_ == External)
              (*handler_)(emd_obj.emd());
            else if (emd_storage_ == FullSymmetric)
              emds_[i*nevB_ + j] = emds_[j*nevB_ + i] = emd_obj.emd();
            else std::cerr << "Should never get here\n";
          }
        }
      }

      // update and do printing
      begin = emd_counter_;
      print_update();
    }

    if (throw_on_error_ && error_messages_.size())
      throw std::runtime_error(error_messages_.front());
  }

private:

  // determine the number of threads to use
  int determine_num_threads(int num_threads) {
    #ifdef _OPENMP
      if (num_threads == -1 || num_threads > omp_get_max_threads())
        return omp_get_max_threads();
      return num_threads;
    #else
      return 1;
    #endif
  }

  // init from constructor
  void setup(long long print_every, unsigned verbose,
             bool store_sym_emds_flattened, bool throw_on_error,
             std::ostream * os) {
    
    // store arguments, from constructor
    print_every_ = print_every;
    verbose_ = verbose;
    store_sym_emds_flattened_ = store_sym_emds_flattened;
    throw_on_error_ = throw_on_error;
    print_stream_ = os;
    handler_ = nullptr;

    // turn off request mode by default
    request_mode_ = false;
    set_omp_dynamic_chunksize(10);

    // print_every of 0 is equivalent to -1
    if (print_every_ == 0)
      print_every_ = -1;

    // setup stringstream for printing
    oss_ = std::ostringstream(std::ios_base::ate);
    oss_.setf(std::ios_base::fixed, std::ios_base::floatfield);

    // turn off timing in EMD objects
    for (EMD & emd_obj : emd_objs_) emd_obj.do_timing_ = false;

    // clear is meant to be used between computations, call it here for consistency
    clear(false);
  }

  // store events
  template<class ProtoEvent>
  void store_proto_events(const std::vector<ProtoEvent> & proto_events) {
    for (const ProtoEvent & proto_event : proto_events) {
      events_.emplace_back(proto_event);
      preprocess_back_event();
    }
  }

  void record_failure(EMDStatus status, size_t i, size_t j) {
    std::ostringstream message;
    message << "PairwiseEMD::compute - Issue with EMD between events ("
            << i << ", " << j << "), error code " << int(status);
    error_messages_.push_back(message.str());

    // acquire Python GIL if in SWIG in order to print message
    #ifdef SWIG
      SWIG_PYTHON_THREAD_BEGIN_BLOCK;
        std::cerr << error_messages().back() << '\n';
      SWIG_PYTHON_THREAD_END_BLOCK;
    #else
      std::cerr << error_messages().back() << '\n';
    #endif
  }

  // indexes lower triangle of symmetric matrix with zeros on diagonal that has been flattened into 1D
  static size_t index_symmetric(size_t i, size_t j) {
    return i*(i - 1)/2 + j;
  }

  void print_update() {

    // prepare message
    if (verbose_) {
      oss_.str("  ");
      oss_ << std::setw(num_emds_width_) << emd_counter_ << " / "
           << std::setw(num_emds_width_) << num_emds_ << "  EMDs computed  - "
           << std::setprecision(2) << std::setw(6) << double(emd_counter_)/num_emds_*100
           << "% completed - "
           << std::setprecision(3) << emd_objs_[0].store_duration() << 's';  
    }

    // acquire Python GIL if in SWIG in order to check for signals and print message
    #ifdef SWIG
      SWIG_PYTHON_THREAD_BEGIN_BLOCK;
      if (verbose_) *print_stream_ << oss_.str() << std::endl;
      if (PyErr_CheckSignals() != 0)
        throw std::runtime_error("KeyboardInterrupt received in PairwiseEMD::compute");
      SWIG_PYTHON_THREAD_END_BLOCK;
    #else
      if (verbose_) *print_stream_ << oss_.str() << std::endl;
    #endif
  }

}; // PairwiseEMD

END_EMD_NAMESPACE

#endif // WASSERSTEIN_EMD_HH
