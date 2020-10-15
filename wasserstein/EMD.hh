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

#ifndef WASSERSTEIN_EMD_HH
#define WASSERSTEIN_EMD_HH

// C++ standard library
#include <algorithm>
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

template<class E, class PD, class NetworkSimplex = lemon::NetworkSimplex<>>
#ifdef SWIG
class EMD : public EMDBase<Value> {
#else
class EMD : public EMDBase<typename NetworkSimplex::Value> {
#endif
public:

  // allow passing FastJetParticleWeight as first template parameter
  #ifdef __FASTJET_PSEUDOJET_HH__
  typedef E ParticleWeight;
  typedef typename std::conditional<std::is_base_of<FastJetParticleWeight, E>::value, FastJetEvent<E>, E>::type Event;
  #else
  typedef E Event;
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
    EMDBase(norm, do_timing, external_dists),

    // initialize contained objects
    pairwise_distance_(R, beta),
    network_simplex_(n_iter_max, epsilon_large_factor, epsilon_small_factor)
  {
    // setup units correctly (only relevant here if norm = true)
    scale_ = 1;
  }

  // virtual destructor
  virtual ~EMD() {}

  // return a description of this object
  std::string description(bool write_preprocessors = true) const {
    std::ostringstream oss;
    oss << "EMD" << '\n'
        << "  " << Event::name() << '\n'
        << "    norm - " << (norm_ ? "true" : "false") << '\n'
        << '\n'
        << pairwise_distance_.description()
        << network_simplex_.description();

    if (write_preprocessors)
      output_preprocessors(oss);

    return oss.str();
  }

  // add preprocessor to internal list
  template<template<class> class P, typename... Args>
  EMD & preprocess(Args && ... args) {
    preprocessors_.emplace_back(new P<Self>(std::forward<Args>(args)...));
    return *this;
  }

  // access underlying network simplex and pairwise distance objects
  const NetworkSimplex & network_simplex() const { return network_simplex_; }
  const PairwiseDistance & pairwise_distance() const { return pairwise_distance_; }

  // access/set R and beta parameters
  Value R() const { return pairwise_distance_.R(); }
  Value beta() const { return pairwise_distance_.beta(); }
  void set_R(Value r) { pairwise_distance_.set_R(r); }
  void set_beta(Value beta) { pairwise_distance_.set_beta(beta); }

  // set network simplex parameters
  void set_network_simplex_params(unsigned n_iter_max, Value epsilon_large_factor, Value epsilon_small_factor) {
    network_simplex_.set_params(n_iter_max, epsilon_large_factor, epsilon_small_factor);
  }

  // free all dynamic memory help by this object
  void clear() {
    preprocessors_.clear();
    network_simplex_.free();
  }

  // runs computation from anything that an Event can be constructed from
  // this includes preprocessing the events
  template<class ProtoEvent0, class ProtoEvent1>
  Value operator()(const ProtoEvent0 & pev0, const ProtoEvent1 & pev1) {
    Event ev0(pev0), ev1(pev1);
    check_emd_status(compute(preprocess(ev0), preprocess(ev1)));
    return emd_;
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
    if (do_timing_) start_timing();

    // grab number of particles
    n0_ = ws0.size();
    n1_ = ws1.size();

    // handle adding fictitious particle
    weightdiff_ = ev1.total_weight() - ev0.total_weight();

    // for norm or already equal or custom distance, don't add particle
    if (norm_ || external_dists() || weightdiff_ == 0) {
      extra_ = ExtraParticle::Neither;
      weights().resize(n0() + n1() + 1); // + 1 is to match what network simplex will do anyway
      std::copy(ws1.begin(), ws1.end(), std::copy(ws0.begin(), ws0.end(), weights().begin()));
    }

    // total weights unequal, add extra particle to event0 as it has less total weight
    else if (weightdiff_ > 0) {
      extra_ = ExtraParticle::Zero;
      n0_++;
      weights().resize(n0() + n1() + 1); // +1 is to match what network simplex will do anyway

      // put weight diff after ws0
      auto it(std::copy(ws0.begin(), ws0.end(), weights().begin()));
      *it = weightdiff_;
      std::copy(ws1.begin(), ws1.end(), ++it);
    }

    // total weights unequal, add extra particle to event1 as it has less total weight
    else {
      extra_ = ExtraParticle::One;
      n1_++;
      weights().resize(n0() + n1() + 1); // +1 is to match what network simplex will do anyway
      *std::copy(ws1.begin(), ws1.end(), std::copy(ws0.begin(), ws0.end(), weights().begin())) = -weightdiff_;
    }

    // if not norm, prepare to scale each weight by the max total
    if (!norm_) {
      scale_ = std::max(ev0.total_weight(), ev1.total_weight());
      for (Value & w : weights()) w /= scale_;
    }

    // store distances in network simplex if not externally provided
    if (!external_dists())
      pairwise_distance_.fill_distances(ev0.particles(), ev1.particles(), dists(), extra());

    // run the EarthMoversDistance at this point
    status_ = network_simplex_.compute(n0(), n1());
    emd_ = network_simplex_.total_cost();

    // account for weight scale if not normed
    if (status_ == EMDStatus::Success && !norm_)
      emd_ *= scale_;

    // end timing and get duration
    if (do_timing_)
      store_duration();

    // return status
    return status_;
  }

  // access dists
  ValueVector dists() const {
    return ValueVector(network_simplex_.dists().begin(),
                       network_simplex_.dists().begin() + n0_*n1_);
  }

  // emd flow values between particle i in event0 and particle j in event1
  Value flow(std::size_t i, std::size_t j) const {
    if (i >= n0_ || j >= n1_)
      throw std::out_of_range("EMD::flow - Indices out of range");

    return network_simplex_.flows()[i*n1_ + j] * scale_;
  }

  // returns all flows 
  ValueVector flows() const {

    // copy flows in the valid range
    ValueVector unscaled_flows(network_simplex_.flows().begin(), 
                               network_simplex_.flows().begin() + n0_*n1_);
    // unscale all values
    for (Value & f: unscaled_flows)
      f *= scale_;

    return unscaled_flows;
  }

#ifdef SWIG_WASSERSTEIN
  // make dists available publicly (avoid name conflict in SWIG with leading underscore)
  ValueVector & _dists() { return dists(); }

private:
#endif

  // writeable dists
  ValueVector & dists() { return network_simplex_.dists(); }

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
    if (norm_)
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
  long long chunksize_;
  ExternalEMDHandler * handler_;
  bool do_pairwise_timing_, store_sym_emds_flattened_, throw_on_error_;
  std::ostream * print_stream_;
  std::ostringstream oss_;

  // vectors of events and EnergyMoversDistances
  EventVector events_;
  ValueVector emds_, full_emds_;
  std::vector<std::string> error_messages_;

  // info about stored events
  std::size_t nevA_, nevB_, emd_counter_, num_emds_, num_emds_width_;
  EMDPairsStorage emd_storage_;
  bool two_event_sets_;

public:

  // contructor that initializes the EMD object, uses the same default arguments
  PairwiseEMD(Value R = 1, Value beta = 1, bool norm = false,
              int num_threads = -1,
              long long chunksize = 0,
              bool do_timing = true,
              bool store_sym_emds_flattened = true,
              bool throw_on_error = false,
              unsigned n_iter_max = 100000,
              Value epsilon_large_factor = 10000,
              Value epsilon_small_factor = 1,
              std::ostream & os = std::cout) :
    num_threads_(determine_num_threads(num_threads)),
    emd_objs_(num_threads_, EMD(R, beta, norm, do_timing, false,
                                n_iter_max, epsilon_large_factor, epsilon_small_factor))
  {
    setup(chunksize, &os, store_sym_emds_flattened, throw_on_error);
  }

// avoid overloading constructor so swig can handle keyword arguments
#ifndef SWIG

  // contructor uses existing EMD instance
  PairwiseEMD(const EMD & emd,
              int num_threads = -1,
              long long chunksize = 0,
              bool store_sym_emds_flattened = true,
              bool throw_on_error = false,
              std::ostream & os = std::cout) :
    num_threads_(determine_num_threads(num_threads)),
    emd_objs_(num_threads_, emd)
  {
    if (emd.external_dists())
      throw std::invalid_argument("Cannot use PairwiseEMD with external distances");

    setup(chunksize, &os, store_sym_emds_flattened, throw_on_error);
  }

#endif // SWIG

  // virtual destructor
  virtual ~PairwiseEMD() {}

  // return a description of this object as a string
  std::string description() const {
    std::ostringstream oss;
    oss << "Pairwise" << emd_objs_[0].description(false) << '\n'
        << "  num_threads - " << num_threads_ << '\n'
        << "  chunksize - ";

    // handle chunksize logic
    if (chunksize_ > 0) oss << chunksize_;
    else {
      oss << "auto, " << (chunksize_ != 0 ? std::abs(chunksize_) : 1) << " total chunks";
      if (chunksize_ == 0) oss << ", no printing";
    }

    oss << '\n'
        << "  store_sym_emds_flattened - " << (store_sym_emds_flattened_ ? "true\n" : "false\n")
        << "  throw_on_error - " << (throw_on_error_ ? "true\n" : "false\n")
        << '\n'
        << (handler_ ? handler_->description() : "  Pairwise EMD distance matrix stored internally\n");
        
    emd_objs_[0].output_preprocessors(oss);

    return oss.str();
  }

  // add preprocessor to internal list
  template<template<class> class P, typename... Args>
  PairwiseEMD & preprocess(Args && ... args) {
    for (EMD & emd_obj : emd_objs_)
      emd_obj.template preprocess<P>(std::forward<Args>(args)...);
    return *this;
  }

  // externally set the number of EMD evaluations that will be spooled to each OpenMP thread at a time
  void set_omp_dynamic_chunksize(int chunksize) {
    omp_dynamic_chunksize_ = std::abs(chunksize);
  }

  // set a handler to process EMDs on the fly instead of storing them
  void set_external_emd_handler(ExternalEMDHandler & handler) {
    handler_ = &handler;
  }

  // clears internal storage
  void clear(bool free_memory = true) {

    events_.clear();
    emds_.clear();
    full_emds_.clear();
    error_messages_.clear();

    emd_storage_ = External;
    nevA_ = nevB_ = emd_counter_ = num_emds_ = 0;
    omp_dynamic_chunksize_ = 10;

    if (free_memory) {
      delete handler_;
      handler_ = nullptr;
      EventVector().swap(events_);
      ValueVector().swap(emds_);
      ValueVector().swap(full_emds_);
      std::vector<std::string>().swap(error_messages_);
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

  // access a specific emd
  Value emd(std::ptrdiff_t i, std::ptrdiff_t j) const {

    // check for External handling, in which case we don't have any emds stored
    if (handler_ != nullptr)
      throw std::logic_error("EMD requested but external handler provided, so no EMDs stored");

    // allow for negative indices
    if (i < 0) i += nevA_;
    if (j < 0) j += nevB_;

    // check for improper indexing
    if (std::size_t(i) >= nevA_ || std::size_t(j) >= nevB_) {
      std::ostringstream message("PairwiseEMD::emd - Accessing emd value at (", std::ios_base::ate);
      message << i << ", " << j << ") exceeds allowed range";
      throw std::out_of_range(message.str());
    }

    // index into emd vector
    if (emd_storage_ == FlattenedSymmetric)
      return (i == j ? 0 : emds_[i > j ? index_symmetric(i, j) : index_symmetric(j, i)]);
    else return emds_[i*nevB_ + j];
  }

  // access all emds as a matrix flattened into a vector
  const ValueVector & emds() {

    // check if we need to construct a new full matrix from a flattened symmetric one
    if (emd_storage_ == FlattenedSymmetric) {

      // allocate a new vector for holding the full emds
      full_emds_.resize(nevA_*nevB_);

      // zeros on the diagonal
      for (std::size_t i = 0; i < nevA_; i++)
        full_emds_[i*i] = 0;

      // fill out matrix
      for (std::size_t i = 0; i < nevA_; i++)
        for (std::size_t j = 0; j < i; j++)
          full_emds_[i*nevB_ + j] = full_emds_[j*nevB_ + i] = emds_[index_symmetric(i, j)];

      return full_emds_;
    }

    // no emds stored
    else if (emd_storage_ == External) return full_emds_;

    // full emds stored
    else return emds_;
  }

  // error reporting
  bool errored() const { return error_messages_.size() > 0; }
  const std::vector<std::string> & error_messages() const { return error_messages_; }
  void report_errors(std::ostream & os = std::cerr) const {
    for (const std::string & err : error_messages_)
      os << err << '\n';
    os << std::flush;
  }

  // number of unique emds computed
  std::size_t num_emds() const { return num_emds_; }

  // access events
  std::size_t nevA() const { return nevA_; }
  std::size_t nevB() const { return nevB_; }
  const EventVector & events() const { return events_; }

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
  void init(std::size_t nev) {

    clear(false);
    nevA_ = nevB_ = nev;
    two_event_sets_ = false;

    // start clock for overall timing
    if (do_pairwise_timing_)
      emd_objs_[0].start_timing();

    // resize emds
    num_emds_ = nevA_*(nevA_ - 1)/2;
    if (handler_ == nullptr) {
      emd_storage_ = (store_sym_emds_flattened_ ? FlattenedSymmetric : FullSymmetric);
      emds_.resize(emd_storage_ == FullSymmetric ? nevA_*nevB_ : num_emds_);
    }

    // reserve space for events
    events_.reserve(nevA_);
  }

  // init pairs
  void init(std::size_t nevA, std::size_t nevB) {

    clear(false);
    nevA_ = nevA;
    nevB_ = nevB;
    two_event_sets_ = true;
    
    // start clock for overall timing
    if (do_pairwise_timing_)
      emd_objs_[0].start_timing();

    // resize emds
    num_emds_ = nevA_*nevB_;
    if (handler_ == nullptr) {
      emd_storage_ = Full;
      emds_.resize(num_emds_);  
    }

    // reserve space for events
    events_.reserve(nevA_ + nevB_);
  }

  void compute() {

    num_emds_width_ = std::to_string(num_emds_).size();

    long long chunksize(chunksize_);
    if (chunksize < 0) {
      chunksize = num_emds_/std::abs(chunksize_);
      if (num_emds_ % std::abs(chunksize_) != 0)
        chunksize++;
    }
    else if (chunksize == 0)
      chunksize = num_emds_;

    if (chunksize_ != 0) {
      oss_.str("Finished preprocessing ");
      oss_ << events_.size() << " events";
      if (do_pairwise_timing_)
        oss_ << " in " << std::setprecision(4) << emd_objs_[0].store_duration() << 's';
      *print_stream_ << oss_.str() << std::endl;
    }

    int omp_for_dynamic_chunksize(omp_dynamic_chunksize_);
    if (omp_for_dynamic_chunksize < chunksize/num_threads_)
      omp_for_dynamic_chunksize = chunksize/num_threads_;

    // iterate over emd pairs
    std::mutex failure_mutex;
    std::size_t begin(0);
    while (emd_counter_ < num_emds_ && !(throw_on_error_ && error_messages_.size())) {
      emd_counter_ += std::size_t(chunksize);
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

          std::size_t i(k/nevB_), j(k%nevB_);
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
      if (chunksize_ != 0)
        print_update();
    }

    if (throw_on_error_ && error_messages_.size())
      throw std::runtime_error(error_messages_.front());
  }

private:

  // determine the number of threads to use
  int determine_num_threads(int num_threads) {
    #ifdef _OPENMP
      return num_threads == -1 ? omp_get_max_threads() : num_threads;
    #else
      return 1;
    #endif
  }

  // init from constructor
  void setup(long long chunksize, std::ostream * os,
             bool store_sym_emds_flattened, bool throw_on_error) {
    
    // store arguments, from constructor
    chunksize_ = chunksize;
    store_sym_emds_flattened_ = store_sym_emds_flattened;
    throw_on_error_ = throw_on_error;
    print_stream_ = os;
    handler_ = nullptr;

    // setup stringstream for printing
    oss_ = std::ostringstream(std::ios_base::ate);
    oss_.setf(std::ios_base::fixed, std::ios_base::floatfield);

    // handle timing
    do_pairwise_timing_ = emd_objs_[0].do_timing_;
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

  void record_failure(EMDStatus status, std::size_t i, std::size_t j) {
    std::ostringstream message;
    message << "PairwiseEMD::compute - Issue with EMD ("
            << i << ", " << j << "), error code " << int(status);
    error_messages_.push_back(message.str());
    *print_stream_ << error_messages_.back() << '\n';
  }

  // indexes lower triangle of symmetric matrix with zeros on diagonal that has been flattened into 1D
  static std::size_t index_symmetric(std::size_t i, std::size_t j) {
    return i*(i - 1)/2 + j;
  }

  void print_update() {

    // output to oss
    oss_.str("  ");
    oss_ << std::setw(num_emds_width_) << emd_counter_ << " / "
         << std::setw(num_emds_width_) << num_emds_ << "  EMDs computed  - "
         << std::setprecision(2) << std::setw(6) << double(emd_counter_)/num_emds_*100
         << "% completed";

    // check for timing
    if (do_pairwise_timing_) {
      oss_ << "  -  " << std::setprecision(3) << emd_objs_[0].store_duration() << 's';
    }

    // acquire Python GIL if in SWIG in order to check for signals
    #ifdef SWIG
      SWIG_PYTHON_THREAD_BEGIN_BLOCK;
      *print_stream_ << oss_.str() << std::endl;
      if (PyErr_CheckSignals() != 0)
        throw std::runtime_error("KeyboardInterrupt received in PairwiseEMD::compute");
      SWIG_PYTHON_THREAD_END_BLOCK;
    #else
      *print_stream_ << oss_.str() << std::endl;
    #endif
  }

}; // PairwiseEMD

END_EMD_NAMESPACE

#endif // WASSERSTEIN_EMD_HH
