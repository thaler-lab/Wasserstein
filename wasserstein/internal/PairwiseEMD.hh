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

/*  _____        _____ _______          _______  _____ ______ 
 * |  __ \ /\   |_   _|  __ \ \        / /_   _|/ ____|  ____|
 * | |__) /  \    | | | |__) \ \  /\  / /  | | | (___ | |__   
 * |  ___/ /\ \   | | |  _  / \ \/  \/ /   | |  \___ \|  __|  
 * | |  / ____ \ _| |_| | \ \  \  /\  /   _| |_ ____) | |____ 
 * |_| /_/    \_\_____|_|  \_\  \/  \/   |_____|_____/|______|
 *   ______ __  __ _____  
 * |  ____|  \/  |  __ \
 * | |__  | \  / | |  | |
 * |  __| | |\/| | |  | |
 * | |____| |  | | |__| |
 * |______|_|  |_|_____/
 */

#ifndef WASSERSTEIN_PAIRWISEEMD_HH
#define WASSERSTEIN_PAIRWISEEMD_HH

// C++ standard library
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// OpenMP for multithreading
#ifdef _OPENMP
#include <omp.h>
#endif

#include "EMDUtils.hh"
#include "ExternalEMDHandler.hh"

BEGIN_EMD_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// PairwiseEMD - Facilitates computing EMDs between all event-pairs in a set
////////////////////////////////////////////////////////////////////////////////

template<class EMD, typename Value = typename EMD::value_type>
class PairwiseEMD {
public:

  typedef Value value_type;
  typedef typename EMD::Event Event;
  typedef std::vector<Event> EventVector;

private:

  // needs to be before emd_objs_ in order to determine actual number of threads
  int num_threads_, omp_dynamic_chunksize_;

  // EMD objects
  std::vector<EMD> emd_objs_;

  // variables local to this class
  index_type print_every_;
  ExternalEMDHandler<Value> * handler_;
  unsigned verbose_;
  bool store_sym_emds_flattened_, throw_on_error_, request_mode_;
  std::ostream * print_stream_;
  std::ostringstream oss_;

  // vectors of events and EnergyMoversDistances
  EventVector events_;
  std::vector<Value> emds_, full_emds_;
  std::vector<std::string> error_messages_;

  // info about stored events
  index_type nevA_, nevB_, num_emds_, emd_counter_, num_emds_width_;
  EMDPairsStorage emd_storage_;
  bool two_event_sets_;

public:

  // contructor that initializes the EMD object, uses the same default arguments
  PairwiseEMD(Value R = 1, Value beta = 1, bool norm = false,
              int num_threads = -1,
              index_type print_every = -10,
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
              index_type print_every = -10,
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
  void set_external_emd_handler(ExternalEMDHandler<Value> & handler) {
    handler_ = &handler;
  }
  ExternalEMDHandler<Value> * external_emd_handler() {
    if (!have_external_emd_handler())
      throw std::logic_error("no external emd handler set");

    return handler_;
  }
  bool have_external_emd_handler() const {
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

  // query storage mode
  EMDPairsStorage storage() const {
    return emd_storage_;
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

    emd_storage_ = EMDPairsStorage::External;
    nevA_ = nevB_ = emd_counter_ = num_emds_ = 0;

    // start clock for overall timing
    emd_objs_[0].start_timing();

    if (free_memory) {
      handler_ = nullptr;
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
  void operator()(const std::vector<ProtoEvent> & proto_events,
                  const std::vector<Value> & event_weights = {}) {
    init(proto_events.size());
    store_proto_events(proto_events, event_weights);
    compute();
  }

  // compute EMDs between two sets of proto events, including preprocessing
  template<class ProtoEventA, class ProtoEventB>
  void operator()(const std::vector<ProtoEventA> & proto_eventsA,
                  const std::vector<ProtoEventB> & proto_eventsB,
                  const std::vector<Value> & event_weightsA = {},
                  const std::vector<Value> & event_weightsB = {}) {
    init(proto_eventsA.size(), proto_eventsB.size());
    store_proto_events(proto_eventsA, event_weightsA);
    store_proto_events(proto_eventsB, event_weightsB);
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
    events_.reserve(nevA() + nevB());
    events_.insert(events_.end(), eventsA.begin(), eventsA.end());
    events_.insert(events_.end(), eventsB.begin(), eventsB.end());
    compute();
  }

  // access events
  index_type nevA() const { return nevA_; }
  index_type nevB() const { return nevB_; }
  const EventVector & events() const { return events_; }

  // number of unique EMDs computed
  index_type num_emds() const { return num_emds_; }

  // error reporting
  bool errored() const { return error_messages_.size() > 0; }
  const std::vector<std::string> & error_messages() const { return error_messages_; }

  // access timing information
  double duration() const { return emd_objs_[0].duration(); }

  // access all emds as a matrix flattened into a vector
  const std::vector<Value> & emds(bool flattened = false) {

    // check for having no emds stored
    if (emd_storage_ == EMDPairsStorage::External)
      throw std::logic_error("No EMDs stored");

    // check if we need to construct a new full matrix from a flattened symmetric one
    if (emd_storage_ == EMDPairsStorage::FlattenedSymmetric && !flattened) {

      // allocate a new vector for holding the full emds
      full_emds_.resize(nevA()*nevB());

      // zeros on the diagonal
      for (index_type i = 0; i < nevA(); i++)
        full_emds_[i*i] = 0;

      // fill out matrix (index into upper triangular part)
      for (index_type i = 0; i < nevA(); i++)
        for (index_type j = i + 1; j < nevB(); j++)
          full_emds_[i*nevB() + j] = full_emds_[j*nevB() + i] = emds_[index_symmetric(i, j)];

      return full_emds_;
    }

    // return stored emds_
    return emds_;
  }

  // access a specific emd
  Value emd(index_type i, index_type j, int thread = 0) {

    // allow for negative indices
    if (i < 0) i += nevA();
    if (j < 0) j += nevB();

    // check for improper indexing
    if (i >= nevA() || j >= nevB() || i < 0 || j < 0) {
      std::ostringstream message;
      message << "PairwiseEMD::emd - Accessing emd value at ("
              << i << ", " << j << ") exceeds allowed range";
      throw std::out_of_range(message.str());
    }

    // calculate EMD if in request mode
    if (request_mode()) {

      if (thread >= num_threads_)
        throw std::out_of_range("invalid thread index");

      // run and check for failure
      const Event & eventA(events_[i]), & eventB(events_[two_event_sets_ ? nevA() + j : j]);
      check_emd_status(emd_objs_[thread].compute(eventA, eventB));
      if (handler_)
        (*handler_)(emd_objs_[thread].emd(), eventA.event_weight() * eventB.event_weight());
      return emd_objs_[thread].emd();
    }

    // check for External handling, in which case we don't have any emds stored
    if (emd_storage_ == EMDPairsStorage::External)
      throw std::logic_error("EMD requested but external handler provided, so no EMDs stored");

    // index into emd vector (j always bigger than i because upper triangular storage)
    if (emd_storage_ == EMDPairsStorage::FlattenedSymmetric)
      return (i == j ? 0 : emds_[index_symmetric(i, j)]);

    else return emds_[i*nevB() + j];
  }

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
  void init(index_type nev) {

    if (!request_mode())
      clear(false);

    nevA_ = nevB_ = nev;
    two_event_sets_ = false;

    // resize emds
    num_emds_ = nev*(nev - 1)/2;
    if (!have_external_emd_handler() && !request_mode()) {
      emd_storage_ = (store_sym_emds_flattened_ ? EMDPairsStorage::FlattenedSymmetric : EMDPairsStorage::FullSymmetric);
      emds_.resize(emd_storage_ == EMDPairsStorage::FullSymmetric ? nevA()*nevB() : num_emds());
    }

    // reserve space for events
    events_.reserve(nevA());
  }

  // init pairs
  void init(index_type nevA, index_type nevB) {

    if (!request_mode())
      clear(false);

    nevA_ = nevA;
    nevB_ = nevB;
    two_event_sets_ = true;

    // resize emds
    num_emds_ = nevA * nevB;
    if (!have_external_emd_handler() && !request_mode()) {
      emd_storage_ = EMDPairsStorage::Full;
      emds_.resize(num_emds());  
    }

    // reserve space for events
    events_.reserve(nevA + nevB);
  }

  void compute() {

    // check that we're not in request mode
    if (request_mode())
      throw std::runtime_error("cannot compute pairwise EMDs in request mode");

    num_emds_width_ = std::to_string(num_emds()).size();

    // note that print_every == 0 is handled in setup()
    index_type print_every(print_every_);
    if (print_every < 0) {
      print_every = num_emds()/std::abs(print_every_);
      if (print_every == 0 || num_emds() % std::abs(print_every_) != 0)
        print_every++;
    }

    if (verbose_) {
      oss_.str("Finished preprocessing ");
      oss_ << events_.size() << " events in "
           << std::setprecision(4) << emd_objs_[0].store_duration() << 's';
      *print_stream_ << oss_.str() << std::endl;
    }

    // iterate over emd pairs
    std::mutex failure_mutex;
    index_type begin(0);
    while (emd_counter_ < num_emds() && !(throw_on_error_ && error_messages().size())) {
      emd_counter_ += print_every;
      if (emd_counter_ > num_emds()) emd_counter_ = num_emds();

      #pragma omp parallel num_threads(num_threads_) default(shared)
      {
        // grab EMD object for this thread
        EMD & emd_obj(emd_objs_[get_thread_id()]);

        // parallelize loop over EMDs
        #pragma omp for schedule(dynamic, omp_dynamic_chunksize())
        for (index_type k = begin; k < emd_counter_; k++) {

          index_type i(k/nevB()), j(k%nevB());
          if (two_event_sets_) {

            // run and check for failure
            const Event & eventA(events_[i]), & eventB(events_[nevA() + j]);
            EMDStatus status(emd_obj.compute(eventA, eventB));
            if (status != EMDStatus::Success) {
              std::lock_guard<std::mutex> failure_lock(failure_mutex);
              record_failure(status, i, j);
            }

            if (handler_)
              (*handler_)(emd_obj.emd(), eventA.event_weight() * eventB.event_weight());
            else emds_[k] = emd_obj.emd(); 
          }
          else {

            // this properly sets indexing for symmetric case
            if (j >= ++i) {
              i = nevA() - i;
              j = nevA() - j - 1;
            }

            // run and check for failure
            const Event & eventA(events_[i]), & eventB(events_[j]);
            EMDStatus status(emd_obj.compute(eventA, eventB));
            if (status != EMDStatus::Success) {
              std::lock_guard<std::mutex> failure_lock(failure_mutex);
              record_failure(status, i, j);
            }

            // store emd value
            if (emd_storage_ == EMDPairsStorage::FlattenedSymmetric)
              emds_[index_symmetric(i, j)] = emd_obj.emd();

            else if (emd_storage_ == EMDPairsStorage::External)
              (*handler_)(emd_obj.emd(), eventA.event_weight() * eventB.event_weight());

            else if (emd_storage_ == EMDPairsStorage::FullSymmetric)
              emds_[i*nevB() + j] = emds_[j*nevB() + i] = emd_obj.emd();

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

  // get thread id
  int get_thread_id() const {
    #ifdef _OPENMP
      return omp_get_thread_num();
    #else
      return 0;
    #endif
  }

  // init from constructor
  void setup(index_type print_every, unsigned verbose,
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
  void store_proto_events(const std::vector<ProtoEvent> & proto_events,
                          const std::vector<Value> & event_weights) {

    if (proto_events.size() != event_weights.size()) {
      if (event_weights.size() == 0)
        for (const ProtoEvent & proto_event : proto_events) {
          events_.emplace_back(proto_event);
          preprocess_back_event();
        }
      else throw std::invalid_argument("length of event_weights does not match proto_events");
    }
    else
      for (unsigned i = 0; i < proto_events.size(); i++) {
        events_.emplace_back(proto_events[i], event_weights[i]);
        preprocess_back_event();
      }
  }

  void record_failure(EMDStatus status, index_type i, index_type j) {
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

  // indexes upper triangle of symmetric matrix with zeros on diagonal that has been flattened into 1D
  // see scipy's squareform function
  index_type index_symmetric(index_type i, index_type j) {

    // treat i as the row and j as the column
    if (j > i)
      return num_emds() - (nevA() - i)*(nevA() - i - 1)/2 + j - i - 1;

    // treat i as the column and j as the row
    if (i > j)
      return num_emds() - (nevA() - j)*(nevA() - j - 1)/2 + i - j - 1;

    return -1;
  }

  void print_update() {

    // prepare message
    if (verbose_) {
      oss_.str("  ");
      oss_ << std::setw(num_emds_width_) << emd_counter_ << " / "
           << std::setw(num_emds_width_) << num_emds() << "  EMDs computed  - "
           << std::setprecision(2) << std::setw(6) << double(emd_counter_)/num_emds()*100
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

#endif // WASSERSTEIN_PAIRWISEEMD_HH