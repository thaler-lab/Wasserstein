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
#include <iterator>
#include <stdexcept>
#include <type_traits>

#include "PairwiseEMDBase.hh"


BEGIN_WASSERSTEIN_NAMESPACE

template<typename InputIterator>
using RequireInputIterator = typename std::enable_if<std::is_convertible<typename
                                      std::iterator_traits<InputIterator>::iterator_category,
                                      std::input_iterator_tag>::value>::type;

////////////////////////////////////////////////////////////////////////////////
// PairwiseEMD - Facilitates computing EMDs between all event-pairs in a set
////////////////////////////////////////////////////////////////////////////////

template<class EMD, typename Value = typename EMD::value_type>
class PairwiseEMD : public PairwiseEMDBase<Value> {
public:

  typedef Value value_type;
  typedef typename EMD::Event Event;
  typedef PairwiseEMDBase<Value> Base;

private:

  // EMD objects
  std::vector<EMD> emd_objs_;

  // vector of events
  std::vector<Event> events_;
  bool two_event_sets_;

  std::ostringstream oss_;
  index_type emd_counter_;

#ifdef WASSERSTEIN_SERIALIZATION
  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive & ar, const unsigned int version) const {
    ar & boost::serialization::base_object<Base>(*this)
       & emd_objs_ & two_event_sets_ & emd_counter_;
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int version) {
    ar & boost::serialization::base_object<Base>(*this)
       & emd_objs_ & two_event_sets_ & emd_counter_;

    construct();
    events().resize(nevA() + nevB());
  }
#endif

public:

  // contructor that initializes the EMD object, uses the same default arguments
  PairwiseEMD(Value R = 1, Value beta = 1, bool norm = false,
              int num_threads = -1,
              index_type print_every = -10,
              unsigned verbose = 1,
              bool request_mode = false,
              bool store_sym_emds_raw = true,
              bool throw_on_error = false,
              unsigned omp_dynamic_chunksize = 10,
              std::size_t n_iter_max = 100000,
              Value epsilon_large_factor = 1000,
              Value epsilon_small_factor = 1,
              std::ostream & os = std::cout) :
    Base(num_threads, print_every,
         verbose, omp_dynamic_chunksize,
         request_mode, store_sym_emds_raw, throw_on_error,
         os),
    emd_objs_(this->num_threads(), EMD(R, beta, norm, false, false,
                                       n_iter_max, epsilon_large_factor, epsilon_small_factor))
  {
    clear(false);
  }

// avoid overloading constructor so swig can handle keyword arguments
#ifndef SWIG

  // contructor uses existing EMD instance
  PairwiseEMD(const EMD & emd,
              int num_threads = -1,
              index_type print_every = -10,
              unsigned verbose = 1,
              bool request_mode = false,
              bool store_sym_emds_raw = true,
              bool throw_on_error = false,
              unsigned omp_dynamic_chunksize = 10,
              std::ostream & os = std::cout) :
    Base(num_threads, print_every,
         verbose, omp_dynamic_chunksize,
         request_mode, store_sym_emds_raw, throw_on_error,
         os),
    emd_objs_(this->num_threads(), emd)
  {
    if (emd.external_dists())
      throw std::invalid_argument("Cannot use PairwiseEMD with external distances");

    clear(false);
  }

#endif // SWIG

  virtual ~PairwiseEMD() = default;

  // these avoid needing this-> with base class attributes
  #ifndef SWIG_PREPROCESSOR
    using Base::nevA;
    using Base::nevB;
    using Base::num_emds;
  #endif

  // add preprocessor to internal list
  template<template<class> class P, typename... Args>
  PairwiseEMD & preprocess(Args && ... args) {
    for (EMD & emd_obj : emd_objs_)
      emd_obj.template preprocess<P>(std::forward<Args>(args)...);
    return *this;
  }

  // return a description of this object as a string
  std::string description() const {
    std::ostringstream oss;
    oss << std::boolalpha
        << "Pairwise" << emd_objs_[0].description(false) << '\n'
        << "  num_threads - " << this->num_threads() << '\n'
        << "  print_every - ";

    // handle print_every logic
    if (this->print_every_ > 0) oss << this->print_every_;
    else
      oss << "auto, " << std::abs(this->print_every_) << " total chunks";

    oss << '\n'
        << "  request_mode - " << this->request_mode() << '\n'
        << "  store_sym_emds_raw - " << this->store_sym_emds_raw_ << '\n'
        << "  throw_on_error - " << this->throw_on_error_ << '\n'
        << "  omp_dynamic_chunksize - " << this->omp_dynamic_chunksize() << '\n'
        << '\n'
        << (this->handler_ ? this->handler_->description() : "  Pairwise EMD distance matrix stored internally\n");
      
    // this will not print preprocessors if there aren't any  
    emd_objs_[0].output_preprocessors(oss);

    return oss.str();
  }

#ifndef SWIG_PREPROCESSOR

  // compute EMDs between all pairs of proto events, including preprocessing
  template<class ProtoEvent>
  void operator()(const std::vector<ProtoEvent> & proto_events,
                  const std::vector<Value> & event_weights = {}) {

    assert(std::distance(proto_events.begin(), proto_events.end()) == proto_events.size());
    operator()(proto_events.begin(), proto_events.end(), event_weights);
  }

  // version taking iterators
  template<class ProtoEventIt, typename = RequireInputIterator<ProtoEventIt>>
  void operator()(ProtoEventIt proto_events_first, ProtoEventIt proto_events_last,
                  const std::vector<Value> & event_weights = {}) {

    init(std::distance(proto_events_first, proto_events_last));
    store_proto_events(proto_events_first, proto_events_last, event_weights);
    compute();
  }

  // compute EMDs between two sets of proto events, including preprocessing
  template<class ProtoEventA, class ProtoEventB>
  void operator()(const std::vector<ProtoEventA> & proto_eventsA,
                  const std::vector<ProtoEventB> & proto_eventsB,
                  const std::vector<Value> & event_weightsA = {},
                  const std::vector<Value> & event_weightsB = {}) {

    operator()(proto_eventsA.begin(), proto_eventsA.end(),
               proto_eventsB.begin(), proto_eventsB.end(),
               event_weightsA, event_weightsB);
  }

  // version taking iterators
  template<class ProtoEventAIt, class ProtoEventBIt,
           typename = RequireInputIterator<ProtoEventAIt>,
           typename = RequireInputIterator<ProtoEventBIt>>
  void operator()(ProtoEventAIt proto_eventsA_first, ProtoEventAIt proto_eventsA_last,
                  ProtoEventBIt proto_eventsB_first, ProtoEventBIt proto_eventsB_last,
                  const std::vector<Value> & event_weightsA = {},
                  const std::vector<Value> & event_weightsB = {}) {

    init(std::distance(proto_eventsA_first, proto_eventsA_last),
         std::distance(proto_eventsB_first, proto_eventsB_last));
    store_proto_events(proto_eventsA_first, proto_eventsA_last, event_weightsA);
    store_proto_events(proto_eventsB_first, proto_eventsB_last, event_weightsB);
    compute();
  }

#endif // SWIG_PREPROCESSOR

  // compute pairs among same set of events (no preprocessing)
  void compute(const std::vector<Event> & events) {
    init(events.size());
    events_ = events;
    compute();
  }

  // compute pairs between different sets of events
  void compute(const std::vector<Event> & eventsA, const std::vector<Event> & eventsB) {
    init(eventsA.size(), eventsB.size());
    events().reserve(nevA() + nevB());
    events().insert(events().end(), eventsA.begin(), eventsA.end());
    events().insert(events().end(), eventsB.begin(), eventsB.end());
    compute();
  }

  // access events
  const std::vector<Event> & events() const { return events_; }

  // clears internal storage
  void clear(bool free_memory = true) {

    Base::clear(free_memory);

    events().clear();
    emd_counter_ = 0;

    if (free_memory) {
      free_vector(events());
      for (EMD & emd_obj : emd_objs_)
        emd_obj.clear();
    }

    // turn off timing in EMD objects (harmless to call this each time, only needed in constructor)
    for (EMD & emd_obj : emd_objs_)
      emd_obj.do_timing_ = false;

    // initialization that needs to happen when object is created
    construct();
  }

// these should be private for the SWIG Python wrappers and public otherwise
#ifdef SWIG
private:
#endif

  // get/set EMD parameters of actual EMD objects
  Value R() const { return emd_objs_[0].R(); }
  void set_R(Value R) {
    for (EMD & emd_obj : emd_objs_) emd_obj.set_R(R);
  }
  Value beta() const { return emd_objs_[0].beta(); }
  void set_beta(Value beta) {
    for (EMD & emd_obj : emd_objs_) emd_obj.set_beta(beta);
  }
  bool norm() const { return emd_objs_[0].norm(); }
  void set_norm(bool norm) {
    for (EMD & emd_obj : emd_objs_) emd_obj.set_norm(norm);
  }
  void set_network_simplex_params(std::size_t n_iter_max,
                                   Value epsilon_large_factor,
                                   Value epsilon_small_factor) {
    for (EMD & emd_obj : emd_objs_)
      emd_obj.set_network_simplex_params(n_iter_max, epsilon_large_factor, epsilon_small_factor);
  }

  // timing
  double duration() const { return emd_objs_[0].duration(); }

// Wasserstein needs access to these functions in the SWIG Python wrapper, else they can be private
#ifdef SWIG
public:
#else
private:
#endif

  // access modifiable events
  std::vector<Event> & events() { return events_; }

  void preprocess_back_event() {
    emd_objs_[0].preprocess(events_.back());
  }

  // init self pairs
  void init(index_type nev) {

    if (!this->request_mode())
      clear(false);

    this->nevA_ = this->nevB_ = nev;
    two_event_sets_ = false;

    // resize emds
    this->num_emds_ = nev*(nev - 1)/2;
    if (!this->have_external_emd_handler() && !this->request_mode()) {
      this->emd_storage_ = (this->store_sym_emds_raw_ ? EMDPairsStorage::FlattenedSymmetric : EMDPairsStorage::FullSymmetric);
      this->emds_.resize(this->emd_storage_ == EMDPairsStorage::FullSymmetric ? nevA()*nevB() : num_emds());
    }

    // reserve space for events
    events().reserve(nevA());
  }

  // init pairs
  void init(index_type nevA, index_type nevB) {

    if (!this->request_mode())
      clear(false);

    this->nevA_ = nevA;
    this->nevB_ = nevB;
    two_event_sets_ = true;

    // resize emds
    this->num_emds_ = nevA * nevB;
    if (!this->have_external_emd_handler() && !this->request_mode()) {
      this->emd_storage_ = EMDPairsStorage::Full;
      this->emds_.resize(num_emds());  
    }

    // reserve space for events
    events().reserve(nevA + nevB);
  }

  void compute() {

    // check that we're not in request mode
    if (this->request_mode())
      throw std::runtime_error("cannot compute pairwise EMDs in request mode");

    // note that print_every == 0 is handled in finish_setup()
    index_type print_every(this->print_every_);
    if (print_every < 0) {
      print_every = num_emds()/std::abs(this->print_every_);
      if (print_every == 0 || num_emds() % std::abs(this->print_every_) != 0)
        print_every++;
    }

    if (this->verbose_) {
      oss_.str("Finished preprocessing ");
      oss_ << events_.size() << " events in "
           << std::setprecision(4) << emd_objs_[0].store_duration() << 's';
      *(this->print_stream_) << oss_.str() << std::endl;
    }

    // iterate over emd pairs
    std::mutex failure_mutex;
    index_type begin(0);
    while (emd_counter_ < num_emds() && !(this->throw_on_error_ && this->errored())) {

      emd_counter_ += print_every;
      if (emd_counter_ > num_emds())
        emd_counter_ = num_emds();

      #pragma omp parallel num_threads(this->num_threads()) default(shared)
      {
        // grab EMD object for this thread
        EMD & emd_obj(emd_objs_[get_thread_id()]);

        // parallelize loop over EMDs
        #pragma omp for schedule(dynamic, this->omp_dynamic_chunksize())
        for (index_type k = begin; k < emd_counter_; k++) {

          index_type i(k/nevB()), j(k%nevB());
          if (two_event_sets_) {

            // run and check for failure
            const Event & eventA(events()[i]), & eventB(events()[nevA() + j]);
            EMDStatus status(emd_obj.compute(eventA, eventB));
            if (status != EMDStatus::Success)
              record_failure(failure_mutex, status, i, j);

            if (this->emd_storage_ == EMDPairsStorage::External)
              (*(this->handler_))(emd_obj.emd(), eventA.event_weight() * eventB.event_weight());
            else this->emds_[k] = emd_obj.emd(); 
          }
          else {

            // this properly sets indexing for symmetric case
            if (j >= ++i) {
              i = nevA() - i;
              j = nevA() - j - 1;
            }

            // run and check for failure
            const Event & eventA(events()[i]), & eventB(events()[j]);
            EMDStatus status(emd_obj.compute(eventA, eventB));
            if (status != EMDStatus::Success)
              record_failure(failure_mutex, status, i, j);

            // store emd value
            if (this->emd_storage_ == EMDPairsStorage::FlattenedSymmetric)
              this->emds_[this->index_symmetric(i, j)] = emd_obj.emd();

            else if (this->emd_storage_ == EMDPairsStorage::External)
              (*(this->handler_))(emd_obj.emd(), eventA.event_weight() * eventB.event_weight());

            else if (this->emd_storage_ == EMDPairsStorage::FullSymmetric)
              this->emds_[i*nevB() + j] = this->emds_[j*nevB() + i] = emd_obj.emd();

            else std::cerr << "Should never get here\n";
          }
        }
      }

      // update and do printing
      begin = emd_counter_;
      print_update();
    }

    if (this->throw_on_error_ && this->errored())
      throw std::runtime_error(this->error_messages().front());
  }

private:

  void construct() {

    // start clock for overall timing
    emd_objs_[0].start_timing();

    // setup stringstream for printing
    oss_ = std::ostringstream(std::ios_base::ate);
    oss_.setf(std::ios_base::fixed, std::ios_base::floatfield);
  }

  Value _evaluate_emd(index_type i, index_type j, int thread) {
    
    // run and check for failure
    const Event & eventA(events_[i]), & eventB(events_[two_event_sets_ ? nevA() + j : j]);
    check_emd_status(emd_objs_[thread].compute(eventA, eventB));
    if (this->handler_)
      (*(this->handler_))(emd_objs_[thread].emd(), eventA.event_weight() * eventB.event_weight());
    return emd_objs_[thread].emd();
  }

  // store events
  template<class ProtoEventIt>
  void store_proto_events(ProtoEventIt proto_events_first,
                          ProtoEventIt proto_events_last,
                          const std::vector<Value> & event_weights) {

    std::size_t nev(std::distance(proto_events_first, proto_events_last));
    ProtoEventIt p(proto_events_first);

    if (nev != event_weights.size()) {
      if (event_weights.size() == 0)
        do {
          events().emplace_back(*p);
          preprocess_back_event();
        } while (++p != proto_events_last);

      else throw std::invalid_argument("length of event_weights does not match proto_events");
    }

    else {
      for (index_type i = 0; p != proto_events_last; i++, ++p) {
        events().emplace_back(*p, event_weights[i++]);
        preprocess_back_event();
      }
    }
  }

  static int get_thread_id() {
    #ifdef _OPENMP
      return omp_get_thread_num();
    #else
      return 0;
    #endif
  }

  void record_failure(std::mutex & failure_mutex, EMDStatus status, index_type i, index_type j) {
    std::lock_guard<std::mutex> failure_lock(failure_mutex);

    std::ostringstream message;
    message << "PairwiseEMD::compute - Issue with EMD between events ("
            << i << ", " << j << "), error code " << int(status);
    this->error_messages_.push_back(message.str());

    // acquire Python GIL if in SWIG in order to print message
    #ifdef SWIG
      SWIG_PYTHON_THREAD_BEGIN_BLOCK;
        std::cerr << this->error_messages().back() << '\n';
      SWIG_PYTHON_THREAD_END_BLOCK;
    #else
      std::cerr << this->error_messages().back() << '\n';
    #endif
  }

  void print_update() {

    // prepare message
    if (this->verbose_) {
      unsigned num_emds_width(std::to_string(num_emds()).size());
      oss_.str("  ");
      oss_ << std::setw(num_emds_width) << emd_counter_ << " / "
           << std::setw(num_emds_width) << num_emds() << "  EMDs computed  - "
           << std::setprecision(2) << std::setw(6) << double(emd_counter_)/num_emds()*100
           << "% completed - "
           << std::setprecision(3) << emd_objs_[0].store_duration() << 's';  
    }

    // acquire Python GIL if in SWIG in order to check for signals and print message
    #ifdef SWIG
      SWIG_PYTHON_THREAD_BEGIN_BLOCK;
      if (this->verbose_) *(this->print_stream_) << oss_.str() << std::endl;
      if (PyErr_CheckSignals() != 0)
        throw std::runtime_error("KeyboardInterrupt received in PairwiseEMD::compute");
      SWIG_PYTHON_THREAD_END_BLOCK;
    #else
      if (this->verbose_) *(this->print_stream_) << oss_.str() << std::endl;
    #endif
  }

}; // PairwiseEMD

END_WASSERSTEIN_NAMESPACE

#endif // WASSERSTEIN_PAIRWISEEMD_HH
