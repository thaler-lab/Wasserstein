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
 *  ______ __  __ _____  ____           _____ ______ 
 * |  ____|  \/  |  __ \|  _ \   /\    / ____|  ____|
 * | |__  | \  / | |  | | |_) | /  \  | (___ | |__   
 * |  __| | |\/| | |  | |  _ < / /\ \  \___ \|  __|  
 * | |____| |  | | |__| | |_) / ____ \ ____) | |____ 
 * |______|_|  |_|_____/|____/_/    \_\_____/|______|
 */

#ifndef WASSERSTEIN_PAIRWISEEMDBASE_HH
#define WASSERSTEIN_PAIRWISEEMDBASE_HH

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// OpenMP for multithreading
#ifdef _OPENMP
#include <omp.h>
#endif

#include "EMDUtils.hh"
#include "ExternalEMDHandler.hh"


BEGIN_WASSERSTEIN_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// EMDBase - base class to reduce wrapper code for simple EMD access functions
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
class PairwiseEMDBase {
protected:

  // variables corresopnding to constructor arguments
  int num_threads_;
  index_type print_every_;
  unsigned verbose_, omp_dynamic_chunksize_;
  bool request_mode_, store_sym_emds_raw_, throw_on_error_;
  std::ostream * print_stream_;
  
  // variables initialized in/by constructor
  ExternalEMDHandler<Value> * handler_;

  // vectors of EMDs
  std::vector<Value> emds_, full_emds_;
  std::vector<std::string> error_messages_;

  // info about stored events
  index_type nevA_, nevB_, num_emds_;
  EMDPairsStorage emd_storage_;

private:

#ifdef WASSERSTEIN_SERIALIZATION
  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive & ar, const unsigned int version) const {
    ar & num_threads_ & print_every_
       & verbose_ & omp_dynamic_chunksize_
       & request_mode_ & store_sym_emds_raw_ & throw_on_error_
       & emds_ & error_messages_
       & nevA_ & nevB_ & num_emds_ & emd_storage_;
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int version) {
    ar & num_threads_ & print_every_
       & verbose_ & omp_dynamic_chunksize_
       & request_mode_ & store_sym_emds_raw_ & throw_on_error_
       & emds_ & error_messages_
       & nevA_ & nevB_ & num_emds_ & emd_storage_;

    handler_ = nullptr;
    print_stream_ = &std::cout;
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER()
#endif

public:

  PairwiseEMDBase(int num_threads,
                  index_type print_every,
                  unsigned verbose,
                  unsigned omp_dynamic_chunksize,
                  bool request_mode,
                  bool store_sym_emds_raw,
                  bool throw_on_error,
                  std::ostream & os) :
    num_threads_(determine_num_threads(num_threads)),
    print_every_(print_every),
    verbose_(verbose),
    omp_dynamic_chunksize_(omp_dynamic_chunksize),
    request_mode_(request_mode),
    store_sym_emds_raw_(store_sym_emds_raw),
    throw_on_error_(throw_on_error),
    print_stream_(&os),
    handler_(nullptr)
  {
    // print_every of 0 is equivalent to -1
    if (print_every_ == 0)
      print_every_ = -1;
  }

  virtual ~PairwiseEMDBase() = default;

  // public get/set EMD parameters
  virtual Value R() const = 0;
  virtual void set_R(Value R) = 0;
  virtual Value beta() const = 0;
  virtual void set_beta(Value beta) = 0;
  virtual bool norm() const = 0;
  virtual void set_norm(bool norm) = 0;
  virtual void set_network_simplex_params(std::size_t n_iter_max=100000,
                                          Value epsilon_large_factor=1000,
                                          Value epsilon_small_factor=1) = 0;

  // set a handler to process EMDs on the fly instead of storing them
  void set_external_emd_handler(ExternalEMDHandler<Value> & handler) {
    handler_ = &handler;
  }
  bool have_external_emd_handler() const {
    return handler_ != nullptr;
  }
  template<class EMDHandler>
  EMDHandler * external_emd_handler() {
    if (!have_external_emd_handler())
      throw std::invalid_argument("no external emd handler set");

    return dynamic_cast<EMDHandler>(handler_);
  }

  // get number of threads/print info
  int num_threads() const { return num_threads_; }

  // externally set the number of EMD evaluations that will be spooled to each OpenMP thread at a time
  void set_omp_dynamic_chunksize(int chunksize) {
    omp_dynamic_chunksize_ = std::abs(chunksize);
  }
  int omp_dynamic_chunksize() const {
    return omp_dynamic_chunksize_;
  }

  // turn on or off request mode, where nothing is stored or handled but
  // EMD distances can be queried and computed on the fly
  void set_request_mode(bool mode) { request_mode_ = mode; }
  bool request_mode() const { return request_mode_; }

  // access timing information
  virtual double duration() const = 0;

  // access event info
  index_type nevA() const { return nevA_; }
  index_type nevB() const { return nevB_; }

  // number of unique EMDs computed
  index_type num_emds() const { return num_emds_; }

  // query storage mode
  EMDPairsStorage storage() const { return emd_storage_; }


  // error reporting
  bool errored() const { return error_messages_.size(); }
  const std::vector<std::string> & error_messages() const { return error_messages_; }

  // access all emds as a matrix raw into a vector
  const std::vector<Value> & emds(bool raw = false) {

    // check for having no emds stored
    if (emd_storage_ == EMDPairsStorage::External)
      throw std::invalid_argument("No EMDs stored");

    // check if we need to construct a new full matrix from a raw symmetric one
    if (emd_storage_ == EMDPairsStorage::FlattenedSymmetric && !raw) {

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
      _evaluate_emd(i, j, thread);
    }

    // check for External handling, in which case we don't have any emds stored
    if (emd_storage_ == EMDPairsStorage::External)
      throw std::invalid_argument("EMD requested but external handler provided, so no EMDs stored");

    // index into emd vector (j always bigger than i because upper triangular storage)
    if (emd_storage_ == EMDPairsStorage::FlattenedSymmetric)
      return (i == j ? 0 : emds_[index_symmetric(i, j)]);

    else return emds_[i*nevB() + j];
  }

protected:

  void clear(bool free_memory) {

    emds_.clear();
    full_emds_.clear();
    error_messages_.clear();

    emd_storage_ = EMDPairsStorage::External;
    nevA_ = nevB_ = num_emds_ = 0;

    if (free_memory) {
      handler_ = nullptr;
      free_vector(emds_);
      free_vector(full_emds_);
      free_vector(error_messages_);
    }
  }

  virtual Value _evaluate_emd(index_type i, index_type j, int thread) = 0;

  // indexes upper triangle of symmetric matrix with zeros on diagonal that has been raw into 1D
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

private:

  // determine the number of threads to use
  static int determine_num_threads(int num_threads) {
    #ifdef _OPENMP
      if (num_threads == -1 || num_threads > omp_get_max_threads())
        return omp_get_max_threads();
      return num_threads;
    #else
      return 1;
    #endif
  }

}; // PairwiseEMDBase

END_WASSERSTEIN_NAMESPACE

#endif // WASSERSTEIN_PAIRWISEEMDBASE_HH
