//------------------------------------------------------------------------
// This file is part of Wasserstein, a C++ library with a Python wrapper
// that evaluates the Wasserstein/EMD distance. If you use it for academic
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

/*  ________   _________ ______ _____  _   _          _      
 * |  ____\ \ / /__   __|  ____|  __ \| \ | |   /\   | |     
 * | |__   \ V /   | |  | |__  | |__) |  \| |  /  \  | |     
 * |  __|   > <    | |  |  __| |  _  /| . ` | / /\ \ | |     
 * | |____ / . \   | |  | |____| | \ \| |\  |/ ____ \| |____ 
 * |______/_/ \_\  |_|  |______|_|  \_\_| \_/_/    \_\______|
 *  ______ __  __ _____  _    _          _   _ _____  _      ______ _____  
 * |  ____|  \/  |  __ \| |  | |   /\   | \ | |  __ \| |    |  ____|  __ \ 
 * | |__  | \  / | |  | | |__| |  /  \  |  \| | |  | | |    | |__  | |__) |
 * |  __| | |\/| | |  | |  __  | / /\ \ | . ` | |  | | |    |  __| |  _  / 
 * | |____| |  | | |__| | |  | |/ ____ \| |\  | |__| | |____| |____| | \ \ 
 * |______|_|  |_|_____/|_|  |_/_/    \_\_| \_|_____/|______|______|_|  \_\
 */

#ifndef WASSERSTEIN_EXTERNALEMDHANDLER_HH
#define WASSERSTEIN_EXTERNALEMDHANDLER_HH

#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "EMDUtils.hh"


BEGIN_WASSERSTEIN_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// ExternalEMDHandler - base class for handling values from a PairwiseEMD on the fly
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
class ExternalEMDHandler {
public:

  ExternalEMDHandler() : num_calls_(0) {}
  virtual ~ExternalEMDHandler() = default;

  virtual std::string description() const = 0;
  std::size_t num_calls() const { return num_calls_; }

  // call external handler on a single emd value
  void operator()(Value emd, Value weight = 1) {
    std::lock_guard<std::mutex> handler_guard(mutex_);
    handle(emd, weight);
    num_calls_++;
  }

  // call emd handler on several emds at once (given as a vector)
  void evaluate(const std::vector<Value> & emds, const std::vector<Value> & weights = {}) {

    if (emds.size() != weights.size()) {
      if (weights.size() == 0)
        evaluate(emds.data(), emds.size());
      else throw std::invalid_argument("length of weights must match that of emds or be 0");
    }
    else
      evaluate(emds.data(), emds.size(), weights.data());
  }

  // call emd handler on several emds at once (given as raw pointers)
  void evaluate(const Value * emds, std::size_t num_emds, const Value * weights = nullptr) {
    
    std::lock_guard<std::mutex> handler_guard(mutex_);
    if (weights == nullptr)
      for (std::size_t i = 0; i < num_emds; i++)
        handle(emds[i], 1);
    else
      for (std::size_t i = 0; i < num_emds; i++)
        handle(emds[i], weights[i]);

    num_calls_ += num_emds;
  }

  // here, weights are length nev and emds are length nev(nev-1)/2, given as vectors
  void evaluate_symmetric(const std::vector<Value> & emds, const std::vector<Value> & weights) {

    if (emds.size() != weights.size()*(weights.size() - 1)/2)
      throw std::invalid_argument("length of emds should be length of weights choose 2");

    evaluate_symmetric(emds.data(), weights.size(), weights.data());
  }

  // here, weights are length nev and emds are length nev(nev-1)/2, given as raw pointers
  void evaluate_symmetric(const Value * emds, std::size_t nev, const Value * weights) {
    
    std::lock_guard<std::mutex> handler_guard(mutex_);
    for (std::size_t i = 0, k = 0; i < nev; i++) {
      Value weight_i(weights[i]);

      for (std::size_t j = i + 1; j < nev; j++, k++)
        handle(emds[k], weight_i * weights[j]);
    }

    num_calls_ += nev*(nev - 1)/2;
  }

protected:

  virtual void handle(Value emd, Value weight) = 0; 

private:

  std::mutex mutex_;
  std::size_t num_calls_;

#ifdef WASSERSTEIN_SERIALIZATION
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & num_calls_;
  }
#endif

}; // ExternalEMDHandler

END_WASSERSTEIN_NAMESPACE

#endif // WASSERSTEIN_EXTERNALEMDHANDLER_HH
