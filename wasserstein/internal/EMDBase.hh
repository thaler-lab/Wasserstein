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

/*  ______ __  __ _____  ____           _____ ______ 
 * |  ____|  \/  |  __ \|  _ \   /\    / ____|  ____|
 * | |__  | \  / | |  | | |_) | /  \  | (___ | |__   
 * |  __| | |\/| | |  | |  _ < / /\ \  \___ \|  __|  
 * | |____| |  | | |__| | |_) / ____ \ ____) | |____ 
 * |______|_|  |_|_____/|____/_/    \_\_____/|______|
 */

#ifndef WASSERSTEIN_EMDBASE_HH
#define WASSERSTEIN_EMDBASE_HH

#include <chrono>
#include <utility>
#include <vector>

#include "EMDUtils.hh"


BEGIN_WASSERSTEIN_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// EMDBase - base class to reduce wrapper code for simple EMD access functions
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
class EMDBase {
protected:

  // boolean options
  bool norm_, do_timing_, external_dists_;

  // number of particles
  index_type n0_, n1_;
  ExtraParticle extra_;

  // emd value and status
  Value weightdiff_, scale_, emd_;
  EMDStatus status_;

  // timing
  std::chrono::steady_clock::time_point start_;
  double duration_;

public:

  EMDBase(bool norm, bool do_timing, bool external_dists) :
    norm_(norm),
    do_timing_(do_timing),
    external_dists_(external_dists),
    n0_(0), n1_(0),
    extra_(ExtraParticle::Neither),
    emd_(0),
    status_(EMDStatus::Empty),
    duration_(0)
  {}

  virtual ~EMDBase() = default;

  // access/set R and beta parameters
  virtual Value R() const = 0;
  virtual Value beta() const = 0;
  virtual void set_R(Value R) = 0;
  virtual void set_beta(Value beta) = 0;

  // set network simplex parameters
  virtual void set_network_simplex_params(std::size_t n_iter_max=100000,
                                          Value epsilon_large_factor=1000,
                                          Value epsilon_small_factor=1) = 0;

  bool norm() const { return norm_; }
  void set_norm(bool norm) { norm_ = norm; } 

  bool do_timing() const { return do_timing_; }
  void set_do_timing(bool timing) { do_timing_ = timing; }

  bool external_dists() const { return external_dists_; }
  void set_external_dists(bool exdists) { external_dists_ = exdists; }

  // number of particles in each event (after possible addition of extra particles)
  index_type n0() const { return n0_; }
  index_type n1() const { return n1_; }

  // which event, 0 or 1, got an extra particle (-1 if no event got one)
  ExtraParticle extra() const { return extra_; }

  // returns emd scale, value and status
  Value emd() const { return emd_; }
  EMDStatus status() const { return status_; }
  Value weightdiff() const { return weightdiff_; }
  Value scale() const { return scale_; }
  virtual std::size_t n_iter() const = 0;

  virtual std::vector<Value> dists() const = 0;
  virtual std::vector<Value> flows() const = 0;
  virtual Value flow(index_type i, index_type j) const = 0;
  virtual Value flow(std::size_t ind) const = 0;
  virtual std::pair<std::vector<Value>, std::vector<Value>> node_potentials() const = 0;

#ifdef SWIG
  // needed by the python wrapper
  virtual std::vector<Value> & ground_dists() = 0;
  virtual const std::vector<Value> & raw_flows() const = 0;
#endif

  // return timing info
  double duration() const { return duration_; }

  // free all dynamic memory help by this object
  virtual void clear() = 0;

protected:

  // duration of emd computation in seconds
  void start_timing() { start_ = std::chrono::steady_clock::now(); }
  double store_duration() {
    auto diff(std::chrono::steady_clock::now() - start_);
    duration_ = std::chrono::duration_cast<std::chrono::duration<double>>(diff).count();
    return duration_;
  }

private:

#ifdef WASSERSTEIN_SERIALIZATION
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & norm_ & do_timing_ & external_dists_;
  }
#endif

}; // EMDBase

END_WASSERSTEIN_NAMESPACE

#endif // WASSERSTEIN_EMDBASE_HH
