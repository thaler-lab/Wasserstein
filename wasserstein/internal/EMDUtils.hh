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

/*  ______ __  __ _____  _    _ _______ _____ _       _____ 
 * |  ____|  \/  |  __ \| |  | |__   __|_   _| |     / ____|
 * | |__  | \  / | |  | | |  | |  | |    | | | |    | (___  
 * |  __| | |\/| | |  | | |  | |  | |    | | | |     \___ \
 * | |____| |  | | |__| | |__| |  | |   _| |_| |____ ____) |
 * |______|_|  |_|_____/ \____/   |_|  |_____|______|_____/
 */

#ifndef WASSERSTEIN_EMDUTILS_HH
#define WASSERSTEIN_EMDUTILS_HH

// C++ standard library
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// namespace macros
#ifndef BEGIN_EMD_NAMESPACE
#define EMDNAMESPACE emd
#define BEGIN_EMD_NAMESPACE namespace EMDNAMESPACE {
#define END_EMD_NAMESPACE }
#endif

// fastjet macros
#ifdef __FASTJET_PSEUDOJET_HH__
#define WASSERSTEIN_FASTJET
#endif

BEGIN_EMD_NAMESPACE

// set these globally so SWIG knows how to parse them
#ifdef SWIG
typedef double Value;
typedef std::vector<double> ValueVector;
#endif

// enum with possible return statuses from the NetworkSimplex solver
enum class EMDStatus { 
  Success = 0,
  Empty = 1,
  SupplyMismatch = 2,
  Unbounded = 3,
  MaxIterReached = 4,
  Infeasible = 5
};

// enum for which event got an extra particle
enum class ExtraParticle { Neither = -1, Zero = 0, One = 1 };

const double PI = 3.14159265358979323846;
const double TWOPI = 2*PI;

inline double phi_fix(double phi, double ref_phi) {
  double diff(phi - ref_phi);
  if (diff > PI) phi -= TWOPI;
  else if (diff < -PI) phi += TWOPI;
  return phi; 
}

// function that raises appropriate error from a status code
inline void check_emd_status(EMDStatus status) {
  if (status != EMDStatus::Success)
    switch (status) {
      case EMDStatus::Empty:
        throw std::runtime_error("EMDStatus - Empty");
        break;
      case EMDStatus::SupplyMismatch:
        throw std::runtime_error("EMDStatus - SupplyMismatch, consider increasing epsilon_large_factor");
        break;
      case EMDStatus::Unbounded:
        throw std::runtime_error("EMDStatus - Unbounded");
        break;
      case EMDStatus::MaxIterReached:
        throw std::runtime_error("EMDStatus - MaxIterReached, consider increasing n_iter_max");
        break;
      case EMDStatus::Infeasible:
        throw std::runtime_error("EMDStatus - Infeasible");
        break;
      default:
        throw std::runtime_error("EMDStatus - Unknown");
    }
}

// helps with freeing memory
template<typename T>
void free_vector(std::vector<T> & vec) {
  std::vector<T>().swap(vec);
}

////////////////////////////////////////////////////////////////////////////////
// EMDBase - base class to reduce wrapper code for simple EMD access functions
////////////////////////////////////////////////////////////////////////////////

template<typename V>
class EMDBase {
protected:

  // these have been defined globally if in SWIG
  #ifndef SWIG
  typedef V Value;
  typedef std::vector<Value> ValueVector;
  #endif

  // boolean options
  bool norm_, do_timing_, external_dists_;

  // number of particles
  std::size_t n0_, n1_;
  ExtraParticle extra_;

  // emd value and status
  Value weightdiff_, scale_, emd_;
  EMDStatus status_;

  // timing
  std::chrono::steady_clock::time_point start_;
  double duration_;

public:

  // always defined, so that external classes can access the value
  typedef Value ValuePublic;

  // constructor from two boolean options
  EMDBase(bool norm = false, bool do_timing = false, bool external_dists = false) :
    norm_(norm),
    do_timing_(do_timing),
    external_dists_(external_dists),
    n0_(0), n1_(0),
    extra_(ExtraParticle::Neither),
    emd_(0),
    status_(EMDStatus::Empty),
    duration_(0)
  {}

  // virtual desctructor
  virtual ~EMDBase() {}

  // the norm setting
  bool norm() const { return norm_; }
  void set_norm(bool nrm) { norm_ = nrm; } 

  // timing setting
  bool do_timing() const { return do_timing_; }
  void set_do_timing(bool timing) { do_timing_ = timing; }

  // get/set whether we're currently setup to use external distances
  bool external_dists() const { return external_dists_; }
  void set_external_dists(bool exdists) { external_dists_ = exdists; }

  // number of particles in each event (after possible addition of extra particles)
  std::size_t n0() const { return n0_; }
  std::size_t n1() const { return n1_; }

  // which event, 0 or 1, got an extra particle (-1 if no event got one)
  ExtraParticle extra() const { return extra_; }

  // returns emd scale, value and status
  Value emd() const { return emd_; }
  EMDStatus status() const { return status_; }
  Value weightdiff() const { return weightdiff_; }
  Value scale() const { return scale_; }

  // return timing info
  double duration() const { return duration_; }

protected:

  // duration of emd computation in seconds
  void start_timing() { start_ = std::chrono::steady_clock::now(); }
  double store_duration() {
    auto diff(std::chrono::steady_clock::now() - start_);
    duration_ = std::chrono::duration_cast<std::chrono::duration<double>>(diff).count();
    return duration_;
  }

}; // EMDBase

////////////////////////////////////////////////////////////////////////////////
// ExternalEMDHandler - base class for making an emd operation thread-safe
////////////////////////////////////////////////////////////////////////////////

class ExternalEMDHandler {
public:
  ExternalEMDHandler() : num_calls_(0) {}
  virtual ~ExternalEMDHandler() {}
  virtual std::string description() const = 0;
  void operator()(double emd, double weight = 1) {
    std::lock_guard<std::mutex> handler_guard(mutex_);
    handle(emd, weight);
    num_calls_++;
  }
  std::size_t num_calls() const { return num_calls_; }

protected:
  virtual void handle(double, double) = 0; 

private:
  std::mutex mutex_;
  std::size_t num_calls_;

}; // ExternalEMDHandler

////////////////////////////////////////////////////////////////////////////////
// EuclideanParticles - structs that work with the euclidean pairwise distance
////////////////////////////////////////////////////////////////////////////////

template<unsigned N, typename V = double>
struct EuclideanParticleND {
  typedef V Value;
  typedef std::array<Value, N> Coords;
  typedef EuclideanParticleND<N, Value> Self;

  static_assert(std::is_floating_point<V>::value, "Template parameter must be floating point.");

  // constructor from weight and coord array
  EuclideanParticleND(Value weight, const std::array<Value, N> & xs) : weight_(weight), xs_(xs) {}

  Value weight() const { return weight_; }
  Coords & coords() { return xs_; }
  const Coords & coords() const { return xs_; }

  static Value plain_distance(const Self & p0, const Self & p1) {
    Value d(0);
    for (unsigned i = 0; i < N; i++) {
      Value dx(p0.xs_[i] - p1.xs_[i]);
      d += dx*dx;
    }
    return d;
  }

  static std::string name() {
    std::ostringstream oss;
    oss << "EuclideanParticle" << N << "D<" << sizeof(V) << "-byte float>";
    return oss.str();
  }
  static std::string distance_name() {
    std::ostringstream oss;
    oss << "EuclideaDistance" << N << 'D';
    return oss.str();
  }

protected:

  // store particle info
  Value weight_;
  Coords xs_;

}; // EuclideanParticleND

template<typename V = double>
struct EuclideanParticle2D : public EuclideanParticleND<2, V> {
  typedef typename EuclideanParticleND<2, V>::Value Value;
  typedef EuclideanParticle2D<Value> Self;

  // constructor from weight, x, y
  EuclideanParticle2D(Value weight, Value x, Value y) :
    EuclideanParticleND<2>(weight, {x, y}) {}
  EuclideanParticle2D(Value weight, const std::array<Value, 2> & xs) :
    EuclideanParticleND<2>(weight, xs) {}

  // overload this to avoid for loop
  static Value plain_distance(const Self & p0, const Self & p1) {
    Value dx(p0.xs_[0] - p1.xs_[0]), dy(p0.xs_[1] - p1.xs_[1]);
    return dx*dx + dy*dy;
  }

}; // EuclideanParticle2D

template<typename V = double>
struct EuclideanParticle3D : public EuclideanParticleND<3, V> {
  typedef typename EuclideanParticleND<3, V>::Value Value;
  typedef EuclideanParticle3D<Value> Self;

  // constructor from weight, x, y
  EuclideanParticle3D(Value weight, Value x, Value y, Value z) :
    EuclideanParticleND<3>(weight, {x, y, z}) {}
  EuclideanParticle3D(Value weight, const std::array<Value, 3> & xs) :
    EuclideanParticleND<3>(weight, xs) {}

  // overload this to avoid for loop
  static Value plain_distance(const Self & p0, const Self & p1) {
    Value dx(p0.xs_[0] - p1.xs_[0]), dy(p0.xs_[1] - p1.xs_[1]), dz(p0.xs_[2] - p1.xs_[2]);
    return dx*dx + dy*dy + dz*dz;
  }

}; // EuclideanParticle3D

////////////////////////////////////////////////////////////////////////////////
// Preprocessor - base class for preprocessing operations
////////////////////////////////////////////////////////////////////////////////

// base class for preprocessing events
template<class EMD>
class Preprocessor {
public:
  typedef typename EMD::Event Event;

  virtual ~Preprocessor() {}

  // returns description
  virtual std::string description() const { return "Preprocessor"; };

  // call this preprocessor on event
  virtual Event & operator()(Event & event) const { return event; };

}; // Preprocessor

////////////////////////////////////////////////////////////////////////////////
// Preprocessor - base class for preprocessing operations
////////////////////////////////////////////////////////////////////////////////

// forward declare fastjet event
class FastJetEventBase;
template<class ParticleWeight> struct FastJetEvent;

// center generic event according to weighted centroid
template<class EMD>
class CenterWeightedCentroid : public Preprocessor<typename EMD::Self> {
public:
  typedef typename EMD::Event Event;
  typedef typename EMD::ValuePublic Value;
  typedef typename Event::ParticleCollection ParticleCollection;
  typedef typename Event::WeightCollection WeightCollection;

  std::string description() const { return "Center according to weighted centroid"; }
  Event & operator()(Event & event) const {
    return center(event);
  }

private:

  // this version will be used for everything that's not FastJetEvent
  template<class E>
  typename std::enable_if<!std::is_base_of<FastJetEventBase, E>::value, E &>::type
  center(E & event) const {
    static_assert(std::is_same<E, Event>::value, "Event must match that of the EMD class");

    // determine weighted centroid
    typename E::Particle::Coords coords;
    coords.fill(0);
    for (const auto & particle : event.particles())
      for (unsigned i = 0; i < coords.size(); i++) 
        coords[i] += particle.weight() * particle.coords()[i];

    for (unsigned i = 0; i < coords.size(); i++)
      coords[i] /= event.total_weight();

    // center the particles
    for (auto & particle : event.particles())
      for (unsigned i = 0; i < coords.size(); i++) {
        particle.coords()[i] -= coords[i];
      }

    return event;
  }

  // enable overloaded operator for FastJetEvent
#ifdef WASSERSTEIN_FASTJET
  FastJetEvent<typename EMD::ParticleWeight> & center(FastJetEvent<typename EMD::ParticleWeight> & event) const;
#endif

}; // CenterWeightedCentroid

END_EMD_NAMESPACE

#endif // WASSERSTEIN_EMDUTILS_HH
