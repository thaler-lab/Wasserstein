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

#ifndef WASSERSTEIN_EMDUTILS_HH
#define WASSERSTEIN_EMDUTILS_HH

// C++ standard library
#include <chrono>
#include <cstddef>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// enum with possible return statuses from the NetworkSimplex solver
enum class EMDStatus : char { 
  Success = 0,
  Empty = 1,
  SupplyMismatch = 2,
  Unbounded = 3,
  MaxIterReached = 4,
  Infeasible = 5
};

// use fastjet::contrib if part of FastJet, otherwise emd
#ifdef __FASTJET_PSEUDOJET_HH__
  #define BEGIN_WASSERSTEIN_NAMESPACE FASTJET_BEGIN_NAMESPACE namespace contrib {
  #define END_WASSERSTEIN_NAMESPACE } FASTJET_END_NAMESPACE
#else
  #define BEGIN_WASSERSTEIN_NAMESPACE namespace emd {
  #define END_WASSERSTEIN_NAMESPACE }
#endif

BEGIN_WASSERSTEIN_NAMESPACE

// enum for which event got an extra particle
enum class ExtraParticle : char { Neither = -1, Zero = 0, One = 1 };

// ensure that phi utils are provided only once
#ifndef PIPHIUTILS
#define PIPHIUTILS

const double PI = 3.14159265358979323846;
const double TWOPI = 2*PI;

inline double phi_fix(double phi, double ref_phi) {
  double diff(phi - ref_phi);
  if (diff > PI) phi -= TWOPI;
  else if (diff < -PI) phi += TWOPI;
  return phi; 
}
#endif // PIPHIUTILS

// set these globally so SWIG knows how to parse them
#ifdef SWIG
typedef double Value;
typedef std::vector<double> ValueVector;
#endif

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
        throw std::runtime_error("EMDStatus - NoSuccess");
    }
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
  Value scale_, emd_;
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

  // get/set whether we're currently setup to use external distances
  bool external_dists() const { return external_dists_; }
  void set_external_dists(bool exdists) { external_dists_ = exdists; }

  // which event, 0 or 1, got an extra particle (-1 if no event got one)
  ExtraParticle extra() const { return extra_; }

  // number of particles in each event (after possible addition of extra particles)
  std::size_t n0() const { return n0_; }
  std::size_t n1() const { return n1_; }

  // returns emd scale, value and status
  Value scale() const { return scale_; }
  Value emd() const { return emd_; }
  EMDStatus status() const { return status_; }

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
// ArrayWeightCollection - implements a "smart" 1D array from a plain array
////////////////////////////////////////////////////////////////////////////////

template<typename V>
struct ArrayWeightCollection {
  typedef V value_type;

  // contructor, int is used for compatibility with SWIG's numpy.i
  ArrayWeightCollection(V * array, int size) : array_(array), size_(size) {}

  int size() const { return size_; }
  V * begin() { return array_; }
  V * end() { return array_ + size_; }
  const V * begin() const { return array_; }
  const V * end() const { return array_ + size_; }

private:
  V * array_;
  int size_;

}; // ArrayWeightCollection

////////////////////////////////////////////////////////////////////////////////
// ArrayParticleCollection - implements a "smart" 2D array from a plain array
////////////////////////////////////////////////////////////////////////////////

template<typename V>
struct ArrayParticleCollection {

  template<typename T>
  class templated_iterator {
    T * ptr_;
    int stride_;

  public:
    templated_iterator(T * ptr, int stride) : ptr_(ptr), stride_(stride) {}
    templated_iterator<T> & operator++() { ptr_ += stride_; return *this; }
    T * operator*() const { return ptr_; }
    bool operator!=(const templated_iterator & other) const { return ptr_ != other.ptr_; }
    int stride() const { return stride_; }
  };
  //using iterator = templated_iterator<V>;
  using const_iterator = templated_iterator<const V>;
  using value_type = const_iterator;

  // contructor, int is used for compatibility with SWIG's numpy.i
  ArrayParticleCollection(V * array, int size, int stride) :
    array_(array), size_(size), stride_(stride)
  {}

  std::size_t size() const { return size_; }
  int stride() const { return stride_; }
  //iterator begin() { return iterator(array_, stride_); }
  //iterator end() { return iterator(array_ + size_*stride_, stride_); }
  const_iterator begin() const { return const_iterator(array_, stride_); }
  const_iterator end() const { return const_iterator(array_ + size_*stride_, stride_); }

private:
  V * array_;
  int size_, stride_;

}; // ArrayParticleCollection

////////////////////////////////////////////////////////////////////////////////
// EuclideanParticles - structs that work with the euclidean pairwise distance
////////////////////////////////////////////////////////////////////////////////

template<typename V = double>
struct EuclideanParticle2D {
  typedef V Value;
  typedef EuclideanParticle2D<Value> Self;

  static_assert(std::is_floating_point<V>::value, "Template parameter must be floating point.");

  // data storage
  Value weight, x, y;

  static Value plain_distance(const Self & p0, const Self & p1) {
    Value dx(p0.x - p1.x), dy(p0.y - p1.y);
    return dx*dx + dy*dy;
  }

  static std::string name() {
    std::ostringstream oss;
    oss << "EuclideanParticle2D<" << sizeof(V) << "-byte float>";
    return oss.str();
  }
  static std::string distance_name() { return "EuclideanDistance2D"; }

}; // EuclideanParticle2D

template<typename V = double>
struct EuclideanParticle3D {
  typedef V Value;
  typedef EuclideanParticle3D<Value> Self;

  static_assert(std::is_floating_point<V>::value, "Template parameter must be floating point.");

  // data storage
  Value weight, x, y, z;
  
  static Value plain_distance(const Self & p0, const Self & p1) {
    Value dx(p0.x - p1.x), dy(p0.y - p1.y), dz(p0.z - p1.z);
    return dx*dx + dy*dy + dz*dz;
  }

  static std::string name() {
    std::ostringstream oss;
    oss << "EuclideanParticle3D<" << sizeof(V) << "-byte float>";
    return oss.str();
  }
  static std::string distance_name() { return "EuclideanDistance3D"; }

}; // EuclideanParticle3D

template<unsigned int N, typename V = double>
struct EuclideanParticleND {
  typedef V Value;
  typedef EuclideanParticleND<N, Value> Self;

  static_assert(std::is_floating_point<V>::value, "Template parameter must be floating point.");

  // data storage
  Value weight, xs[N];
  
  static Value plain_distance(const Self & p0, const Self & p1) {
    Value d(0);
    for (unsigned int i = 0; i < N; i++) {
      Value dx(p0.xs[i] - p1.xs[i]);
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

}; // EuclideanParticleND

END_WASSERSTEIN_NAMESPACE

#endif // WASSERSTEIN_EMDUTILS_HH
