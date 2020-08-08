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

#ifndef EVENTGEOMETRY_EMDUTILS_HH
#define EVENTGEOMETRY_EMDUTILS_HH

#include <sstream>
#include <string>
#include <type_traits>

#include "EventGeometryUtils.hh"

#ifdef __FASTJET_PSEUDOJET_HH__
FASTJET_BEGIN_NAMESPACE
namespace contrib {
#else
namespace emd {
#endif

// encapsulates plain 1D array with structure to enable range-based iteration
// note: int is used for compatibility with SWIG's numpy.i
template<typename V>
struct ArrayWeightCollection {
  typedef V value_type;

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

// encapsulates plain 2D array with structure to enable range-based iteration
// note: int is used for compatibility with SWIG's numpy.i
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

//-----------------------------------------------------------------------------
// Particle structs that work with the euclidean distance
//-----------------------------------------------------------------------------

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
};

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
};

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
};

#ifdef __FASTJET_PSEUDOJET_HH__
} // namespace contrib
FASTJET_END_NAMESPACE
#else
} // namespace emd
#endif

#endif // EVENTGEOMETRY_EMDUTILS_HH
