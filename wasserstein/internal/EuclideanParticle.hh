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

/*  ______ _    _  _____ _      _____ _____  ______          _   _ 
 * |  ____| |  | |/ ____| |    |_   _|  __ \|  ____|   /\   | \ | |
 * | |__  | |  | | |    | |      | | | |  | | |__     /  \  |  \| |
 * |  __| | |  | | |    | |      | | | |  | |  __|   / /\ \ | . ` |
 * | |____| |__| | |____| |____ _| |_| |__| | |____ / ____ \| |\  |
 * |______|\____/ \_____|______|_____|_____/|______/_/    \_\_| \_|
 *  _____        _____ _______ _____ _____ _      ______ 
 * |  __ \ /\   |  __ \__   __|_   _/ ____| |    |  ____|
 * | |__) /  \  | |__) | | |    | || |    | |    | |__   
 * |  ___/ /\ \ |  _  /  | |    | || |    | |    |  __|  
 * | |  / ____ \| | \ \  | |   _| || |____| |____| |____ 
 * |_| /_/    \_\_|  \_\ |_|  |_____\_____|______|______|
 */

#ifndef WASSERSTEIN_EUCLIDEANPARTICLE_HH
#define WASSERSTEIN_EUCLIDEANPARTICLE_HH

#include <array>
#include <sstream>
#include <string>

#include "EMDUtils.hh"


BEGIN_WASSERSTEIN_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// EuclideanParticleND - struct to hold an N-dimensional vector and a weight
////////////////////////////////////////////////////////////////////////////////

template<unsigned N, typename Value>
struct EuclideanParticleND {

  typedef Value value_type;
  typedef std::array<Value, N> Coords;
  typedef EuclideanParticleND<N, Value> Self;

  // constructor from weight and coord array
  EuclideanParticleND(Value weight, const Coords & xs) : weight_(weight), xs_(xs) {}

  Value weight() const { return weight_; }

  const Value & operator[](index_type i) const { return xs_[i]; }
  Value & operator[](index_type i) { return xs_[i]; }

  static index_type dimension() { return N; }

  static Value plain_distance(const Self & p0, const Self & p1) {
    Value d(0);
    for (unsigned i = 0; i < N; i++) {
      Value dx(p0[i] - p1[i]);
      d += dx*dx;
    }
    return d;
  }

  static std::string name() {
    std::ostringstream oss;
    oss << "EuclideanParticle" << N << "D<" << sizeof(Value) << "-byte float>";
    return oss.str();
  }

  static std::string distance_name() {
    std::ostringstream oss;
    oss << "EuclideanDistance" << N << 'D';
    return oss.str();
  }

protected:

  // store particle info
  Value weight_;
  Coords xs_;

}; // EuclideanParticleND

// specialized to N=2
template<typename Value>
struct EuclideanParticle2D : public EuclideanParticleND<2, Value> {

  typedef Value value_type;
  typedef EuclideanParticleND<2, Value> Base;
  typedef EuclideanParticle2D<Value> Self;

  // constructor from weight, x, y
  EuclideanParticle2D(Value weight, Value x, Value y) :
    Base(weight, {x, y}) {}
  EuclideanParticle2D(Value weight, const std::array<Value, 2> & xs) :
    Base(weight, xs) {}

  // overloaded to avoid for loop
  static Value plain_distance(const Self & p0, const Self & p1) {
    Value dx(p0[0] - p1[0]), dy(p0[1] - p1[1]);
    return dx*dx + dy*dy;
  }

}; // EuclideanParticle2D

// specialized to N=3
template<typename Value>
struct EuclideanParticle3D : public EuclideanParticleND<3, Value> {

  typedef Value value_type;
  typedef EuclideanParticleND<3, Value> Base;
  typedef EuclideanParticle3D<Value> Self;

  // constructor from weight, x, y
  EuclideanParticle3D(Value weight, Value x, Value y, Value z) :
    Base(weight, {x, y, z}) {}
  EuclideanParticle3D(Value weight, const std::array<Value, 3> & xs) :
    Base(weight, xs) {}

  // overloaded to avoid for loop
  static Value plain_distance(const Self & p0, const Self & p1) {
    Value dx(p0[0] - p1[0]), dy(p0[1] - p1[1]), dz(p0[2] - p1[2]);
    return dx*dx + dy*dy + dz*dz;
  }

}; // EuclideanParticle3D

END_WASSERSTEIN_NAMESPACE

#endif // WASSERSTEIN_EUCLIDEANPARTICLE_HH
