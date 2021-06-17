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
 *  _____ _____  _____ _______       _   _  _____ ______ 
 * |  __ \_   _|/ ____|__   __|/\   | \ | |/ ____|  ____|
 * | |  | || | | (___    | |  /  \  |  \| | |    | |__   
 * | |  | || |  \___ \   | | / /\ \ | . ` | |    |  __|  
 * | |__| || |_ ____) |  | |/ ____ \| |\  | |____| |____ 
 * |_____/_____|_____/   |_/_/    \_\_| \_|\_____|______|
 */

#ifndef WASSERSTEIN_PAIRWISEDISTANCE_HH
#define WASSERSTEIN_PAIRWISEDISTANCE_HH

// C++ standard library
#include <cmath>

#include "EMDUtils.hh"


BEGIN_WASSERSTEIN_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// PairwiseDistanceBase - implements (theta_ij/R)^beta between particles i and j
////////////////////////////////////////////////////////////////////////////////

template <class PairwiseDistance, class ParticleCollection, typename Value>
class PairwiseDistanceBase {
public:

  typedef typename ParticleCollection::value_type Particle;
  typedef typename ParticleCollection::const_iterator ParticleIterator;
  typedef Value value_type;

  // default constructor
  PairwiseDistanceBase() {

    // check that we properly have passed a derived class (must be here because derived class is incomplete)
    static_assert(std::is_base_of<PairwiseDistanceBase<PairwiseDistance, ParticleCollection, Value>,
                                  PairwiseDistance>::value, 
                  "Template parameter must be a derived class of PairwiseDistanceBase.");
  }

  // constructor with parameters
  PairwiseDistanceBase(Value R, Value beta) :
    PairwiseDistanceBase()
  {
    set_R(R);
    set_beta(beta);    
  }

  // description of class
  std::string description() const {
    std::ostringstream oss;
    oss << "  " << PairwiseDistance::name() << '\n'
        << "    R - " << R() << '\n'
        << "    beta - " << beta() << '\n'
        << '\n';
    return oss.str();
  }

  // access/set parameters
  Value R() const { return R_; }
  Value beta() const { return beta_; }
  void set_R(Value r) {
    if (r <= 0) throw std::invalid_argument("R must be positive.");
    R_ = r;
    R2_ = r*r;
  }
  void set_beta(Value beta) {
    if (beta < 0) throw std::invalid_argument("beta must be non-negative.");
    beta_ = beta;
    halfbeta_ = beta/2;
  }

  // computes pairwise distances between two particle collections
  void fill_distances(const ParticleCollection & ps0, const ParticleCollection & ps1,
                      std::vector<Value> & dists, ExtraParticle extra) {

    std::size_t k(0);

    if (extra == ExtraParticle::Neither) {
      dists.resize(ps0.size() * ps1.size());
      for (ParticleIterator p0 = ps0.begin(), end0 = ps0.end(), end1 = ps1.end(); p0 != end0; ++p0)
        for (ParticleIterator p1 = ps1.begin(); p1 != end1; ++p1)
          dists[k++] = distance(p0, p1);
    }

    else if (extra == ExtraParticle::Zero) {
      dists.resize((ps0.size() + 1) * ps1.size());
      for (ParticleIterator p0 = ps0.begin(), end0 = ps0.end(), end1 = ps1.end(); p0 != end0; ++p0)
        for (ParticleIterator p1 = ps1.begin(); p1 != end1; ++p1)
          dists[k++] = distance(p0, p1);
      for (std::size_t j = 0, end = ps1.size(); j < end; j++)
        dists[k++] = 1;
    }

    // extra == ExtraParticle::One
    else {
      dists.resize(ps0.size() * (ps1.size() + 1));
      for (ParticleIterator p0 = ps0.begin(), end0 = ps0.end(), end1 = ps1.end(); p0 != end0; ++p0) {
        for (ParticleIterator p1 = ps1.begin(); p1 != end1; ++p1)
          dists[k++] = distance(p0, p1);
        dists[k++] = 1;
      }
    }
  }

  // returns the distance divided by R, all to beta power
  Value distance(const ParticleIterator & p0, const ParticleIterator & p1) const {
    Value pd(PairwiseDistance::plain_distance_(p0, p1));
    return (beta_ == 1.0 ? std::sqrt(pd)/R_ : (beta_ == 2.0 ? pd/R2_ : std::pow(pd/R2_, halfbeta_)));
  }

  // return the plain distance, without the square root
  static Value plain_distance_(const ParticleIterator & p0, const ParticleIterator & p1) {
    return PairwiseDistance::plain_distance(*p0, *p1);
  }

  // ensure this method is overloaded
  static Value plain_distance(const Particle & p0, const Particle & p1) {
    throw std::runtime_error("called pairwise distance without any particles");
    return -1;
  }

protected:

  ~PairwiseDistanceBase() = default;

private:

  Value R_, R2_, beta_, halfbeta_;

#ifdef WASSERSTEIN_SERIALIZATION
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & R_ & R2_ & beta_ & halfbeta_;
  }
#endif

}; // PairwiseDistanceBase


////////////////////////////////////////////////////////////////////////////////
// DefaultPairwiseDistance - does nothing by default, for use with external dists
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
class DefaultPairwiseDistance : public PairwiseDistanceBase<DefaultPairwiseDistance<Value>,
                                                             std::vector<Value>,
                                                             Value> {
public:

  typedef Value value_type;
  typedef std::vector<Value> ParticleCollection;
  typedef typename ParticleCollection::const_iterator ParticleIterator;

  using PairwiseDistanceBase<DefaultPairwiseDistance<Value>, ParticleCollection, Value>::PairwiseDistanceBase;

  static std::string name() { return "DefaultPairwiseDistance (none)"; }
  static Value plain_distance_(const ParticleIterator & p0, const ParticleIterator & p1) {
    return -1;
  }

}; // DefaultPairwiseDistance


////////////////////////////////////////////////////////////////////////////////
// EuclideanArrayDistance - euclidean distance between two particle arrays
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
class EuclideanArrayDistance : public PairwiseDistanceBase<EuclideanArrayDistance<Value>,
                                                            ArrayParticleCollection<Value>,
                                                            Value> {
public:

  typedef Value value_type;
  typedef ArrayParticleCollection<Value> ParticleCollection;
  typedef typename ParticleCollection::value_type Particle;
  typedef typename ParticleCollection::const_iterator ParticleIterator;

  using PairwiseDistanceBase<EuclideanArrayDistance<Value>, ParticleCollection, Value>::PairwiseDistanceBase;

  static std::string name() { return "EuclideanArrayDistance"; }
  static Value plain_distance_(const ParticleIterator & p0, const ParticleIterator & p1) {
    if (p0.stride() == 2) {
      Value dx((*p0)[0] - (*p1)[0]), dy((*p0)[1] - (*p1)[1]);
      return dx*dx + dy*dy;
    }
    else {
      Value d(0);
      for (int i = 0; i < p0.stride(); i++) {
        Value dx((*p0)[i] - (*p1)[i]);
        d += dx*dx;
      }
      return d;
    }
  }
}; // EuclideanArrayDistance


////////////////////////////////////////////////////////////////////////////////
// YPhiArrayDistance - euclidean distance in (y,phi) plane, accounting for periodicity
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
class YPhiArrayDistance : public PairwiseDistanceBase<YPhiArrayDistance<Value>,
                                                       Array2ParticleCollection<Value>,
                                                       Value> {
public:

  typedef Value value_type;
  typedef Array2ParticleCollection<Value> ParticleCollection;

  typedef typename ParticleCollection::value_type Particle;
  typedef typename ParticleCollection::const_iterator ParticleIterator;

  using PairwiseDistanceBase<YPhiArrayDistance<Value>, ParticleCollection, Value>::PairwiseDistanceBase;

  static std::string name() { return "YPhiArrayDistance"; }
  static Value plain_distance_(const ParticleIterator & p0, const ParticleIterator & p1) {
    Value dy((*p0)[0] - (*p1)[0]), dphi(std::fabs((*p0)[1] - (*p1)[1]));
    if (dphi > PI) dphi = TWOPI - dphi;
    return dy*dy + dphi*dphi;
  }
}; // EuclideanArrayDistance


////////////////////////////////////////////////////////////////////////////////
// EuclideanParticleDistance
////////////////////////////////////////////////////////////////////////////////

template<class _Particle>
class EuclideanParticleDistance : public PairwiseDistanceBase<EuclideanParticleDistance<_Particle>,
                                                     std::vector<_Particle>,
                                                     typename _Particle::value_type> {
public:

  typedef _Particle Particle;
  typedef typename Particle::value_type value_type;

  using PairwiseDistanceBase<EuclideanParticleDistance<Particle>,
                             std::vector<Particle>,
                             typename Particle::value_type>::PairwiseDistanceBase;

  static std::string name() { return Particle::distance_name(); }
  static value_type plain_distance(const Particle & p0, const Particle & p1) {
    return Particle::plain_distance(p0, p1);
  }
}; // EuclideanParticleDistance

////////////////////////////////////////////////////////////////////////////////
// YPhiParticleDistance - handles periodicity in phi
////////////////////////////////////////////////////////////////////////////////

template<typename Value>
class YPhiParticleDistance : public PairwiseDistanceBase<YPhiParticleDistance<Value>,
                                                         std::vector<EuclideanParticle2D<Value>>,
                                                         Value> {
public:

  typedef EuclideanParticle2D<Value> Particle;
  typedef typename Particle::value_type value_type;

  using PairwiseDistanceBase<YPhiParticleDistance<Value>,
                             std::vector<Particle>,
                             Value>::PairwiseDistanceBase;

  static std::string name() { return "YPhiParticleDistance"; }
  static Value plain_distance(const Particle & p0, const Particle & p1) {
    Value dy(p0[0] - p1[0]), dphi(p0[1] - p1[1]);
    if (dphi > PI) dphi = TWOPI - dphi;
    return dy*dy + dphi*dphi;
  }
}; // EuclideanParticleDistance

END_WASSERSTEIN_NAMESPACE

#endif // WASSERSTEIN_PAIRWISEDISTANCE_HH
