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

#ifndef EVENTGEOMETRY_MEASURES_HH
#define EVENTGEOMETRY_MEASURES_HH

#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "EMDUtils.hh"

#ifdef __FASTJET_PSEUDOJET_HH__
FASTJET_BEGIN_NAMESPACE
namespace contrib {
#else
namespace emd {
#endif

//-----------------------------------------------------------------------------
// PairwiseDistance - implements theta_ij between particles
//-----------------------------------------------------------------------------

// base class declaring an interface for a pairwise distance
template <class PairwiseDistance, class ParticleCollection, class Value>
struct PairwiseDistanceBase {
  typedef typename ParticleCollection::value_type Particle;
  typedef typename ParticleCollection::const_iterator ParticleIterator;

  PairwiseDistanceBase(Value R, Value beta) :
    R_(R), beta_(beta), halfbeta_(beta_/2), R_to_beta_(std::pow(R_, beta_)), inv_R_to_beta_(1/R_to_beta_)
  {

    // check that we properly have passed a derived class
    static_assert(std::is_base_of<PairwiseDistanceBase<PairwiseDistance, ParticleCollection, Value>, PairwiseDistance>::value, 
                  "Template parameter must be a derived class of PairwiseDistanceBase.");

    // check parameters
    if (beta_ < 0) throw std::invalid_argument("beta must be non-negative.");
    if (R_ <= 0) throw std::invalid_argument("R must be positive.");
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

  // access parameters
  Value R() const { return R_; }
  Value beta() const { return beta_; }
  Value R_to_beta() const { return R_to_beta_; }
  Value unscale_factor() const { return inv_R_to_beta_; }

  void fill_distances(const ParticleCollection & ps0, const ParticleCollection & ps1,
                      std::vector<Value> & dists, ExtraParticle extra) {

    //static_cast<const PairwiseDistance &>(*this).init(ps0, ps1);
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
      for (std::size_t j = 0; j < ps1.size(); j++)
        dists[k++] = R_to_beta_;
    }

    // extra == ExtraParticle::One
    else {
      dists.resize(ps0.size() * (ps1.size() + 1));
      for (ParticleIterator p0 = ps0.begin(), end0 = ps0.end(), end1 = ps1.end(); p0 != end0; ++p0) {
        for (ParticleIterator p1 = ps1.begin(); p1 != end1; ++p1)
          dists[k++] = distance(p0, p1);
        dists[k++] = R_to_beta_;
      }
    }
  }

  // returns the distance to the appropriate power
  Value distance(const ParticleIterator & p0, const ParticleIterator & p1) const {
    Value pd(PairwiseDistance::plain_distance_(p0, p1));
    return (beta_ == 1.0 ? std::sqrt(pd) : (beta_ == 2.0 ? pd : std::pow(pd, halfbeta_)));
  }

  // return the plain distance, without the square root
  static Value plain_distance_(const ParticleIterator & p0, const ParticleIterator & p1) {
    return PairwiseDistance::plain_distance(*p0, *p1);
  }

private:
  Value R_, beta_, halfbeta_, R_to_beta_, inv_R_to_beta_;
};

// FastJet specific pairwise distances
#ifdef __FASTJET_PSEUDOJET_HH__

// Hadronic Delta_R measure with proper checking for phi
struct DeltaR : public PairwiseDistanceBase<DeltaR, std::vector<PseudoJet>, double> {
  typedef PseudoJet Particle;
  typedef double Value;

  DeltaR(Value R, Value beta) : PairwiseDistanceBase<DeltaR, std::vector<PseudoJet>, double>(R, beta) {}
  static Value plain_distance(const PseudoJet & p0, const PseudoJet & p1) {
    Value dphiabs(std::fabs(p0.phi() - p1.phi()));
    Value dy(p0.rap() - p1.rap()), dphi(dphiabs > PI ? TWOPI - dphiabs : dphiabs);
    return dy*dy + dphi*dphi;
  }
  static const char * name() { return "DeltaR"; }
};

// Massless dot product measure normalized with transverse momenta
struct HadronicDot : public PairwiseDistanceBase<HadronicDot, std::vector<PseudoJet>, double> {
  typedef PseudoJet Particle;
  typedef double Value;

  HadronicDot(Value R, Value beta) : PairwiseDistanceBase<HadronicDot, std::vector<PseudoJet>, double>(R, beta) {}
  static Value plain_distance(const PseudoJet & p0, const PseudoJet & p1) {
    Value d(2*(p0.E()*p1.E() - p0.px()*p1.px() - p0.py()*p1.py() - p0.pz()*p1.pz())/(p0.pt()*p1.pt()));
    return (d > 0 ? d : 0);
  }
  static const char * name() { return "HadronicDot"; }
};

// Massless dot product measure normalized by total momenta
struct EEDot : public PairwiseDistanceBase<EEDot, std::vector<PseudoJet>, double> {
  typedef PseudoJet Particle;
  typedef double Value;

  EEDot(Value R, Value beta) : PairwiseDistanceBase<EEDot, std::vector<PseudoJet>, double>(R, beta) {}
  static Value plain_distance(const PseudoJet & p0, const PseudoJet & p1) {
    Value d(2 - 2*(p0.px()*p1.px() + p0.py()*p1.py() + p0.pz()*p1.pz())/(p0.modp()*p1.modp()));
    return (d > 0 ? d : 0);
  }
  static const char * name() { return "EEDot"; }
};

// Massive dot product measure normalized with transverse energies
struct HadronicDotMassive : public PairwiseDistanceBase<HadronicDotMassive, std::vector<PseudoJet>, double> {
  typedef PseudoJet Particle;
  typedef double Value;

  HadronicDotMassive(Value R, Value beta) :
    PairwiseDistanceBase<HadronicDotMassive, std::vector<PseudoJet>, double>(R, beta)
  {}
  static Value plain_distance(const PseudoJet & p0, const PseudoJet & p1) {
    Value d(2*(p0.E()*p1.E() - p0.px()*p1.px() - p0.py()*p1.py() - p0.pz()*p1.pz())/(p0.Et()*p1.Et())
             - p0.m2()/p0.Et2() - p1.m2()/p1.Et2());
    return (d > 0 ? d : 0);
  }
  static const char * name() { return "HadronicDotMassive"; }
};

// Massive dot product measure normalized with energies
struct EEDotMassive : public PairwiseDistanceBase<EEDotMassive, std::vector<PseudoJet>, double> {
  typedef PseudoJet Particle;
  typedef double Value;

  EEDotMassive(Value R, Value beta) : PairwiseDistanceBase<EEDotMassive, std::vector<PseudoJet>, double>(R, beta) {}
  static Value plain_distance(const PseudoJet & p0, const PseudoJet & p1) {
    Value d(2 - 2*(p0.px()*p1.px() + p0.py()*p1.py() + p0.pz()*p1.pz())/(p0.E()*p1.E())
               - p0.m2()/(p0.E()*p0.E()) - p1.m2()/(p1.E()*p1.E()));
    return (d > 0 ? d : 0);
  }
  static const char * name() { return "EEDotMassive"; }
};

// Arc length between momentum vectors
struct EEArcLength : public PairwiseDistanceBase<EEArcLength, std::vector<PseudoJet>, double> {
  typedef PseudoJet Particle;
  typedef double Value;

  EEArcLength(Value R, Value beta) : PairwiseDistanceBase<EEArcLength, std::vector<PseudoJet>, double>(R, beta) {}
  static Value plain_distance(const PseudoJet & p0, const PseudoJet & p1) {
    Value dot((p0.px()*p1.px() + p0.py()*p1.py() + p0.pz()*p1.pz())/(p0.modp()*p1.modp()));
    return (dot > 1 ? 0 : (dot < -1 ? pi : std::acos(dot)));
  }
  static const char * name() { return "EEArcLength"; }
};

// Arc length between momentum vectors, normalized by the energy
struct EEArcLengthMassive : public PairwiseDistanceBase<EEArcLengthMassive, std::vector<PseudoJet>, double> {
  typedef PseudoJet Particle;
  typedef double Value;

  EEArcLengthMassive(Value R, Value beta) :
    PairwiseDistanceBase<EEArcLengthMassive, std::vector<PseudoJet>, double>(R, beta)
  {}
  static Value plain_distance(const PseudoJet & p0, const PseudoJet & p1) {
    Value dot((p0.px()*p1.px() + p0.py()*p1.py() + p0.pz()*p1.pz())/(p0.E()*p1.E()));
    return (dot > 1 ? 0 : (dot < -1 ? pi : std::acos(dot)));
  }
  static const char * name() { return "EEArcLengthMassive"; }
};

#endif // __FASTJET_PSEUDOJET_HH__

// generic distance
template<class P>
struct GenericDistance : public PairwiseDistanceBase<GenericDistance<P>, std::vector<P>, typename P::Value> {
  typedef P Particle;
  typedef typename Particle::Value Value;

  GenericDistance(Value R, Value beta) :
    PairwiseDistanceBase<GenericDistance, std::vector<P>, Value>(R, beta)
  {}
  static Value plain_distance(const Particle & p0, const Particle & p1) {
    return Particle::plain_distance(p0, p1);
  }
  static const char * name() { return Particle::distance_name(); }
};

// euclidean distances with double-precision particles
using EuclideanDistance2D = GenericDistance<EuclideanParticle2D<>>;
using EuclideanDistance3D = GenericDistance<EuclideanParticle3D<>>;
template<unsigned int N>
using EuclideanDistanceND = GenericDistance<EuclideanParticleND<N>>;

template<typename V = double>
struct EuclideanArrayDistance : PairwiseDistanceBase<EuclideanArrayDistance<V>, ArrayParticleCollection<V>, V> {
  typedef ArrayParticleCollection<V> ParticleCollection;
  typedef typename ParticleCollection::value_type Particle;
  typedef typename ParticleCollection::const_iterator ParticleIterator;
  typedef V Value;

  EuclideanArrayDistance(Value R, Value beta) :
    PairwiseDistanceBase<EuclideanArrayDistance<V>, ArrayParticleCollection<V>, V>(R, beta)
  {}

  static Value plain_distance_(const ParticleIterator & p0, const ParticleIterator & p1) {
    Value d(0);
    for (int i = 0; i < p0.stride(); i++) {
      Value dx((*p0)[i] - (*p1)[i]);
      d += dx*dx;
    }
    return d;
  }
  static const char * name() { return "EuclideanArrayDistance"; }
};

template<typename V = double>
struct CustomArrayDistance : PairwiseDistanceBase<CustomArrayDistance<V>, ArrayParticleCollection<V>, V> {
  typedef ArrayParticleCollection<V> ParticleCollection;
  typedef typename ParticleCollection::value_type Particle;
  typedef V Value;

  CustomArrayDistance(Value R, Value beta) :
    PairwiseDistanceBase<CustomArrayDistance<V>, ArrayParticleCollection<V>, V>(R, beta)
  {}
  static Value plain_distance(const Particle & p0, const Particle & p1) {
    throw std::runtime_error("Should never call this function.");
    return -1;
  }
  static const char * name() { return "CustomArrayDistance"; }

  // ensure emd isn't unscaled
  Value unscale(Value val) const { return val; }

  // override method of base class
  void fill_distances(const ParticleCollection & ps0, const ParticleCollection & ps1,
                      std::vector<Value> & dists, ExtraParticle extra)
  {}
};

#ifdef __FASTJET_PSEUDOJET_HH__
} // namespace contrib
FASTJET_END_NAMESPACE
#else
} // namespace emd
#endif

#endif // EVENTGEOMETRY_MEASURES_HH
