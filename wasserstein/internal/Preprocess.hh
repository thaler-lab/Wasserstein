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

#ifndef WASSERSTEIN_PREPROCESS_HH
#define WASSERSTEIN_PREPROCESS_HH

// C++ standard library
#include <algorithm>
#include <cassert>

#include "Event.hh"

BEGIN_WASSERSTEIN_NAMESPACE

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

// currently, all preprocessors require FastJet
#ifdef __FASTJET_PSEUDOJET_HH__

// center all the particles in a vector according to a given rapidity and azimuth
template<class ParticleWeight>
void center_event(FastJetEvent<ParticleWeight> & event, Value rap, Value phi) {
  PseudoJet & axis(event.axis());
  axis.reset_momentum_PtYPhiM(axis.pt(), axis.rap() - rap, phi_fix(axis.phi(), phi) - phi, axis.m());
  for (PseudoJet & pj: event.particles())
    pj.reset_momentum_PtYPhiM(pj.pt(), pj.rap() - rap, phi_fix(pj.phi(), phi) - phi, pj.m());
}

////////////////////////////////////////////////////////////////////////////////
// CenterEScheme - center all the particles according to their Escheme axis
////////////////////////////////////////////////////////////////////////////////

template<class EMD>
class CenterEScheme : public Preprocessor<typename EMD::Self> {
public:
  typedef typename EMD::Event Event;

  static_assert(std::is_base_of<FastJetEventBase, Event>::value,
                "CenterEScheme works only with FastJet events.");

  std::string description() const { return "Center according to E-scheme axis"; }
  Event & operator()(Event & event) const {

    // set pj to Escheme axis if it isn't already
  #ifdef __FASTJET_JETDEFINITION_HH__
    if (!event.axis().has_valid_cs() || event.axis().validated_cs()->jet_def().recombination_scheme() != E_scheme)
  #endif
    {
      event.axis().reset_momentum_PtYPhiM(0, 0, 0, 0);
      for (const PseudoJet & pj : event.particles())
        event.axis() += pj;
    }

    // center the particles
    center_event(event, event.axis().rap(), event.axis().phi());

    return event;
  }

}; // CenterEScheme

////////////////////////////////////////////////////////////////////////////////
// CenterPtCentroid - center all the particles according to their pT centroid
////////////////////////////////////////////////////////////////////////////////

template<class EMD>
class CenterPtCentroid : public Preprocessor<typename EMD::Self> {
public:
  typedef typename EMD::Event Event;
  typedef typename EMD::ValuePublic Value;

  static_assert(std::is_base_of<FastJetEventBase, Event>::value,
                "CenterPtCentroid works only with FastJet events.");

  std::string description() const { return "Center according to pT centroid"; }
  Event & operator()(Event & event) const {

    // determine pt centroid
    Value pttot(0), y(0), phi(0);
    for (const PseudoJet & pj : event.particles()) {
      Value pt(pj.pt());
      pttot += pt;
      y += pt * pj.rap();
      phi += pt * phi_fix(pj.phi(), event.particles()[0].phi());
    }
    y /= pttot;
    phi /= pttot;

    // set PtCentroid as axis
    event.axis().reset_momentum_PtYPhiM(pttot, y, phi, 0);

    // center the particles
    center_event(event, y, phi);

    return event;
  }

}; // CenterPtCentroid

////////////////////////////////////////////////////////////////////////////////
// CenterWeightedCentroid - center all the particles according to their weighted centroid
////////////////////////////////////////////////////////////////////////////////

template<class EMD>
class CenterWeightedCentroid : public Preprocessor<typename EMD::Self> {
public:
  typedef typename EMD::Event Event;
  typedef typename EMD::ValuePublic Value;
  typedef typename Event::ParticleCollection ParticleCollection;
  typedef typename Event::WeightCollection WeightCollection;

  std::string description() const { return "Center according to weighted centroid"; }
  Event & operator()(Event & event) const {
    event.ensure_weights();

    const ParticleCollection & ps(event.particles());
    const WeightCollection & ws(event.weights());

    // determine weighted centroid
    Value x(0), y(0);
    for (std::size_t i = 0; i < ps.size(); i++) {
      x += ws[i] * ps[i].rap();
      y += ws[i] * phi_fix(ps[i].phi(), ps[0].phi());
    }
    x /= event.total_weight();
    y /= event.total_weight();

    // set PtCentroid as pj of the event
    event.axis().reset_momentum_PtYPhiM(event.total_weight(), x, y, 0);

    // center the particles
    center_event(event, x, y);

    return event;
  }

}; // CenterWeightedCentroid

// mask out particles farther than a certain distance from the pseudojet of the event
// since this uses the pseudojet, make sure it is properly set first
template<class EMD>
class MaskCircleRapPhi : public Preprocessor<typename EMD::Self> {
public:
  typedef typename EMD::Event Event;
  typedef typename EMD::ValuePublic Value;

  MaskCircleRapPhi(Value R) : R_(R), R2_(R*R) {}
  std::string description() const { return "Mask particles farther than " + std::to_string(R_) + " from axis"; }
  Event & operator()(Event & event) const {

    std::vector<PseudoJet> & ps(event.particles());

    // get indices of particles to remove
    std::vector<std::size_t> inds;
    for (std::size_t i = 0; i < ps.size(); i++)
      if (EMD::PairwiseDistance::plain_distance(event.axis(), ps[i]) > R2_)
        inds.push_back(i);

    // remove particles and weights if they exist
    if (!inds.empty()) {
      std::reverse(inds.begin(), inds.end());

      for (std::size_t i : inds)
        ps.erase(ps.begin() + i);

      if (event.has_weights()) {
        for (std::size_t i : inds) {
          event.total_weight() -= event.weights()[i];
          event.weights().erase(event.weights().begin() + i);
        }
      }
    }

    return event;
  }

private:

  Value R_, R2_;

}; // MaskCircleRapPhi

#endif // __FASTJET_PSEUDOJET_HH__

END_WASSERSTEIN_NAMESPACE

#endif // WASSERSTEIN_PREPROCESS_HH
