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

// Wasserstein library
#include "Wasserstein.hh"

// classes and functions for reading/preparing events
#include "ExampleUtils.hh"

// `EMDFloat64` uses `double` for the floating-point type
// first template parameter is an Event type, second is a PairwiseDistance type
using EMD = emd::EMDFloat64<emd::EuclideanEvent2D, emd::YPhiParticleDistance>;

// `PairwiseEMD` is a templated class accepting a fully qualified `EMD` type
using PairwiseEMD = emd::PairwiseEMD<EMD>;

// the `EuclideanParticle[N]D` classes provide a simple container for weighted particles
using EMDParticle = emd::EuclideanParticle2D<>;

int main(int argc, char** argv) {

  // load events
  EventProducer * evp(load_events(argc, argv));
  if (evp == nullptr)
    return 1;

  // demonstrate some EMD usage
  double EMD_R = 0.4;
  double EMD_beta = 1;
  bool EMD_norm = true;
  PairwiseEMD pairwise_emd_obj(EMD_R, EMD_beta, EMD_norm);

  // preprocess events to center
  pairwise_emd_obj.preprocess<emd::CenterWeightedCentroid>();

  // print description
  std::cout << pairwise_emd_obj.description() << std::endl;

  // get vector of events
  std::vector<std::vector<EMDParticle>> events;

  // loop over events and compute the EMD between each successive pair
  evp->reset();
  while (evp->next())
    events.push_back(convert2event<EMDParticle>(evp->particles()));

  // run computation
  pairwise_emd_obj(events.begin(), events.begin() + evp->num_accepted()/2,
                   events.begin() + evp->num_accepted()/2, events.end());

  // get max and min EMD value
  const std::vector<double> & emds_raw(pairwise_emd_obj.emds());
  std::cout << "Min. EMD - " << *std::min_element(emds_raw.begin(), emds_raw.end()) << '\n'
            << "Max. EMD - " << *std::max_element(emds_raw.begin(), emds_raw.end()) << '\n'
            << emds_raw.size() << " emds\n"
            << '\n';

  // setup EMD object to compute cross section mover's distance
  double SigmaMD_R = 1;
  double SigmaMD_beta = 1;
  bool SigmaMD_norm = true;
  bool SigmaMD_do_timing = true;

  // external dists are used by the default configuration
  emd::EMDFloat64<> sigmamd_obj(SigmaMD_R, SigmaMD_beta, SigmaMD_norm, SigmaMD_do_timing);

  std::cout << sigmamd_obj.description() << '\n';

  // set distances
  auto emds(pairwise_emd_obj.emds());
  sigmamd_obj.ground_dists().resize(emds.size());
  for (std::size_t i = 0; i < emds.size(); i++)
    sigmamd_obj.ground_dists()[i] = emds[i];

  // form datasets
  std::vector<double> weights0(pairwise_emd_obj.nevA(), 1),
                      weights1(pairwise_emd_obj.nevB(), 1);

  std::cout << "Running computation ..." << std::endl;

  // run computation
  std::cout << "Cross-section Mover's Distance : " << sigmamd_obj(weights0, weights1) << '\n'
            << "Done in " << sigmamd_obj.duration() << "s\n";

  return 0;
}
