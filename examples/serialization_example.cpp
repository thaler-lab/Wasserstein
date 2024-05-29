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
#define WASSERSTEIN_SERIALIZATION
#include "Wasserstein.hh"

// classes and functions for reading/preparing events
#include "ExampleUtils.hh"

// `EMDFloat64` uses `double` for the floating-point type
// first template parameter is an Event type, second is a PairwiseDistance type
using EMD = emd::EMDFloat64<emd::EuclideanEvent2D, emd::YPhiParticleDistance>;

// `PairwiseEMD` is a templated class accepting a fully qualified `EMD` type
using PairwiseEMD = emd::PairwiseEMD<EMD>;

// empty angle brackets use the default floating-point type `double`
using CorrelationDimension = emd::CorrelationDimension<>;

// the `EuclideanParticle[N]D` classes provide a simple container for weighted particles
using EMDParticle = emd::EuclideanParticle2D<>;

int main(int argc, char** argv) {

  // load events
  EventProducer * evp(load_events(argc, argv));
  if (evp == nullptr)
    return 1;

  ////////////////////////////
  // EMD Serialization
  ////////////////////////////

  // demonstrate calculating single EMDs
  EMD emd_obj(0.4, 1.0, true);

  // preprocess events to center
  emd_obj.preprocess<emd::CenterWeightedCentroid>();

  // print description
  std::cout << emd_obj.description() << std::endl;

  // serialize
  std::stringstream ss;
  boost::archive::text_oarchive oa(ss);
  oa << emd_obj;

  // extract
  boost::archive::text_iarchive ia(ss);
  EMD new_emd_obj;
  ia >> new_emd_obj;

  // print EMD description
  std::cout << new_emd_obj.description()
            << "Note that preprocessors are not currently preserved in serialized objects"
            << std::endl;

  /////////////////////////////////////////////////////
  // PairwiseEMD and CorrelationDimension Serialization
  /////////////////////////////////////////////////////

  // demonstrate calculating pairwise EMDs
  PairwiseEMD pairwise_emd_obj(0.4, 1.0, false);

  // preprocess events to center
  pairwise_emd_obj.preprocess<emd::CenterWeightedCentroid>();

  // print description
  std::cout << pairwise_emd_obj.description() << std::endl;

  // get vector of events
  std::vector<std::vector<EMDParticle>> events;

  // loop over events and compute the EMD between each successive pair
  evp->reset();
  for (int i = 0; i < 1000 && evp->next(); i++)
    events.push_back(convert2event<EMDParticle>(evp->particles()));

  // setup correlation dimension
  CorrelationDimension corrdim(50, 10., 250.);
  pairwise_emd_obj.set_external_emd_handler(corrdim);

  // run computation
  pairwise_emd_obj(events);

  // serialize
  std::stringstream ss2;
  boost::archive::text_oarchive oa2(ss2);
  oa2 << pairwise_emd_obj << corrdim;

  // define empty objects
  PairwiseEMD new_pairwise_emd_obj;
  CorrelationDimension new_corrdim;

  // extract
  boost::archive::text_iarchive ia2(ss2);
  ia2 >> new_pairwise_emd_obj >> new_corrdim;

  // print PairwiseEMD description
  std::cout << new_pairwise_emd_obj.description()
            << "Note that preprocessors are not currently preserved in serialized objects"
            << std::endl;

  // print out correlation dimensions
  auto corrdims(new_corrdim.corrdims());
  auto corrdim_bins(new_corrdim.corrdim_bins());
  std::cout << "\nEMD         Corr. Dim.  Error\n" << std::left;
  for (unsigned i = 0; i < corrdims.first.size(); i++)
    std::cout << std::setw(12) << corrdim_bins[i]
              << std::setw(12) << corrdims.first[i]
              << std::setw(12) << corrdims.second[i]
              << '\n';
}
