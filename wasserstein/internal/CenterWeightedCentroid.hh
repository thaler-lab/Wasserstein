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

/*   _____ ______ _   _ _______ ______ _____  
 *  / ____|  ____| \ | |__   __|  ____|  __ \ 
 * | |    | |__  |  \| |  | |  | |__  | |__) |
 * | |    |  __| | . ` |  | |  |  __| |  _  / 
 * | |____| |____| |\  |  | |  | |____| | \ \ 
 *  \_____|______|_| \_|  |_|  |______|_|  \_\
 * __          ________ _____ _____ _    _ _______ ______ _____  
 * \ \        / /  ____|_   _/ ____| |  | |__   __|  ____|  __ \ 
 *  \ \  /\  / /| |__    | || |  __| |__| |  | |  | |__  | |  | |
 *   \ \/  \/ / |  __|   | || | |_ |  __  |  | |  |  __| | |  | |
 *    \  /\  /  | |____ _| || |__| | |  | |  | |  | |____| |__| |
 *     \/  \/   |______|_____\_____|_|  |_|  |_|  |______|_____/ 
 *   _____ ______ _   _ _______ _____   ____ _____ _____  
 *  / ____|  ____| \ | |__   __|  __ \ / __ \_   _|  __ \ 
 * | |    | |__  |  \| |  | |  | |__) | |  | || | | |  | |
 * | |    |  __| | . ` |  | |  |  _  /| |  | || | | |  | |
 * | |____| |____| |\  |  | |  | | \ \| |__| || |_| |__| |
 *  \_____|______|_| \_|  |_|  |_|  \_\\____/_____|_____/ 
 */

#ifndef WASSERSTEIN_CENTERWEIGHTEDCENTROID_HH
#define WASSERSTEIN_CENTERWEIGHTEDCENTROID_HH

#include <string>
#include <type_traits>
#include <vector>

#include "EMDUtils.hh"


BEGIN_WASSERSTEIN_NAMESPACE

// center generic event according to weighted centroid
template<class EMD>
class CenterWeightedCentroid : public Preprocessor<typename EMD::Self> {
public:

  typedef typename EMD::Event Event;

  std::string description() const { return "Center according to weighted centroid"; }
  Event & operator()(Event & event) const {
    return center(event);
  }

private:

  typedef typename Event::WeightCollection WeightCollection;
  typedef typename Event::ParticleCollection ParticleCollection;
  typedef typename ParticleCollection::const_iterator const_iterator;
  typedef typename ParticleCollection::iterator iterator;

  // this version will be used for everything that's not FastJetEvent
  template<class E>
  typename std::enable_if<!std::is_base_of<FastJetEventBase, E>::value, E &>::type
  center(E & event) const {
    static_assert(std::is_same<E, Event>::value, "Event must match that of the EMD class");
    event.ensure_weights();

    // determine weighted centroid
    index_type dim(event.dimension());
    std::vector<typename Event::value_type> coords(dim, 0);

    index_type k(0);
    for (const_iterator particle = event.particles().cbegin(), end = event.particles().cend();
         particle != end; ++particle) {
      for (unsigned i = 0; i < dim; i++) 
        coords[i] += event.weights()[k] * (*particle)[i];
      k++;
    }

    for (unsigned i = 0; i < dim; i++)
      coords[i] /= event.total_weight();

    // center the particles
    for (iterator particle = event.particles().begin(), end = event.particles().end();
         particle != end; ++particle)
      for (unsigned i = 0; i < dim; i++) {
        (*particle)[i] -= coords[i];
      }

    return event;
  }

  // enable overloaded operator for FastJetEvent
#ifdef WASSERSTEIN_FASTJET
  FastJetEvent<typename EMD::ParticleWeight> &
  center(FastJetEvent<typename EMD::ParticleWeight> & event) const;
#endif

}; // CenterWeightedCentroid

END_WASSERSTEIN_NAMESPACE

#endif // WASSERSTEIN_CENTERWEIGHTEDCENTROID_HH
