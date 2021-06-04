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

#include <cstdlib>

#include "NPZEventProducer.hh"

inline EventProducer * load_events(int argc, char** argv) {

  // get number of events from command line
  long num_events(1000);
  EventType evtype(All);
  if (argc >= 2)
    num_events = atol(argv[1]);
  if (argc >= 3)
    evtype = atoi(argv[2]) == 1 ? Quark : Gluon;

  // get energyflow samples
  const char * home(std::getenv("HOME"));
  if (home == NULL)
    throw std::invalid_argument("Error: cannot get HOME environment variable");

  // form path
  std::string filepath(home);
  filepath += "/.energyflow/datasets/QG_jets.npz";
  std::cout << "Filepath: " << filepath << '\n';

  // open file
  NPZEventProducer * npz(nullptr);
  try {
    npz = new NPZEventProducer(filepath, num_events, evtype);
  }
  catch (std::exception & e) {
    std::cerr << "Error: cannot open file " << filepath << ", try running "
              << "`python3 -c \"import energyflow as ef; ef.qg_jets.load()\"`\n";
    return nullptr;
  }

  return npz;
}

template<class P>
inline std::vector<P> convert2event(const std::vector<Particle> & particles) {
  std::vector<P> euclidean_particles;
  euclidean_particles.reserve(particles.size());

  for (const Particle & particle : particles)
    euclidean_particles.push_back(P(particle.pt, {particle.y, particle.phi}));

  return euclidean_particles;
}