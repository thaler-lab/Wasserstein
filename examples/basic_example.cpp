#include <cstdlib>

#include "NPZEventProducer.hh"

#include "EMD.hh"
#include "CorrelationDimension.hh"

template<class P>
std::vector<P> convert2event(const std::vector<Particle> & particles) {
  std::vector<P> euclidean_particles;
  euclidean_particles.reserve(particles.size());
  for (const Particle & particle : particles)
    euclidean_particles.push_back(P(particle.pt, {particle.y, particle.phi}));
  return euclidean_particles;
}

void single_emds(EventProducer * evp) {

  // get EMD object with R = 0.4, beta = 1.0, norm = false
  emd::EMD<emd::EuclideanEventND<2>, emd::EuclideanDistanceND<2>> emd_obj(0.4, 1.0, false);

  // preprocess events to center
  emd_obj.preprocess<emd::CenterWeightedCentroid>();

  // print description
  std::cout << emd_obj.description() << std::endl;

  // container to hold emd values
  std::vector<double> emds;

  // loop over events and compute the EMD between each successive pair
  evp->reset();
  while (true) {

    // get first event
    if (!evp->next()) break;
    std::vector<emd::EuclideanParticleND<2>> event0(convert2event<emd::EuclideanParticleND<2>>(evp->particles()));

    // get second event
    if (!evp->next()) break;
    std::vector<emd::EuclideanParticleND<2>> event1(convert2event<emd::EuclideanParticleND<2>>(evp->particles()));

    // compute emd and add it to vector
    emds.push_back(emd_obj(event0, event1));
  }

  // get max and min EMD value
  std::cout << emds.size() << " EMDs computed\n"
            << "Min. EMD - " << *std::min_element(emds.begin(), emds.end()) << '\n'
            << "Max. EMD - " << *std::max_element(emds.begin(), emds.end()) << '\n'
            << '\n';
}

void pairwise_emds(EventProducer * evp) {

  // get EMD object with R = 0.4, beta = 1.0, norm = false
  emd::PairwiseEMD<emd::EMD<emd::EuclideanEvent2D, emd::EuclideanDistance2D>>pairwise_emd_obj(0.4, 1.0, true);

  // preprocess events to center
  pairwise_emd_obj.preprocess<emd::CenterWeightedCentroid>();

  // print description
  std::cout << pairwise_emd_obj.description() << std::endl;

  // get vector of events
  std::vector<std::vector<emd::EuclideanParticle2D<>>> events;

  // loop over events and compute the EMD between each successive pair
  evp->reset();
  for (int i = 0; i < 1000 && evp->next(); i++)
    events.push_back(convert2event<emd::EuclideanParticle2D<>>(evp->particles()));

  // run computation
  pairwise_emd_obj(events);  

  // get max and min EMD value
  const std::vector<double> & emds(pairwise_emd_obj.emds(true));
  std::cout << "Min. EMD - " << *std::min_element(emds.begin(), emds.end()) << '\n'
            << "Max. EMD - " << *std::max_element(emds.begin(), emds.end()) << '\n'
            << '\n';

  // setup correlation dimension
  emd::CorrelationDimension corrdim(0.04, 1, 40);
  pairwise_emd_obj.set_external_emd_handler(corrdim);

  // rerun computation
  pairwise_emd_obj(events);

  // print out correlation dimensions
  auto corrdims(corrdim.corrdims());
  auto corrdim_bins(corrdim.corrdim_bins());
  std::cout << "\nEMD         Corr. Dim.  Error\n" << std::left;
  for (unsigned i = 0; i < corrdims.first.size(); i++)
    std::cout << std::setw(12) << corrdim_bins[i]
              << std::setw(12) << corrdims.first[i]
              << std::setw(12) << corrdims.second[i]
              << '\n';
}

EventProducer * load_events(int argc, char** argv) {

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

int main(int argc, char** argv) {

  // load events
  EventProducer * evp(load_events(argc, argv));
  if (evp == nullptr)
    return 1;

  // demonstrate some EMD usage
  single_emds(evp);
  pairwise_emds(evp);

  return 0;
}
