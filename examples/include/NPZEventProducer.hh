#ifndef NPZ_EVENT_PRODUCER_HH
#define NPZ_EVENT_PRODUCER_HH

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include "cnpy.h"

#include "EventProducer.hh"

// produce events from numpy file
class NPZEventProducer : public EventProducer {

  // arguments to class constructor
  std::string filepath_;
  double particle_pt_min_;

  // npyarray object
  cnpy::NpyArray X_, y_;
  double * X_data_, * y_data_;
  std::size_t num_particles_in_event_, particle_size_, event_size_;

public:

  NPZEventProducer(const std::string & filepath, long num_events = -1,
                   EventType event_type = All,
                   double particle_pt_min = 0,
                   unsigned print_every = 10000) :
    EventProducer(num_events, print_every, event_type),

    // arguments
    filepath_(filepath),
    particle_pt_min_(particle_pt_min)
  {
    // load npz
    cnpy::npz_t npz(cnpy::npz_load(filepath_));

    // X
    X_ = npz["X"];
    X_data_ = X_.data<double>();

    // y
    y_ = npz["y"];
    y_data_ = y_.data<double>();

    if (sizeof(double) != X_.word_size)
      throw std::runtime_error("double does not match number of bytes expected");
    if (X_.shape.size() != 3)
      throw std::runtime_error("expected a 3d array");

    tot_num_events_ = X_.shape[0];
    num_particles_in_event_ = X_.shape[1];
    particle_size_ = X_.shape[2];
    event_size_ = num_particles_in_event_ * particle_size_;

    if (tot_num_events_ != y_.shape[0])
      throw std::runtime_error("mismatch between X and y for total number of events");

    // print loaded message
    print_loaded();
  }

  bool next() {

    // iterate until we get a good event
    while (true) {

      // check if we're done with events
      if (iEvent_ == num_events_ || iEvent_ == tot_num_events_) {
        if (iEvent_ % print_every_ != 0)
          print_progress(true);
        break;
      }

      std::size_t event_start(iEvent_ * event_size_);

      // make sure to increment iEvent_ before we could possibly continue
      iEvent_++;
      int flavor(y_data_[iEvent_] == 0 ? 21 : 1);
      bool good_event(true);

      // check flavor
      if ((event_type_ == Gluon && flavor != 21) || (event_type_ == Quark && (flavor == 21 || flavor == 0)))
        good_event = false;

      // print update
      if (iEvent_ % print_every_ == 0)
          print_progress(true);

      // check for bad event
      if (!good_event) continue;

      // update number of accepted events
      iAccept_++;

      // fill up vector of particles
      particles_.clear();
      for (unsigned i = 0; i < num_particles_in_event_; i++) {
        std::size_t particle_start(event_start + i * particle_size_);
        double pt(X_data_[particle_start]);

        // append non-zero particles
        if (pt > particle_pt_min_)
          particles_.emplace_back(pt, X_data_[particle_start + 1], X_data_[particle_start + 2]);
      }

      return true;
    }

    return false;
  }

}; // NPZEventProducer

#endif // NPZ_EVENT_PRODUCER_HH
