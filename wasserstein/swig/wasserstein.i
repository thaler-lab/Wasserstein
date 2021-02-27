// -*- C -*-
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

%module("threads"=1) wasserstein
%nothreadallow;

#define EMDNAMESPACE emd

// this ensures SWIG parses class members properly
#define SWIG_WASSERSTEIN

// use numpy
#define SWIG_NUMPY

%{
// this controls some class access modifiers
#ifndef SWIG_WASSERSTEIN
#define SWIG_WASSERSTEIN
#endif
%}

// Python imports at the top of the module
%pythonbegin %{
import itertools
import numpy as np
%}

%feature("autodoc", "1");

// include common wasserstein wrappers
%include "wasserstein_common.i"

// add functionality to get flows and dists as numpy arrays
%extend emd::EMD<emd::ArrayEvent<>, emd::EuclideanArrayDistance<>> {

  // provide weights and particle coordinates as numpy arrays
  double operator()(double* weights0, int n0,
                    double* coords0, int n00, int n01,
                    double* weights1, int n1,
                    double* coords1, int n10, int n11) {

    if (n0 != n00 || n1 != n10)
      throw std::invalid_argument("Number of weights does not match number of coordinates");
    if (n01 != n11)
      throw std::invalid_argument("Coordinate dimensions do not match");

    $self->set_external_dists(false);

    return (*$self)(std::make_tuple(coords0, weights0, n0, n01), std::make_tuple(coords1, weights1, n1, n11));
  }

  // provide weights and pairwise distances as numpy arrays
  double operator()(double* weights0, int n0,
                    double* weights1, int n1,
                    double* external_dists, int d0, int d1) {

    if (n0 != d0 || n1 != d1)
      throw std::invalid_argument("Weights and distance matrix are incompatible");

    // copy distances into vector for network simplex
    std::size_t ndists(std::size_t(d0) * std::size_t(d1));
    $self->ground_dists().resize(ndists);
    std::copy(external_dists, external_dists + ndists, $self->ground_dists().begin());

    $self->set_external_dists(true);

    return (*$self)(std::make_tuple(nullptr, weights0, n0, -1), std::make_tuple(nullptr, weights1, n1, -1));
  }
}

%pythoncode %{

# function for storing events in a pairwise_emd object
def _store_events(pairwise_emd, events, gdim, mask):

    if mask:
        R2 = pairwise_emd.R()**2

    for event in events:

        # ensure event is 2d
        event = np.atleast_2d(event)

        # consider gdim
        if gdim is not None:
            event = event[:,:gdim+1]

        # consider mask
        if mask:
            event = event[np.sum(event**2, axis=1) <= R2]

        # extract weights and coords
        weights = np.asarray(event[:,0], dtype=np.double, order='C')
        coords = np.asarray(event[:,1:], dtype=np.double, order='C')
        
        # ensure that the lifetime of these arrays lasts through the computation
        pairwise_emd.event_arrs.append((weights, coords))

        # store individual event
        pairwise_emd._add_event(weights, coords)
%}

%extend emd::PairwiseEMD {

  // add a single event to the PairwiseEMD object
  void _add_event(double* weights, int n0, double* coords, int n1, int d) {
    $self->events().emplace_back(coords, weights, n1, d);
    $self->preprocess_back_event();
  }

  void _reset_B_events() {
    $self->events().resize($self->nevA());
  }

  // python function to get events from container of 2d arrays, first column becomes the weights
  %pythoncode %{

    def __call__(self, eventsA, eventsB=None, gdim=None, mask=False):

        if eventsB is None:
            self.init(len(eventsA))
            eventsB = []
        else:
            self.init(len(eventsA), len(eventsB))

        self.event_arrs = []
        _store_events(self, itertools.chain(eventsA, eventsB), gdim, mask)

        # run actual computation
        if not self.request_mode():
            self.compute()

    def set_new_eventsB(self, eventsB, gdim=None, mask=False):

        # check that we have been initialized before
        if not hasattr(self, 'event_arrs'):
            raise RuntimeError('PairwiseEMD object must be called on some events before the B events can be reset')

        # clear away old B events in underlying object
        self._reset_B_events()

        # clear B events from python array
        del self.event_arrs[self.nevA():]

        # reinitialize
        self.init(self.nevA(), len(eventsB))
        _store_events(self, eventsB, gdim, mask)
  %}
}

// ensure that python array of events is deleted also
%feature("shadow") emd::PairwiseEMD::clear %{
  def clear(self):
      $action(self)
      self.event_arrs = []
%}

// instantiate specific (Pairwise)EMD templates
%template(EMD) emd::EMD<emd::ArrayEvent<>, emd::EuclideanArrayDistance<>>;
%template(PairwiseEMD) emd::PairwiseEMD<emd::EMD<emd::ArrayEvent<>, emd::EuclideanArrayDistance<>>>;
