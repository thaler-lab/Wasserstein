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

%module wasserstein

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

// include common wasserstein wrappers
%include "wasserstein_common.i"

// add functionality to get flows and dists as numpy arrays
%extend emd::EMD<emd::ArrayEvent<>, emd::EuclideanArrayDistance<>> {

  // provide weights and particle coordinates as numpy arrays
  double operator()(double* weights0, int n0,
                    double* coords0, int n00, int n01,
                    double* weights1, int n1,
                    double* coords1, int n10, int n11) {

    if (n0 != n00 || n1 != n10) {
        PyErr_SetString(PyExc_ValueError, "Number of weights does not match number of coordinates");
        return -1;
    }
    if (n01 != n11) {
      PyErr_Format(PyExc_ValueError,
                   "Dimension of coordinates must match, coords0 has dimension %i while coords1 has dimension %i",
                   n01, n11);
      return -1;
    }

    $self->set_external_dists(false);

    return (*$self)(std::make_tuple(coords0, weights0, n0, n01), std::make_tuple(coords1, weights1, n1, n11));
  }

  // provide weights and pairwise distances as numpy arrays
  double operator()(double* weights0, int n0,
                    double* weights1, int n1,
                    double* external_dists, int d0, int d1) {

    if (n0 != d0 || n1 != d1) {
      PyErr_Format(PyExc_ValueError,
                   "Incompatible distance matrix of shape (%i, %i) whereas weights0 shape is (%i,) and weights1 shape is (%i,)",
                   d0, d1, n0, n1);
      return -1;
    }

    // copy distances into vector for network simplex
    std::size_t ndists(std::size_t(d0) * std::size_t(d1));
    $self->_dists().resize(ndists);
    std::copy(external_dists, external_dists + ndists, $self->_dists().begin());

    $self->set_external_dists(true);

    return (*$self)(std::make_tuple(nullptr, weights0, n0, -1), std::make_tuple(nullptr, weights1, n1, -1));
  }
}

%extend emd::PairwiseEMD {

  // add a single event to the PairwiseEMD object
  void _add_event(double* weights, int n0, double* coords, int n1, int d) {
    $self->events().emplace_back(coords, weights, n1, d);
    $self->preprocess_back_event();
  }

  // python function to get events from container of 2d arrays, first column becomes the weights
  %pythoncode %{

    def __call__(self, events0, events1=None, gdim=None):

        if events1 is None:
            self.init(len(events0))
            events1 = []
        else:
            self.init(len(events0), len(events1))

        self.event_arrs = []
        for event in itertools.chain(events0, events1):

            # extract weights and coords from 
            event = np.atleast_2d(event)
            weights = np.ascontiguousarray(event[:,0], dtype=np.double)
            coords = np.ascontiguousarray(event[:,1:] if gdim is None else event[:,1:gdim+1])
            
            # ensure that the lifetime of these arrays lasts through the computation
            self.event_arrs.append((weights, coords))

            # store individual event
            self._add_event(weights, coords)

        # run actual computation
        self.compute()
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
