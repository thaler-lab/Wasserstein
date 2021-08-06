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
// astd::ptrdiff_t with this program.  If not, see <https://www.gnu.org/licenses/>.
//------------------------------------------------------------------------

%module("threads"=1) wasserstein
%nothreadallow;

#define WASSERSTEIN_NAMESPACE wasserstein

// this ensures SWIG parses class members properly
#define SWIG_WASSERSTEIN

%{
// this controls some class access modifiers
#ifndef SWIG_WASSERSTEIN
# define SWIG_WASSERSTEIN
#endif

namespace wasserstein {}
using namespace wasserstein;
%}

%feature("autodoc", "1");

// include common wasserstein wrappers
%include "wasserstein_common.i"

%define WASSERSTEIN_EMD_NUMPY_FUNCS(F)

  // provide weights and particle coordinates as numpy arrays
  F operator()(F* weights0, std::ptrdiff_t n0,
               F* coords0,  std::ptrdiff_t n00, std::ptrdiff_t n01,
               F* weights1, std::ptrdiff_t n1,
               F* coords1,  std::ptrdiff_t n10, std::ptrdiff_t n11) {

    if (n0 != n00 || n1 != n10)
      throw std::invalid_argument("Number of weights does not match number of coordinates");
    if (n01 != n11)
      throw std::invalid_argument("Coordinate dimensions do not match");

    $self->set_external_dists(false);

    return (*$self)(std::make_tuple(weights0, coords0, n0, n01), std::make_tuple(weights1, coords1, n1, n11));
  }

  // provide weights and pairwise distances as numpy arrays
  F operator()(F* weights0, std::ptrdiff_t n0,
               F* weights1, std::ptrdiff_t n1,
               F* external_dists, std::ptrdiff_t d0, std::ptrdiff_t d1) {

    if (n0 != d0 || n1 != d1)
      throw std::invalid_argument("Weights and distance matrix are incompatible");

    // copy distances into vector for network simplex
    std::size_t ndists(std::size_t(d0) * std::size_t(d1));
    $self->ground_dists().resize(ndists);
    std::copy(external_dists, external_dists + ndists, $self->ground_dists().begin());

    $self->set_external_dists(true);

    return (*$self)(std::make_tuple(weights0, nullptr, n0, -1), std::make_tuple(weights1, nullptr, n1, -1));
  }
%enddef

%pythoncode %{

# function for storing events in a pairwise_emd object
def _store_events(pairwise_emd, events, event_weights, gdim, mask):

    if mask:
        R2 = pairwise_emd.R()**2

    for event, event_weight in zip(events, event_weights):

        # ensure event is 2d
        event = np.atleast_2d(event)

        # consider gdim
        if gdim is not None:
            event = event[:,:gdim+1]

        # consider mask
        if mask:
            event = event[np.sum(event[:,1:]**2, axis=1) <= R2]

        # extract weights and coords
        weights = np.asarray(event[:,0], dtype=np.double, order='C')
        coords = np.asarray(event[:,1:], dtype=np.double, order='C')
        
        # ensure that the lifetime of these arrays lasts through the computation
        pairwise_emd.event_arrs.append((weights, coords))

        # store individual event
        pairwise_emd._add_event(weights, coords, event_weight)
%}

%define WASSERSTEIN_PAIRWISE_EMD_NUMPY_FUNCS(F)
  // add a single event to the PairwiseEMD object
  void _add_event(F* weights, std::ptrdiff_t n,
                  F* coords, std::ptrdiff_t n1, std::ptrdiff_t d,
                  F event_weight = 1) {

    $self->events().emplace_back(weights, coords, n1, d, event_weight);
    $self->preprocess_back_event();
  }
%enddef

namespace WASSERSTEIN_NAMESPACE {

  %extend PairwiseEMD {

    void _reset_B_events() {
      $self->events().resize($self->nevA());
    }

    // python function to get events from container of 2d arrays, first column becomes the weights
    %pythoncode %{

      # ensure proper destruction of objects held by this instance
      def __del__(self):
          super().__del__()
          if hasattr(self, 'event_arrs'):
              del self.event_arrs

      def set_new_eventsB(self, eventsB, gdim=None, mask=False, event_weightsB=None):

          # check that we have been initialized before
          if not hasattr(self, 'event_arrs'):
              raise RuntimeError('PairwiseEMD object must be called on some events before the B events can be reset')

          # check that we are in request mode
          if not self.request_mode():
              raise RuntimeError('PairwiseEMD object must be in request mode in order to set new eventsB')

          if event_weightsB is None:
              event_weightsB = np.ones(len(eventsB), dtype=np.double)
          elif len(event_weightsB) != len(eventsB):
              raise ValueError('length of `event_weightsB` does not match length of `eventsB`')

          # clear away old B events in underlying object
          self._reset_B_events()

          # clear B events from python array
          del self.event_arrs[self.nevA():]

          # reinitialize
          self.init(self.nevA(), len(eventsB))
          _store_events(self, eventsB, event_weightsB, gdim, mask)
    %}

    // ensure that python array of events is deleted also
    %feature("shadow") clear %{
      def clear(self):
          $action(self)
          self.event_arrs = []
    %}
  }

  // extend/instantiate specific EMD classes
  %extend EMD<double, DefaultArrayEvent,  EuclideanArrayDistance> { WASSERSTEIN_EMD_NUMPY_FUNCS(double) }
  %extend EMD<float,  DefaultArrayEvent,  EuclideanArrayDistance> { WASSERSTEIN_EMD_NUMPY_FUNCS(float) }
  %extend EMD<double, DefaultArray2Event, YPhiArrayDistance> { WASSERSTEIN_EMD_NUMPY_FUNCS(double) }
  %extend EMD<float,  DefaultArray2Event, YPhiArrayDistance> { WASSERSTEIN_EMD_NUMPY_FUNCS(float) }
  %template(EMDFloat64)     EMD<double, DefaultArrayEvent,  EuclideanArrayDistance>;
  %template(EMDFloat32)     EMD<float,  DefaultArrayEvent,  EuclideanArrayDistance>;
  %template(EMDYPhiFloat64) EMD<double, DefaultArray2Event, YPhiArrayDistance>;
  %template(EMDYPhiFloat32) EMD<float,  DefaultArray2Event, YPhiArrayDistance>;

  // extend/instantiate specific PairwiseEMD classes
  %extend PairwiseEMD<EMD<double, DefaultArrayEvent,  EuclideanArrayDistance>, double> { WASSERSTEIN_PAIRWISE_EMD_NUMPY_FUNCS(double) }
  %extend PairwiseEMD<EMD<float,  DefaultArrayEvent,  EuclideanArrayDistance>, float> {  WASSERSTEIN_PAIRWISE_EMD_NUMPY_FUNCS(float) }
  %extend PairwiseEMD<EMD<double, DefaultArray2Event, YPhiArrayDistance>, double> { WASSERSTEIN_PAIRWISE_EMD_NUMPY_FUNCS(double) }
  %extend PairwiseEMD<EMD<float,  DefaultArray2Event, YPhiArrayDistance>, float> {  WASSERSTEIN_PAIRWISE_EMD_NUMPY_FUNCS(float) }
  %template(PairwiseEMDFloat64)     PairwiseEMD<EMD<double, DefaultArrayEvent,  EuclideanArrayDistance>, double>;
  %template(PairwiseEMDFloat32)     PairwiseEMD<EMD<float,  DefaultArrayEvent,  EuclideanArrayDistance>, float>;
  %template(PairwiseEMDYPhiFloat64) PairwiseEMD<EMD<double, DefaultArray2Event, YPhiArrayDistance>, double>;
  %template(PairwiseEMDYPhiFloat32) PairwiseEMD<EMD<float,  DefaultArray2Event, YPhiArrayDistance>, float>;

} // namespace WASSERSTEIN_NAMESPACE

DECLARE_PYTHON_FUNC_VARIABLE_FLOAT_TYPE(EMD)
DECLARE_PYTHON_FUNC_VARIABLE_FLOAT_TYPE(EMDYPhi)
DECLARE_PYTHON_FUNC_VARIABLE_FLOAT_TYPE(PairwiseEMD)
DECLARE_PYTHON_FUNC_VARIABLE_FLOAT_TYPE(PairwiseEMDYPhi)
