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

// C++ standard library wrappers
%include "std_pair.i"
%include "std_string.i"
%include "std_vector.i"
%include "exception.i"

// vector templates
%template(vectorDouble) std::vector<double>;
%template(vectorString) std::vector<std::string>;
%template(pairVectorDouble) std::pair<std::vector<double>, std::vector<double>>;

// this ensures SWIG parses class members properly
#define SWIG_WASSERSTEIN

%{
// include this to avoid needing to define it at compile time 
#ifndef SWIG
#define SWIG
#endif

// this controls some class access modifiers
#ifndef SWIG_WASSERSTEIN
#define SWIG_WASSERSTEIN
#endif

// needed by numpy.i
#define SWIG_FILE_WITH_INIT

// standard library headers we need
#include <cstdlib>
#include <cstring>

// the main library headers
#include "EMD.hh"
#include "CorrelationDimension.hh"
%}

// Python imports at the top of the module
%pythonbegin %{
import itertools
import numpy as np
%}

// numpy wrapping and initialization
%include numpy.i
%init %{
import_array();
%}

// numpy typemaps
%apply (double* IN_ARRAY1, int DIM1) {(double* weights0, int n0), (double* weights1, int n1)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* coords0, int n00, int n01),
                                                (double* coords1, int n10, int n11),
                                                (double* external_dists, int d0, int d1)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* weights, int n0)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* coords, int n1, int d)}
%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1) {(double** arr_out0, int* n0), (double** arr_out1, int* n1)}
%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(double** arr_out, int* n0, int* n1)}

// allow threads in PairwiseEMD computation
%threadallow emd::PairwiseEMD::compute();

// basic exception handling for all functions
%exception {
  try { $action }
  catch (std::exception & e) { SWIG_exception(SWIG_SystemError, e.what()); }
}

// EMD exceptions
%exception emd::EMD::EMD {
  try { $action }
  catch (std::invalid_argument & e) { SWIG_exception(SWIG_ValueError, e.what()); }
  catch (std::exception & e) { SWIG_exception(SWIG_SystemError, e.what()); }
}
%exception emd::EMD::operator() {
  try { $action }
  catch (std::runtime_error & e) { SWIG_exception(SWIG_RuntimeError, e.what()); }
  catch (std::exception & e) { SWIG_exception(SWIG_SystemError, e.what()); }
  if (PyErr_Occurred() != NULL)
    SWIG_fail;
}
%exception emd::EMD::flow {
  try { $action }
  catch (std::out_of_range & e) { SWIG_exception(SWIG_IndexError, e.what()); }
  catch (std::exception & e) { SWIG_exception(SWIG_SystemError, e.what()); }
}

// PairwiseEMD exceptions
%exception emd::PairwiseEMD::PairwiseEMD {
  try { $action }
  catch (std::invalid_argument & e) { SWIG_exception(SWIG_ValueError, e.what()); }
  catch (std::exception & e) { SWIG_exception(SWIG_SystemError, e.what()); }
}
%exception emd::PairwiseEMD::emd {
  try { $action }
  catch (std::out_of_range & e) { SWIG_exception(SWIG_IndexError, e.what()); }
  catch (std::logic_error & e) { SWIG_exception(SWIG_RuntimeError, e.what()); }
  catch (std::exception & e) { SWIG_exception(SWIG_SystemError, e.what()); }
}
%exception emd::PairwiseEMD::compute() {
  try { $action }
  catch (std::runtime_error & e) { SWIG_exception(SWIG_RuntimeError, e.what()); }
  catch (std::exception & e) { SWIG_exception(SWIG_SystemError, e.what()); }
}

// EMDUtils exceptions
%exception emd::check_emd_status {
  try { $action }
  catch (std::runtime_error & e) { SWIG_exception(SWIG_RuntimeError, e.what()); }
  catch (std::exception & e) { SWIG_exception(SWIG_SystemError, e.what()); } 
}

// Histogram exceptions
%exception emd::Histogram1DHandler::Histogram1DHandler {
  try { $action }
  catch (std::invalid_argument & e) { SWIG_exception(SWIG_ValueError, e.what()); }
  catch (std::exception & e) { SWIG_exception(SWIG_SystemError, e.what()); }  
}

// CorrelationDimension exceptions
%exception emd::CorrelationDimension::CorrelationDimension {
  try { $action }
  catch (std::invalid_argument & e) { SWIG_exception(SWIG_ValueError, e.what()); }
  catch (std::exception & e) { SWIG_exception(SWIG_SystemError, e.what()); }
}

// ignore certain functions
%ignore emd::EMD::compute;
%ignore emd::EMD::network_simplex;
%ignore emd::EMD::pairwise_distance;
%ignore emd::EMD::_dists;
%ignore emd::PairwiseEMD::compute(const EventVector & events);
%ignore emd::PairwiseEMD::compute(const EventVector & eventsA, const EventVector & eventsB);
%ignore emd::PairwiseEMD::events;
%ignore emd::Histogram1DHandler::print_axis;
%ignore emd::Histogram1DHandler::print_hist;

// include EMD utilities
%include "internal/EMDUtils.hh"

// handle templated base class
%template(EMDBaseDouble) emd::EMDBase<double>;

// include main EMD code
%include "EMD.hh"

// include histogram code
#define SWIG_PREPROCESSOR
%include "internal/HistogramUtils.hh"

// prepare to extend classes by renaming some methods
%rename(flows_vec) emd::EMD::flows;
%rename(dists_vec) emd::EMD::dists;
%rename(flows) emd::EMD::npy_flows;
%rename(dists) emd::EMD::npy_dists;
%rename(emds_vec) emd::PairwiseEMD::emds;
%rename(emds) emd::PairwiseEMD::npy_emds;
%rename(bin_centers_vec) emd::Histogram1DHandler::bin_centers;
%rename(bin_edges_vec) emd::Histogram1DHandler::bin_edges;
%rename(hist_vals_errs_vec) emd::Histogram1DHandler::hist_vals_errs;
%rename(bin_centers) emd::Histogram1DHandler::npy_bin_centers;
%rename(bin_edges) emd::Histogram1DHandler::npy_bin_edges;
%rename(hist_vals_errs) emd::Histogram1DHandler::npy_hist_vals_errs;
%rename(cumulative_vals_vars_vec) emd::CorrelationDimension::cumulative_vals_vars;
%rename(corrdim_bins_vec) emd::CorrelationDimension::corrdim_bins;
%rename(corrdims_vec) emd::CorrelationDimension::corrdims;
%rename(cumulative_vals_vars) emd::CorrelationDimension::npy_cumulative_vals_vars;
%rename(corrdim_bins) emd::CorrelationDimension::npy_corrdim_bins;
%rename(corrdims) emd::CorrelationDimension::npy_corrdims;

// makes python class printable from a description method
%define ADD_STR_FROM_DESCRIPTION
std::string __str__() const {
  return $self->description();
}
%enddef

// mallocs a 1D array of doubles of the specified size
%define MALLOC_1D_VALUE_ARRAY(arr_out, n, size, nbytes)
  *n = size;
  size_t nbytes = size_t(*n)*sizeof(double);
  *arr_out = (double *) malloc(nbytes);
  if (*arr_out == NULL) {
    PyErr_Format(PyExc_MemoryError, "Failed to allocate %zu bytes", nbytes);
    return;
  }
%enddef

%define RETURN_1DNUMPY_FROM_VECTOR(pyname, cppname, size)
void pyname(double** arr_out0, int* n0) {
  MALLOC_1D_VALUE_ARRAY(arr_out0, n0, size, nbytes)
  memcpy(*arr_out0, $self->cppname().data(), nbytes);
}
%enddef

%define PAIRED_1DNUMPY_FROM_VECPAIR(cppfunccall, size)
MALLOC_1D_VALUE_ARRAY(arr_out0, n0, size, nbytes0)
MALLOC_1D_VALUE_ARRAY(arr_out1, n1, size, nbytes1)
std::pair<std::vector<double>, std::vector<double>> vecpair($self->cppfunccall);
memcpy(*arr_out0, vecpair.first.data(), nbytes0);
memcpy(*arr_out1, vecpair.second.data(), nbytes1);
%enddef

%define RETURN_PAIRED_1DNUMPY_FROM_VECPAIR(pyname, cppfunccall, size)
void pyname(double** arr_out0, int* n0, double** arr_out1, int* n1) {
  PAIRED_1DNUMPY_FROM_VECPAIR(cppfunccall, size)
}
%enddef

// mallocs a 2D array of doubles of the specified size
%define MALLOC_2D_VALUE_ARRAY(a, b)
  *n0 = a;
  *n1 = b;
  size_t num_elements = size_t(*n0)*size_t(*n1);
  size_t nbytes = num_elements*sizeof(double);
  double * values = (double *) malloc(nbytes);
  if (values == NULL) {
    PyErr_Format(PyExc_MemoryError, "Failed to allocate %zu bytes", nbytes);
    return;
  }
  *arr_out = values;
%enddef

// add functionality to get flows and dists as numpy arrays
%extend emd::EMD<emd::ArrayEvent<>, emd::EuclideanArrayDistance<>> {
  ADD_STR_FROM_DESCRIPTION

  void npy_flows(double** arr_out, int* n0, int* n1) {
    MALLOC_2D_VALUE_ARRAY($self->n0(), $self->n1())
    memcpy(*arr_out, $self->network_simplex().flows().data(), nbytes);
    
    for (size_t i = 0; i < num_elements; i++)
      values[i] *= $self->scale();
  }

  void npy_dists(double** arr_out, int* n0, int* n1) {
    MALLOC_2D_VALUE_ARRAY($self->n0(), $self->n1())
    memcpy(*arr_out, $self->network_simplex().dists().data(), nbytes);
  }

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
  ADD_STR_FROM_DESCRIPTION

  // add a single event to the PairwiseEMD object
  void _add_event(double* weights, int n0, double* coords, int n1, int d) {
    $self->events().emplace_back(coords, weights, n1, d);
    $self->preprocess_back_event();
  }

  void npy_emds(double** arr_out, int* n0, int* n1) {
    MALLOC_2D_VALUE_ARRAY($self->nevA(), $self->nevB())
    memcpy(*arr_out, $self->emds().data(), nbytes);
  }

  // python function to get events from container of 2d arrays, first column becomes the weights
  %pythoncode %{

    # ensure proper destruction of objects held by this instance
    def __del__(self):
        if hasattr(self, 'event_arrs'):
            del self.event_arrs
        if hasattr(self, 'external_emd_handler'):
            self.external_emd_handler.thisown = 1
            del self.external_emd_handler

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

// ensure that python array of events is deleted also
%feature("shadow") emd::PairwiseEMD::set_external_emd_handler %{
  def set_external_emd_handler(self, handler):
      if not handler.thisown:
          raise RuntimeError('ExternalEMDHandler must own itself; perhaps it is already in use elsewhere')
      handler.thisown = 0
      $action(self, handler)
      self.external_emd_handler = handler
%}

// instantiate specific (Pairwise)EMD templates
%template(EMD) emd::EMD<emd::ArrayEvent<>, emd::EuclideanArrayDistance<>>;
%template(PairwiseEMD) emd::PairwiseEMD<emd::EMD<emd::ArrayEvent<>, emd::EuclideanArrayDistance<>>>;

// extend Histogram1DHandler
%extend emd::Histogram1DHandler {
  ADD_STR_FROM_DESCRIPTION
  RETURN_1DNUMPY_FROM_VECTOR(npy_bin_centers, bin_centers, $self->nbins())
  RETURN_1DNUMPY_FROM_VECTOR(npy_bin_edges, bin_edges, $self->nbins() + 1)

  void npy_hist_vals_errs(double** arr_out0, int* n0, double** arr_out1, int* n1, bool overflows = true) {
    unsigned int nbins = $self->nbins() + (overflows ? 2 : 0);
    PAIRED_1DNUMPY_FROM_VECPAIR(hist_vals_errs(overflows), nbins)
  }
}

// extend CorrelationDimension
%extend emd::CorrelationDimension {
  ADD_STR_FROM_DESCRIPTION
  RETURN_1DNUMPY_FROM_VECTOR(npy_corrdim_bins, corrdim_bins, $self->nbins() - 1)
  RETURN_PAIRED_1DNUMPY_FROM_VECPAIR(npy_corrdims, corrdims(), $self->nbins() - 1)
  RETURN_PAIRED_1DNUMPY_FROM_VECPAIR(npy_cumulative_vals_vars, cumulative_vals_vars(), $self->nbins())
}

// instantiate explicit Histogram1DHandler classes
%template(Histogram1DHandler) emd::Histogram1DHandler<>;
%template(Histogram1DHandlerLog) emd::Histogram1DHandler<boost::histogram::axis::transform::log>;

// include correlation dimension
%include "CorrelationDimension.hh"
