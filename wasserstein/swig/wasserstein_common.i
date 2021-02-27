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

// C++ standard library wrappers
%include <exception.i>
%include <std_pair.i>
%include <std_string.i>
%include <std_vector.i>

// vector templates
%template(vectorDouble) std::vector<double>;
%template(vectorString) std::vector<std::string>;
%template(pairVectorDouble) std::pair<std::vector<double>, std::vector<double>>;

%{
// include this to avoid needing to define it at compile time 
#ifndef SWIG
#define SWIG
#endif

// needed by numpy.i, harmless otherwise
#define SWIG_FILE_WITH_INIT

// standard library headers we need
#include <cstdlib>
#include <cstring>

// the main library headers
#include "wasserstein/EMD.hh"
#include "wasserstein/CorrelationDimension.hh"
%}

// numpy wrapping and initialization
#ifdef SWIG_NUMPY

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

// prepare to extend classes by renaming some methods
%rename(flows_vec) EMDNAMESPACE::EMD::flows;
%rename(dists_vec) EMDNAMESPACE::EMD::dists;
%rename(flows) EMDNAMESPACE::EMD::npy_flows;
%rename(dists) EMDNAMESPACE::EMD::npy_dists;
%rename(emds_vec) EMDNAMESPACE::PairwiseEMD::emds;
%rename(emds) EMDNAMESPACE::PairwiseEMD::npy_emds;
%rename(bin_centers_vec) EMDNAMESPACE::Histogram1DHandler::bin_centers;
%rename(bin_edges_vec) EMDNAMESPACE::Histogram1DHandler::bin_edges;
%rename(hist_vals_errs_vec) EMDNAMESPACE::Histogram1DHandler::hist_vals_errs;
%rename(bin_centers) EMDNAMESPACE::Histogram1DHandler::npy_bin_centers;
%rename(bin_edges) EMDNAMESPACE::Histogram1DHandler::npy_bin_edges;
%rename(hist_vals_errs) EMDNAMESPACE::Histogram1DHandler::npy_hist_vals_errs;
%rename(cumulative_vals_vars_vec) EMDNAMESPACE::CorrelationDimension::cumulative_vals_vars;
%rename(corrdim_bins_vec) EMDNAMESPACE::CorrelationDimension::corrdim_bins;
%rename(corrdims_vec) EMDNAMESPACE::CorrelationDimension::corrdims;
%rename(cumulative_vals_vars) EMDNAMESPACE::CorrelationDimension::npy_cumulative_vals_vars;
%rename(corrdim_bins) EMDNAMESPACE::CorrelationDimension::npy_corrdim_bins;
%rename(corrdims) EMDNAMESPACE::CorrelationDimension::npy_corrdims;

#endif // SWIG_NUMPY

// basic exception handling for all functions
%exception {
  try { $action }
  SWIG_CATCH_STDEXCEPT
  catch (...) {
    SWIG_exception_fail(SWIG_UnknownError, "unknown exception");
  }
}

namespace EMDNAMESPACE {

// allow threads in PairwiseEMD computation
%threadallow PairwiseEMD::compute;

// ignore certain functions
%ignore EMD::compute;
%ignore EMD::network_simplex;
%ignore EMD::pairwise_distance;
%ignore EMD::ground_dists;
%ignore PairwiseEMD::compute(const EventVector & events);
%ignore PairwiseEMD::compute(const EventVector & eventsA, const EventVector & eventsB);
%ignore PairwiseEMD::events;
%ignore Histogram1DHandler::print_axis;
%ignore Histogram1DHandler::print_hist;

} // namespace EMDNAMESPACE

// include EMD utilities
%include "wasserstein/internal/EMDUtils.hh"

// handle templated base class
%template(EMDBaseDouble) EMDNAMESPACE::EMDBase<double>;

// include main EMD code and histogram code
#define SWIG_PREPROCESSOR
%include "wasserstein/EMD.hh"
%include "wasserstein/internal/HistogramUtils.hh"

// makes python class printable from a description method
%define ADD_STR_FROM_DESCRIPTION
std::string __str__() const {
  return $self->description();
}
std::string __repr__() const {
  return $self->description();
}
%enddef

// mallocs a 1D array of doubles of the specified size
%define MALLOC_1D_VALUE_ARRAY(arr_out, n, size, nbytes)
  *n = size;
  size_t nbytes = size_t(*n)*sizeof(double);
  *arr_out = (double *) malloc(nbytes);
  if (*arr_out == NULL) {
    throw std::runtime_error("Failed to allocate " + std::to_string(nbytes) + " bytes");
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
  if (values == NULL)
    throw std::runtime_error("Failed to allocate " + std::to_string(nbytes) + " bytes");
  *arr_out = values;
%enddef

// add functionality to get flows and dists as numpy arrays
%extend EMDNAMESPACE::EMD {
  ADD_STR_FROM_DESCRIPTION

  #ifdef SWIG_NUMPY
  void npy_flows(double** arr_out, int* n0, int* n1) {
    MALLOC_2D_VALUE_ARRAY($self->n0(), $self->n1())
    memcpy(*arr_out, $self->network_simplex().flows().data(), nbytes);
    
    for (size_t i = 0; i < num_elements; i++)
      values[i] *= $self->scale();
  }
  void npy_dists(double** arr_out, int* n0, int* n1) {
    MALLOC_2D_VALUE_ARRAY($self->n0(), $self->n1())
    memcpy(*arr_out, $self->ground_dists().data(), nbytes);
  }
  #endif // SWIG_NUMPY
}

%extend EMDNAMESPACE::PairwiseEMD {
  ADD_STR_FROM_DESCRIPTION

  #ifdef SWIG_NUMPY
  void npy_emds(double** arr_out, int* n0, int* n1) {
    MALLOC_2D_VALUE_ARRAY($self->nevA(), $self->nevB())
    memcpy(*arr_out, $self->emds().data(), nbytes);
  }
  #endif // SWIG_NUMPY

  // python function to get events from container of 2d arrays, first column becomes the weights
  %pythoncode %{

    # ensure proper destruction of objects held by this instance
    def __del__(self):
        if hasattr(self, 'event_arrs'):
            del self.event_arrs
        if hasattr(self, 'external_emd_handler'):
            self.external_emd_handler.thisown = 1
            del self.external_emd_handler
  %}
}

// ensure that python array of events is deleted also
%feature("shadow") EMDNAMESPACE::PairwiseEMD::set_external_emd_handler %{
  def set_external_emd_handler(self, handler):
      if not handler.thisown:
          raise RuntimeError('ExternalEMDHandler must own itself; perhaps it is already in use elsewhere')
      handler.thisown = 0
      $action(self, handler)
      self.external_emd_handler = handler
%}

// extend Histogram1DHandler
%extend EMDNAMESPACE::Histogram1DHandler {
  ADD_STR_FROM_DESCRIPTION

  #ifdef SWIG_NUMPY
  RETURN_1DNUMPY_FROM_VECTOR(npy_bin_centers, bin_centers, $self->nbins())
  RETURN_1DNUMPY_FROM_VECTOR(npy_bin_edges, bin_edges, $self->nbins() + 1)
  void npy_hist_vals_errs(double** arr_out0, int* n0, double** arr_out1, int* n1, bool overflows = true) {
    unsigned int nbins = $self->nbins() + (overflows ? 2 : 0);
    PAIRED_1DNUMPY_FROM_VECPAIR(hist_vals_errs(overflows), nbins)
  }
  #endif // SWIG_NUMPY
}

// extend CorrelationDimension
%extend EMDNAMESPACE::CorrelationDimension {
  ADD_STR_FROM_DESCRIPTION

  #ifdef SWIG_NUMPY
  RETURN_1DNUMPY_FROM_VECTOR(npy_corrdim_bins, corrdim_bins, $self->nbins() - 1)
  RETURN_PAIRED_1DNUMPY_FROM_VECPAIR(npy_corrdims, corrdims(), $self->nbins() - 1)
  RETURN_PAIRED_1DNUMPY_FROM_VECPAIR(npy_cumulative_vals_vars, cumulative_vals_vars(), $self->nbins())
  #endif // SWIG_NUMPY
}

// instantiate explicit Histogram1DHandler classes
%template(Histogram1DHandler) EMDNAMESPACE::Histogram1DHandler<>;
%template(Histogram1DHandlerLog) EMDNAMESPACE::Histogram1DHandler<boost::histogram::axis::transform::log>;

// include correlation dimension
%include "wasserstein/CorrelationDimension.hh"
