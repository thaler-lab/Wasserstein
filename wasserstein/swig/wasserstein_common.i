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
%template(vectorFloat) std::vector<float>;
%template(vectorString) std::vector<std::string>;
%template(pairVectorDouble) std::pair<std::vector<double>, std::vector<double>>;
%template(pairVectorFloat) std::pair<std::vector<float>, std::vector<float>>;

// this helps to exclude some problematic code from swig
#define SWIG_PREPROCESSOR

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
#include "wasserstein/Wasserstein.hh"

using EMDNAMESPACE::DefaultNetworkSimplex;
%}

// numpy wrapping and initialization
#ifdef SWIG_NUMPY

  %include numpy.i
  %init %{
  import_array();
  %}

  // Python imports at the top of the module
  %pythonbegin %{
  import numpy as np
  %}

  %numpy_typemaps(double, NPY_DOUBLE, std::ptrdiff_t)
  %numpy_typemaps(float,  NPY_FLOAT,  std::ptrdiff_t)

  // numpy typemaps
  %define WASSERSTEIN_NUMPY_TYPEMAPS(F)
    %apply (F* IN_ARRAY1, std::ptrdiff_t DIM1) {(F* weights0, std::ptrdiff_t n0),
                                                (F* weights1, std::ptrdiff_t n1),
                                                (F* emds, std::ptrdiff_t n0),
                                                (F* event_weights, std::ptrdiff_t n1)}
    %apply (F* IN_ARRAY2, std::ptrdiff_t DIM1, std::ptrdiff_t DIM2) {(F* coords0, std::ptrdiff_t n00, std::ptrdiff_t n01),
                                                                     (F* coords1, std::ptrdiff_t n10, std::ptrdiff_t n11),
                                                                     (F* external_dists, std::ptrdiff_t d0, std::ptrdiff_t d1)}

    %apply (F* INPLACE_ARRAY1, std::ptrdiff_t DIM1) {(F* weights, std::ptrdiff_t n)}
    %apply (F* INPLACE_ARRAY2, std::ptrdiff_t DIM1, std::ptrdiff_t DIM2) {(F* coords, std::ptrdiff_t n1, std::ptrdiff_t d)}

    %apply (F** ARGOUTVIEWM_ARRAY1, std::ptrdiff_t* DIM1) {(F** arr_out0, std::ptrdiff_t* n0), (F** arr_out1, std::ptrdiff_t* n1)}
    %apply (F** ARGOUTVIEWM_ARRAY2, std::ptrdiff_t* DIM1, std::ptrdiff_t* DIM2) {(F** arr_out, std::ptrdiff_t* n0, std::ptrdiff_t* n1)}
  %enddef

  WASSERSTEIN_NUMPY_TYPEMAPS(double)
  WASSERSTEIN_NUMPY_TYPEMAPS(float)

  // prepare to extend classes by renaming some methods
  namespace EMDNAMESPACE {
    %rename(flows_vec) _EMD::flows;
    %rename(dists_vec) _EMD::dists;
    %rename(flows) _EMD::npy_flows;
    %rename(dists) _EMD::npy_dists;
    %rename(emds_vec) PairwiseEMD::emds;
    %rename(emds) PairwiseEMD::npy_emds;
    %rename(bin_centers_vec) Histogram1DHandler::bin_centers;
    %rename(bin_edges_vec) Histogram1DHandler::bin_edges;
    %rename(hist_vals_vars_vec) Histogram1DHandler::hist_vals_vars;
    %rename(bin_centers) Histogram1DHandler::npy_bin_centers;
    %rename(bin_edges) Histogram1DHandler::npy_bin_edges;
    %rename(hist_vals_vars) Histogram1DHandler::npy_hist_vals_vars;
    %rename(cumulative_vals_vars_vec) CorrelationDimension::cumulative_vals_vars;
    %rename(corrdim_bins_vec) CorrelationDimension::corrdim_bins;
    %rename(corrdims_vec) CorrelationDimension::corrdims;
    %rename(cumulative_vals_vars) CorrelationDimension::npy_cumulative_vals_vars;
    %rename(corrdim_bins) CorrelationDimension::npy_corrdim_bins;
    %rename(corrdims) CorrelationDimension::npy_corrdims;
  }

#endif // SWIG_NUMPY

// makes python class printable from a description method
%define ADD_STR_FROM_DESCRIPTION(...)
  std::string __str__() const {
    return $self->description(__VA_ARGS__);
  }
  std::string __repr__() const {
    return $self->description(__VA_ARGS__);
  }
%enddef

// mallocs a 1D array of doubles of the specified size
%define MALLOC_1D_VALUE_ARRAY(arr_out, n, size, nbytes, F)
  *n = size;
  size_t nbytes = size_t(*n)*sizeof(F);
  *arr_out = (F *) malloc(nbytes);
  if (*arr_out == NULL) {
    throw std::runtime_error("Failed to allocate " + std::to_string(nbytes) + " bytes");
    return;
  }
%enddef

%define RETURN_1DNUMPY_FROM_VECTOR(pyname, cppname, size, F)
  void pyname(F** arr_out0, std::ptrdiff_t* n0) {
    MALLOC_1D_VALUE_ARRAY(arr_out0, n0, size, nbytes, F)
    memcpy(*arr_out0, $self->cppname().data(), nbytes);
  }
%enddef

%define PAIRED_1DNUMPY_FROM_VECPAIR(cppfunccall, size, F)
  MALLOC_1D_VALUE_ARRAY(arr_out0, n0, size, nbytes0, F)
  MALLOC_1D_VALUE_ARRAY(arr_out1, n1, size, nbytes1, F)
  std::pair<std::vector<F>, std::vector<F>> vecpair($self->cppfunccall);
  memcpy(*arr_out0, vecpair.first.data(), nbytes0);
  memcpy(*arr_out1, vecpair.second.data(), nbytes1);
%enddef

%define RETURN_PAIRED_1DNUMPY_FROM_VECPAIR(pyname, cppfunccall, size, F)
  void pyname(F** arr_out0, std::ptrdiff_t* n0, F** arr_out1, std::ptrdiff_t* n1) {
    PAIRED_1DNUMPY_FROM_VECPAIR(cppfunccall, size, F)
  }
%enddef

// mallocs a 2D array of doubles of the specified size
%define MALLOC_2D_VALUE_ARRAY(a, b, F)
  *n0 = a;
  *n1 = b;
  size_t num_elements = size_t(*n0)*size_t(*n1);
  size_t nbytes = num_elements*sizeof(F);
  F * values = (F *) malloc(nbytes);
  if (values == NULL)
    throw std::runtime_error("Failed to allocate " + std::to_string(nbytes) + " bytes");
  *arr_out = values;
%enddef

// add functionality to get flows and dists as numpy arrays
%define EMD_NUMPY_FUNCS(F)
  void npy_flows(F** arr_out, std::ptrdiff_t* n0, std::ptrdiff_t* n1) {
    MALLOC_2D_VALUE_ARRAY($self->n0(), $self->n1(), F)
    memcpy(*arr_out, $self->network_simplex().flows().data(), nbytes);
    
    for (size_t i = 0; i < num_elements; i++)
      values[i] *= $self->scale();
  }
  void npy_dists(F** arr_out, std::ptrdiff_t* n0, std::ptrdiff_t* n1) {
    MALLOC_2D_VALUE_ARRAY($self->n0(), $self->n1(), F)
    memcpy(*arr_out, $self->ground_dists().data(), nbytes);
  }
%enddef

%define PAIRWISE_EMD_NUMPY_FUNCS(F)
  void npy_emds(F** arr_out, std::ptrdiff_t* n0, std::ptrdiff_t* n1) {
    MALLOC_2D_VALUE_ARRAY($self->nevA(), $self->nevB(), F)
    memcpy(*arr_out, $self->emds().data(), nbytes);
  }
  void flat_emds(F** arr_out0, std::ptrdiff_t* n0) {
    if ($self->storage() != EMDNAMESPACE::EMDPairsStorage::FlattenedSymmetric)
      throw std::runtime_error("flattened emds only available with flattened symmetric storage");

    MALLOC_1D_VALUE_ARRAY(arr_out0, n0, $self->num_emds(), nbytes, F)
    memcpy(*arr_out0, $self->emds(true).data(), nbytes);
  }
%enddef

%define EXTERNAL_EMD_HANDLER_NUMPY_FUNCS(F)
  %#ifdef SWIG_NUMPY
    void compute(F* emds, std::ptrdiff_t n0) {
      $self->compute(emds, n0);
    }
    void compute(F* emds, std::ptrdiff_t n0, F* event_weights, std::ptrdiff_t n1) {
      if (n0 != n1)
        throw std::invalid_argument("length of `emds` should match lengh of `event_weights`");
      $self->compute(emds, n0, event_weights);
    }
    void compute_symmetric(F* emds, std::ptrdiff_t n0) {
      $self->compute(emds, n0);
    }
    void compute_symmetric(F* emds, std::ptrdiff_t n0, F* event_weights, std::ptrdiff_t n1) {
      if (n0 != n1*(n1 - 1)/2)
        throw std::invalid_argument("length of `emds` should be lengh of `event_weights` choose 2");
      $self->compute_symmetric(emds, n1, event_weights);
    }
  %#endif // SWIG_NUMPY
%enddef

%define HISTOGRAM_1D_HANDLER_NUMPY_FUNCS(F)
  %#ifdef SWIG_NUMPY
    RETURN_1DNUMPY_FROM_VECTOR(npy_bin_centers, bin_centers, $self->nbins(), F)
    RETURN_1DNUMPY_FROM_VECTOR(npy_bin_edges, bin_edges, $self->nbins() + 1, F)
    void npy_hist_vals_vars(F** arr_out0, std::ptrdiff_t* n0, F** arr_out1, std::ptrdiff_t* n1, bool overflows = true) {
      unsigned nbins = $self->nbins() + (overflows ? 2 : 0);
      PAIRED_1DNUMPY_FROM_VECPAIR(hist_vals_vars(overflows), nbins, F)
    }

    %pythoncode %{
      def hist_vals_errs(self, overflows=True):
          vals, vars = self.hist_vals_vars(overflows)
          return vals, np.sqrt(vars)
    %}
  %#endif // SWIG_NUMPY
%enddef

%define CORRELATION_DIMENSION_NUMPY_FUNCS(F)
  %#ifdef SWIG_NUMPY
    RETURN_1DNUMPY_FROM_VECTOR(npy_corrdim_bins, corrdim_bins, $self->nbins() - 1, F)
    RETURN_PAIRED_1DNUMPY_FROM_VECPAIR(npy_corrdims, corrdims(), $self->nbins() - 1, F)
    RETURN_PAIRED_1DNUMPY_FROM_VECPAIR(npy_cumulative_vals_vars, cumulative_vals_vars(), $self->nbins(), F)
  %#endif // SWIG_NUMPY
%enddef

%define ADD_EXPLICIT_PREPROCESSORS
void preprocess_CenterWeightedCentroid() { $self->preprocess<EMDNAMESPACE::CenterWeightedCentroid>(); }
%enddef

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
  %ignore ExternalEMDHandler::compute;
  %ignore ExternalEMDHandler::compute_symmetric;
  %ignore Histogram1DHandler::print_axis;
%ignore Histogram1DHandler::print_hist;

} // namespace EMDNAMESPACE

// include EMD utilities
%include "wasserstein/internal/EMDUtils.hh"
%include "wasserstein/internal/EMDBase.hh"
%include "wasserstein/internal/ExternalEMDHandler.hh"

namespace EMDNAMESPACE {

  // handle templated base class
  %template(EMDBaseFloat64) EMDBase<double>;
  %template(EMDBaseFloat32) EMDBase<float>;
  %template(ExternalEMDHandlerFloat64) ExternalEMDHandler<double>;
  %template(ExternalEMDHandlerFloat32) ExternalEMDHandler<float>;

} // namespace EMDNAMESPACE

// include main EMD code and histogram code
%include "wasserstein/internal/CenterWeightedCentroid.hh"
%include "wasserstein/internal/EMD.hh"
%include "wasserstein/internal/HistogramUtils.hh"
%include "wasserstein/internal/PairwiseEMD.hh"

namespace EMDNAMESPACE {
  %extend _EMD {
    ADD_STR_FROM_DESCRIPTION(false)
    ADD_EXPLICIT_PREPROCESSORS
  }

  %extend PairwiseEMD {
    ADD_STR_FROM_DESCRIPTION(false)
    ADD_EXPLICIT_PREPROCESSORS

    // ensure proper destruction of objects held by this instance
    %pythoncode %{
      def __del__(self):
          if hasattr(self, 'event_arrs'):
              del self.event_arrs
          if hasattr(self, '_external_emd_handler'):
              self._external_emd_handler.thisown = 1
              del self._external_emd_handler
    %}
  }

  // ensure that external handler ownership is handled correctly
  %feature("shadow") PairwiseEMD::set_external_emd_handler %{
    def set_external_emd_handler(self, handler):
        if not handler.thisown:
            raise RuntimeError('ExternalEMDHandler must own itself; perhaps it is already in use elsewhere')
        handler.thisown = 0
        $action(self, handler)
        self._external_emd_handler = handler
  %}

  %extend Histogram1DHandler { ADD_STR_FROM_DESCRIPTION() }
  %extend CorrelationDimension { ADD_STR_FROM_DESCRIPTION() }

  // instantiate explicit ExternalEMDHandler and Histogram1DHandler classes
  %extend ExternalEMDHandler<double> { EXTERNAL_EMD_HANDLER_NUMPY_FUNCS(double) }
  %extend ExternalEMDHandler<float> { EXTERNAL_EMD_HANDLER_NUMPY_FUNCS(float) }

  %extend Histogram1DHandler<boost::histogram::axis::transform::log, double> {
    HISTOGRAM_1D_HANDLER_NUMPY_FUNCS(double)
  }
  %extend Histogram1DHandler<boost::histogram::axis::transform::log, float> {
    HISTOGRAM_1D_HANDLER_NUMPY_FUNCS(float)
  }

  %template(Histogram1DHandlerLogFloat64) Histogram1DHandler<boost::histogram::axis::transform::log, double>;
  %template(Histogram1DHandlerLogFloat32) Histogram1DHandler<boost::histogram::axis::transform::log, float>;
}

// include correlation dimension
%include "wasserstein/internal/CorrelationDimension.hh"

namespace EMDNAMESPACE {
  %extend CorrelationDimension<double> { CORRELATION_DIMENSION_NUMPY_FUNCS(double) }
  %extend CorrelationDimension<float> { CORRELATION_DIMENSION_NUMPY_FUNCS(float) }

  %template(CorrelationDimensionFloat64) CorrelationDimension<double>;
  %template(CorrelationDimensionFloat32) CorrelationDimension<float>;
}

%define DECLARE_PYTHON_FUNC_VARIABLE_FLOAT_TYPE(func)
%pythoncode %{
def func(*args, **kwargs):
    dtype = kwargs.pop('dtype', 'float64')
    if dtype == 'float64' or dtype == np.float64:
        return func##Float64(*args, **kwargs)
    elif dtype == 'float32' or dtype == np.float32:
        return func##Float32(*args, **kwargs)
    else:
      raise TypeError('`dtype` {} not supported'.format(dtype))
%}
%enddef

DECLARE_PYTHON_FUNC_VARIABLE_FLOAT_TYPE(CorrelationDimension)
