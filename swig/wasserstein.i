%module(threads=1) wasserstein

// though module is built with threads=1, turn that off for now
%nothreadallow;

%include "std_string.i"
%include "std_vector.i"
%include "exception.i"

// include vector of doubles so we can do testing
%template(vectorDouble) std::vector<double>;
%template(vectorString) std::vector<std::string>;

// this ensures SWIG parses class membes properly
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

// the main library header
#include "EMD.hh"
#include "CorrelationDimension.hh"
%}

// define a subset of std exceptions to be caught
%define SWIG_CATCH_SOME_STDEXCEPT(T)
%exception T {
    try { $action }
    catch (std::out_of_range & e) { SWIG_exception(SWIG_IndexError, e.what()); }
    catch (std::invalid_argument & e) { SWIG_exception(SWIG_ValueError, e.what()); }
    catch (std::exception & e) { SWIG_exception(SWIG_SystemError, e.what()); }
}
%enddef

// catch exceptions in methods that throw them
SWIG_CATCH_SOME_STDEXCEPT(emd::EMD::EMD)
SWIG_CATCH_SOME_STDEXCEPT(emd::PairwiseEMD::PairwiseEMD)
SWIG_CATCH_SOME_STDEXCEPT(emd::EMD::flow)
SWIG_CATCH_SOME_STDEXCEPT(emd::PairwiseEMD::emd)

// check for python errors raised by other methods
%exception emd::EMD::operator() {
  $action
  if (PyErr_Occurred() != NULL)
    SWIG_fail;
}

%exception emd::PairwiseEMD::compute() {
  try { $action }
  catch (std::exception & e) { SWIG_exception(SWIG_SystemError, e.what()); }
}

// some extra python code at the beginning
%pythonbegin %{
import itertools
import numpy as np
%}

%include numpy.i

%init %{
import_array();
%}

%apply (double* IN_ARRAY1, int DIM1) {(double* weights0, int n0), (double* weights1, int n1)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* coords0, int n00, int n01),
                                                (double* coords1, int n10, int n11),
                                                (double* external_dists, int d0, int d1)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* weights, int n0)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* coords, int n1, int d)}
%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(double** arr_out, int* n0, int* n1)}

%threadallow emd::PairwiseEMD::compute();

%ignore check_emd_status;
%ignore emd::EMD::compute;
%ignore emd::EMD::_dists;
%ignore emd::PairwiseEMD::compute(const EventVector & events);
%ignore emd::PairwiseEMD::compute(const EventVector & eventsA, const EventVector & eventsB);
%ignore emd::PairwiseEMD::events;

%include "internal/EventGeometryUtils.hh"

#define SWIG_PREPROCESSOR
%include "internal/HistogramUtils.hh"
%template(Histogram1DHandler) emd::Histogram1DHandler<>;
%template(Histogram1DHandlerLog) emd::Histogram1DHandler<boost::histogram::axis::transform::log>;

%include "CorrelationDimension.hh"
%include "EMD.hh"

%define ADD_STR_FROM_DESCRIPTION
std::string __str__() const {
  return $self->description();
}
%enddef

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

%rename(flows_vec) emd::EMD::flows;
%rename(dists_vec) emd::EMD::dists;
%rename(emds_vec) emd::PairwiseEMD::emds;
%rename(flows) emd::EMD::npy_flows;
%rename(dists) emd::EMD::npy_dists;
%rename(emds) emd::EMD::npy_emds;

%extend emd::EMD {
  ADD_STR_FROM_DESCRIPTION

  void npy_flows(double** arr_out, int* n0, int* n1) {
    MALLOC_2D_VALUE_ARRAY($self->n0(), $self->n1())
    memcpy(*arr_out, $self->network_simplex().flows().data(), nbytes);
    
    double unscale_factor = $self->pairwise_distance().unscale_factor();
    for (size_t i = 0; i < num_elements; i++)
      values[i] *= unscale_factor;
  }

  void npy_dists(double** arr_out, int* n0, int* n1) {
    MALLOC_2D_VALUE_ARRAY($self->n0(), $self->n1())
    memcpy(*arr_out, $self->network_simplex().dists().data(), nbytes);
  }
}

%extend emd::EMD<emd::ArrayEvent<>, emd::EuclideanArrayDistance<>> {
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
    return (*$self)(std::make_tuple(coords0, weights0, n0, n01), std::make_tuple(coords1, weights1, n1, n11));
  }
}

%extend emd::EMD<emd::ArrayEvent<>, emd::CustomArrayDistance<>> {
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

    return (*$self)(std::make_tuple(nullptr, weights0, n0, -1), std::make_tuple(nullptr, weights1, n1, -1));
  }
}

%extend emd::PairwiseEMD {
  ADD_STR_FROM_DESCRIPTION

  // add a single event to the PairwiseEMD object
  void _add_event(double* weights, int n0, double* coords, int n1, int d) {
    $self->events().emplace_back(coords, weights, n1, d);
  }

  void npy_emds(double** arr_out, int* n0, int* n1) {
    MALLOC_2D_VALUE_ARRAY($self->nevA(), $self->nevB())
    memcpy(*arr_out, $self->emds().data(), nbytes);
  }

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

%feature("shadow") emd::PairwiseEMD::clear %{
  def clear(self):
      $action(self)
      self.event_arrs = []
%}

%template(EMDArrayEuclidean) emd::EMD<emd::ArrayEvent<>, emd::EuclideanArrayDistance<>>;
%template(EMDArray) emd::EMD<emd::ArrayEvent<>, emd::CustomArrayDistance<>>;
%template(PairwiseEMDArrayEuclidean) emd::PairwiseEMD<emd::EMD<emd::ArrayEvent<>, emd::EuclideanArrayDistance<>>>;
