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

/*  _    _ _____  _____ _______ ____   _____ _____            __  __ 
 * | |  | |_   _|/ ____|__   __/ __ \ / ____|  __ \     /\   |  \/  |
 * | |__| | | | | (___    | | | |  | | |  __| |__) |   /  \  | \  / |
 * |  __  | | |  \___ \   | | | |  | | | |_ |  _  /   / /\ \ | |\/| |
 * | |  | |_| |_ ____) |  | | | |__| | |__| | | \ \  / ____ \| |  | |
 * |_|  |_|_____|_____/   |_|  \____/ \_____|_|  \_\/_/    \_\_|  |_|
 *  _    _ _______ _____ _       _____ 
 * | |  | |__   __|_   _| |     / ____|
 * | |  | |  | |    | | | |    | (___  
 * | |  | |  | |    | | | |     \___ \ 
 * | |__| |  | |   _| |_| |____ ____) |
 *  \____/   |_|  |_____|______|_____/ 
 */

#ifndef WASSERSTEIN_HISTOGRAMUTILS_HH
#define WASSERSTEIN_HISTOGRAMUTILS_HH

// C++ standard library
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Boost histogram headers
#include <boost/histogram.hpp>

#include "ExternalEMDHandler.hh"


BEGIN_WASSERSTEIN_NAMESPACE

namespace hist {

// gets bin centers from an axis
template<typename V, class Axis>
std::vector<V> get_bin_centers(const Axis & axis) {

  std::vector<V> bin_centers_vec(axis.size());
  for (int i = 0; i < axis.size(); i++)
    bin_centers_vec[i] = axis.bin(i).center();

  return bin_centers_vec;
}

// gets bin edges from an axis
template<typename V, class Axis>
std::vector<V> get_bin_edges(const Axis & axis) {

  if (axis.size() == 0) return std::vector<V>();

  std::vector<V> bins_vec(axis.size() + 1);
  bins_vec[0] = axis.bin(0).lower();
  for (int i = 0; i < axis.size(); i++)
    bins_vec[i+1] = axis.bin(i).upper();

  return bins_vec;
}

template<class Axis>
int get_1d_hist_size(const Axis & axis, bool overflows) {
  return axis.size() + (overflows ? 2 : 0);
}

template<typename V, class Hist>
std::pair<std::vector<V>, std::vector<V>>
get_1d_hist(const Hist & hist, bool overflows) {

  // setup containers to hold histogram values
  int size(get_1d_hist_size(hist.template axis<0>(), overflows)),
      nbins(hist.template axis<0>().size());
  std::vector<V> hist_vals(size), hist_vars(size);

  for (int i = (overflows ? -1 : 0), a = 0;
       i < nbins + (overflows ? 1 : 0); i++, a++) {
    hist_vals[a] = hist.at(i).value();
    hist_vars[a] = hist.at(i).variance();
  }

  return std::make_pair(hist_vals, hist_vars);
}

template<class Transform>
std::string name_transform() {
  if (std::is_same<Transform, boost::histogram::axis::transform::log>::value)
    return "log";
  else if (std::is_same<Transform, boost::histogram::axis::transform::id>::value)
    return "id";
  else if (std::is_same<Transform, boost::histogram::axis::transform::sqrt>::value)
    return "sqrt";
  else if (std::is_same<Transform, boost::histogram::axis::transform::pow>::value)
    return "pow";
  else
    return "unknown";
}

template<class Axis>
std::string print_axis(const Axis & axis) {

  std::ostringstream oss;
  oss << std::setprecision(16);

  for (int i = 0; i < axis.size(); i++) {
    const auto & bin(axis.bin(i));
    oss << i << " : " << bin.lower() << ' ' << bin.center() << ' ' << bin.upper() << '\n';
  }
  oss << '\n';

  return oss.str();
}

template<class Hist>
std::string print_1d_hist(const Hist & hist) {

  std::ostringstream oss;
  oss << std::setprecision(16);

  for (const auto && x : boost::histogram::indexed(hist, boost::histogram::coverage::all))
    oss << x.index(0) << " : " << x->value() << ' ' << std::sqrt(x->variance()) << '\n';
  oss << '\n';

  return oss.str();
}

} // namespace hist (anonymous)

// use boost::histogram::axis::transform::id if no axis transform desired
template<class _Transform, typename Value>
class Histogram1DHandler : public ExternalEMDHandler<Value> {
public:

  typedef _Transform Transform;
  typedef Value value_type;
  typedef boost::histogram::axis::regular<Value, Transform> Axis;

#ifndef SWIG_PREPROCESSOR
  protected:
    Axis axis_;

  public:
    typedef decltype(boost::histogram::make_weighted_histogram(axis_)) Hist;

    // default constructor, useful for deserialization
    Histogram1DHandler() = default;

  protected:
    Hist hist_;
#endif // SWIG_PREPROCESSOR

public:

  // constructor that will set up the histogram
  Histogram1DHandler(unsigned nbins, Value axis_min, Value axis_max) {

    if (nbins == 0)
      throw std::invalid_argument("Number of histogram bins should be a positive integer");
    if (axis_min >= axis_max)
      throw std::invalid_argument("axis_min should be less than axis_max");

    axis() = Axis(nbins, axis_min, axis_max);
    hist() = boost::histogram::make_weighted_histogram(axis());
  }

  virtual ~Histogram1DHandler() = default;

  // access the constructor arguments
  unsigned nbins() const { return axis().size(); }
  Value axis_min() const { return axis().value(0); }
  Value axis_max() const { return axis().value(axis().size()); }

  std::string description() const {
    std::ostringstream oss;
    oss << std::setprecision(8)
        << "  ExternalEMDHandler - " << name() << '\n'
        << "    bins - " << nbins() << '\n'
        << "    range - [" << axis_min() << ", " << axis_max() << ")\n"
        << "    axis_transform - " << hist::name_transform<Transform>() << '\n';

    return oss.str();
  }

  // SWIG preprocessor complains about these, so hide them
  #ifndef SWIG_PREPROCESSOR
    auto & hist() { return hist_; }
    auto & axis() { return axis_; }
    const auto & hist() const { return hist_; }
    const auto & axis() const { return axis_; }
  #endif

  // get histogram values and errors
  std::pair<std::vector<Value>, std::vector<Value>> hist_vals_vars(bool overflows = true) const {
    return hist::get_1d_hist<Value>(hist_, overflows);
  }

  // get bins
  std::vector<Value> bin_centers() const { return hist::get_bin_centers<Value>(axis()); }
  std::vector<Value> bin_edges() const { return hist::get_bin_edges<Value>(axis()); }

  // return textual representation of axis/hist
  std::string print_axis() const { return hist::print_axis(axis()); }
  std::string print_hist() const { return hist::print_1d_hist(hist()); }

  Histogram1DHandler<Transform, Value> &
  operator+=(const Histogram1DHandler<Transform, Value> & other) {
    if (other.axis() != axis())
      throw std::invalid_argument("`other` does not have the same axis and so cannot be added");
    
    hist() += other.hist();
    return *this;
  }

protected:

  void handle(Value emd, Value weight) {
    hist()(boost::histogram::weight(weight), emd);
  }

  virtual std::string name() const { return "Histogram1DHandler"; }

private:

#ifdef WASSERSTEIN_SERIALIZATION
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & boost::serialization::base_object<ExternalEMDHandler<Value>>(*this)
       & axis() & hist();
  }
#endif

}; // Histogram1DHandler

END_WASSERSTEIN_NAMESPACE

#endif // WASSERSTEIN_HISTOGRAMUTILS_HH
