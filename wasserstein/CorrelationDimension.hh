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

/*   _____ ____  _____  _____  ______ _            _______ _____ ____  _   _ 
 *  / ____/ __ \|  __ \|  __ \|  ____| |        /\|__   __|_   _/ __ \| \ | |
 * | |   | |  | | |__) | |__) | |__  | |       /  \  | |    | || |  | |  \| |
 * | |   | |  | |  _  /|  _  /|  __| | |      / /\ \ | |    | || |  | | . ` |
 * | |___| |__| | | \ \| | \ \| |____| |____ / ____ \| |   _| || |__| | |\  |
 *  \_____\____/|_|  \_\_|  \_\______|______/_/    \_\_|  |_____\____/|_| \_|
 *  _____ _____ __  __ ______ _   _  _____ _____ ____  _   _ 
 * |  __ \_   _|  \/  |  ____| \ | |/ ____|_   _/ __ \| \ | |
 * | |  | || | | \  / | |__  |  \| | (___   | || |  | |  \| |
 * | |  | || | | |\/| |  __| | . ` |\___ \  | || |  | | . ` |
 * | |__| || |_| |  | | |____| |\  |____) |_| || |__| | |\  |
 * |_____/_____|_|  |_|______|_| \_|_____/|_____\____/|_| \_|
 */

#ifndef WASSERSTEIN_EXTERNALHANDLERS_HH
#define WASSERSTEIN_EXTERNALHANDLERS_HH

#include "internal/HistogramUtils.hh"

BEGIN_EMD_NAMESPACE

class CorrelationDimension : public Histogram1DHandler<boost::histogram::axis::transform::log> {
public:

  CorrelationDimension(unsigned nbins, double axis_min, double axis_max) :
    Histogram1DHandler(nbins, axis_min, axis_max)
  {}

  CorrelationDimension() {}
  ~CorrelationDimension() {}

  // calculates the correlation dimensions
  std::pair<std::vector<double>, std::vector<double>> corrdims(double eps = 1e-100) const {

    const auto cum_vals_vars(cumulative_vals_vars());
    const std::vector<double> & cum_vals(cum_vals_vars.first), & cum_vars(cum_vals_vars.second);

    // containers for remaining computations
    std::vector<double> midbins(bin_centers()), dims(midbins.size() - 1), dim_errs(dims.size());

    // compute dims and dim_errs
    for (std::size_t i = 0; i < dims.size(); i++) {
      double dmidbin(std::log(midbins[i + 1]/midbins[i]));

      dims[i] = std::log(cum_vals[i + 1]/(cum_vals[i] + eps) + eps)/dmidbin;
      dim_errs[i] = std::sqrt(cum_vars[i + 1]/(cum_vals[i + 1]*cum_vals[i + 1] + eps) + 
                              cum_vars[i]    /(cum_vals[i]    *cum_vals[i]     + eps)   )/dmidbin;
    }

    return std::make_pair(dims, dim_errs);
  }

  // using geometric mean here instead of arithmetic mean due to logarithmic axis
  std::vector<double> corrdim_bins() const {
    std::vector<double> midbins(bin_centers());

    for (std::size_t i = 0; i < midbins.size() - 1; i++)
      midbins[i] = std::sqrt(midbins[i] * midbins[i + 1]);
    midbins.resize(midbins.size() - 1);

    return midbins;
  }

  // obtains the cuulative histogram of EFM values and their variances
  std::pair<std::vector<double>, std::vector<double>> cumulative_vals_vars() const {

    std::size_t size(axis_.size());
    std::vector<double> hist_vals(size), hist_vars(size);

    hist_vals[0] = hist_.at(0).value();
    hist_vars[0] = hist_.at(0).variance();

    for (std::size_t i = 1; i < size; i++) {
      const auto & bin(hist_.at(i));
      hist_vals[i] = bin.value() + hist_vals[i - 1];
      hist_vars[i] = bin.variance() + hist_vars[i - 1];
    }

    return std::make_pair(hist_vals, hist_vars);
  }

private:

  std::string name() const { return "CorrelationDimension"; }

}; // CorrelationDimension

END_EMD_NAMESPACE

#endif // WASSERSTEIN_EXTERNALHANDLERS_HH