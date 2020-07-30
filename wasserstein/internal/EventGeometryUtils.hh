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

#ifndef EVENTGEOMETRY_EVENTGEOMETRYUTILS_HH
#define EVENTGEOMETRY_EVENTGEOMETRYUTILS_HH

#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

// enum with possible return statuses from the NetworkSimplex solver
enum EMDStatus : char { 
  Success = 0,
  Empty = 1,
  SupplyMismatch = 2,
  Unbounded = 3,
  MaxIterReached = 4,
  Infeasible = 5
};

// enum for which event got an extra particle
enum class ExtraParticle : char { Neither = -1, Zero = 0, One = 1 };

inline void check_emd_status(EMDStatus status) {
  if (status != Success)
    switch (status) {
      case Empty:
        throw std::runtime_error("EMDStatus - Empty");
        break;
      case SupplyMismatch:
        throw std::runtime_error("EMDStatus - SupplyMismatch, consider increasing epsilon_large_factor");
        break;
      case Unbounded:
        throw std::runtime_error("EMDStatus - Unbounded");
        break;
      case MaxIterReached:
        throw std::runtime_error("EMDStatus - MaxIterReached, consider increasing n_iter_max");
        break;
      case Infeasible:
        throw std::runtime_error("EMDStatus - Infeasible");
        break;
      default:
        throw std::runtime_error("EMDStatus - NoSuccess");
    }
}

class ExternalEMDHandler {
public:
  ExternalEMDHandler() {}
  virtual ~ExternalEMDHandler() {}

  void operator()(double emd, double weight = 1) {
    std::lock_guard<std::mutex> handler_guard(mutex_);
    handle(emd, weight);
  }

  virtual std::string description() const = 0;

protected:
  virtual void handle(double, double) = 0; 

private:
  std::mutex mutex_;
};

#ifdef __FASTJET_FASTJET_BASE_HH__
FASTJET_BEGIN_NAMESPACE
namespace contrib {

typedef unsigned int uint;

const double PI = 3.14159265358979323846;
const double TWOPI = 2*PI;

inline double phi_fix(double phi, double ref_phi) {
  double diff(phi - ref_phi);
  if (diff > PI) phi -= TWOPI;
  else if (diff < -PI) phi += TWOPI;
  return phi; 
}

// enum to hold type of quantity to be monitored during grooming
enum GroomingType : char { GroomByArea, GroomByAreaTrackEMD, GroomByEMD };

} // namespace contrib
FASTJET_END_NAMESPACE
#endif // __FASTJET_FASTJET_BASE_HH__

#endif // EVENTGEOMETRY_EVENTGEOMETRYUTILS_HH
