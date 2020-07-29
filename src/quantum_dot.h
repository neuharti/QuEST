#ifndef QUANTUM_DOT_H
#define QUANTUM_DOT_H

#include <Eigen/Dense>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common.h"

class QuantumDot;

typedef std::vector<QuantumDot> DotVector;
typedef std::pair<DotVector::iterator, DotVector::iterator> DotRange;
typedef std::pair<DotVector::const_iterator, DotVector::const_iterator>
    const_DotRange;
typedef Eigen::Vector2cd matrix_elements;
typedef std::function<Eigen::Vector2cd(const Eigen::Vector2cd,
                                       const std::complex<double>)>
    BlochFunctionType;
enum MatrixElement { RHO_00, RHO_01 };
class QuantumDot {
 public:
  QuantumDot() = default;
  QuantumDot(const Eigen::Vector3d &pos)
      : QuantumDot(pos, 0, {0.0, 0.0}, Eigen::Vector3d::Zero()){};
  QuantumDot(const Eigen::Vector3d &pos, const Eigen::Vector3d &dip)
      : QuantumDot(pos, 0, {0.0, 0.0}, dip){};
  QuantumDot(const Eigen::Vector3d &,
             const double,
             const std::pair<double, double> &,
             const Eigen::Vector3d &);

  matrix_elements liouville_rhs(const matrix_elements &,
                                const cmplx,
                                const double,
                                const bool) const;

  const Eigen::Vector3d &position() const { return pos; }
  const Eigen::Vector3d &dipole() const { return dip; }
  friend Eigen::Vector3d separation(const QuantumDot &, const QuantumDot &);
  friend int max_transit_steps_between_dots(const std::shared_ptr<DotVector>,
                                            const double,
                                            const double);
  friend inline double dyadic_product(const QuantumDot &obs,
                                      const Eigen::Matrix3d &dyad,
                                      const QuantumDot &src)
  {
    return obs.dip.transpose() * dyad * src.dip;
  }

  friend std::ostream &operator<<(std::ostream &, const QuantumDot &);
  friend std::istream &operator>>(std::istream &, QuantumDot &);

 private:
  Eigen::Vector3d pos;
  double freq;
  std::pair<double, double> damping;
  Eigen::Vector3d dip;
};

DotVector import_dots(const std::string &);
std::vector<BlochFunctionType> rhs_functions(const DotVector &,
                                             const double,
                                             bool);

#endif
