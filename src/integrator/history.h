#ifndef HISTORY_H
#define HISTORY_H

#include <Eigen/Dense>
#include <boost/multi_array.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include "../quantum_dot.h"

namespace Integrator {
  template <class soltype>
  class History;

  template <class soltype>
  using soltype_array = boost::multi_array<soltype, 3>;

  inline namespace history_enums {
    enum DIMENSION { PARTICLES, TIMES, DERIVATIVES };
    enum ORDER { DERIV_0, DERIV_1 };
  }  // namespace history_enums
}  // namespace Integrator

template <class soltype>
class Integrator::History {
 public:
  History(const int, const int, const int, const int = 2);
  History(const int,
          const int,
          const int,
          const double,
          const double,
          std::shared_ptr<DotVector>,
          const int = 2);
  ~History();
  soltype_array<soltype> array_;  // TODO: make private

  void fill(const soltype &);
  void initialize_past(const soltype &);
  soltype &set_value(const int, const int, const int);
  soltype get_value(const int, const int, const int) const;

  void write_step_to_file(const int);

  int num_particles;
  int num_timesteps;

 private:
  std::ofstream outfile;
  template <class B = soltype>
  typename std::enable_if<std::is_same<B, Eigen::Vector2cd>::value,
                          Eigen::RowVector2cd>::type
      prep_for_output(Eigen::Vector2cd);

  template <class B = soltype>
  typename std::enable_if<!std::is_same<B, Eigen::Vector2cd>::value,
                          soltype>::type prep_for_output(soltype);
};

template <class soltype>
Integrator::History<soltype>::History(const int num_particles,
                                      const int window,
                                      const int num_timesteps,
                                      const int num_derivatives)
    : num_particles(num_particles),
      num_timesteps(num_timesteps),
      array_(boost::extents[num_particles][
          typename soltype_array<soltype>::extent_range(-window, num_timesteps)]
                           [num_derivatives])
{
  // outfile = std::make_unique<std::ofstream>("output.dat");
  outfile.open("output.dat");
  outfile << std::scientific << std::setprecision(15);
}

template <class soltype>
Integrator::History<soltype>::History(const int num_particles,
                                      const int window,
                                      const int num_timesteps,
                                      const double dt,
                                      const double c0,
                                      std::shared_ptr<DotVector> dots,
                                      const int num_derivatives)
    : num_particles(num_particles),
      num_timesteps(num_timesteps),
      array_(boost::extents[num_particles][
          typename soltype_array<soltype>::extent_range(-window, num_timesteps)]
                           [num_derivatives])
{
  // int timesteps_to_keep = longest_time_between_dots(...)
  array_.resize(boost::extents[num_particles][
      typename soltype_array<soltype>::extent_range(-window, num_timesteps)]
                              [num_derivatives]);
  outfile.open("output.dat");
  outfile << std::scientific << std::setprecision(15);
}

template <class soltype>
Integrator::History<soltype>::~History()
{
  outfile.close();
}

template <class soltype>
void Integrator::History<soltype>::fill(const soltype &val)
{
  std::fill(array_.data(), array_.data() + array_.num_elements(), val);
}

template <class soltype>
void Integrator::History<soltype>::initialize_past(const soltype &val)
{
  for(int n = 0; n < static_cast<int>(array_.shape()[PARTICLES]); ++n) {
    for(int t = array_.index_bases()[TIMES]; t <= 0; ++t) {
      array_[n][t][DERIV_0] = val;
    }
  }
}

template <class soltype>
soltype Integrator::History<soltype>::get_value(const int particle_idx,
                                                const int time_idx,
                                                const int derivative_idx) const
{
  int modular_time_idx = time_idx % static_cast<int>(array_.shape()[TIMES]);
  return array_[particle_idx][modular_time_idx][derivative_idx];
}

template <class soltype>
soltype &Integrator::History<soltype>::set_value(const int particle_idx,
                                                 const int time_idx,
                                                 const int derivative_idx)
{
  int modular_time_idx = time_idx % static_cast<int>(array_.shape()[TIMES]);
  return array_[particle_idx][time_idx][derivative_idx];
}

template <class soltype>
template <class B>
typename std::enable_if<std::is_same<B, Eigen::Vector2cd>::value,
                        Eigen::RowVector2cd>::type
Integrator::History<soltype>::prep_for_output(Eigen::Vector2cd element)
{
  return element.transpose();
}

template <class soltype>
template <class B>
typename std::enable_if<!std::is_same<B, Eigen::Vector2cd>::value,
                        soltype>::type
Integrator::History<soltype>::prep_for_output(soltype element)
{
  return element;
}

template <class soltype>
void Integrator::History<soltype>::write_step_to_file(const int timestep)
{
  for(int n = 0; n < num_particles; ++n)
    outfile << prep_for_output(get_value(n, timestep, 0)) << " ";

  outfile << "\n";
  if(timestep == num_timesteps - 1)
    outfile.flush();  // QUESTION: why is this needed?
}

#endif
