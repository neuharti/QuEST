#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <fstream>

#include "../src/common.h"
#include "../src/configuration.h"
#include "../src/integrator/history.h"
#include "../src/interactions/direct_interaction.h"
#include "../src/interactions/green_function.h"
#include "../src/math_utils.h"
#include "../src/quantum_dot.h"

#define NEW_HISTORY 1

const double c0(1);
const double dt(1);
const int num_timesteps(500);
const int window(22);
const double omega(0.1);

typedef Eigen::Array<cmplx, Eigen::Dynamic, 1> ResultArray;

int main(int argc, char *argv[])
{
  // define source and obs functions
  auto src_fn = [](const double t, const double omega) {
    return std::cos(omega * t);
  };
  auto obs_fn = [](const double t, const double omega, const double delay) {
    return std::cos(omega * (t - delay));
  };

  // create dot vector
  auto dots = std::make_shared<DotVector>();
  dots->emplace_back(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 1));
  dots->emplace_back(Eigen::Vector3d(2.5, 0, 0), Eigen::Vector3d(0, 0, 1));

  double delay = separation((*dots)[0], (*dots)[1]).norm() / c0;

  // initialize identity kernel
  Propagation::Identity<cmplx> identity_kernel;

  // initialize history array
  int interp;
  if(argc > 1)
    interp = std::atoi(argv[1]);
  else
    interp = 4;
  std::cout << "interpolation order: " << interp << std::endl;

  int min_time_to_keep = max_transit_steps_between_dots(dots, c0, dt) + interp;
 
  std::shared_ptr<Integrator::History<Eigen::Vector2cd>> history;
  if( NEW_HISTORY ) {
    history = std::make_shared<Integrator::History<Eigen::Vector2cd>>(
        (*dots).size(), window, num_timesteps, min_time_to_keep, 2);
  } else {
    exit(0);
  }

  std::cout << "timesteps between dots: " 
            << min_time_to_keep - interp 
            << std::endl;

  history->fill(Eigen::Vector2cd(0, 0));
  for(int i = -window; i < 0; i++)
    history->set_value(0, i, 0) = Eigen::Vector2cd(0, src_fn(i * dt, omega));

  // initialize interaction
  DirectInteraction direct_interaction(dots, history, identity_kernel, interp,
                                       c0, dt);

  // main loop
  ResultArray calculated_field;
  double total_diff(0), max_diff(0), analytic_field;
  
  std::ofstream outfile("new_history_static.dat");
  for(int i = 0; i < num_timesteps; i++) {
    history->set_value(0, i, 0) = Eigen::Vector2cd(0, src_fn(i * dt, omega));
    calculated_field = direct_interaction.evaluate(i);
    analytic_field = obs_fn(i * dt, omega, delay); 
   
    outfile << calculated_field[1].real() << "\n";
    total_diff += std::abs(calculated_field[1].real() - analytic_field);
    max_diff = std::max(max_diff, calculated_field[1].real() - analytic_field);
  }

  std::cout << "total error: " << total_diff << std::endl;
  std::cout << "max error: " << max_diff << std::endl;
  outfile.close();
  return 0;
}
