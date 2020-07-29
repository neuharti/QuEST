#include <Eigen/Dense>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

#include "../src/configuration.h"
#include "../src/integrator/history.h"
#include "../src/interactions/AIM/aim_interaction.h"
#include "../src/interactions/AIM/expansion.h"
#include "../src/interactions/AIM/grid.h"
#include "../src/interactions/AIM/normalization.h"
#include "../src/interactions/direct_interaction.h"
#include "../src/interactions/green_function.h"
#include "../src/math_utils.h"
#include "../src/quantum_dot.cpp"
#include "../src/quantum_dot.h"

// namespace po = boost::program_options;

const double c0 = 299.792458, mu0 = 2.0133545e-04, hbar = 0.65821193;
const double omega = 2278.9013;
const double prop_constant =  // 1.00 / hbar;
    mu0 / (4 * M_PI * hbar);

Eigen::Vector2cd source(double t, double mu, double sigsqr)
{
  return Eigen::Vector2cd(0, exp(-std::pow(t - mu, 2) / (2.0 * sigsqr)));
}

Eigen::Vector3d efld_d0_source(double t, double mu, double sigsqr, double delay)
{
  return Eigen::Vector3d(exp(-std::pow(t - mu - delay, 2) / (2.0 * sigsqr)), 0,
                         0);
}

Eigen::Vector3d efld_d1_source(double t, double mu, double sigsqr, double delay)
{
  return Eigen::Vector3d(-(t - mu - delay) / sigsqr *
                             exp(-std::pow(t - mu - delay, 2) / (2.0 * sigsqr)),
                         0, 0);
}

Eigen::Vector3d efld_d2_source(double t, double mu, double sigsqr, double delay)
{
  return Eigen::Vector3d((std::pow(t - mu - delay, 2) - sigsqr) /
                             pow(sigsqr, 2) *
                             exp(-std::pow(t - mu - delay, 2) / (2.0 * sigsqr)),
                         0, 0);
}

Eigen::Vector3d analytic_EFIE_interaction(Eigen::Vector3d &efld_d0,
                                          Eigen::Vector3d &efld_d1,
                                          Eigen::Vector3d &efld_d2,
                                          Eigen::Vector3d &dr,
                                          double c0,
                                          double dist)
{
  Eigen::Matrix3d rr = dr * dr.transpose() / dr.squaredNorm();
  Eigen::Matrix3d irr = Eigen::Matrix3d::Identity() - rr;
  Eigen::Matrix3d i3rr = Eigen::Matrix3d::Identity() - 3 * rr;

  return -pow(c0, 2) * prop_constant * hbar *
         (i3rr * efld_d0 / std::pow(dist, 3) +
          i3rr * efld_d1 / (c0 * std::pow(dist, 2)) +
          irr * efld_d2 / (std::pow(c0, 2) * dist));
}

Eigen::Vector3cd analytic_rotatingEFIE_interaction(Eigen::Vector3d &efld_d0,
                                                   Eigen::Vector3d &efld_d1,
                                                   Eigen::Vector3d &efld_d2,
                                                   Eigen::Vector3d &dr,
                                                   double c0,
                                                   double dist)
{
  Eigen::Matrix3d rr = dr * dr.transpose() / dr.squaredNorm();
  Eigen::Matrix3d irr = Eigen::Matrix3d::Identity() - rr;
  Eigen::Matrix3d i3rr = Eigen::Matrix3d::Identity() - 3 * rr;

  return -pow(c0, 2) * prop_constant * hbar *
         (i3rr.cast<cmplx>() * efld_d0 / std::pow(dist, 3) +
          i3rr.cast<cmplx>() * (efld_d1 + iu * omega * efld_d0) /
              (c0 * std::pow(dist, 2)) +
          irr.cast<cmplx>() *
              (efld_d2 + 2.0 * iu * efld_d1 - std::pow(omega, 2) * efld_d0) /
              (std::pow(c0, 2) * dist)) *
         std::exp(-iu * omega * dist / c0);
}
/*
Eigen::Vector3cd analytic_Laplace_interaction(Eigen::Vector3cd &efld_d0, double
dist){ return prop_constant * hbar * Eigen::Matrix3d::Identity() * efld_d0 /
dist;
}

Eigen::Vector3cd analytic_Helmholtz_interaction(Eigen::Vector3cd &efld_d0,
double c0, double dist){ const std::complex<double> iu(0, 1); return
prop_constant * hbar * Eigen::Matrix3cd::Identity() * efld_d0 * std::exp( -iu *
omega / c0 * dist) / dist;
}*/

std::vector<std::complex<double>> analytic_evaluate(
    std::shared_ptr<DotVector> dots, int i, double dt, double mu, double sigsqr)
{
  int ndots = (*dots).size();
  std::vector<std::complex<double>> fld_anlytc(ndots);
  double dist, delay;

  for(int itrg = 0; itrg < ndots; itrg++) {
    for(int isrc = 0; isrc < ndots; ++isrc) {
      if(itrg != isrc) {
        Eigen::Vector3d dr(separation((*dots)[itrg], (*dots)[isrc]));
        dist = dr.norm();
        delay = dist / c0;

        Eigen::Vector3d efld_d0 = efld_d0_source(i * dt, mu, sigsqr, delay);
        Eigen::Vector3d efld_d1 = efld_d1_source(i * dt, mu, sigsqr, delay);
        Eigen::Vector3d efld_d2 = efld_d2_source(i * dt, mu, sigsqr, delay);

        fld_anlytc[itrg] +=
            analytic_EFIE_interaction(efld_d0, efld_d1, efld_d2, dr, c0, dist)
                .dot((*dots)[itrg].dipole()) /
            hbar;
      }
    }
  }
  return fld_anlytc;
}

int main(int argc, char *argv[])
{
  const int interp = 5;
  const int steps = atoi(argv[1]);
  const double dt = 0.1;

  const double tmax = steps * dt;
  const double mu = tmax / 2.0;
  const double sig = tmax / 12.0;
  const double sigsqr = sig * sig;

  const double lambda = 2 * M_PI * c0 / omega;

  // set up dots and history of dots
  auto dots = std::make_shared<DotVector>(import_dots("dots.cfg"));
  const int ndots = (*dots).size();
  std::cout << "Running with " << ndots << " dots" << std::endl;

  auto history =
      std::make_shared<Integrator::History<Eigen::Vector2cd>>(ndots, 22, steps);
  history->fill(Eigen::Vector2cd::Zero());

  for(int n = 0; n < ndots; ++n)
    for(int i = -22; i < steps; ++i)
      // if ( n != OBS )
      history->array_[n][i][0] =
          source(i * dt, mu, sigsqr) / (*dots)[n].dipole().norm();

  // set up propagator
  auto propagator = Propagation::EFIE<cmplx>(c0, prop_constant);
  // auto propagator = Propagation::RotatingEFIE(c0, prop_constant, omega);

  // calculate analytic and direct solution
  auto direct_interaction = std::make_shared<DirectInteraction>(
      dots, history, propagator, interp, c0, dt);

  std::vector<std::vector<std::complex<double>>> fld_anlytc(
      steps, std::vector<std::complex<double>>(ndots));
  std::vector<std::vector<std::complex<double>>> fld_dir(
      steps, std::vector<std::complex<double>>(ndots));

  for(int i = 0; i < steps; ++i) {
    fld_anlytc[i] = analytic_evaluate(dots, i, dt, mu, sigsqr);
    // fld_dir[i] = direct_interaction->evaluate(i);

    const InteractionBase::ResultArray array_dir =
        direct_interaction->evaluate(i);

    for(int itrg = 0; itrg < ndots; itrg++)
      fld_dir[i][itrg] =
          array_dir[itrg];  // * hbar / (*dots)[nsrcs+itrg].dipole().norm();
  }

  // output
  std::cout << "Elapsed time: "
            << (std::clock() - start_time) / (double)CLOCKS_PER_SEC << "s"
            << std::endl;

  std::ofstream outfile1, outfile2;
  outfile1.open("outtest/direct_interaction_fld1.dat");
  outfile1 << std::scientific << std::setprecision(15);

  outfile2.open("outtest/error.dat", std::ios_base::app);
  outfile3 << std::scientific << std::setprecision(15);

  double normdiff_dir = 0;
  double analytic_sum = 0;
  double dir_sum = 0;
  for(int itrg = 0; itrg < ndots; ++itrg) {
    for(int i = 0; i < steps; ++i) {
      outfile1 << abs(fld_dir[i][itrg]) << " " << abs(fld_anlytc[i][itrg])
               << " " << abs(fld_dir[i][itrg]) / abs(fld_anlytc[i][itrg])
               << std::endl;
      normdiff_dir += pow(abs(fld_dir[i][itrg]) - abs(fld_anlytc[i][itrg]), 2);

      analytic_sum += abs(fld_anlytc[i][itrg]);
      dir_sum += abs(fld_dir[i][itrg]);
    }
  }

  std::cout << analytic_sum << std::endl;

  outfile2 << expansion << " " << ds / lambda << " "
           << sqrt(normdiff_dir) / analytic_sum << std::endl;

  outfile1.close();
  outfile2.close();
}
