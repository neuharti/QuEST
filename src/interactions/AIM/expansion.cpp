#include "expansion.h"

AIM::Expansions::ExpansionTable
AIM::Expansions::LeastSquaresExpansionSolver::get_expansions(
    const int box_order, const Grid &grid, const std::vector<QuantumDot> &dots)
{
  return LeastSquaresExpansionSolver(box_order, grid).table(dots);
}

AIM::Expansions::ExpansionTable
AIM::Expansions::LeastSquaresExpansionSolver::table(
    const std::vector<QuantumDot> &dots) const
{
  using namespace enums;
  AIM::Expansions::ExpansionTable table(boost::extents[dots.size()][num_pts]);

  for(auto dot_idx = 0u; dot_idx < dots.size(); ++dot_idx) {
    const auto &pos = dots.at(dot_idx).position();
    Eigen::FullPivLU<Eigen::MatrixXd> lu(w_matrix(pos));

    Eigen::Array4Xd weights(4, num_pts);
    weights.row(D_0) = lu.solve(q_vector({{0, 0, 0}}));
    weights.row(D_X) = lu.solve(q_vector({{1, 0, 0}}));
    weights.row(D_Y) = lu.solve(q_vector({{0, 1, 0}}));
    weights.row(D_Z) = lu.solve(q_vector({{0, 0, 1}}));

    const auto indices = grid.expansion_box_indices(pos, box_order);
    for(auto w = 0; w < num_pts; ++w) {
      table[dot_idx][w].index = indices[w];
      Eigen::Map<Eigen::Array4d>(table[dot_idx][w].weights.data()) =
          weights.col(w);
    }
  }

  return table;
}

Eigen::VectorXd AIM::Expansions::LeastSquaresExpansionSolver::q_vector(
    const std::array<int, 3> &derivatives) const
{
  Eigen::VectorXd q_vec(num_pts);

  // The nx, ny, nz loops correspond to Taylor expansions in the various
  // dimensions. The ternary if statements zero out the constant-term entries of
  // qvec that disappear after differentiation.

  Eigen::Vector3d arg = Eigen::Vector3d::Zero();

  int i = 0;
  for(int nx = 0; nx <= box_order; ++nx) {
    double x_term =
        nx < derivatives[0] ? 0 : falling_factorial(nx, derivatives[0]) *
                                      std::pow(arg(0), nx - derivatives[0]);
    for(int ny = 0; ny <= box_order; ++ny) {
      double y_term =
          ny < derivatives[1] ? 0 : falling_factorial(ny, derivatives[1]) *
                                        std::pow(arg(1), ny - derivatives[1]);
      for(int nz = 0; nz <= box_order; ++nz) {
        double z_term =
            nz < derivatives[2] ? 0 : falling_factorial(nz, derivatives[2]) *
                                          std::pow(arg(2), nz - derivatives[2]);
        q_vec(i++) = x_term * y_term * z_term;
      }
    }
  }

  return q_vec;
}

Eigen::MatrixXd AIM::Expansions::LeastSquaresExpansionSolver::w_matrix(
    const Eigen::Vector3d &pos) const
{
  Eigen::MatrixXd w_mat = Eigen::MatrixXd::Zero(num_pts, num_pts);

  auto expansion_indices = grid.expansion_box_indices(pos, box_order);

  for(int col = 0; col < num_pts; ++col) {
    Eigen::Vector3d dr =
        grid.spatial_coord_of_box(expansion_indices.at(col)) - pos;
    int row = 0;
    for(int nx = 0; nx <= box_order; ++nx) {
      double x_term = std::pow(dr(0), nx);
      for(int ny = 0; ny <= box_order; ++ny) {
        double y_term = std::pow(dr(1), ny);
        for(int nz = 0; nz <= box_order; ++nz) {
          double z_term = std::pow(dr(2), nz);
          w_mat(row++, col) = x_term * y_term * z_term;
        }
      }
    }
  }

  return w_mat;
}
