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
  AIM::Expansions::ExpansionTable table(boost::extents[dots.size()][num_pts]);

  Eigen::VectorXd q_vec = Eigen::VectorXd::Zero(num_pts);
  q_vec(0) = 1;

  for(auto dot_idx = 0u; dot_idx < dots.size(); ++dot_idx) {
    const auto &pos = dots.at(dot_idx).position();
    Eigen::FullPivLU<Eigen::MatrixXd> lu(w_matrix(pos));

    Eigen::VectorXd weights = lu.solve(q_vec);

    const auto indices = grid.expansion_indices(pos);
    for(auto w = 0; w < num_pts; ++w) {
      table[dot_idx][w].index = indices[w];
      table[dot_idx][w].weight = weights(w);
    }
  }

  return table;
}

Eigen::MatrixXd AIM::Expansions::LeastSquaresExpansionSolver::w_matrix(
    const Eigen::Vector3d &pos) const
{
  Eigen::MatrixXd w_mat = Eigen::MatrixXd::Zero(num_pts, num_pts);

  auto expansion_indices = grid.expansion_indices(pos);

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
