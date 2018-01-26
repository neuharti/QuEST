#include "nearfield_interaction.h"

AIM::NearfieldInteraction::NearfieldInteraction(
    std::shared_ptr<const DotVector> dots,
    std::shared_ptr<const Integrator::History<Eigen::Vector2cd>> history,
    Propagation::RotatingFramePropagator propagator,
    const int interp_order,
    const double c0,
    const double dt,
    Grid grid)
    : HistoryInteraction(
          std::move(dots), std::move(history), interp_order, c0, dt),
      propagator(std::move(propagator)),
      grid(std::move(grid)),
      interaction_pairs(build_pair_list()),
      floor_delays(interaction_pairs.size()),
      coefficients(boost::extents[interaction_pairs.size()][interp_order + 1])
{
  build_coefficient_table();
}

std::vector<std::pair<int, int>> AIM::NearfieldInteraction::build_pair_list()
    const
{
  std::vector<std::pair<int, int>> pairs;
  auto box_contents = grid.box_contents_map();

  auto get_dot_idx = [&](const DotVector::const_iterator &d) {
    return std::distance<DotVector::const_iterator>(dots->begin(), d);
  };

  for(auto box1 = box_contents.begin(); box1 != box_contents.end(); ++box1) {
    // start iter = end iter, thus empty box
    if(box1->first == box1->second) continue;

    for(auto src_dot = box1->first; src_dot != box1->second - 1; ++src_dot) {
      for(auto obs_dot = src_dot + 1; obs_dot != box1->second; ++obs_dot) {
        pairs.push_back({get_dot_idx(src_dot), get_dot_idx(obs_dot)});
      }
    }

    for(auto box2 = box1 + 1; box2 < box_contents.end(); ++box2) {
      if(box2->first == box2->second) continue;

      // box_contents[i] and [j] yield *pairs of DotVector iterators*
      // corresponding to the range of particles within the box (which assumes
      // the DotVector is sorted). If the *.first and *.second iterators are
      // equal, then the box is empty (checked above), otherwise it contains
      // particles that can ALL equivalently determine the box's position and
      // thus its nearfield neighbors.
      bool is_in_nearfield = grid.is_nearfield_pair(box1->first->position(),
                                                    box2->first->position());
      if(!is_in_nearfield) continue;

      for(auto src_dot = box1->first; src_dot != box1->second; ++src_dot) {
        for(auto obs_dot = box2->first; obs_dot != box2->second; ++obs_dot) {
          pairs.push_back({get_dot_idx(src_dot), get_dot_idx(obs_dot)});
        }
      }
    }
  }

  pairs.shrink_to_fit();
  return pairs;
}

void AIM::NearfieldInteraction::build_coefficient_table()
{
  Interpolation::UniformLagrangeSet lagrange(interp_order);

  for(auto pair_idx = 0u; pair_idx < interaction_pairs.size(); ++pair_idx) {
    std::pair<int, int> &pair = interaction_pairs[pair_idx];

    Eigen::Vector3d dr(separation((*dots)[pair.first], (*dots)[pair.second]));
    auto delay = split_double(dr.norm() / (c0 * dt));

    floor_delays[pair_idx] = delay.first;
    lagrange.evaluate_derivative_table_at_x(delay.second, dt);

    std::vector<Eigen::Matrix3cd> interp_dyads(
        propagator.coefficients(dr, lagrange));

    for(int i = 0; i <= interp_order; ++i) {
      coefficients[pair_idx][i] = (*dots)[pair.second].dipole().dot(
          interp_dyads[i] * (*dots)[pair.first].dipole());
    }
  }
}
