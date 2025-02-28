#pragma once
#include <limits>

namespace formation_control
{
  struct FormationControlParameters {
    int max_agents = std::numeric_limits<int>::signaling_NaN();
    double max_velocity = std::numeric_limits<double>::signaling_NaN();
    int neighbor_validity_ms = std::numeric_limits<int>::signaling_NaN();
    int max_obstacles = std::numeric_limits<int>::signaling_NaN();
    double robot_safe_distance = std::numeric_limits<double>::signaling_NaN();
    double robot_avoidance_gain = std::numeric_limits<double>::signaling_NaN();
    double obstacle_safe_distance = std::numeric_limits<double>::signaling_NaN();
    double obstacle_avoidance_gain = std::numeric_limits<double>::signaling_NaN();
    bool clf_enabled = false;
    double formation_clf_gain = std::numeric_limits<double>::signaling_NaN();
    double formation_radius = std::numeric_limits<double>::signaling_NaN();
    bool verbose = false;
  };
} // namespace formation_control
