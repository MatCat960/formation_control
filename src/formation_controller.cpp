#include <fmt/format.h>
#include <formation_control/formation_controller.hpp>
#include <iostream>
#include <limits>
extern "C" {
#include <qpOASES_e/QProblem.h>
}

namespace formation_control
{

  FormationController::FormationController(std::shared_ptr<FormationControlParameters> params, std::function<int64_t()> now_fn) :
      params_(params), now_fn_(now_fn)
  {
    hessian_.setIdentity();
    hessian_(3, 3) = 100.0; // weight of slack var
    if (params_->verbose)
      std::cout << "Hessian matrix initialized" << std::endl;
    gradient_vector_.setZero();
  }

  FormationController::Return FormationController::applyCbf(Eigen::Vector2d& uopt, Eigen::Vector2d& ustar,
                                                            const geometry_msgs::msg::Pose& pose,
                                                            const std::vector<nav_msgs::msg::Odometry>& neighbors,
                                                            const std::vector<geometry_msgs::msg::PointStamped>& obstacles,
                                                            const Eigen::Vector2d& target, std::vector<double>& h_out,
                                                            const Eigen::Vector2d& target_velocity)
  {
    setNeighborsAndObstacles(pose, neighbors, obstacles);
    size_t neighbors_number = std::min(neighbors_.size(), (size_t)params_->max_agents);
    size_t obstacles_number = std::min(obstacles_.size(), (size_t)params_->max_obstacles);
    size_t constraints_number = neighbors_number + obstacles_number + (params_->clf_enabled ? 1 : 0);

    if (constraints_number == 0) {
      if (params_->verbose)
        std::cout << "[formation control] no neighbors or obstacles: returning desired" << std::endl;
      uopt = ustar;

      return FormationController::Return::SUCCESS;
    }
    constraint_matrix_.resize(constraints_number, Eigen::NoChange);
    constraint_upperbound_.resize(constraints_number);
    constraint_lowerbound_.resize(constraints_number);
    constraint_lowerbound_.setConstant(-std::numeric_limits<double>::infinity());
    constraint_upperbound_.setConstant(std::numeric_limits<double>::infinity());
    lowerbound_.head(2).setConstant(-params_->max_velocity);
    lowerbound_(2) = -std::numeric_limits<double>::epsilon();
    upperbound_.head(2).setConstant(params_->max_velocity);
    upperbound_(2) = std::numeric_limits<double>::infinity();
    gradient_vector_.head(2) = -ustar;
    gradient_vector_(2) = 0.0; // slack variable
    Eigen::Vector2d my_position{ pose.position.x, pose.position.y };
    Eigen::Vector2d center = my_position;
    Eigen::Vector2d neighbors_velocity_sum{ 0.0, 0.0 };
    // Collision avoidance with other robots
    double robot_safe_distance_squared = pow(params_->robot_safe_distance, 2);
    for (size_t i = 0; i < neighbors_number; i++) {
      Eigen::Vector2d p_i_j{ neighbors_.at(i).pose.pose.position.x, neighbors_.at(i).pose.pose.position.y };
      center += my_position - p_i_j;
      neighbors_velocity_sum += Eigen::Vector2d{ neighbors_.at(i).twist.twist.linear.x, neighbors_.at(i).twist.twist.linear.y };
      constraint_matrix_(i, 0) = 2 * p_i_j.x();
      constraint_matrix_(i, 1) = 2 * p_i_j.y();
      constraint_matrix_(i, 2) = 0.0; // slack var
      double h_i = p_i_j.dot(p_i_j) - robot_safe_distance_squared;
      constraint_lowerbound_(i) = -params_->robot_avoidance_gain * pow(h_i, 3);
      h_out.push_back(h_i);
    }

    // Obstacle avoidance constraints
    double obstacle_safe_distance_squared = pow(params_->obstacle_safe_distance, 2);
    for (size_t i = 0; i < obstacles_number; i++) {
      Eigen::Vector2d o_i_j{ obstacles_.at(i).x, obstacles_.at(i).y };
      constraint_matrix_(neighbors_number + i, 0) = 2 * o_i_j.x();
      constraint_matrix_(neighbors_number + i, 1) = 2 * o_i_j.y();
      constraint_matrix_(neighbors_number + i, 2) = 0.0; // slack var
      double h_i = o_i_j.dot(o_i_j) - obstacle_safe_distance_squared;
      constraint_lowerbound_(neighbors_number + i) = -params_->obstacle_avoidance_gain * pow(h_i, 3);
      h_out.push_back(h_i);
    }

    // CLF for desired distance from center
    if (params_->clf_enabled) {
      if (params_->formation_type == 0) { // Circle formation
        center /= neighbors_number + 1;
        double V, d;
        d = (my_position - center).norm();
        V = pow(d - params_->formation_radius, 2);
        if (params_->verbose) {
          std::cout << "[formation control] Center: " << center << std::endl;
          std::cout << "[formation control] Distance from center: " << d << std::endl;
        }
        Eigen::Vector2d K;
        K = 2 * ((d - params_->formation_radius) / d) * (my_position - center) / (neighbors_number + 1.0);
        constraint_matrix_(constraints_number - 1, 0) = neighbors_number * K(0);
        constraint_matrix_(constraints_number - 1, 1) = neighbors_number * K(1);
        constraint_matrix_(constraints_number - 1, 2) = -1.0;

        constraint_upperbound_(constraints_number - 1) = K.dot(neighbors_velocity_sum) - params_->formation_clf_gain * V;
        h_out.push_back(V);
      } else if (params_->formation_type == 1) { // Line formation
        double V;
        Eigen::Vector2d z = (my_position - target);
        V = z.squaredNorm();
        constraint_matrix_(constraints_number - 1, 0) = 2 * z(0);
        constraint_matrix_(constraints_number - 1, 1) = 2 * z(1);
        constraint_matrix_(constraints_number - 1, 2) = -1.0;
        constraint_upperbound_(constraints_number - 1) = 2*z.dot(target_velocity)-params_->formation_clf_gain * V;
        h_out.push_back(V);
      }
    }
    if (params_->verbose) {
      std::cout << fmt::format("[collision avoidance] h: [{}]", fmt::join(h_out.begin(), h_out.end(), ",")) << std::endl;
    }

    if (params_->verbose) {
      std::cout << "[collision avoidance] Constraint Matrix: " << constraint_matrix_ << std::endl;
      std::cout << "[collision avoidance] Constraint Upperbound: " << constraint_upperbound_ << std::endl;
      std::cout << "[collision avoidance] Constraint Lowerbound: " << constraint_lowerbound_ << std::endl;
    }
    USING_NAMESPACE_QPOASES
    real_t xOpt[qp_prob_size];
    static Options options;
    std::unique_ptr<QProblem> problem(QProblem_createMemory(qp_prob_size, constraints_number));

    int nWSR = 10;
    real_t cputime = 0.003; // 5ms max of cpu time
    QProblemCON(problem.get(), qp_prob_size, constraints_number, HST_POSDEF);
    Options_setToDefault(&options);
    QProblem_setOptions(problem.get(), options);
    /* Solve QP. */

    returnValue retval =
        QProblem_init(problem.get(), hessian_.data(), gradient_vector_.data(), constraint_matrix_.data(), lowerbound_.data(),
                      upperbound_.data(), constraint_lowerbound_.data(), constraint_upperbound_.data(), &nWSR, &cputime);
    switch (retval) {
      case returnValue::SUCCESSFUL_RETURN:
        QProblem_getPrimalSolution(problem.get(), xOpt);
        uopt.x() = xOpt[0];
        uopt.y() = xOpt[1];
        return Return::SUCCESS;
        break;
      case returnValue::RET_INIT_FAILED_INFEASIBILITY:
        if (params_->verbose)
          std::cout << fmt::format("[formation control] QP INFEASIBLE, nWSR: {}, cputime: {}", nWSR, cputime) << std::endl;
        return Return::INFEASIBLE;
      case returnValue::RET_MAX_NWSR_REACHED:
        if (params_->verbose)
          std::cout << fmt::format("[formation control] QP RET_MAX_NWSR_REACHED, nWSR: {}, cputime: {}", nWSR, cputime) << std::endl;
        return Return::SOLVER_ERROR;
      default:
        if (params_->verbose)
          std::cout << fmt::format("[formation control] qp retval {}, nWSR: {}, cputime: {}", (int)retval, nWSR, cputime) << std::endl;
        return Return::SOLVER_ERROR;
    }
  }

  void FormationController::setNeighborsAndObstacles(const geometry_msgs::msg::Pose& pose,
                                                     const std::vector<nav_msgs::msg::Odometry>& neighbors,
                                                     const std::vector<geometry_msgs::msg::PointStamped>& obstacles)
  {
    neighbors_.clear();
    neighbors_.reserve(neighbors.size());
    for (auto n : neighbors) {
      if (now_fn_() - (n.header.stamp.sec * 1e9 + n.header.stamp.nanosec) > params_->neighbor_validity_ms * 1e6) {
        if (params_->verbose)
          std::cout << fmt::format("[collision avoidance] Skipping neighbor {}: too old with time {}", n.header.frame_id,
                                   n.header.stamp.sec * 1e9 + n.header.stamp.nanosec)
                    << std::endl;
        continue;
      }
      nav_msgs::msg::Odometry new_neighbor;
      new_neighbor.pose.pose.position.x = pose.position.x - n.pose.pose.position.x;
      new_neighbor.pose.pose.position.y = pose.position.y - n.pose.pose.position.y;
      new_neighbor.twist = n.twist;
      auto it = std::lower_bound(neighbors_.begin(), neighbors_.end(), new_neighbor, [](const auto& a, const auto& b) {
        return sqrt(pow(a.pose.pose.position.x, 2) + pow(a.pose.pose.position.y, 2)) <
               sqrt(pow(b.pose.pose.position.x, 2) + pow(b.pose.pose.position.y, 2));
      });
      neighbors_.insert(it, new_neighbor);
    }
    std::cout << "[Collision avoidance] number of neighbors: " << neighbors_.size() << std::endl;
    obstacles_.clear();
    obstacles_.reserve(obstacles.size());
    for (auto o : obstacles) {
      geometry_msgs::msg::Point new_obstacle;
      new_obstacle.x = pose.position.x - o.point.x;
      new_obstacle.y = pose.position.y - o.point.y;
      auto it = std::lower_bound(obstacles_.begin(), obstacles_.end(), new_obstacle, [](const auto& a, const auto& b) {
        return sqrt(pow(a.x, 2) + pow(a.y, 2)) < sqrt(pow(b.x, 2) + pow(b.y, 2));
      });
      obstacles_.insert(it, new_obstacle);
    }
  }
} // namespace formation_control
