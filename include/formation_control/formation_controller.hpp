#pragma once

#include <arrc_interfaces/msg/neighbors.hpp>
#include <eigen3/Eigen/Dense>
#include <formation_control/formation_controller_parameters.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <memory>

namespace formation_control
{
  class FormationController
  {
  private:
    static const uint qp_prob_size = 3;
    /**
     * @brief Hessian matrix of QP Problem
     *
     */
    Eigen::Matrix<double, qp_prob_size, qp_prob_size, Eigen::RowMajor> hessian_;

    /**
     * @brief Constraint matrix of the QP Problem
     *
     */
    Eigen::Matrix<double, Eigen::Dynamic, qp_prob_size, Eigen::RowMajor> constraint_matrix_;
    /**
     * @brief gradient vector of QP Problem, size 3
     *
     */
    Eigen::Vector<double, qp_prob_size> gradient_vector_; // vector of optimization problem

    /**
     * @brief vector of lower bounds for the qp problem's variables
     *
     */
    Eigen::Vector<double, qp_prob_size> lowerbound_;
    /**
     * @brief vector of upper bounds for the qp problem's variables
     *
     */

    Eigen::Vector<double, qp_prob_size> upperbound_;

    /**
     * @brief vector of lower bounds for the qp problem's constraints
     *
     */
    Eigen::VectorXd constraint_lowerbound_;
    /**
     * @brief vector of upper bounds for the qp problem's constraints
     *
     */

    Eigen::VectorXd constraint_upperbound_;

    std::shared_ptr<FormationControlParameters> params_;
    std::function<int64_t()> now_fn_;
    std::vector<geometry_msgs::msg::Point> neighbors_;
    std::vector<geometry_msgs::msg::Point> obstacles_;

  public:
    /**
     * @brief Return code
     *
     */
    enum class Return {
      SUCCESS = 0,
      INFEASIBLE,
      SOLVER_ERROR
    };
    explicit FormationController(std::shared_ptr<FormationControlParameters> params, std::function<int64_t()> now_fn);
    /**
     * @brief Apply the control law
     *
     * @param uopt optimal control input
     * @param ustar desired control input
     * @param h_out output of the control law
     * @return Return
     */
    Return applyCbf(Eigen::Vector2d& uopt, Eigen::Vector2d& ustar, const geometry_msgs::msg::Pose& pose,
                    const std::vector<geometry_msgs::msg::PointStamped>& neighbors,
                    const std::vector<geometry_msgs::msg::PointStamped>& obstacles, std::vector<double>& h_out);
    /**
     * @brief Set the neighbors and obstacles
     *
     * @param pose current pose of the drone in world frame
     * @param neighbors neighbors of the drone in world frame
     * @param obstacles obstacles in world frame
     * @return Return
     */
    void setNeighborsAndObstacles(const geometry_msgs::msg::Pose& pose, const std::vector<geometry_msgs::msg::PointStamped>& neighbors,
                                  const std::vector<geometry_msgs::msg::PointStamped>& obstacles);
  };
} // namespace formation_control
