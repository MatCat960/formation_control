#include <algorithm>
#include <arrc_interfaces/msg/detail/neighbors__struct.hpp>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <fmt/format.h>
#include <geometry_msgs/msg/detail/point_stamped__struct.hpp>
#include <nav_msgs/msg/detail/odometry__struct.hpp>
#include <string>
#include <tf2/utils.h>

// ROS includes
#include "arrc_interfaces/msg/neighbors.hpp"
#include "arrc_interfaces/msg/uav_vel_acc.hpp"
#include "formation_control/formation_controller.hpp"
#include "formation_control/formation_controller_parameters.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;
using namespace formation_control;
class FormationNode : public rclcpp::Node
{
private:
  /**
   * Callback function for odometry messages
   * @param msg Odometry message
   */
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  /**
   * Callback function for neighbors messages
   * @param msg Neighbors message
   */
  void targetCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  /**
   * Callback function for neighbors messages
   * @param msg Neighbors message
   */
  void neighborsCallback(const arrc_interfaces::msg::Neighbors::SharedPtr& msg);
  /**
   * Callback function for obstacles messages
   * @param msg Obstacles message
   */
  void obstaclesCallback(const geometry_msgs::msg::PoseArray::SharedPtr& msg);
  /**
   * Main loop function
   */
  void loop();
  /**
   * Declare and initialize parameters
   */
  void declareAndInitParams();
  /**
   * Parameters callback function
   * @param parameters Vector of parameters
   * @return SetParametersResult instance
   */
  rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter>& parameters);

  std::string uav_name_;
  std::string gps_origin_frame_;
  nav_msgs::msg::Odometry odometry_;
  nav_msgs::msg::Odometry target_odometry_;
  std::shared_ptr<FormationController> formation_controller;
  std::shared_ptr<FormationControlParameters> formation_parameters;
  OnSetParametersCallbackHandle::SharedPtr parameters_ch_;
  std::vector<geometry_msgs::msg::PointStamped> neighbors_;
  std::vector<geometry_msgs::msg::PointStamped> obstacles_;
  // publishers
  rclcpp::Publisher<arrc_interfaces::msg::UavVelAcc>::SharedPtr vel_pub_;
  // subscribers
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr target_sub_;
  rclcpp::Subscription<arrc_interfaces::msg::Neighbors>::SharedPtr neighbors_sub_;
  // timers
  rclcpp::TimerBase::SharedPtr main_timer_;

public:
  explicit FormationNode();
};
FormationNode::FormationNode() : Node("formation_controller")
{
  RCLCPP_INFO(this->get_logger(), "Starting Formation Controller...");
  // ----------- params ----------
  declareAndInitParams();
  // -----------formation controller---------
  formation_controller = std::make_shared<FormationController>(formation_parameters, [this] { return now().nanoseconds(); });
  // ---------- subscriptions ---------
  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "odometry", 1, [this](nav_msgs::msg::Odometry::SharedPtr msg) { this->odomCallback(msg); });
  target_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "target", 1, [this](nav_msgs::msg::Odometry::SharedPtr msg) { this->targetCallback(msg); });
  neighbors_sub_ = this->create_subscription<arrc_interfaces::msg::Neighbors>(
      "neighbors_odometry", 1, [this](arrc_interfaces::msg::Neighbors::SharedPtr msg) { this->neighborsCallback(msg); });
  // ---------- publishers ----------
  vel_pub_ = this->create_publisher<arrc_interfaces::msg::UavVelAcc>("command/setVelocityAcceleration", 1);
  // ---------- timers ----------
  main_timer_ = this->create_wall_timer(100ms, [this]() { loop(); });
}

void FormationNode::declareAndInitParams()
{
  uav_name_ = get_namespace();
  uav_name_.erase(0, 1);
  gps_origin_frame_ = uav_name_ + "/gps_origin";
  declare_parameter("max_agents", 10);
  declare_parameter("max_velocity", 3.0);
  declare_parameter("neighbor_validity_ms", 2000);
  declare_parameter("max_obstacles", 10);
  declare_parameter("robot_safe_distance", 2.0);
  declare_parameter("robot_avoidance_gain", 5.0);
  declare_parameter("obstacle_safe_distance", 5.0);
  declare_parameter("obstacle_avoidance_gain", 1.0);
  declare_parameter("clf_enabled", true);
  declare_parameter("formation_clf_gain", 0.1);
  declare_parameter("formation_radius", 3.0);
  declare_parameter("target_radius", 5.0);
  declare_parameter("target_period", 60.0);
  declare_parameter("target_p_gain", 1.0);
  declare_parameter("target_d_gain", 0.1);
  declare_parameter("verbose", true);

  formation_parameters = std::make_shared<FormationControlParameters>();
  formation_parameters->max_agents = get_parameter("max_agents").as_int();
  formation_parameters->max_velocity = get_parameter("max_velocity").as_double();
  formation_parameters->neighbor_validity_ms = get_parameter("neighbor_validity_ms").as_int();
  formation_parameters->max_obstacles = get_parameter("max_obstacles").as_int();
  formation_parameters->robot_safe_distance = get_parameter("robot_safe_distance").as_double();
  formation_parameters->robot_avoidance_gain = get_parameter("robot_avoidance_gain").as_double();
  formation_parameters->obstacle_safe_distance = get_parameter("obstacle_safe_distance").as_double();
  formation_parameters->obstacle_avoidance_gain = get_parameter("obstacle_avoidance_gain").as_double();
  formation_parameters->clf_enabled = get_parameter("clf_enabled").as_bool();
  formation_parameters->formation_clf_gain = get_parameter("formation_clf_gain").as_double();
  formation_parameters->formation_radius = get_parameter("formation_radius").as_double();
  formation_parameters->verbose = get_parameter("verbose").as_bool();
  parameters_ch_ =
      add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter>& parameters) { return parametersCallback(parameters); });
}

rcl_interfaces::msg::SetParametersResult FormationNode::parametersCallback(const std::vector<rclcpp::Parameter>& parameters)
{
  for (const auto& param : parameters) {
    auto param_name = param.get_name();
    if (param_name == "max_agents") {
      formation_parameters->max_agents = param.as_int();
      RCLCPP_INFO(get_logger(), "Max agents set to %i", formation_parameters->max_agents);
    }
    if (param_name == "max_velocity") {
      formation_parameters->max_velocity = param.as_double();
      RCLCPP_INFO(get_logger(), "Max velocity set to %f", formation_parameters->max_velocity);
    }
    if (param_name == "neighbor_validity_ms") {
      formation_parameters->neighbor_validity_ms = param.as_int();
      RCLCPP_INFO(get_logger(), "Neighbor validity set to %i ms", formation_parameters->neighbor_validity_ms);
    }
    if (param_name == "max_obstacles") {
      formation_parameters->max_obstacles = param.as_int();
      RCLCPP_INFO(get_logger(), "Max obstacles set to %i", formation_parameters->max_obstacles);
    }
    if (param_name == "robot_safe_distance") {
      formation_parameters->robot_safe_distance = param.as_double();
      RCLCPP_INFO(get_logger(), "Robot safe distance set to %f", formation_parameters->robot_safe_distance);
    }
    if (param_name == "robot_avoidance_gain") {
      formation_parameters->robot_avoidance_gain = param.as_double();
      RCLCPP_INFO(get_logger(), "Robot avoidance gain set to %f", formation_parameters->robot_avoidance_gain);
    }
    if (param_name == "obstacle_safe_distance") {
      formation_parameters->obstacle_safe_distance = param.as_double();
      RCLCPP_INFO(get_logger(), "Obstacle safe distance set to %f", formation_parameters->obstacle_safe_distance);
    }
    if (param_name == "obstacle_avoidance_gain") {
      formation_parameters->obstacle_avoidance_gain = param.as_double();
      RCLCPP_INFO(get_logger(), "Obstacle avoidance gain set to %f", formation_parameters->obstacle_avoidance_gain);
    }
    if (param_name == "clf_enabled") {
      formation_parameters->clf_enabled = param.as_bool();
      RCLCPP_INFO(get_logger(), "CLF enabled set to %s", formation_parameters->clf_enabled ? "true" : "false");
    }
    if (param_name == "formation_clf_gain") {
      formation_parameters->formation_clf_gain = param.as_double();
      RCLCPP_INFO(get_logger(), "Formation CLF gain set to %f", formation_parameters->formation_clf_gain);
    }
    if (param_name == "formation_radius") {
      formation_parameters->formation_radius = param.as_double();
      RCLCPP_INFO(get_logger(), "Formation radius set to %f", formation_parameters->formation_radius);
    }
    if (param_name == "verbose") {
      formation_parameters->verbose = param.as_bool();
      RCLCPP_INFO(get_logger(), "Verbose mode set to %s", formation_parameters->verbose ? "true" : "false");
    }
  }
  return rcl_interfaces::msg::SetParametersResult();
}
void FormationNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  odometry_ = *msg;
}
void FormationNode::targetCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  target_odometry_ = *msg;
}
void FormationNode::neighborsCallback(const arrc_interfaces::msg::Neighbors::SharedPtr& msg)
{
  neighbors_.clear();
  std::transform(msg->neighbors.begin(), msg->neighbors.end(), std::back_inserter(neighbors_), [](const nav_msgs::msg::Odometry& neighbor) {
    geometry_msgs::msg::PointStamped point;
    point.header = neighbor.header;
    point.point = neighbor.pose.pose.position;
    return point;
  });
}
void FormationNode::loop()
{
  Eigen::Vector2d p_i{ odometry_.pose.pose.position.x, odometry_.pose.pose.position.y };
  RCLCPP_INFO(get_logger(), "Max agents set to %i", formation_parameters->max_agents);

  // // Eigen::Matrix3d R;
  // // R << cos(yaw), sin(yaw), 0, -sin(yaw), cos(yaw), 0, 0, 0, 1;
  // auto time = this->get_clock()->now().seconds();
  // geometry_msgs::msg::Point target_msg;
  // double r_traj = get_parameter("target_radius").as_double();
  // double T_traj = get_parameter("target_period").as_double();
  // target_msg.x = r_traj * cos(2 * M_PI * time / T_traj);
  // target_msg.y = r_traj * sin(2 * M_PI * time / T_traj);
  // target_pub_->publish(target_msg);
  // RCLCPP_INFO(get_logger(), "Target: x: %.2f, y: %.2f", target_msg.x, target_msg.y);

  Eigen::Vector2d x_target{ target_odometry_.pose.pose.position.x, target_odometry_.pose.pose.position.y };
  RCLCPP_INFO(get_logger(), "Target:  %f, %f", x_target(0), x_target(1));
  Eigen::Vector2d x_target_local = (x_target - p_i);

  Eigen::Vector2d config_centroid;
  config_centroid.setZero();
  for (auto n : neighbors_) {
    config_centroid += (Eigen::Vector2d{ n.point.x, n.point.y } - p_i);
  }

  config_centroid /= (neighbors_.size() + 1);

  Eigen::Vector2d uopt, u_star;
  u_star = x_target_local - config_centroid;
  std::vector<double> h_out;
  if (FormationController::Return::SUCCESS ==
      formation_controller->applyCbf(uopt, u_star, odometry_.pose.pose, neighbors_, obstacles_, h_out)) {
    arrc_interfaces::msg::UavVelAcc vel_msg;
    vel_msg.header.frame_id = gps_origin_frame_;
    vel_msg.velocity.x = uopt.x();
    vel_msg.velocity.y = uopt.y();
    vel_msg.velocity.z = 0.0;
    vel_msg.acceleration.x = std::nan("1");
    vel_msg.acceleration.y = std::nan("1");
    vel_msg.acceleration.z = std::nan("1");
    vel_msg.yaw = atan2(x_target_local(1), x_target_local(0));
    vel_msg.yaw_rate = std::nan("1");
    vel_pub_->publish(vel_msg);
  } else {
    RCLCPP_WARN(this->get_logger(), "CBF FAILED.");
  }
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FormationNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
