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
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;
class TargetNode : public rclcpp::Node
{
private:
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

  // publishers
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr target_pub_;
  // timers
  rclcpp::TimerBase::SharedPtr main_timer_;

public:
  explicit TargetNode();
};
TargetNode::TargetNode() : Node("target_node")
{
  RCLCPP_INFO(this->get_logger(), "Starting Target Node...");
  // ----------- params ----------
  declareAndInitParams();
  // ---------- publishers ----------
  target_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("target", 1);
  // ---------- timers ----------
  main_timer_ = this->create_wall_timer(100ms, [this]() { loop(); });
}

void TargetNode::declareAndInitParams()
{
  declare_parameter("target_radius", 5.0);
  declare_parameter("target_period", 60.0);
  declare_parameter("target_p_gain", 1.0);
  declare_parameter("target_d_gain", 0.1);
}


void TargetNode::loop()
{
  auto time = this->get_clock()->now().seconds();
  nav_msgs::msg::Odometry target_msg;
  double r_traj = get_parameter("target_radius").as_double();
  double T_traj = get_parameter("target_period").as_double();
  target_msg.header.stamp = this->get_clock()->now();
  target_msg.header.frame_id = "common_origin";
  target_msg.pose.pose.position.x = r_traj * cos(2 * M_PI * time / T_traj);
  target_msg.pose.pose.position.y = r_traj * sin(2 * M_PI * time / T_traj);
  target_msg.twist.twist.linear.x = -2 * M_PI / T_traj * r_traj * sin(2 * M_PI * time / T_traj);
  target_msg.twist.twist.linear.y = 2 * M_PI / T_traj * r_traj * cos(2 * M_PI * time / T_traj);
  target_pub_->publish(target_msg);
  RCLCPP_INFO(get_logger(), "Target: x: %.2f, y: %.2f", target_msg.pose.pose.position.x, target_msg.pose.pose.position.y);
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TargetNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
