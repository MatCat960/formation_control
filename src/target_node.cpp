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
  std::string uav_name_;
  uint32_t uav_id_;
  std::vector<int> uav_team_;
  std::string gps_origin_frame_;
  double target_phase_;
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
  uav_name_ = get_namespace();
  uav_name_.erase(0, 1);
  gps_origin_frame_ = uav_name_ + "/gps_origin";
  RCLCPP_INFO(this->get_logger(), "UAV name: %s", uav_name_.c_str());
  // Extract UAV ID from name (format: "Drone{id}")
  try {
    uav_id_ = std::stoul(uav_name_.substr(5)); // Skip "Drone" prefix and convert remaining digits
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to extract UAV ID from name '%s': %s", uav_name_.c_str(), e.what());
    uav_id_ = 0;
  }
  declare_parameter("target_radius", 5.0);
  declare_parameter("target_period", 60.0);
  declare_parameter("target_p_gain", 1.0);
  declare_parameter("target_d_gain", 0.1);
  declare_parameter<std::vector<int>>("team_sizes", { 1 });
  declare_parameter<std::vector<int>>("team_ids", { 1 });
  declare_parameter<std::vector<int>>("team_phases", { 0 });
  std::vector<int64_t> team_sizes = get_parameter("team_sizes").as_integer_array();
  std::vector<int64_t> team_ids = get_parameter("team_ids").as_integer_array();
  std::vector<int64_t> team_phases = get_parameter("team_phases").as_integer_array();

  // Create vector of teams
  std::vector<std::vector<int>> teams;
  size_t id_idx = 0;
  for (size_t i = 0; i < team_sizes.size(); i++) {
    std::vector<int> team;
    for (size_t j = 0; j < static_cast<size_t>(team_sizes[i]); j++) {
      if (id_idx < team_ids.size()) {
        team.push_back(team_ids[id_idx++]);
      }
    }
    teams.push_back(team);
  }

  // Find my team
  uav_team_.clear();
  target_phase_ = 0;
  for (size_t i = 0; i < teams.size(); i++) {
    if (std::find(teams[i].begin(), teams[i].end(), uav_id_) != teams[i].end()) {
      uav_team_ = teams[i];
      target_phase_ = team_phases[i];
      break;
    }
  }
  
  if (uav_team_.empty()) {
    RCLCPP_WARN(this->get_logger(), "UAV ID %d not found in any team!", uav_id_);
  }else{
    RCLCPP_INFO_STREAM(this->get_logger(), fmt::format("UAV team: {}", fmt::join(uav_team_, ",")));
  }
}


void TargetNode::loop()
{
  auto time = this->get_clock()->now().seconds();
  nav_msgs::msg::Odometry target_msg;
  double r_traj = get_parameter("target_radius").as_double();
  double T_traj = get_parameter("target_period").as_double();
  target_msg.header.stamp = this->get_clock()->now();
  target_msg.header.frame_id = "common_origin";
  target_msg.pose.pose.position.x = r_traj * cos(target_phase_ + 2 * M_PI * time / T_traj);
  target_msg.pose.pose.position.y = r_traj * sin(target_phase_ + 2 * M_PI * time / T_traj);
  target_msg.twist.twist.linear.x = -2 * M_PI / T_traj * r_traj * sin(target_phase_ + 2 * M_PI * time / T_traj);
  target_msg.twist.twist.linear.y = 2 * M_PI / T_traj * r_traj * cos(target_phase_ + 2 * M_PI * time / T_traj);
  target_pub_->publish(target_msg);
  // RCLCPP_INFO(get_logger(), "Target: x: %.2f, y: %.2f", target_msg.pose.pose.position.x, target_msg.pose.pose.position.y);
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TargetNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
