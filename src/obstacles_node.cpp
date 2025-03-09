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
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;
class ObstaclesNode : public rclcpp::Node
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
   
  // obs msg
  geometry_msgs::msg::PoseArray obs_msg_;
  // publishers
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr obs_pub_;
  // timers
  rclcpp::TimerBase::SharedPtr main_timer_;

public:
  explicit ObstaclesNode();
};
ObstaclesNode::ObstaclesNode() : Node("obstacles_node")
{
  RCLCPP_INFO(this->get_logger(), "Starting Obstacles Node...");
  // ----------- params ----------
  declareAndInitParams();
  // ---------- publishers ----------
  obs_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("obstacles", 1);
  // ---------- timers ----------
  main_timer_ = this->create_wall_timer(100ms, [this]() { loop(); });
}

void ObstaclesNode::declareAndInitParams()
{
  declare_parameter<std::vector<double>>("x_obstacles", {});
  declare_parameter<std::vector<double>>("y_obstacles", {});
  declare_parameter<std::vector<double>>("z_obstacles", {});

  std::vector<double> x_obs = get_parameter("x_obstacles").as_double_array();
  std::vector<double> y_obs = get_parameter("y_obstacles").as_double_array();
  std::vector<double> z_obs = get_parameter("z_obstacles").as_double_array();
  if (x_obs.size() != y_obs.size() || x_obs.size() != z_obs.size()){
    RCLCPP_ERROR(this->get_logger(), "Obstacles sizes do not match!");
  }
  // std::cout << "Obstacles x: \n" << x_obs << std::endl;
  // std::cout << "Obstacles y: \n" << y_obs << std::endl;
  obs_msg_.header.frame_id = "common_origin";
  for (int i = 0; i < x_obs.size(); i++){
    std::cout << "obs " << i  << ": " << x_obs[i] << ", " << y_obs[i] << ", " << z_obs[i] << std::endl;
    geometry_msgs::msg::Pose pose_msg;
    pose_msg.position.x = x_obs[i];
    pose_msg.position.y = y_obs[i];
    pose_msg.position.z = z_obs[i];
    obs_msg_.poses.push_back(pose_msg);
  }
  

}


void ObstaclesNode::loop()
{
  obs_msg_.header.stamp = this->get_clock()->now();
  obs_pub_->publish(obs_msg_);
  for (int i = 0; i < obs_msg_.poses.size(); i++){
    RCLCPP_INFO(get_logger(), "Obstacle %i: x: %.2f, y: %.2f, z: %.2f", i, obs_msg_.poses[i].position.x, obs_msg_.poses[i].position.y, obs_msg_.poses[i].position.z);  
  }
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ObstaclesNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
