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
#include <rclcpp/logging.hpp>
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
   * Callback function for obstacles messages
   * @param A First vertex of the segment
   * @param B Second vertex of the segment
   * @param robot Robot position
   * @return Closest point
   */
  Eigen::Vector2d closestPointOnSegment(const geometry_msgs::msg::PointStamped& v1, const geometry_msgs::msg::PointStamped& v2, const Eigen::Vector2d& robot);
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
  uint32_t uav_id_;
  std::vector<int> uav_team_;
  std::string gps_origin_frame_;
  nav_msgs::msg::Odometry odometry_;
  nav_msgs::msg::Odometry target_odometry_;
  std::shared_ptr<FormationController> formation_controller;
  std::shared_ptr<FormationControlParameters> formation_parameters;
  OnSetParametersCallbackHandle::SharedPtr parameters_ch_;
  std::vector<geometry_msgs::msg::PointStamped> neighbors_;
  std::vector<geometry_msgs::msg::PointStamped> obstacles_;
  std::vector<geometry_msgs::msg::PointStamped> vertices_; 

  // publishers
  rclcpp::Publisher<arrc_interfaces::msg::UavVelAcc>::SharedPtr vel_pub_;
  // subscribers
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr target_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr obs_sub_;
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
  RCLCPP_INFO(this->get_logger(), "UAV name: %s", uav_name_.c_str());
  // Extract UAV ID from name (format: "Drone{id}")
  try {
    uav_id_ = std::stoul(uav_name_.substr(5)); // Skip "Drone" prefix and convert remaining digits
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to extract UAV ID from name '%s': %s", uav_name_.c_str(), e.what());
    uav_id_ = 0;
  }
  RCLCPP_INFO(this->get_logger(), "UAV ID: %d", uav_id_);
  declare_parameter("max_agents", 10);
  declare_parameter("max_velocity", 3.0);
  declare_parameter("neighbor_validity_ms", 2000);
  declare_parameter("max_obstacles", 10);
  declare_parameter("robot_safe_distance", 2.0);
  declare_parameter("robot_avoidance_gain", 5.0);
  declare_parameter("obstacle_safe_distance", 2.0);
  declare_parameter("obstacle_avoidance_gain", 1.0);
  declare_parameter("clf_enabled", true);
  declare_parameter("formation_clf_gain", 0.1);
  declare_parameter("formation_radius", 3.0);
  declare_parameter("formation_type", 0);
  declare_parameter("verbose", true);
  declare_parameter<std::vector<double>>("x_obstacles", { 100.0 });
  declare_parameter<std::vector<double>>("y_obstacles", { 100.0 });
  declare_parameter<std::vector<double>>("z_obstacles", { 100.0 });
  declare_parameter<std::vector<double>>("x_segment", { 100.0 });
  declare_parameter<std::vector<double>>("y_segment", { 100.0 });
  declare_parameter<std::vector<double>>("z_segment", { 100.0 });
  declare_parameter<std::vector<int>>("team_sizes", { 1 });
  declare_parameter<std::vector<int>>("team_ids", { 1 });
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
  formation_parameters->formation_type = get_parameter("formation_type").as_int();
  formation_parameters->verbose = get_parameter("verbose").as_bool();

  std::vector<double> x_obs = get_parameter("x_obstacles").as_double_array();
  std::vector<double> y_obs = get_parameter("y_obstacles").as_double_array();
  std::vector<double> z_obs = get_parameter("z_obstacles").as_double_array();

  std::vector<double> x_segment = get_parameter("x_segment").as_double_array();
  std::vector<double> y_segment = get_parameter("y_segment").as_double_array();
  std::vector<double> z_segment = get_parameter("z_segment").as_double_array();
  if (x_obs.size() != y_obs.size() || x_obs.size() != z_obs.size()) {
    RCLCPP_WARN(this->get_logger(), "Obstacles sizes do not match! will take minimum common number");
  }
  auto obs_num = std::min(x_obs.size(), std::min(y_obs.size(), z_obs.size()));
  obstacles_.clear();
  obstacles_.reserve(obs_num);
  for (size_t i = 0; i < obs_num; ++i) {
    std::cout << "obs " << i << ": " << x_obs[i] << ", " << y_obs[i] << ", " << z_obs[i] << std::endl;
    geometry_msgs::msg::PointStamped point_msg;
    point_msg.header.frame_id = gps_origin_frame_;
    point_msg.point.x = x_obs[i];
    point_msg.point.y = y_obs[i];
    point_msg.point.z = z_obs[i];
    obstacles_.push_back(point_msg);
  }

  // Define segments
  size_t vert_num = std::min(x_segment.size(), std::min(y_segment.size(), z_segment.size()));
  vertices_.clear();
  vertices_.reserve(vert_num);
  for (size_t i = 0; i < vert_num; i++){
        std::cout << "Vertex " << i << ": " << x_segment[i] << ", " << y_segment[i] << ", " << z_segment[i] << std::endl;
        geometry_msgs::msg::PointStamped point_msg;
        point_msg.header.frame_id = gps_origin_frame_;
        point_msg.point.x = x_segment[i];
        point_msg.point.y = y_segment[i];
        point_msg.point.z = y_segment[i];
        vertices_.push_back(point_msg);
  }

  std::vector<int64_t> team_sizes = get_parameter("team_sizes").as_integer_array();
  std::vector<int64_t> team_ids = get_parameter("team_ids").as_integer_array();

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
  for (const auto& team : teams) {
    if (std::find(team.begin(), team.end(), uav_id_) != team.end()) {
      uav_team_ = team;
      break;
    }
  }
  
  if (uav_team_.empty()) {
    RCLCPP_WARN(this->get_logger(), "UAV ID %d not found in any team!", uav_id_);
  }else{
    RCLCPP_INFO_STREAM(this->get_logger(), fmt::format("UAV team: {}", fmt::join(uav_team_, ",")));
  }
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
    if (param_name == "formation_type") {
      formation_parameters->formation_type = param.as_int();
      RCLCPP_INFO(get_logger(), "Formation type set to %i", formation_parameters->formation_type);
    }
    if (param_name == "verbose") {
      formation_parameters->verbose = param.as_bool();
      RCLCPP_INFO(get_logger(), "Verbose mode set to %s", formation_parameters->verbose ? "true" : "false");
    }
  }
  // obstacles_.resize(formation_parameters->num_obstacles);
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
Eigen::Vector2d FormationNode::closestPointOnSegment(const geometry_msgs::msg::PointStamped& v1, const geometry_msgs::msg::PointStamped& v2, const Eigen::Vector2d& robot)
{
  Eigen::Vector2d A{v1.point.x, v1.point.y};
  Eigen::Vector2d B{v2.point.x, v2.point.y};
  Eigen::Vector2d AB = B - A;
  Eigen::Vector2d AR = robot - A;
  double t = AR.dot(AB) / AB.dot(AB);     // project AR onto AB
  t = std::clamp(t, 0.0, 1.0);
  Eigen::Vector2d p = A + t * AB;
  return p;
}
void FormationNode::loop()
{
  if (target_odometry_.header.frame_id.empty()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Waiting for target odometry");
    return;
  }
  if (odometry_.header.frame_id.empty()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Waiting for target odometry");
    return;
  }

  Eigen::Vector2d p_i{ odometry_.pose.pose.position.x, odometry_.pose.pose.position.y };
  Eigen::Vector2d x_target{ target_odometry_.pose.pose.position.x, target_odometry_.pose.pose.position.y };
  Eigen::Vector2d x_target_local;

  std::vector<geometry_msgs::msg::PointStamped> neighbors_team;
  std::vector<geometry_msgs::msg::PointStamped> all_obstacles;
  for (auto o : obstacles_) {
    all_obstacles.push_back(o);
  }
  for (auto n : neighbors_) {
    std::string frame_id = n.header.frame_id;
    size_t start = frame_id.find("Drone") + 5; // Skip "Drone"
    size_t end = frame_id.find("/", start);
    auto id = std::stoi(frame_id.substr(start, end - start));
    if (std::find(uav_team_.begin(), uav_team_.end(), id) != uav_team_.end()) {
      neighbors_team.push_back(n);
    } else {
      all_obstacles.push_back(n);
    }
  }

  Eigen::Vector2d uopt, u_star;
  if (formation_parameters->formation_type == 0){
    x_target_local = (x_target - p_i);
    Eigen::Vector2d config_centroid;
    config_centroid.setZero();
    for (auto n : neighbors_) {
      config_centroid += (Eigen::Vector2d{ n.point.x, n.point.y } - p_i);
    }
    config_centroid /= (neighbors_.size() + 1);
    u_star = x_target_local - config_centroid;
  } else {
    // Find closest point on segment
    Eigen::Vector2d p_closest{100.0, 100.0};
    for (size_t i = 0; i < vertices_.size()-1; i++){
      Eigen::Vector2d p = closestPointOnSegment(vertices_[i], vertices_[i+1], p_i);
      if ((p - p_i).norm() < (p_closest - p_i).norm()){
        p_closest = p;
      }
    }
    std::cout << "Closest point to Drone " << uav_id_ << " on segment: " << p_closest.transpose() << std::endl; 
    x_target_local = p_closest - p_i;
    u_star.setZero();
  }
  std::vector<double> h_out;
  if (FormationController::Return::SUCCESS ==
      formation_controller->applyCbf(uopt, u_star, odometry_.pose.pose, neighbors_team, all_obstacles, x_target_local, h_out)) {
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
