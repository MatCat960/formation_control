#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/impl/utils.h>
#include <tf2/utils.h>
#include <chrono>
#include <string>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include <stdio.h>
#include <math.h>
#include <limits>

// ROS includes
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "formation_control/FormationController.h"
#include "nav_msgs/msg/odometry.hpp"
#include "arrc_interfaces/msg/uav_vel_acc.hpp"


#define M_PI   3.14159265358979323846  /*pi*/

using namespace std::chrono_literals;
using std::placeholders::_1;

class FormationNode : public rclcpp::Node
{
public:
    FormationNode(): Node("formation_node"), controller(2.09, 2.0, 8.0, 4, 0)
    {
        // ----------- params ----------
        RCLCPP_INFO_STREAM(this->get_logger(), "FormationNode constructor called.");
        this->declare_parameter<int>("ROBOTS_NUM", 5);
        this->get_parameter("ROBOTS_NUM", ROBOTS_NUM);
        std::cout << "Robots number: " << ROBOTS_NUM << std::endl;
        UAV_NAME = std::getenv("UAV_NAME");
        std::cout << "NAME: " << UAV_NAME << std::endl;
        ID = UAV_NAME[5] - '0';
        std::cout << "I'm UAV " << ID << std::endl;

        // ---------- pub/sub ---------
        for (int i = 1; i < ROBOTS_NUM+1; i++)
        {
            odomSubs_.push_back(this->create_subscription<nav_msgs::msg::Odometry>("/Drone" + std::to_string(i) + "/odometry", 1,  [this, i](nav_msgs::msg::Odometry::SharedPtr msg) {this->odomCallback(msg,i);}));
        }
        velPub_ = this->create_publisher<arrc_interfaces::msg::UavVelAcc>("/Drone" + std::to_string(ID) + "/command/setVelocityAcceleration", 1);
        targetPub_ = this->create_publisher<geometry_msgs::msg::Point>("/target_"+std::to_string(ID), 1);
        timer_ = this->create_wall_timer(20ms, std::bind(&FormationNode::loop, this));

        robots.resize(3, ROBOTS_NUM);

        x_target << 0.0, 0.0, 0.0;

        // init CBF controller
        controller.setGamma(1.0, 5.0, 0.1);
        controller.setVelBounds(-3.0, 3.0);
        controller.setVerbose(false);
    }

    ~FormationNode()
    {
        RCLCPP_INFO_STREAM(this->get_logger(), "Shutting down the formation node ....");
    }

    void stop();
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg, int id);
    void loop();

private:
    std::string UAV_NAME = "Drone1";
    int ID;
    int ROBOTS_NUM;
    Eigen::MatrixXd robots;
    Eigen::Vector3d p_i;
    Eigen::Vector3d x_target;
    formation_control::FormationController controller;
    rclcpp::Publisher<arrc_interfaces::msg::UavVelAcc>::SharedPtr velPub_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr targetPub_;
    std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> odomSubs_;
    rclcpp::TimerBase::SharedPtr timer_;
};


void FormationNode::stop()
{
    RCLCPP_INFO_STREAM(this->get_logger(), "shutting down the controller, stopping the robot.");
    this->timer_->cancel();
    rclcpp::sleep_for(100000000ns);

    RCLCPP_INFO_STREAM(this->get_logger(), "controller has been closed and robot has been stopped");
    rclcpp::sleep_for(100000000ns);
}


void FormationNode::loop()
{
    p_i = robots.col(ID-1);
    std::cout << "Im in " << p_i.transpose() << std::endl;
    Eigen::Matrix3d R;
    R << cos(p_i(2)), sin(p_i(2)), 0,
        -sin(p_i(2)), cos(p_i(2)), 0,
        0           , 0          , 1;

    auto time = this->get_clock()->now().seconds();
    std::cout << "time: " << time << std::endl;
    double r_traj = 5.0;
    double w_traj = 0.001;
    x_target(0) = r_traj * cos(2*M_PI*w_traj*time);
    x_target(1) = r_traj * sin(2*M_PI*w_traj*time);
    geometry_msgs::msg::Point pt_msg;
    pt_msg.x = x_target(0);
    pt_msg.y = x_target(1);
    targetPub_->publish(pt_msg);
    std::cout << "Target: " << x_target.transpose() << std::endl;
    Eigen::MatrixXd p_js_i;
    p_js_i.resize(3, ROBOTS_NUM);
    p_js_i = R*(robots.colwise() - p_i);
    
    // Remove col ID-1
    unsigned int numRows = p_js_i.rows();
    unsigned int numCols = p_js_i.cols() - 1;
    unsigned int colToRemove = ID-1;
    if (colToRemove < numCols)
    {
        p_js_i.block(0, colToRemove, numRows, numCols-colToRemove) = p_js_i.block(0, colToRemove+1, numRows, numCols-colToRemove);    
    }
    p_js_i.conservativeResize(numRows, numCols);
    // std::cout << "p_js_i: " << p_js_i.transpose() << std::endl;
    if (p_js_i.rowwise().norm().norm() < 0.1)
    {
        std::cout << "waiting for initialization ... \n";
        return;
    }

    Eigen::Vector3d x_target_local = R * (x_target - p_i);

    Eigen::Vector3d config_centroid;
    config_centroid.setZero();
    for (int i = 0; i < p_js_i.cols(); i++)
    {
        config_centroid += p_js_i.col(i);
    }
    config_centroid = config_centroid / ROBOTS_NUM;

    Eigen::Vector3d uopt, u_star, uopt_global;
    u_star = x_target_local - config_centroid;
    u_star(2) = atan2(x_target_local(1), x_target_local(0));
    std::cout << "udes: " << u_star.transpose() << std::endl;

    arrc_interfaces::msg::UavVelAcc vel_msg;
    Eigen::VectorXd h_out;
    // Dummy obstacles
    Eigen::MatrixXd obs_i;
    obs_i.resize(2,2);
    obs_i.setZero();
    if (!controller.applyCbf(uopt, u_star, p_js_i, obs_i, h_out))
    {
        std::cout << "Local optimal vel: " << uopt.transpose() << std::endl;
        uopt_global.head(2) = R.transpose() * uopt.head(2);
        uopt_global(2) = uopt(2);
        std::cout << "Global optimal vel: " << uopt_global.transpose() << std::endl;
        vel_msg.header.frame_id = "common_origin";
        vel_msg.velocity.x = uopt_global(0);
        vel_msg.velocity.y = uopt_global(1);
        vel_msg.velocity.z = 0.0;
        vel_msg.acceleration.x = std::nan("1");
        vel_msg.acceleration.y = std::nan("1");
        vel_msg.acceleration.z = std::nan("1");
        vel_msg.yaw = std::nan("1");
        vel_msg.yaw_rate = uopt_global(2);
        velPub_->publish(vel_msg);
    } else {
        RCLCPP_INFO_STREAM(this->get_logger(), "CBF FAILED.");
    }

    std::cout << "h_out : " << h_out.transpose() << std::endl;

    

}

void FormationNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg, int id)
{
    tf2::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    robots.col(id-1) << msg->pose.pose.position.x, msg->pose.pose.position.y, yaw;
}




//alternatively to a global variable to have access to the method you can make STATIC the class method interested, 
//but some class function may not be accessed: "this->" method cannot be used

std::shared_ptr<FormationNode> globalobj_signal_handler;     //the signal function requires only one argument {int}, so the class and its methods has to be global to be used inside the signal function.
void nodeobj_wrapper_function(int){
    std::cout<<"signal handler function CALLED"<<std::endl;
    globalobj_signal_handler->stop();
}

int main(int argc, char **argv)
{
    signal(SIGINT, nodeobj_wrapper_function);

    rclcpp::init(argc, argv);
    auto node = std::make_shared<FormationNode>();

    globalobj_signal_handler = node;    //to use the ros function publisher, ecc the global pointer has to point to the same node object.


    rclcpp::spin(node);

    rclcpp::sleep_for(100000000ns);
    rclcpp::shutdown();

    return 0;
}