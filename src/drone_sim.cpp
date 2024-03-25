// STL
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/impl/utils.h>
#include <functional>
#include <memory>
#include <string>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <netinet/in.h>
#include <sys/types.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <math.h>
#include <signal.h>
// SFML
// #include <SFML/Graphics.hpp>
// #include <SFML/OpenGL.hpp>
// ROS includes
#include <ros/ros.h>
// #include "std_msgs/msg/string.hpp"
// #include "geometry_msgs/msg/point.hpp"
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Odometry.h>
#include <std_srvs/Empty.h>


#define M_PI   3.14159265358979323846  /*pi*/

using namespace std::chrono_literals;
using std::placeholders::_1;

sig_atomic_t volatile node_shutdown_request = 0;    //signal manually generated when ctrl+c is pressed

std::vector<std::string> readFile(std::string path){
    // Create and open a stream to read the file contents by giving the path to the file
    std::ifstream file(path);
    std::vector<std::string> lines;
    std::string line;
    
    if(file.is_open()) {
        // Read the file line by line
        while(getline(file, line)) {
            lines.push_back(line);
        }
        
        file.close();
    }else {
        std::cerr << "Unable to open file: " << path << std::endl;
    }

    return lines;
}

std::vector<double> getNumbers(const std::string &line)
{
    // get numbers even if I dont know the number of digits
    std::vector<double> numbers;
    int start = 0;
    int length = 5;

    std::string n0 = line.substr(start, length);
    std::string n1 = line.substr(start + length + 1, length);
    std::string n2 = line.substr(start + 2*length + 2, length);

    numbers.push_back(std::stod(n0));
    numbers.push_back(std::stod(n1));
    numbers.push_back(std::stod(n2));

    return numbers;
}

class Controller
{

public:
    Controller() : nh_priv_("~")
    {
        //------------------------------------------------- ROS parameters ---------------------------------------------------------
        // ID of the controlled robot
        this->nh_priv_.getParam("ROBOT_ID", ROBOT_ID);

        // Name of the controlled robot
        this->nh_priv_.getParam("NAME", name);

        // File containing positions
        this->nh_priv_.getParam("FILE", filepath);

        

        //--------------------------------------------------- Subscribers and Publishers ----------------------------------------------------    
        odomPub_ = nh_.advertise<nav_msgs::Odometry>("/robot" + std::to_string(ROBOT_ID) + "/odom", 1);
        velSub_ = nh_.subscribe<geometry_msgs::TwistStamped>("/robot" + std::to_string(ROBOT_ID) + "/cmd_vel", 1, std::bind(&Controller::velCallback, this, std::placeholders::_1));
        timer_ = nh_.createTimer(ros::Duration(0.01), std::bind(&Controller::loop, this));

        // Get starting position from file
        std::vector<std::string> lines = readFile(filepath);
        std::vector<double> numbers = getNumbers(lines[ROBOT_ID]);

        px = numbers[0];
        py = numbers[1];
        pth = numbers[2];

        auto I4 = Eigen::Matrix<double, 4, 4>::Identity(4, 4);
        auto zero4 = Eigen::Matrix<double, 4, 4>::Zero(4, 4);

        Ak << I4, dt*I4, zero4, I4;
        Eigen::Matrix<double, 8, 4> Bk;
        Bk << 0.5 * dt * dt * I4, dt * I4;
        double Kp = 2.0;
        double Kd = 0.8;

        Eigen::Matrix<double, 4, 8> K;
        K << Kp * I4, Kd * I4;
        B_fb = Bk * K;
        A_fb = Ak - B_fb;

        // Initialize odom msg
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = name;

        state_vector_ << px, py, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        ref_vector_ = state_vector_;
        ref_vector_(3) = pth;

        // timerstart = 0.0;

        std::cout << "Hi! I'm robot number " << ROBOT_ID << " in position " << px << " " << py << " " << pth << std::endl;
    
    }
    ~Controller()
    {
        std::cout<<"DESTROYER HAS BEEN CALLED"<<std::endl;
    }

    void stop();
    void velCallback(const geometry_msgs::TwistStamped::ConstPtr msg);
    void loop();


private:
    int ROBOT_ID = 0;
    std::string name = "uav";
    double vx = 0.0;
    double vy = 0.0;
    double vth = 0.0;

    double px = 0.0;
    double py = 0.0;
    double pth = 0.0;

    Eigen::Matrix<double, 8, 1> state_vector_, ref_vector_;
    Eigen::Matrix<double, 4, 1> req_pose_vector_;
    Eigen::Matrix<double, 4, 1> req_vel_vector_;
    Eigen::Matrix<double, 8, 8> Ak, A_fb, B_fb;

    double VMAX = 0.5;

    double dt = 0.01;

    std::string filepath = "path.txt";

    double timerstart = 0.0;

    // ROS
    ros::NodeHandle nh_;
    ros::NodeHandle nh_priv_;
    ros::Subscriber velSub_;
    ros::Publisher odomPub_;
    ros::Timer timer_;

    nav_msgs::Odometry odom_msg;

};


void Controller::stop()
{
    //if (signum == SIGINT || signum == SIGKILL || signum ==  SIGQUIT || signum == SIGTERM)
    ROS_INFO("shutting down the controller, stopping the robots, closing the graphics window");

    ROS_INFO("controller has been closed and robots have been stopped");
    ros::Duration(0.1).sleep();

    ros::shutdown();
}

void Controller::velCallback(const geometry_msgs::TwistStamped::ConstPtr msg)
{
    vx = std::max(-VMAX, std::min(msg->twist.linear.x, VMAX));
    vy = std::max(-VMAX, std::min(msg->twist.linear.y, VMAX));
    vth = std::max(-VMAX, std::min(msg->twist.angular.z, VMAX));

    ref_vector_.block<4, 1>(0, 0) = state_vector_.block<4, 1>(0, 0);
    ref_vector_(4) = vx;
    ref_vector_(5) = vy;
    ref_vector_(6) = 0.0;       // vz
    ref_vector_(7) = vth;
}

void Controller::loop()
{
    // if (timerstart == 0.0)
    // {
    //     timerstart = ros::Time::now().toSec();
    //     return;
    // }
    // dt = ros::Time::now().toSec() - timerstart;
    // // Simulate robot motion
    // std::cout << "dt: " << dt << std::endl;
    // px += vx * dt;
    // py += vy * dt;
    // pth += vth * dt;

    ref_vector_ = Ak * ref_vector_;
    state_vector_ = A_fb * state_vector_ + B_fb * ref_vector_;

    // Publish odometry
    odom_msg.header.stamp = ros::Time::now();
    odom_msg.pose.pose.position.x = state_vector_(0);
    odom_msg.pose.pose.position.y = state_vector_(1);
    odom_msg.pose.pose.position.z = state_vector_(2);
    tf2::Quaternion myQuaternion;
    myQuaternion.setRPY(0, 0, state_vector_(3)); // Create this quaternion from roll/pitch/yaw (in radians)
    odom_msg.pose.pose.orientation.w = myQuaternion.getW();
    odom_msg.pose.pose.orientation.x = myQuaternion.getX();
    odom_msg.pose.pose.orientation.y = myQuaternion.getY();
    odom_msg.pose.pose.orientation.z = myQuaternion.getZ();

    odomPub_.publish(odom_msg);

    // timerstart = ros::Time::now().toSec();
}


//alternatively to a global variable to have access to the method you can make STATIC the class method interested, 
//but some class function may not be accessed: "this->" method cannot be used

std::shared_ptr<Controller> globalobj_signal_handler;     //the signal function requires only one argument {int}, so the class and its methods has to be global to be used inside the signal function.
void nodeobj_wrapper_function(int){
    std::cout<<"signal handler function CALLED"<<std::endl;
    node_shutdown_request = 1;
}

int main(int argc, char **argv)
{
    // signal(SIGINT, nodeobj_wrapper_function);

    ros::init(argc, argv, "drones_sim2");
    auto node = std::make_shared<Controller>();

    // globalobj_signal_handler = node;    //to use the ros function publisher, ecc the global pointer has to point to the same node object.


    // rclcpp::spin(node);

    // rclcpp::sleep_for(100000000ns);
    // rclcpp::shutdown();

    
    while(!node_shutdown_request)
    {
        ros::spinOnce();
    }
    node->stop();

    if (ros::ok())
    {
        ROS_WARN("ROS HAS NOT BEEN PROPERLY SHUTDOWN, it is being shutdown again.");
        ros::shutdown();
    }
    

    return 0;
}
