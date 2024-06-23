// STL
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include <cstdlib>
#include <cmath>
#include <netinet/in.h>
#include <sys/types.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
// My includes
#include "FortuneAlgorithm.h"
#include "Voronoi.h"
// ROS includes
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Odometry.h>
#include <std_srvs/Empty.h>
#include <signal.h>
// #include "coverage_unimore_nyu/PoseVector.h"

//------------------------------------------------------------------------
const int debug = 0;
//------------------------------------------------------------------------
const int shutdown_timer = 5;           //count how many seconds to let the robots stopped before shutting down the node
//------------------------------------------------------------------------
//------------------------------------------------------------------------
sig_atomic_t volatile node_shutdown_request = 0;    //signal manually generated when ctrl+c is pressed
//------------------------------------------------------------------------

bool IsPathExist(const std::string &s)
{
  struct stat buffer;
  return (stat (s.c_str(), &buffer) == 0);
}

class Controller
{
public:
    Controller() : nh_priv_("~")
    {
        ROS_INFO("Node Initialization");
        //-------------------------------------------------------- ROS parameters -----------------------------------------------------------
        this->nh_priv_.getParam("ids", ids_);
        this->nh_priv_.getParam("ROBOTS_NUM", ROBOTS_NUM);
        this->nh_priv_.getParam("SAVE_POS", SAVE_POS);
        this->nh_priv_.getParam("WAIT_INIT", WAIT_INIT);

        //sensing range single robot (= halfe edge of the local sensing box)
        this->nh_priv_.getParam("ROBOT_RANGE", ROBOT_RANGE);
        this->nh_priv_.getParam("ROBOT_FOV", ROBOT_FOV);

        // Parameters for Gaussian

        // Area parameter
        this->nh_priv_.getParam("AREA_SIZE_x", AREA_SIZE_x);
        this->nh_priv_.getParam("AREA_SIZE_y", AREA_SIZE_y);
        this->nh_priv_.getParam("AREA_LEFT", AREA_LEFT);
        this->nh_priv_.getParam("AREA_BOTTOM", AREA_BOTTOM);

        //view graphical voronoi rapresentation - bool
        //-----------------------------------------------------------------------------------------------------------------------------------
        //--------------------------------------------------- Subscribers and Publishers ----------------------------------------------------
        for (int i = 0; i < ROBOTS_NUM; i++)
        {
            // TODONYU: change the topic name to match the one you use
            poseSub_.push_back(nh_.subscribe<nav_msgs::Odometry>("/turtlebot" + std::to_string(i) + "/odom", 1, std::bind(&Controller::poseCallback, this, i, std::placeholders::_1)));
            neighposePub_.push_back(nh_.advertise<geometry_msgs::PoseArray>("/supervisor/robot" + std::to_string(i) + "/pose", 1));
        }
        controller_timer_ = nh_.createTimer(ros::Duration(0.01), std::bind(&Controller::Emulate_Vision, this));
        init_srv_ = nh_.advertiseService("/init", &Controller::initCallback, this);
        //-----------------------------------------------------------------------------------------------------------------------------------
        //----------------------------------------------------------- init Variables ---------------------------------------------------------
        pose_x = Eigen::VectorXd::Zero(ROBOTS_NUM);
        pose_y = Eigen::VectorXd::Zero(ROBOTS_NUM);
        pose_theta = Eigen::VectorXd::Zero(ROBOTS_NUM);
        //------------------------------------------------------------------------------------------------------------------------------------

        //----------------------------------------------- Graphics window -----------------------------------------------
        //---------------------------------------------------------------------------------------------------------------
        //--------------------------------------------------open log file -----------------------------------------------
        //---------------------------------------------------------------------------------------------------------------
        ROBOT_FOV_rad = 0.5 * ROBOT_FOV /180 * M_PI;
        counter = 0;
        // myfile.open("/home/mattia/.txt");

        if (SAVE_POS)
        {
            this->open_log_file();
        }

    }

    ~Controller()
    {
        std::cout<<"DESTROYER HAS BEEN CALLED"<<std::endl;
        //ros::shutdown();
    }

    void stop();
    void poseCallback(int i, const nav_msgs::Odometry::ConstPtr &msg);
    bool initCallback(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
    {
        if (initCounter < 50)
        {
            initCounter++;
            return false;
        }
        ROS_INFO("INITIALIZING ROBOT COVERAGE SUPERVISOR");
        justStarted = false;
        return true;
    }
    void Emulate_Vision();

    //Graphics -- draw coverage

    //open write and close LOG file
    void open_log_file();
    void write_log_file(std::string text);
    void close_log_file();

private:
    std::vector<int> ids_{1,2,3};
    int ROBOTS_NUM = 3;
    double ROBOT_RANGE = 5;
    double ROBOT_FOV = 100.0;
    double ROBOT_FOV_rad;
    bool SAVE_POS = false;
    bool WAIT_INIT = false;
    bool justStarted = true;
    double vel_linear_x, vel_angular_z;
    Eigen::VectorXd pose_x;
    Eigen::VectorXd pose_y;
    Eigen::VectorXd pose_theta;
    std::vector<Vector2<double>> seeds_xy;

    

    //------------------------------- Ros NodeHandle ------------------------------------
    ros::NodeHandle nh_;
    ros::NodeHandle nh_priv_;
    //-----------------------------------------------------------------------------------
    //------------------------- Publishers and subscribers ------------------------------
    std::vector<ros::Publisher> neighposePub_;
    std::vector<ros::Subscriber> poseSub_;
    ros::ServiceServer init_srv_;
    ros::Timer controller_timer_;
    int initCounter = 0;
    //-----------------------------------------------------------------------------------

    //Rendering with SFML
    //------------------------------ graphics window -------------------------------------
    //------------------------------------------------------------------------------------

    //---------------------------- Environment definition --------------------------------
    double AREA_SIZE_x = 20.0;
    double AREA_SIZE_y = 20.0;
    double AREA_LEFT = -10.0;
    double AREA_BOTTOM = -10.0;

    int counter;
    //------------------------------------------------------------------------------------

    //---------------------- Gaussian Density Function parameters ------------------------
    //------------------------------------------------------------------------------------

    //graphical view - ON/OFF

    //timer - check how long robots are being stopped

    //ofstream on external log file
    std::ofstream log_file;
};

void Controller::stop()
{
    ROS_INFO("shutting down the supervisor controller");
    
    // if (SAVE_POS)
    // {
    //     ROS_INFO("Saving final position of robots");
    //     std::ofstream myfile;
    //     myfile.open ("/home/mattia/final_pos.txt");
    //     for (int i = 0; i < ROBOTS_NUM; i++)
    //     {
    //         myfile << pose_x(i) << " " << pose_y(i) << " " << pose_theta(i) << std::endl;
    //     }
    //     myfile.close();
    //     ROS_INFO("Final position of robots saved");
    // }
    if (SAVE_POS)
    {
        std::cout << "Saving final position of robots ...\n";
        for (int i = 0; i < ROBOTS_NUM; i++)
        {
            std::string txt = std::to_string(this->pose_x(i)) + " " + std::to_string(this->pose_y(i)) + " " + std::to_string(this->pose_theta(i)) + "\n";
            this->write_log_file(txt);
        }
        std::cout << "Final position of robots saved. Closing file.\n";
        
        this->close_log_file();
    }
    

    ros::Duration(0.1).sleep();

    ros::shutdown();
}

void Controller::open_log_file()
{
    std::time_t t = time(0);
    struct tm * now = localtime(&t);
    char buffer [80];

    char *dir = get_current_dir_name();
    std::string dir_str(dir);
    dir_str = dir_str + "/pf_logs";

    if (IsPathExist(dir_str))     //check if the folder exists
    {
        strftime (buffer,80,"/%Y_%m_%d_%H-%M_logfile.txt",now);
    } else {
        system(("mkdir " + (dir_str)).c_str());
        strftime (buffer,80,"/%Y_%m_%d_%H-%M_logfile.txt",now);
    }

    std::cout<<"file name :: "<<dir_str + buffer<<std::endl;
    this->log_file.open(dir_str + buffer,std::ofstream::app);
}

void Controller::write_log_file(std::string text)
{
    if (this->log_file.is_open())
    {
        // std::cout << text;
        this->log_file << text;
    }
}

void Controller::close_log_file()
{
    std::cout<<"Log file is being closed"<<std::endl;
    this->log_file.close();
}

// void Controller::initCallback(const std_srvs::Empty::Request::ConstPtr &req, const std_srvs::Empty::Response::ConstPtr &res)
// {
//     ROS_INFO("INITIALIZING ROBOT COVERAGE SUPERVISOR");
//     justStarted = false;
// }

void Controller::poseCallback(int i, const nav_msgs::Odometry::ConstPtr &msg)
{
    this->pose_x(i) = msg->pose.pose.position.x;
    this->pose_y(i) = msg->pose.pose.position.y;  

    this->pose_theta(i) = tf2::getYaw(msg->pose.pose.orientation);;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-------------------------------Rendering functions - for SFML graphical visualization-----------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void Controller::Emulate_Vision(){

    Box<double> AreaBox{AREA_LEFT, AREA_BOTTOM, AREA_SIZE_x + AREA_LEFT, AREA_SIZE_y + AREA_BOTTOM};
    Box<double> RangeBox{-ROBOT_RANGE, -ROBOT_RANGE, ROBOT_RANGE, ROBOT_RANGE};

    // fake position for non detected robots
    geometry_msgs::Pose fake_pose;
    fake_pose.position.x = 100.0;
    fake_pose.position.y = 100.0;
    fake_pose.position.z = 0.0;

    for (int i = 0; i < ROBOTS_NUM; ++i)
    {
        geometry_msgs::PoseArray neighbors;
        
        neighbors.header.stamp = ros::Time::now();
        std::stringstream ss;
        ss << "hummingbird"<< i<<"/base_link";
        neighbors.header.frame_id = ss.str();

        if (WAIT_INIT && justStarted)
        {
            for (int j = 0; j < ROBOTS_NUM; ++j)
            {
                if (i != j)
                {
                    double dist_x = this->pose_x(j) - this->pose_x(i);
                    double dist_y = this->pose_y(j) - this->pose_y(i);
                    double dist = sqrt(pow(dist_x,2) + pow(dist_y,2));
                    geometry_msgs::Pose msg;
                    msg.position.x = dist_x * cos(this->pose_theta(i)) + dist_y * sin(this->pose_theta(i));
                    msg.position.y = -dist_x * sin(this->pose_theta(i)) + dist_y * cos(this->pose_theta(i));
                    msg.orientation.w = 1.0;
                    neighbors.poses.push_back(msg);
                }
            }
        } else
        {
            for (int j = 0; j < ROBOTS_NUM; ++j)
            {
                if (i != j)
                {
                    // Vector2<double> distance_vect = {(this->pose_x[j] - this->pose_x[i]), (this->pose_y[j] - this->pose_y[i])};
                    double dist_x = this->pose_x(j) - this->pose_x(i);
                    double dist_y = this->pose_y(j) - this->pose_y(i);
                    double dist = sqrt(pow(dist_x,2) + pow(dist_y,2));

                    if (dist <= ROBOT_RANGE)
                    {
                        geometry_msgs::Pose msg;
                        //the distance_vect is already the point neighbor expressed in local coordinates
                        // msg.position.x = cos(this->pose_theta(i))*distance_vect.x + sin(this->pose_theta(i))*distance_vect.y;
                        // msg.position.y = -sin(this->pose_theta(i))*distance_vect.x + cos(this->pose_theta(i))*distance_vect.y;
                        msg.position.x = dist_x * cos(this->pose_theta(i)) + dist_y * sin(this->pose_theta(i));
                        msg.position.y = -dist_x * sin(this->pose_theta(i)) + dist_y * cos(this->pose_theta(i));
                        msg.orientation.w = 1.0;

                        // Filter robots that are not in the FOV
                        // double angle = abs(atan2(msg.position.y, msg.position.x));
                        // if (angle <= ROBOT_FOV_rad &&  msg.position.x > 0.0)
                        //     {neighbors.poses.push_back(msg);}
                        // else
                        // {
                        //     // Filter robots outside FOV
                        //     neighbors.poses.push_back(fake_pose);
                        // }
                        neighbors.poses.push_back(msg);
                    } else {
                        // Filter robots outside range
                        neighbors.poses.push_back(fake_pose);
                    }
                } else
                {
                    // Fake pose in self position
                    neighbors.poses.push_back(fake_pose);
                }
            }
        }

        // save position
        if (SAVE_POS)
        {
            std::string txt = std::to_string(this->pose_x(i)) + " " + std::to_string(this->pose_y(i)) + " " + std::to_string(this->pose_theta(i)) + "\n";
            this->write_log_file(txt);
        }

        

        std::cout<<"robot "<<i<<" has "<<neighbors.poses.size()<<" neighbours"<<std::endl;
        this->neighposePub_[i].publish(neighbors);
    }
    counter++;
}

/*******************************************************************************
* Main function
*******************************************************************************/
//alternatively to a global variable to have access to the method you can make STATIC the class method interested, 
//but some class function may not be accessed: "this->" method cannot be used

void nodeobj_wrapper_function(int){
    ROS_WARN("signal handler function CALLED");
    node_shutdown_request = 1;
}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "robot_coverage_supervisor", ros::init_options::NoSigintHandler);
    signal(SIGINT, nodeobj_wrapper_function);

    //Controller node_controller;
    auto node_controller = std::make_shared<Controller>();

    while (!node_shutdown_request){
        ros::spinOnce();
    }
    node_controller->stop();

    //ros::spin();
    //do pre-shutdown tasks
    if (ros::ok())
    {
        ROS_WARN("ROS HAS NOT BEEN PROPERLY SHUTDOWN, it is being shutdown again.");
        ros::shutdown();
    }

    return 0;
}