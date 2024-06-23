#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Odometry.h>
#include <formation_control/FormationController.h>
#include <tf2/utils.h>
#include <algorithm>

// My includes
#include "gmm_coverage/FortuneAlgorithm.h"
#include "gmm_coverage/Voronoi.h"
#include "gmm_coverage/Diagram.h"
#include "gmm_coverage/Graphics.h"

#include <gmm_msgs/Gaussian.h>
#include <gmm_msgs/GMM.h>

#define M_PI   3.14159265358979323846  /*pi*/

//Robots parameters ------------------------------------------------------
const double MAX_ANG_VEL = 1.0;
const double MAX_LIN_VEL = 1.0;         //set to turtlebot max velocities
const double b = 0.025;                 //for differential drive control (only if we are moving a differential drive robot (e.g. turtlebot))
//------------------------------------------------------------------------
const float CONVERGENCE_TOLERANCE = 0.1;
//------------------------------------------------------------------------
const int shutdown_timer = 10;           //count how many seconds to let the robots stopped before shutting down the node



class ClusterNode
{
private:
    ros::Publisher pub;
    ros::Subscriber sub;
    ros::Subscriber detections_sub;
    ros::Subscriber gmmSub_;
    std::vector<ros::Subscriber> agents_subs;
    std::vector<ros::Subscriber> cluster_subs;
    std::vector<ros::Subscriber> neighbors_subs;
    ros::Subscriber target_sub;
    ros::Subscriber obstacles_sub;
    ros::NodeHandle n, nh_;
    ros::Timer timer;
    // Eigen::MatrixXd p_js_i;                 // local
    // Eigen::Vector2d p_target;               // global
    // Eigen::MatrixXd obs, obs_i;                  // global, local
    Eigen::Vector2d p;                      // global position of virtual agent
    Eigen::MatrixXd p_is;                   // global position of cluster agents
    Eigen::MatrixXd p_js;                   // global position of neighbors
    std::vector<Eigen::Vector2d> enemies;
    gmm_msgs::GMM gmm_msg;
    Eigen::MatrixXd agents;

    int ROBOTS_NUM = 3;
    int CLUSTERS_NUM = 4;
    int CLUSTER_ID = 0;
    int ROBOT_ID = 0;
    int OBSTACLES_NUM = 5;
    double ROBOT_FOV = 120.0;
    double ROBOT_RANGE = 20.0;
    double SAFETY_DIST = 2.0;

    double AREA_BOTTOM = -50.0;
    double AREA_LEFT = -50.0;
    double AREA_SIZE_x = 100.0;
    double AREA_SIZE_y = 100.0;

    double GAUSS_X = -5.0;
    double GAUSS_Y = -5.0;
    double GAUSS_COV = 2.0;

    double dt = 0.01;
    bool got_gmm = false;


    

public:


    ClusterNode(): nh_("~")
    {
        std::cout << "Constructor called" << std::endl;
        nh_.getParam("CLUSTER_ID", CLUSTER_ID);
        nh_.getParam("CLUSTERS_NUM", CLUSTERS_NUM);
        nh_.getParam("ROBOTS_NUM", ROBOTS_NUM);
        nh_.getParam("ROBOT_ID", ROBOT_ID);
        nh_.getParam("OBSTACLES_NUM", OBSTACLES_NUM);

        for (int i = 0; i < ROBOTS_NUM; i++)
        {
            // nh_.subscribe<nav_msgs::Odometry>("/hummingbird" + std::to_string(i) + "/ground_truth/odometry", 1, std::bind(&Controller::realposeCallback, this, std::placeholders::_1, i)))
            cluster_subs.push_back(n.subscribe<nav_msgs::Odometry>("/hummingbird" + std::to_string(CLUSTER_ID*ROBOTS_NUM+i) + "/ground_truth/odometry", 1, std::bind(&ClusterNode::odom_callback, this, std::placeholders::_1, i)));
        }

        for (int i = 0; i < CLUSTERS_NUM; i++)
        {
            int  c = i;
            if (i > CLUSTER_ID) {c = i-1;}
            if (i != CLUSTER_ID)
            {
                agents_subs.push_back(n.subscribe<nav_msgs::Odometry>("/cluster" + std::to_string(i) + "/virtual_agent", 1, std::bind(&ClusterNode::agents_callback, this, std::placeholders::_1, c)));
            }
        }
        detections_sub = n.subscribe<geometry_msgs::PoseArray>("/hummingbird" + std::to_string(ROBOT_ID) + "/detections", 1, &ClusterNode::detectionsCallback, this);
        gmmSub_ = n.subscribe<gmm_msgs::GMM>("/gaussian_mixture_model", 1, std::bind(&ClusterNode::gmm_callback, this, std::placeholders::_1));
        
        // sub = n.subscribe<geometry_msgs::Twist> ("/cmd_vel_in", 1, &ClusterNode::vel_callback, this);
        // neighbors_sub = n.subscribe<geometry_msgs::PoseArray>("/neighbors_topic", 1, &ClusterNode::neigh_callback, this);
        // pose_sub = n.subscribe<nav_msgs::Odometry>("odom", 1, &ClusterNode::odom_callback, this);
        // target_sub = n.subscribe<nav_msgs::Odometry>("/target", 1, &ClusterNode::target_callback, this);
        // obstacles_sub = n.subscribe<geometry_msgs::PoseArray>("/obstacles", 1, &ClusterNode::obs_callback, this);
        pub = n.advertise<nav_msgs::Odometry> ("/virtual_agent", 1);
        timer = n.createTimer(ros::Duration(0.1), &ClusterNode::timerCallback,this);

        // Cluster robots
        p_is.resize(2, ROBOTS_NUM);
        p_is.setZero();

        // Virtual agent
        // p.resize(3);

        // Neighbors
        p_js.resize(2, (CLUSTERS_NUM-1)*ROBOTS_NUM);
        p_js.setZero();

        agents.resize(2, CLUSTERS_NUM-1);
        agents.setZero();

        

        // Collection of enemies by all robots in the cluster ()
        // enemies.resize((CLUSTERS_NUM-1)*ROBOTS_NUM*ROBOTS_NUM, Eigen::Vector2d::Zero());

        // Obstacles (static)
        // obs_i.resize(2, OBSTACLES_NUM);
        // obs_i.setOnes();
        // obs_i = obs_i * 100.0;

        // obs.resize(2, OBSTACLES_NUM);
        // obs.setOnes();
        // obs = obs * 100.0;


        std::cout << "Hi! I'm cluster number " << CLUSTER_ID << std::endl;


    }

    void agents_callback(const nav_msgs::Odometry::ConstPtr &msg, int i)
    {
        agents(0, i) = msg->pose.pose.position.x;
        agents(1, i) = msg->pose.pose.position.y;
        // std::cout << "Agent " << i << " in " << agents.col(i).transpose() << std::endl;
    }

    void odom_callback(const nav_msgs::Odometry::ConstPtr &msg, int i)
    {
        p_is(0, i) = msg->pose.pose.position.x;
        p_is(1, i) = msg->pose.pose.position.y;
    }

    void detectionsCallback(const geometry_msgs::PoseArray::ConstPtr &msg)
    {
        for (int i = 0; i < msg->poses.size(); i++)
        {
            p_js(0, i) = msg->poses[i].position.x;
            p_js(1, i) = msg->poses[i].position.y;
        }
    }

    void gmm_callback(const gmm_msgs::GMM::ConstPtr msg)
    {
        this->gmm_msg.gaussians = msg->gaussians;
        this->gmm_msg.weights = msg->weights;
        this->got_gmm = true;
        // std::cout << "Sto ricevendo il GMM\n";
    }



    void timerCallback(const ros::TimerEvent&)
    {
        if (!this->got_gmm) 
        {
            p = p_is.rowwise().mean();
            return;
        }
        // Calculate position of virtual agent
        // p = p_is.rowwise().mean();
        std::cout << "Virtual agent " << CLUSTER_ID << " position: " << p << std::endl;

        // -------------------- Coverage Control ---------------------
        //Variables
        double vel_x=0.0, vel_y=0.0, vel_z = 0.0;
        std::vector<Vector2<double>> seeds;
        std::vector<std::vector<float>> centroids;
        std::vector<double> vel;
        std::vector<float> centroid;
        int K_gain = 1;                  //Lloyd law gain

        Box<double> AreaBox{AREA_LEFT, AREA_BOTTOM, AREA_SIZE_x + AREA_LEFT, AREA_SIZE_y + AREA_BOTTOM};
        Box<double> RangeBox{-2*ROBOT_RANGE, -2*ROBOT_RANGE, 2*ROBOT_RANGE, 2*ROBOT_RANGE};
        std::vector<Box<double>> ObstacleBoxes = {};

        seeds.push_back({p(0), p(1)});
        for (int i = 0; i < agents.cols(); ++i)
        {
            if ((agents(0, i) != 0.0) && (agents(1, i) != 0.0))
            {
                seeds.push_back({agents(0, i), agents(1, i)});    
            }
        }

        std::cout << "Seeds: " << std::endl;
        for (auto s : seeds)
        {
            std::cout << s.x << " " << s.y << std::endl;
        }

        if ((p(0) != 0.0) && (p(1) != 0.0))
        {
            bool robots_stopped = true;

            // -------- Voronoi --------
            // Transform to local coords (ROBOT_ID is always the 1st elem)
            auto local_seeds_i = reworkPointsVector(seeds, seeds[0]);
            std::cout << "Local seeds: " << std::endl;
            for (auto s : local_seeds_i)
            {
                std::cout << s.x << " " << s.y << std::endl;
            }
            auto flt_seeds = filterPointsVector(local_seeds_i, RangeBox);
            std::cout << "Filtered seeds: " << std::endl;
            for (auto s : flt_seeds)
            {
                std::cout << s.x << " " << s.y << std::endl;
            }
            auto diagram = generateDecentralizedDiagram(flt_seeds, RangeBox, seeds[0], ROBOT_RANGE, AreaBox);
            // auto verts = diagram.getVertices();

            std::vector<Vector2<double>> MEANS;
            MEANS.push_back({GAUSS_X, GAUSS_Y});
            std::vector<double> COVS;
            COVS.push_back(GAUSS_COV);

            centroid = computeGMMPolygonCentroid2(diagram, this->gmm_msg, ObstacleBoxes);

            double norm = sqrt(pow(centroid[0], 2) + pow(centroid[1], 2));
            if (norm > CONVERGENCE_TOLERANCE)
            {
                // vel_x = std::max(-0.2, std::min(K_gain * centroid[0], 0.2));
                // vel_y = std::max(-0.2, std::min(K_gain * centroid[1], 0.2));
                vel_x = K_gain * centroid[0];
                vel_y = K_gain * centroid[1];
                vel_x = std::max(-MAX_LIN_VEL, std::min(vel_x, MAX_LIN_VEL));
                vel_y = std::max(-MAX_LIN_VEL, std::min(vel_y, MAX_LIN_VEL));
                robots_stopped = false;
            } else
            {
                vel_x = 0.0;
                vel_y = 0.0;
            }

            // Calculate motion of virtual agent
            p(0) += vel_x * dt;
            p(1) += vel_y * dt;
        } 

        // Publish virtual agent position
        nav_msgs::Odometry vagent_pos;
        vagent_pos.header.stamp = ros::Time::now();
        vagent_pos.header.frame_id = "odom";
        vagent_pos.pose.pose.position.x = p(0);
        vagent_pos.pose.pose.position.y = p(1);
        pub.publish(vagent_pos);

    }


    

};//End of class SubscribeAndPublish

int main(int argc, char **argv)
{
    ros::init(argc, argv, "front_end");
    ClusterNode node;

    ros::spin();

    return 0;
}