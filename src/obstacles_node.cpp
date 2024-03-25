#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Odometry.h>
#include <formation_control/FormationController.h>
#include <tf2/utils.h>

#define M_PI   3.14159265358979323846  /*pi*/

class ObstaclesNode
{
private:
    ros::Publisher pub;
    ros::NodeHandle n, nh_;
    ros::Timer timer;

    std::vector<double> obs_x;
    std::vector<double> obs_y;

    int OBSTACLES_NUM = 1;


    

public:


    ObstaclesNode(): nh_("~")
    {
        nh_.getParam("OBSTACLES_NUM", OBSTACLES_NUM);
        pub = n.advertise<geometry_msgs::PoseArray> ("/obstacles", 1);
        timer = n.createTimer(ros::Duration(0.01), &ObstaclesNode::timerCallback,this);

        obs_x.resize(OBSTACLES_NUM);
        obs_y.resize(OBSTACLES_NUM);
        
        for (int i = 0; i < OBSTACLES_NUM; i++)
        {
            std::string x = "x" + std::to_string(i);
            std::string y = "y" + std::to_string(i);
            nh_.getParam(x, obs_x[i]);
            nh_.getParam(y, obs_y[i]);
        }
    }


    void timerCallback(const ros::TimerEvent&)
    {
        geometry_msgs::PoseArray obs_msg;
        obs_msg.header.stamp = ros::Time::now();
        obs_msg.header.frame_id = "odom";
        for (int i = 0; i < OBSTACLES_NUM; i++)
        {
            geometry_msgs::Pose obs;
            obs.position.x = obs_x[i];
            obs.position.y = obs_y[i];
            obs.position.z = 0;
            obs.orientation.x = 0;
            obs.orientation.y = 0;
            obs.orientation.z = 0;
            obs.orientation.w = 1;
            obs_msg.poses.push_back(obs);
        }
        pub.publish(obs_msg);
    }


};//End of class SubscribeAndPublish

int main(int argc, char **argv)
{
    ros::init(argc, argv, "obs_node");
    ObstaclesNode node;

    ros::spin();

    return 0;
}