#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <vector>


// ROS
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2/utils.h>

// Voronoi
#include "gmm_coverage/Voronoi.h"
#include "gmm_coverage/Diagram.h"
#include "gmm_coverage/Box.h"

#define M_PI   3.14159265358979323846  /*pi*/

double VMAX = 1.0;
double WMAX = 1.0;

class ExplorationNode
{
private:
    ros::NodeHandle n, nh_;
    ros::Timer timer;
    
    std::vector<ros::Publisher> velPubs;
    std::vector<ros::Subscriber> odomSubs;
    ros::Subscriber gp_sub;
    ros::Publisher test_pub;
    ros::Publisher voro_pub;
    ros::Publisher centr_pub;

    nav_msgs::OccupancyGrid test_map;
    


    int ROBOTS_NUM = 3;
    double AREA_W = 20.0;
    double ROBOT_RANGE = 5.0;
    double FOV_DEG = 120.0;
    double FOV_RAD;
    int GRID_SIZE = 50;
    double RESOLUTION;

    Eigen::MatrixXd robots;
    Eigen::MatrixXd gp;


public:

    ExplorationNode(): nh_("~")
    {
        nh_.getParam("ROBOTS_NUM", ROBOTS_NUM);
        nh_.getParam("AREA_W", AREA_W);
        nh_.getParam("ROBOT_FOV", FOV_DEG);
        nh_.getParam("ROBOT_RANGE", ROBOT_RANGE);
        nh_.getParam("GRID_SIZE", GRID_SIZE);
        FOV_RAD = FOV_DEG * M_PI / 180.0;

        for (int i = 0; i < ROBOTS_NUM; i++)
        {
            odomSubs.push_back(n.subscribe<nav_msgs::Odometry>("/hummingbird" + std::to_string(i)+"/ground_truth/odometry", 1, std::bind(&ExplorationNode::odom_callback, this, std::placeholders::_1, i)));
            velPubs.push_back(n.advertise<geometry_msgs::TwistStamped>("/hummingbird"+std::to_string(i)+"/autopilot/velocity_command", 1));
        }

        RESOLUTION = AREA_W / GRID_SIZE;
        std::cout << "Robots num: " << ROBOTS_NUM << std::endl;

        gp_sub = n.subscribe<nav_msgs::OccupancyGrid>("/posterior_map", 1, &ExplorationNode::gp_callback, this);
        test_pub = n.advertise<nav_msgs::OccupancyGrid>("/test_map", 1);
        voro_pub = n.advertise<visualization_msgs::MarkerArray>("/voronoi", 1);
        centr_pub = n.advertise<visualization_msgs::MarkerArray>("/centroid", 1);
        timer = n.createTimer(ros::Duration(0.1), &ExplorationNode::timer_callback,this);

        // init robots
        robots.resize(ROBOTS_NUM, 3);
        robots.setZero();

        // Probability values of discrete cells
        gp.resize(GRID_SIZE, GRID_SIZE);
        gp.setZero();

        // Init test map
        test_map.header.frame_id = "odom";
        test_map.info.resolution = RESOLUTION;
        test_map.info.width = GRID_SIZE;
        test_map.info.height = GRID_SIZE;
        test_map.info.origin.position.x = -0.5*AREA_W;
        test_map.info.origin.position.y = -0.5*AREA_W;
        for (int i = 0; i < GRID_SIZE*GRID_SIZE; i++)
        {
            test_map.data.push_back(0);
        }

        std::cout << "Init completed." << std::endl;

    
    }


    void publishVoro(const std::vector<Diagram<double>> &diagrams)
    {
        visualization_msgs::MarkerArray voro_msg;
        for (int i = 0; i < diagrams.size(); i++)
        {
            visualization_msgs::Marker diagram_msg;
            diagram_msg.header.frame_id = "odom";
            diagram_msg.id = i;
            diagram_msg.type = 4;          // Linestrip
            diagram_msg.action = visualization_msgs::Marker::ADD;
            diagram_msg.scale.x = 0.2;
            diagram_msg.color.a = 1.0;
            diagram_msg.color.r = 0.0;
            diagram_msg.color.g = 1.0;
            diagram_msg.color.b = 0.0;

            auto halfEdge = diagrams[i].getFaces().at(0).outerComponent;
            geometry_msgs::Point tmp_p;
            // resizing every time because the library does not expose the number of vertices
            do
            {
                tmp_p.x = halfEdge->origin->point.x;
                tmp_p.y = halfEdge->origin->point.y;
                // tmp_p.z = 0.0;
                diagram_msg.points.push_back(tmp_p);
                halfEdge = halfEdge->next; 
                
            } while (halfEdge != diagrams[i].getFaces().at(0).outerComponent);   

            diagram_msg.points.push_back(diagram_msg.points[0]);                // close the polygon

            /*
            for (const auto& site : diagrams[i].getSites())
            {
                Vector2<double> center = site.point;
                auto face = site.face;
                auto halfEdge = face->outerComponent;
                if (halfEdge == nullptr) {continue;}
            }

            while(halfEdge->prev != nullptr)
            {
                halfEdge = halfEdge->prev;
                if (halfEdge == face->outerComponent) {break;}
            }
            auto start = halfEdge;
            while (halfEdge != nullptr)
            {
                if (halfEdge->origin != nullptr && halfEdge->destination != nullptr)
                {
                    auto origin = halfEdge->origin->point;
                    geometry_msgs::Point tmp_p;
                    tmp_p.x = origin.x;
                    tmp_p.y = origin.y;
                    diagram_msg.points.push_back(tmp_p);
                    
                }
                halfEdge = halfEdge->next;
                if(halfEdge == start)
                {
                    geometry_msgs::Point tmp_p;
                    tmp_p.x = origin.x;
                    tmp_p.y = origin.y;
                    diagram_msg.points.push_back(tmp_p);
                    break;
                }
            }
            */

            voro_msg.markers.push_back(diagram_msg);

        }

        voro_pub.publish(voro_msg);
    }

    void publishCentroid(const std::vector<Vector2<double>> centroids)
    {
        visualization_msgs::MarkerArray msg;
        for (int i = 0; i < centroids.size(); i++)
        {
            visualization_msgs::Marker tmp_msg;
            tmp_msg.header.frame_id = "odom";
            tmp_msg.id = i+2;
            tmp_msg.type = 1;          // Cube
            tmp_msg.action = visualization_msgs::Marker::ADD;
            tmp_msg.scale.x = 0.2;
            tmp_msg.scale.y = 0.2;
            tmp_msg.scale.z = 0.2;
            tmp_msg.color.a = 1.0;
            tmp_msg.color.r = 0.0;
            tmp_msg.color.g = 0.0;
            tmp_msg.color.b = 1.0;
            tmp_msg.pose.position.x = centroids[i].x;
            tmp_msg.pose.position.y = centroids[i].y;
            msg.markers.push_back(tmp_msg);
        }

        centr_pub.publish(msg);

    }


    void odom_callback(const nav_msgs::Odometry::ConstPtr &msg, int i)
    {
        robots(i, 0) = msg->pose.pose.position.x;
        robots(i, 1) = msg->pose.pose.position.y;
        tf2::Quaternion q(
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        robots(i, 2) = yaw;
    }

    void gp_callback(const nav_msgs::OccupancyGrid::ConstPtr &msg)
    {
        for (int i = 0; i < msg->data.size(); i++)
        {
            auto dv = std::div(i, GRID_SIZE);       // row = quotient, col = rem
            gp(dv.quot, dv.rem) = msg->data[i];
        }
        std::cout << "Max prob: " << gp.maxCoeff() << std::endl;
    }
    
    void timer_callback(const ros::TimerEvent&)
    {
        if (gp.isZero(0))
        {
            return;
        }
        // ------------------- Coverage --------------
        std::vector<Vector2<double>> seeds;
        Vector2<double> centroid;
        Box<double> AreaBox{-0.5*AREA_W, -0.5*AREA_W, AREA_W, AREA_W};

        for (int i = 0; i < ROBOTS_NUM; ++i)
        {
            seeds.push_back({robots(i, 0), robots(i, 1)});
        }

        bool all_robots_stopped = true;

        std::vector<Diagram<double>> diagrams = generateDecentralizedDiagrams(seeds, AreaBox, AREA_W, AreaBox);

        // for (auto diagram : diagrams)
        // {
        // int robot_id = 0;
        std::vector<Vector2<double>> centr;
        for (int robot_id = 0; robot_id < ROBOTS_NUM; robot_id++)
        {
            auto seed = diagrams[robot_id].getSites()[0];
            auto halfedge = seed.face->outerComponent;

            //trova gli estremi del rettangolo che contengono il poligono
            double x_inf = std::min(halfedge->origin->point.x, halfedge->destination->point.x);
            double x_sup = std::max(halfedge->origin->point.x, halfedge->destination->point.x);
            double y_inf = std::min(halfedge->origin->point.y, halfedge->destination->point.y);
            double y_sup = std::max(halfedge->origin->point.y, halfedge->destination->point.y);
            halfedge = halfedge->next;

            do{
                //------------------ x component --------------------
                if (x_inf > halfedge->destination->point.x)
                {
                    x_inf = halfedge->destination->point.x;

                } else if (x_sup < halfedge->destination->point.x)
                {
                    x_sup = halfedge->destination->point.x;
                }

                //------------------ y component --------------------
                if (y_inf > halfedge->destination->point.y)
                {
                    y_inf = halfedge->destination->point.y;
                } else if (y_sup < halfedge->destination->point.y)
                {
                    y_sup = halfedge->destination->point.y;
                }

                halfedge = halfedge->next;

            } while (halfedge != seed.face->outerComponent);

            /* ---------------- Voronoi RViz ---------------------
            std::vector<Vector2<double>> verts;
            verts.push_back(Vector2<double>{x_inf, y_inf});
            verts.push_back(Vector2<double>{x_sup, y_inf});
            verts.push_back(Vector2<double>{x_sup, y_sup});
            verts.push_back(Vector2<double>{x_inf, y_sup});
            publishVoro(verts);
            */

            

            double dx = (x_sup - x_inf)/2.0 * RESOLUTION;
            double dy = (y_sup - y_inf)/2.0 * RESOLUTION;
            double dA = dx*dy;
            double A = 0;
            double Cx = 0, Cy = 0;


            int i_max = (x_sup + 0.5*AREA_W) / RESOLUTION;
            int i_min = (x_inf + 0.5*AREA_W) / RESOLUTION;
            int j_max = (y_sup + 0.5*AREA_W) / RESOLUTION;
            int j_min = (y_inf + 0.5*AREA_W) / RESOLUTION;

            for (int i = i_min; i < i_max; i++)
            {
                for (int j = j_min; j < j_max; j++)
                {
                    double x_i = i*RESOLUTION - 0.5*AREA_W - robots(robot_id, 0);
                    double y_i = j*RESOLUTION - 0.5*AREA_W - robots(robot_id, 1);
                    bool in = inPolygon(diagrams[robot_id], Vector2<double> {x_i, y_i});
                    if (in)
                    {
                        // double dA_pdf = test_map.data[i+j*GRID_SIZE];
                        double dA_pdf = gp(i, j);
                        A = A + dA_pdf;
                        Cx = Cx + x_i*dA_pdf;
                        Cy = Cy + y_i*dA_pdf;
                    }
                }
            }


            Cx = Cx / A;
            Cy = Cy / A;

            // std::cout << "Centroid: " << Cx << ", " << Cy << std::endl;
            centr.push_back(Vector2<double>{Cx, Cy});

            // Cartesian velocity
            Eigen::Vector2d vel_xy;
            vel_xy(0) = Cx - robots(robot_id, 0);
            vel_xy(1) = Cy - robots(robot_id, 1);

            // SFL
            double b = 0.5*ROBOT_RANGE;
            double Kp = 0.8;
            Eigen::Matrix2d T;
            T << cos(robots(robot_id, 2)), sin(robots(robot_id, 2)),
                -1/b*sin(robots(robot_id, 2)), 1/b*cos(robots(robot_id, 2));

            Eigen::VectorXd vel = T * vel_xy;
            double v = std::max(-VMAX, std::min(VMAX, Kp*vel(0)));
            double w = std::max(-WMAX, std::min(WMAX, Kp*vel(1)));

            

            geometry_msgs::TwistStamped vel_msg;
            vel_msg.header.stamp = ros::Time::now();
            // vel_msg.header.frame_id = "odom";
            vel_msg.twist.linear.x = v*cos(robots(robot_id, 2));
            vel_msg.twist.linear.y = v*sin(robots(robot_id, 2));
            vel_msg.twist.angular.z = w;
            velPubs[robot_id].publish(vel_msg);

            
        }
        publishCentroid(centr);
        publishVoro(diagrams);




        // test_pub.publish(test_map);


    }

};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "expl_node");
    ExplorationNode node;

    ros::spin();

    return 0;
}