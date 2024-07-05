#!/usr/bin/env python3

import numpy as np
import os
import math
import random

# Coverage stuff
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely import Polygon, intersection
from shapely import Point as Pt
from tqdm import tqdm
from pathlib import Path
import pyvoro

# ROS imports
import rospy
from geometry_msgs.msg import Point, Twist, Pose, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Float32, Int32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion

VMAX = 1.0
WMAX = 1.0

class ExplorationNodepy():
    def __init__(self):
        rospy.init_node("exploration_node")
        
        # Params
        self.AREA_W = rospy.get_param("~AREA_W", 20.0)
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.FOV_DEG = rospy.get_param("~ROBOT_FOV", 120.0)
        self.ROBOTS_NUM = rospy.get_param('~ROBOTS_NUM', 3)
        self.FOV_RAD = self.FOV_DEG * math.pi / 180.0
        self.ROBOT_RANGE = rospy.get_param("~ROBOT_RANGE", 3.0)
        self.dt = rospy.get_param("~dt", 0.5)
        self.GRID_SIZE = rospy.get_param("~GRID_SIZE", 50)
        self.RESOLUTION = self.AREA_W / self.GRID_SIZE

        # pubs/subs
        self.odomSubs = [rospy.Subscriber(f"/hummingbird{i}/ground_truth/odometry", Odometry, self.odom_callback, i) for i in range(self.ROBOTS_NUM)]
        self.velPubs = [rospy.Publisher(f"/hummingbird{i}/autopilot/velocity_command", TwistStamped, queue_size=1) for i in range(self.ROBOTS_NUM)]

        self.gp_sub = rospy.Subscriber("/posterior_map", OccupancyGrid, self.gp_callback)
        self.voro_pub = rospy.Publisher("/voronoi", Marker, queue_size=1)
        self.centr_pub = rospy.Publisher("/centroid", Marker, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.timer_callback)


        # init robots
        self.robots = np.zeros((self.ROBOTS_NUM, 3))

        # init GP
        self.gp = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        self.gp_init = False

        print("Init completed. ")
    

    def odom_callback(self, msg, i):
        self.robots[i, 0] = msg.pose.pose.position.x
        self.robots[i, 1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        (roll, pitch, yaw) = euler_from_quaternion(quat)
        self.robots[i, 2] = yaw

    
    def gp_callback(self, msg):
        for i in range(len(msg.data)):
            q = i // self.GRID_SIZE
            r = i % self.GRID_SIZE
            self.gp[q, r] = msg.data[i]
        
        self.gp_init = True
    
    def publishCentroid(self, centroid, id=None):
        tmp_msg = Marker()
        tmp_msg.header.frame_id = self.frame_id
        tmp_msg.id = self.ROBOTS_NUM+id if id is not None else np.random.randint(2, 100)
        tmp_msg.type = 1
        tmp_msg.action = Marker.ADD
        tmp_msg.scale.x = 0.2
        tmp_msg.scale.y = 0.2
        tmp_msg.scale.z = 0.2
        tmp_msg.color.a = 1.0
        tmp_msg.color.r = 0.0
        tmp_msg.color.g = 0.0
        tmp_msg.color.b = 1.0
        tmp_msg.pose.position.x = centroid[0]
        tmp_msg.pose.position.y = centroid[1]

        self.centr_pub.publish(tmp_msg)


    def publishVoroSingle(self, vertices, id=None):
        voro_msg = Marker()
        voro_msg.header.frame_id = self.frame_id
        voro_msg.id = id if id is not None else np.random.randint(2, 100)
        voro_msg.type = 4
        voro_msg.action = Marker.ADD
        voro_msg.scale.x = 0.2
        voro_msg.color.a = 1.0
        voro_msg.color.r = 0.0
        voro_msg.color.g = 1.0
        voro_msg.color.b = 0.0
        for v in vertices:
            tmp_p = Point()
            tmp_p.x = v[0]
            tmp_p.y = v[1]
            voro_msg.points.append(tmp_p)
        
        voro_msg.points.append(voro_msg.points[0])
        self.voro_pub.publish(voro_msg)      




    def timer_callback(self, e):
        if not self.gp_init:
            print("GP not initialized")
            return

        cells = pyvoro.compute_2d_voronoi(self.robots[:, :2], [[-0.5*self.AREA_W, 0.5*self.AREA_W],[-0.5*self.AREA_W, 0.5*self.AREA_W]], 2.0)
        for idx in range(self.ROBOTS_NUM):
            verts = cells[idx]["vertices"]
            self.publishVoroSingle(verts, idx)
            polygon = Polygon(verts)
            xmin, ymin, xmax, ymax = polygon.bounds

            dx = (xmax - xmin) / 2 * self.RESOLUTION
            dy = (ymax - ymin) / 2 * self.RESOLUTION

            i_max = int(xmax + 0.5*self.AREA_W / self.RESOLUTION)
            i_min = int(xmin + 0.5*self.AREA_W / self.RESOLUTION)
            j_max = int(ymax + 0.5*self.AREA_W / self.RESOLUTION)
            j_min = int(ymin + 0.5*self.AREA_W / self.RESOLUTION)

            

            dA = dx*dy
            A = 0.0
            Cx = 0.0; Cy = 0.0

            pts_count = 0
            for i in np.arange(i_min, i_max):
                for j in np.arange(j_min, j_max):
                    pts_count += 1
                    x_i = i*self.RESOLUTION - 0.5*self.AREA_W # - self.robots[idx, 0]
                    y_i = j*self.RESOLUTION - 0.5*self.AREA_W # - self.robots[idx, 1]
                    if polygon.contains(Pt(x_i, y_i)):
                        dA_pdf = self.gp[i,j]
                        A = A + dA_pdf
                        Cx = Cx + x_i * dA_pdf
                        Cy = Cy + y_i * dA_pdf

            Cx = Cx / A
            Cy = Cy / A
            self.publishCentroid(np.array([Cx, Cy]), idx)

            # Cartesian velocity
            vel_xy = np.array([Cx, Cy]) - self.robots[idx, :2]

            # SFL
            b = 0.5 * self.ROBOTS_NUM
            Kp = 0.8
            T = np.array([[math.cos(self.robots[idx, 2]), math.cos(self.robots[idx, 2])],
                            [-1/b * math.sin(self.robots[idx, 2]), 1*b * math.cos(self.robots[idx, 2])]])
            vel = np.matmul(T, vel_xy)
            v = max(-VMAX, min(VMAX, Kp*vel[0]))
            w = max(-WMAX, min(WMAX, Kp*vel[1]))

            vel_msg = TwistStamped()
            vel_msg.header.stamp = rospy.Time.now()
            vel_msg.twist.linear.x = v * math.cos(self.robots[idx, 2])
            vel_msg.twist.linear.y = v * math.sin(self.robots[idx, 2])
            vel_msg.twist.angular.z = w
            print(f"Velocity for robot {idx}: {v}, {w}")
            self.velPubs[idx].publish(vel_msg)
            







if __name__ == '__main__':
  det = ExplorationNodepy()
  rospy.spin()
    




