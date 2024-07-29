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

# CBFs
from scipy.optimize import minimize

# ROS imports
import rospy
from geometry_msgs.msg import Point, Twist, Pose, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Float32, Int32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion

VMAX = 1.0
WMAX = 1.0

def sigmoid(x):
    tmp = np.exp(-x) + 1
    v = 1 / tmp
    print("val shape: ", v.shape)

    return v


def modified_sigmoid(x, k=1):
    return 1 / (1 + np.exp(-k*(x - 50)))


def objective_function(u):
    return np.linalg.norm(u)**2

def safety_constraint(u, A, b):
    return -np.dot(A,u) + b

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))



class ExplorationNodepy():
    def __init__(self):
        rospy.init_node("exploration_node")
        
        # Params
        self.AREA_W = rospy.get_param("~AREA_W", 20.0)
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.FOV_DEG = rospy.get_param("~ROBOT_FOV", 120.0)
        self.ROBOTS_NUM = rospy.get_param('~ROBOTS_NUM', 4)
        self.FOV_RAD = self.FOV_DEG * math.pi / 180.0
        self.ROBOT_RANGE = rospy.get_param("~ROBOT_RANGE", 3.0)
        self.dt = rospy.get_param("~dt", 0.1)
        self.GRID_SIZE = rospy.get_param("~GRID_SIZE", 50)
        self.RESOLUTION = self.AREA_W / self.GRID_SIZE

        # pubs/subs
        self.odomSubs = [rospy.Subscriber(f"/hummingbird{i}/ground_truth/odometry", Odometry, self.odom_callback, i) for i in range(self.ROBOTS_NUM)]
        self.velPubs = [rospy.Publisher(f"/hummingbird{i}/autopilot/velocity_command", TwistStamped, queue_size=1) for i in range(self.ROBOTS_NUM)]
        
        self.gp_sub = rospy.Subscriber("/posterior_map", OccupancyGrid, self.gp_callback)
        self.voro_pub = rospy.Publisher("/voronoi", Marker, queue_size=1)
        self.centr_pub = rospy.Publisher("/centroid", Marker, queue_size=1)
        self.maptest_pub = rospy.Publisher("gp_test", OccupancyGrid, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.timer_callback)


        # init robots
        self.robots = np.zeros((self.ROBOTS_NUM, 3))

        # init GP
        self.gp = np.zeros((self.GRID_SIZE**2))
        self.gp_init = False

        # init test map for GP
        self.gp_map = OccupancyGrid()
        self.gp_map.header.frame_id = self.frame_id
        self.gp_map.info.resolution = self.RESOLUTION
        self.gp_map.info.width = self.GRID_SIZE
        self.gp_map.info.height = self.GRID_SIZE
        self.gp_map.info.origin.position.x = -0.5*self.AREA_W
        self.gp_map.info.origin.position.y = -0.5*self.AREA_W
        self.gp_map.data = [-1 for _ in range(self.GRID_SIZE**2)]

        print("Init completed. ")

    

    def odom_callback(self, msg, i):
        self.robots[i, 0] = msg.pose.pose.position.x
        self.robots[i, 1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        (roll, pitch, yaw) = euler_from_quaternion(quat)
        self.robots[i, 2] = yaw

    
    def gp_callback(self, msg):
        # for i in range(len(msg.data)):
        #     q = i // self.GRID_SIZE
        #     r = i % self.GRID_SIZE
        #     self.gp[q, r] = msg.data[i]
        self.gp = np.array(msg.data)
        print(f"Min/max GP values before sigmoid: {self.gp.min()}, {self.gp.max()}")

        # self.gp = 100*modified_sigmoid(self.gp, k=1.0)
        self.gp = 100*softmax(self.gp)
        # self.gp = self.gp**3
        # self.gp = np.exp(self.gp) - 1
        # self.gp[self.gp > 55] *= 20
        print(f"Min/max GP values after sigmoid: {self.gp.min()}, {self.gp.max()}")

        # self.gp_map.data = np.ravel(self.gp*100/np.max(self.gp)).astype(int).tolist()
        # self.maptest_pub.publish(self.gp_map)
        
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


            i_max = (xmax + 0.5*self.AREA_W) // self.RESOLUTION
            i_min = (xmin + 0.5*self.AREA_W) // self.RESOLUTION
            j_max = (ymax + 0.5*self.AREA_W) // self.RESOLUTION
            j_min = (ymin + 0.5*self.AREA_W) // self.RESOLUTION


            

            dA = dx*dy
            A = 0.0
            Cx = 0.0; Cy = 0.0

            for x_i in np.arange(xmin, xmax, self.RESOLUTION):
                for y_i in np.arange(ymin, ymax, self.RESOLUTION):
                    if polygon.contains(Pt(x_i, y_i)):
                        i = int((x_i + 0.5*self.AREA_W) / self.RESOLUTION)
                        j = int((y_i + 0.5*self.AREA_W) / self.RESOLUTION)
                        dA_pdf = dA*self.gp[i+j*self.GRID_SIZE]
                        A = A + dA_pdf
                        Cx += x_i * dA_pdf
                        Cy += y_i * dA_pdf
            
            Cx = Cx / A
            Cy = Cy / A
            self.publishCentroid(np.array([Cx, Cy]), idx)

            # Cartesian velocity
            fov_ctr = self.robots[idx, :2] + 0.5*self.ROBOT_RANGE * np.array([math.cos(self.robots[idx, 2]), math.sin(self.robots[idx, 2])])
            vel_xy = np.array([Cx, Cy]) - fov_ctr

            # SFL
            b = 0.5 * self.ROBOT_RANGE
            Kp = 0.8
            T = np.array([[math.cos(self.robots[idx, 2]), math.sin(self.robots[idx, 2])],
                            [-1/b * math.sin(self.robots[idx, 2]), 1*b * math.cos(self.robots[idx, 2])]])
            vel = np.matmul(T, vel_xy)
            v = max(-VMAX, min(VMAX, Kp*vel[0]))
            w = max(-WMAX, min(WMAX, Kp*vel[1]))
            
            # CBF
            gamma = 0.1
            vdes = v * np.array([math.cos(self.robots[idx, 2]), math.sin(self.robots[idx, 2])])
            h = -np.linalg.norm(self.robots[idx, :2])**2 + (0.5*self.AREA_W)**2
            Acbf = 2 * self.robots[idx, :2]
            bcbf = gamma * h
            constraint = {"type": "ineq", "fun": lambda u: safety_constraint(u, Acbf, bcbf)}
            obj = lambda u: objective_function(u-vdes)
            result = minimize(obj, vdes, constraints=constraint, bounds=[(-VMAX, VMAX), (-VMAX, VMAX)])
            u_opt = result.x
            print("Desired vel: ", vdes)
            print("Optimal vel: ", u_opt)

            vel_msg = TwistStamped()
            vel_msg.header.stamp = rospy.Time.now()
            vel_msg.twist.linear.x = u_opt[0] # v * math.cos(self.robots[idx, 2])
            vel_msg.twist.linear.y = u_opt[1] # v * math.sin(self.robots[idx, 2])
            vel_msg.twist.angular.z = w
            # print(f"Velocity for robot {idx}: {v}, {w}")
            self.velPubs[idx].publish(vel_msg)
            







if __name__ == '__main__':
  det = ExplorationNodepy()
  rospy.spin()
    




