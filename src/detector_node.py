#!/usr/bin/env python3

import numpy as np
import os


# Coverage stuff
import math

# ROS stuff
import rospy
from geometry_msgs.msg import Point, Twist, Pose
from nav_msgs.msg import Odometry, OccupancyGrid
from tf.transformations import euler_from_quaternion






### UTILITY FUCTIONS


def insideFOV(robot, target, fov, range):
  fov_rad = fov * math.pi / 180.0
  xr = robot[0]
  yr = robot[1]
  phi = robot[2]
  dx = target[0] - xr
  dy = target[1] - yr
  dist = math.sqrt(dx**2 + dy**2)
  if dist > range:
    return 0

  xrel = dx * math.cos(phi) + dy * math.sin(phi)
  yrel = -dx * math.sin(phi) + dy * math.cos(phi)
  angle = abs(math.atan2(yrel, xrel))
  if (angle <= fov_rad*0.5) and (xrel >= 0.0):
    return 1
  else:
    return 0

def plot_fov(x1, theta, fov_deg, radius, color="tab:blue", ax=None):
  # fig = plt.figure(figsize=(6,6))
  # plt.scatter(neighs[:, 0], neighs[:, 1], marker='*')

  # x1 = np.array([0.0, 0.0, 0.0]
  fov = fov_deg * math.pi / 180
  arc_theta = np.arange(theta-0.5*fov, theta+0.5*fov, 0.01*math.pi)
  th = np.arange(fov/2, 2*math.pi+fov/2, 0.01*math.pi)

  # FOV
  xfov = x1[0] + radius * np.cos(arc_theta)
  xfov = np.append(x1[0], xfov)
  xfov = np.append(xfov, x1[0])
  yfov = x1[1] + radius * np.sin(arc_theta)
  yfov = np.append(x1[1], yfov)
  yfov = np.append(yfov, x1[1])
  if ax is not None:
    ax.plot(xfov, yfov, c=color, zorder=10)
  else:
    plt.plot(xfov, yfov)
  
  
class Detector():
  def __init__(self):
    rospy.init_node("detector_node")

    self.AREA_W = rospy.get_param("AREA_W", 20.0)
    self.FOV_DEG = rospy.get_param("ROBOT_FOV", 120.0)
    self.FOV_RAD = self.FOV_DEG * math.pi / 180.0
    self.ROBOT_RANGE = rospy.get_param("ROBOT_RANGE", 3.0)
    self.dt = rospy.get_param("dt", 0.2)
    self.GRID_SIZE = 100
    self.resolution = self.AREA_W / self.GRID_SIZE

    self.robot_sub = rospy.Subscriber("/robot_odom", Odometry, self.robot_callback)
    self.target_sub = rospy.Subscriber("/target/odom", Odometry, self.target_callback)
    self.map_pub = rospy.Publisher("/detections_grid", OccupancyGrid, queue_size=1)
    
    self.map_msg = OccupancyGrid()
    self.map_msg.header.frame_id = "odom"
    self.map_msg.info.resolution = self.resolution
    self.map_msg.info.width = self.GRID_SIZE
    self.map_msg.info.height = self.GRID_SIZE
    self.map_msg.info.origin = Pose()
    self.map_msg.info.origin.position.x = -0.5*self.AREA_W
    self.map_msg.info.origin.position.y = -0.5*self.AREA_W
    self.map_msg.data = [-1 for _ in range(self.GRID_SIZE**2)]




    # self.detections_ = rospy.Subscriber("/detections_grid", Point, self.detections_callback)
    # self.detections = []

    self.timer = rospy.Timer(rospy.Duration(self.dt), self.timer_callback)

    self.robot = np.zeros(3)
    self.target = np.zeros(2)




  def robot_callback(self, msg):
    self.robot[0] = msg.pose.pose.position.x
    self.robot[1] = msg.pose.pose.position.y
    q = msg.pose.pose.orientation
    quat = [q.x, q.y, q.z, q.w]
    (roll, pitch, yaw) = euler_from_quaternion(quat)
    self.robot[2] = yaw
  
  def target_callback(self, msg):
    self.target[0] = msg.pose.pose.position.x
    self.target[1] = msg.pose.pose.position.y

  



  def timer_callback(self, e):
    # check if target inside fov
    for i in range(self.GRID_SIZE):
      for j in range(self.GRID_SIZE):
        x_ij = np.array([i*self.resolution - 0.5*self.AREA_W, -0.5*self.AREA_W + j*self.resolution])
        if insideFOV(self.robot, x_ij, self.FOV_DEG, self.ROBOT_RANGE):
          self.map_msg.data[i+j*self.GRID_SIZE] = 0
    
    if insideFOV(self.robot, self.target, self.FOV_DEG, self.ROBOT_RANGE):
      i_t = (self.target[0] + 0.5*self.AREA_W) // self.resolution
      j_t = (self.target[1] + 0.5*self.AREA_W) // self.resolution
      self.map_msg.data[int(i_t+j_t*self.GRID_SIZE)] = 100



    self.map_msg.header.stamp = rospy.Time.now()
    self.map_pub.publish(self.map_msg)




if __name__ == '__main__':
  det = Detector()
  rospy.spin()