#!/usr/bin/env python3

import numpy as np
import os
import math
import time

# GP stuff
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Product
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression


# ROS stuff
import rospy
from geometry_msgs.msg import Point, Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Float32, Int32MultiArray


### PARAMETERS

GAMMA_RATE = 0.05        # forgetting factor
TIME_VAR = False
MOVING_TARGET = True
TARGETS_NUM = 3
DETECTION_PROB = 0.75



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

def gauss_pdf(x, y, mean, covariance):

  # points = np.column_stack([x.flatten(), y.flatten()])
  points = np.array([x, y])
  points = np.expand_dims(points, 0)
  # Calculate the multivariate Gaussian probability
  exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
  coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
  prob = coefficient * np.exp(exponent)

  return prob

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
  
  
class GPClassifier():
  def __init__(self):
    rospy.init_node("gp_classifier_node")

    self.AREA_W = rospy.get_param("~AREA_W", 20.0)
    self.ROBOTS_NUM = rospy.get_param("~ROBOTS_NUM", 3)
    self.FOV_DEG = rospy.get_param("~ROBOT_FOV", 120.0)
    self.FOV_RAD = self.FOV_DEG * math.pi / 180.0
    self.dt = rospy.get_param("~dt", 1.0)
    self.GRID_SIZE = rospy.get_param("~GRID_SIZE", 50)
    self.DETECTIONS_BUFFER = rospy.get_param("~DETECTIONS_BUFFER", 50)
    self.frame_id = rospy.get_param("~frame_id", "world")
    self.resolution = self.AREA_W / self.GRID_SIZE
    print("Resolution: ", self.resolution)
    print("Grid size: ", self.GRID_SIZE)

    self.measurements_sub = rospy.Subscriber("/measurements", PoseStamped, self.meas_callback)
    self.measurements = []
    self.detections_subs = [rospy.Subscriber(f"robot{i}/detections_grid", OccupancyGrid, self.detections_callback) for i in range(self.ROBOTS_NUM)]
    self.detections = []
    self.prior_map_pub = rospy.Publisher("/prior_map", OccupancyGrid, queue_size=1)
    self.post_map_pub = rospy.Publisher("/posterior_map", OccupancyGrid, queue_size=1)
    self.times_sub = rospy.Subscriber("/times", Int32MultiArray, self.t_callback)
    
    self.prior_map = OccupancyGrid()
    self.prior_map.header.frame_id = self.frame_id
    self.prior_map.info.resolution = self.resolution
    self.prior_map.info.width = self.GRID_SIZE
    self.prior_map.info.height = self.GRID_SIZE
    self.prior_map.info.origin = Pose()
    self.prior_map.info.origin.position.x = -0.5*self.AREA_W
    self.prior_map.info.origin.position.y = -0.5*self.AREA_W
    self.prior_map.data = [-1 for _ in range(self.GRID_SIZE**2)]
    
    self.post_map = OccupancyGrid()
    self.post_map.header.frame_id = self.frame_id
    self.post_map.info.resolution = self.resolution
    self.post_map.info.width = self.GRID_SIZE
    self.post_map.info.height = self.GRID_SIZE
    self.post_map.info.origin = Pose()
    self.post_map.info.origin.position.x = -0.5*self.AREA_W
    self.post_map.info.origin.position.y = -0.5*self.AREA_W
    self.post_map.data = [-1 for _ in range(self.GRID_SIZE**2)]


    self.timer = rospy.Timer(rospy.Duration(self.dt), self.timer_callback)
    self.T_factor = 0.01

    X = np.zeros((self.GRID_SIZE**2, 2))
    self.Y = np.zeros(self.GRID_SIZE**2)
    self.t_start = rospy.Time.now()
    self.t_now = (rospy.Time.now() - self.t_start).secs
    # self.times = -np.inf * np.ones(self.GRID_SIZE**2)
    self.times = np.zeros(self.GRID_SIZE**2)

    for i in range(0, self.GRID_SIZE**2, self.GRID_SIZE):
      X[i:i+self.GRID_SIZE, 0] = self.AREA_W*i/(self.GRID_SIZE**2)
      for j in range(0, self.GRID_SIZE):
        X[i+j, 1] = self.AREA_W*j/self.GRID_SIZE
        
    self.X = X

    self.gp = GaussianProcessClassifier()
    self.values = []
    self.det_ids = []

 

  # detections must be in range [-0.5*W, 0.5*W]
  def detections_callback(self, msg):
    det_ids = [i for i, v in enumerate(msg.data) if v != -1]
    for i in det_ids:
      self.prior_map.data[i] = msg.data[i]
      # self.times[i] = (msg.header.stamp - self.t_start).secs

  
  def meas_callback(self, msg):
    x = msg.pose.position.x
    y = msg.pose.position.y
    time = msg.header.stamp
    self.measurements.append(np.array([x, y, time]))
    if len(self.measurements) > self.DETECTIONS_BUFFER:
      self.measurements.pop(0)


  def t_callback(self, msg):
    det_ids = [i for i, v in enumerate(msg.data) if v != -1]
    new_ts = []
    for i in det_ids:
      self.times[i] = max(msg.data[i], self.times[i])     # keep only the most recent
    
    



  def timer_callback(self, e):
    
    timerstart = time.time()
    self.t_now = (rospy.Time.now() - self.t_start).secs

    # out = [[i,v] for i,v in enumerate(self.prior_map.data) if v != -1]
    # det_ids = [out[i][0] for i in range(len(out))]
    # values = [out[i][1]/100 for i in range(len(out))]
    det_ids = [i for i,v in enumerate(self.prior_map.data) if v != -1]
    for i in det_ids:
      # self.times[i] = self.t_now
      if i not in self.det_ids:
        self.det_ids.append(i)
    
    for i in self.det_ids:
      self.Y[i] = 0

    print("Number of measurements: ", len(self.measurements))
    
    
    for meas in self.measurements:
      i_t = (meas[0] + 0.5*self.AREA_W) // self.resolution # + random.randint(-1, 1)
      j_t = (meas[1] + 0.5*self.AREA_W) // self.resolution # + random.randint(-1, 1)
      self.Y[int(i_t+j_t*self.GRID_SIZE)] = 1
      # self.times[int(i_t+j_t*self.GRID_SIZE)] = (meas[2]).secs
      # self.post_map.data[int(i_t+j_t*self.GRID_SIZE)] = 100
    
    
    """
    for i in range(self.GRID_SIZE):
      for j in range(self.GRID_SIZE):
        x_ij = np.array([i*self.resolution - 0.5*self.AREA_W, -0.5*self.AREA_W + j*self.resolution])
        self.Y[i+j*GRID_SIZE] = gauss_pdf(x_ij[0], x_ij[1], self.measurements[-1], 2.0)
    """
    
    # self.Y[det_ids] = values

    X_train = self.X[self.det_ids]
    y_train = self.Y[self.det_ids]
    t_train = self.T_factor*(self.times[self.det_ids] - self.t_start.secs)
    print("times max: ", np.max(t_train))
    # print("t_train: ", t_train)
    Xst_train = np.concatenate((X_train, np.expand_dims(t_train, 1)), 1)
    print("Max Xst: ", Xst_train.max())
    print("Min Xst: ", Xst_train.min())

    if 1 not in y_train:
      y_train[np.random.randint(len(y_train))] = 1
    
    # Fit
    self.gp.fit(Xst_train, y_train)

    # Prediction
    times_now = self.T_factor * self.t_now * np.ones((self.X.shape[0], 1))
    print("Time now: ", self.t_now)
    X_st = np.concatenate((self.X, times_now), 1)
    y_pred = self.gp.predict(X_st)
    y_prob = self.gp.predict_proba(X_st)
    probs = 100*y_prob[:, 1]
    print("Probs max: ", probs.max())
    # Calc posterior and publish
    self.post_map.data = probs.astype(int).tolist()

    self.prior_map_pub.publish(self.prior_map)
    self.post_map_pub.publish(self.post_map)


    timerend = time.time()
    print("Computation time: ", timerend - timerstart)


if __name__ == '__main__':
  GPClass = GPClassifier()
  rospy.spin()


