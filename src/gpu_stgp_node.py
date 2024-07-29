#!/usr/bin/env python3

import numpy as np
import os
import math
import time

# GP stuff
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpu_classifier import DirichletGPModel, ExactGPModel


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)



  
class GPUClassifier():
  def __init__(self):
    rospy.init_node("gpu_classifier_node")

    self.AREA_W = rospy.get_param("~AREA_W", 20.0)
    self.ROBOTS_NUM = rospy.get_param("~ROBOTS_NUM", 3)
    self.FOV_DEG = rospy.get_param("~ROBOT_FOV", 120.0)
    self.FOV_RAD = self.FOV_DEG * math.pi / 180.0
    self.dt = rospy.get_param("~dt", 0.2)
    self.GRID_SIZE = rospy.get_param("~GRID_SIZE", 50)
    self.DETECTIONS_BUFFER = rospy.get_param("~DETECTIONS_BUFFER", 200)
    self.frame_id = rospy.get_param("~frame_id", "world")
    self.resolution = self.AREA_W / self.GRID_SIZE
    print("Resolution: ", self.resolution)
    print("Grid size: ", self.GRID_SIZE)

    self.times = torch.zeros((self.GRID_SIZE**2)).to(device)

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
    self.FORGET_TIME = 10.0

    X = torch.zeros((self.GRID_SIZE**2, 2))
    self.Y = torch.zeros(self.GRID_SIZE**2).to(device)
    self.t_start = rospy.Time.now()
    self.t_now = (rospy.Time.now() - self.t_start).secs
    # self.times = -np.inf * np.ones(self.GRID_SIZE**2)

    for i in range(0, self.GRID_SIZE**2, self.GRID_SIZE):
      X[i:i+self.GRID_SIZE, 0] = self.AREA_W*i/(self.GRID_SIZE**2)
      for j in range(0, self.GRID_SIZE):
        X[i+j, 1] = self.AREA_W*j/self.GRID_SIZE
        
    self.X = X.to(device)

    self.gp = None
    self.likelihood = None
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
    self.measurements.append(torch.tensor([x, y, time.secs]).to(device))
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

    # Remove old detections
    self.measurements = [m for m in self.measurements if self.t_now - m[2] < self.FORGET_TIME]

    print("Number of measurements: ", len(self.measurements))
    
    
    for meas in self.measurements:
      i_t = (meas[0] + 0.5*self.AREA_W) // self.resolution # + random.randint(-1, 1)
      j_t = (meas[1] + 0.5*self.AREA_W) // self.resolution # + random.randint(-1, 1)
      self.Y[int(i_t+j_t*self.GRID_SIZE)] = 1
      # self.times[int(i_t+j_t*self.GRID_SIZE)] = (meas[2]).secs
      # self.post_map.data[int(i_t+j_t*self.GRID_SIZE)] = 100
    
    
    
    
    # self.Y[det_ids] = values

    X_train = self.X[self.det_ids].long()
    y_train = self.Y[self.det_ids].long()
    t_train = self.T_factor*(self.times[self.det_ids] - self.t_start.secs)
    # print("times max: ", torch.max(t_train))
    # print("times min: ", torch.min(t_train))
    # print("t_train: ", t_train)
    Xst_train = torch.cat((X_train, torch.unsqueeze(t_train, 1)), 1).long()
    # print("Max Xst: ", Xst_train.max())
    # print("Min Xst: ", Xst_train.min())
  
    if 1 not in y_train:
      y_train[torch.randint(len(y_train), (1,))] = 1

    times_now = self.T_factor * self.t_now * torch.ones((self.X.shape[0], 1)).to(device)
    print("Time now: ", self.t_now)
    X_st = torch.cat((self.X, times_now), 1)
    
    # ------- Classification -----
    likelihood = DirichletClassificationLikelihood(y_train, learn_additional_noise=True)
    self.gp = DirichletGPModel(Xst_train, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)
    self.gp = self.gp.to(device)
    
    # Fit
    self.gp.train()
    likelihood.train()
    verbose = False

    optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.1)

    # Loss for GPs - The marginal log-likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self.gp)

    for i in range(50):
      optimizer.zero_grad()
      output = self.gp(Xst_train)
      loss = -mll(output, likelihood.transformed_targets).sum()
      loss.backward()
      if verbose and i % 10 == 0:
          print(f"Iter {i}/{50} - Loss: {loss.item()} - lengthscale: {self.covar_module.base_kernel.lengthscale.mean().item()} - Noise: {self.likelihood.second_noise_covar.noise.mean().item()}")
      optimizer.step()

    # Predict
    self.gp.eval()
    likelihood.eval()

    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        test_dist = self.gp(X_st)
        pred_samples = test_dist.sample(torch.Size((1000,))).exp()
        probs = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)
        probs = probs[1, :]

    
    

    """ #--------- Regression -------------
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    self.gp = ExactGPModel(Xst_train, y_train, likelihood).to(device)

    ### Train
    self.gp.train()
    likelihood.train()

    optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self.gp)

    training_iter = 50
    for i in range(training_iter):
      optimizer.zero_grad()
      output = self.gp(Xst_train)
      loss = -mll(output, y_train)
      loss.backward()
      optimizer.step()

    ### Eval
    self.gp.eval()
    likelihood.eval()
    times_now = self.T_factor * self.t_now * torch.ones((self.X.shape[0], 1)).to(device)
    print("Time now: ", self.t_now)
    X_st = torch.cat((self.X, times_now), 1)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
      pred = self.gp(X_st)
    """

    # Prediction
    """
    y_pred, y_var = self.gp.predict(X_st)                  ######
    y_prob = self.gp.predict_proba(X_st)            ######
    probs = (100*y_prob[1, :]).to(int)
    probs = torch.clamp(pred.loc, min=0.0, max=1.0)
    """
    probs = (100*probs).to(int)
    print("Probs min: ", probs.min())
    print("Probs max: ", probs.max())
    # Calc posterior and publish
    self.post_map.data = probs.tolist() ######

    self.prior_map_pub.publish(self.prior_map)
    self.post_map_pub.publish(self.post_map)


    timerend = time.time()
    print("Computation time: ", timerend - timerstart)


if __name__ == '__main__':
  GPUClass = GPUClassifier()
  rospy.spin()


