#!/usr/bin/env python3

import numpy as np
import os
import math

# GP stuff
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Product
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression


# ROS stuff
import rospy
from geometry_msgs.msg import Point, Twist, Pose
from nav_msgs.msg import Odometry, OccupancyGrid



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

    self.AREA_W = rospy.get_param("AREA_W", 20.0)
    self.ROBOTS_NUM = rospy.get_param("ROBOTS_NUM", 3)
    self.FOV_DEG = rospy.get_param("ROBOT_FOV", 120.0)
    self.FOV_RAD = self.FOV_DEG * math.pi / 180.0
    self.dt = rospy.get_param("dt", 1.0)
    self.GRID_SIZE = 100
    self.resolution = self.AREA_W / self.GRID_SIZE

    self.detections_subs = [rospy.Subscriber(f"robot{i}/detections_grid", OccupancyGrid, self.detections_callback) for i in range(self.ROBOTS_NUM)]
    self.detections = []
    self.map_pub = rospy.Publisher("/global_map", OccupancyGrid, queue_size=1)
    self.global_map = OccupancyGrid()
    self.global_map.header.frame_id = "odom"
    self.global_map.info.resolution = self.resolution
    self.global_map.info.width = self.GRID_SIZE
    self.global_map.info.height = self.GRID_SIZE
    self.global_map.info.origin = Pose()
    self.global_map.info.origin.position.x = -0.5*self.AREA_W
    self.global_map.info.origin.position.y = -0.5*self.AREA_W
    self.global_map.data = [-1 for _ in range(self.GRID_SIZE**2)]


    self.timer = rospy.Timer(rospy.Duration(self.dt), self.timer_callback)

    self.X = np.zeros((self.GRID_SIZE**2, 2))
    self.Y = np.zeros(self.GRID_SIZE**2)
    self.times = -np.inf * np.ones(self.GRID_SIZE**2)

    self.gp = GaussianProcessClassifier()

    """
    # Build X dataset
    for i in range(0, self.GRID_SIZE**2, self.GRID_SIZE):
      X[i:i+self.GRID_SIZE, 0] = self.AREA_W*i/(self.GRID_SIZE**2)
      for j in range(0, self.GRID_SIZE):
        X[i+j, 1] = self.AREA_W * j / self.GRID_SIZE
    """

  # detections must be in range [-0.5*W, 0.5*W]
  def detections_callback(self, msg):
    det_ids = [i for i, v in enumerate(msg.data) if v != -1]
    for i in det_ids:
      self.global_map.data[i] = msg.data[i]



  def timer_callback(self, e):
    """
    detections_ids = []
    # find indexes of detected pts
    for pt in self.detections:
      i = pt[0] // self.discretize_precision
      j = pt[1] // self.discretize_precision
      t = pt[2]
      detections_ids.append(i+j)

    X_train = X[detections_ids]
    Y_train = np.ones(len(detections_ids))
    """

    self.map_pub.publish(self.global_map)



if __name__ == '__main__':
  GPClass = GPClassifier()
  rospy.spin()








### MAIN CODE
"""
NUM_STEPS = 50
ROBOT_RANGE = 3.0
fov_deg = 120.0
ROBOT_FOV = fov_deg*math.pi/180.0


points = 2.0 + (AREA_W-4.0) * np.random.rand(ROBOTS_NUM, 2)
thetas = 2*math.pi * np.random.rand(ROBOTS_NUM)
fov_ctrs = np.zeros_like(points)
fov_ctrs[:, 0] = points[:, 0] + 0.5*ROBOT_RANGE*np.cos(thetas)
fov_ctrs[:, 1] = points[:, 1] + 0.5*ROBOT_RANGE*np.sin(thetas)
robots_hist = np.zeros((1, points.shape[0], points.shape[1]))
robots_hist[0, :, :] = points


# Generate target and dataset
targets = AREA_W * np.random.rand(TARGETS_NUM, 2)
th_targets = 2*np.pi * np.random.rand(TARGETS_NUM)
targets_hist = np.zeros((1, TARGETS_NUM, 2))
targets_hist[0, :, :] = targets
targets.shape
target = targets[0]

train_ids = []
GRID_SIZE = 100
X = np.zeros((GRID_SIZE**2, 2)) 
Y = np.zeros(GRID_SIZE**2)
times = -np.inf * np.ones(GRID_SIZE**2)
t_now = 0.0
'''
for i in range(0, GRID_SIZE**2, GRID_SIZE):
  X[i:i+GRID_SIZE, 0] = AREA_W*i/(GRID_SIZE**2)
  for j in range(0, GRID_SIZE):
    X[i+j, 1] = AREA_W*j/GRID_SIZE
    x_ij = X[i+j, :]
    for target in targets:
        dist_t = np.linalg.norm(target - x_ij)
        if dist_t < 2.0:
            rnd = np.random.rand()
            if rnd < DETECTION_PROB:
              Y[i+j] = 1
'''              

for i in range(0, GRID_SIZE**2, GRID_SIZE):
  X[i:i+GRID_SIZE, 0] = AREA_W*i/(GRID_SIZE**2)
  for j in range(0, GRID_SIZE):
    X[i+j, 1] = AREA_W*j/GRID_SIZE
    x_ij = X[i+j, :2]
    for target in targets:
      target_detected = False
      for rbt in range(ROBOTS_NUM):
        if insideFOV(np.append(points[rbt], thetas[rbt]), target, fov_deg, ROBOT_RANGE):
          target_detected = True
      if target_detected:
        if np.linalg.norm(target - x_ij) < 2.0:
          rnd = np.random.rand()
          if rnd < DETECTION_PROB:             # only 30% of detections
            Y[i+j] = 1
            times[i+j] = t_now

    # if x_ij[0] <= 5.0 and x_ij[1] <= 5.0:
    #   train_ids.append(i+j)

kernel = 1.0 * RBF([1.0, 1.0, 1.0])
gp = GaussianProcessClassifier()

# Initialize forgetting factors
gammas = np.zeros_like(Y)
probs_old = 0.5*np.ones_like(Y)       # initialize previous prob to 0.5



# fig, ax = plt.subplots(1, 1, figsize=(10,10))
# for i in range(ROBOTS_NUM):
#   ax.scatter(points[i, 0], points[i, 1], marker="x", c="tab:blue", label="Robot")
#   ax.scatter(fov_ctrs[i, 0], fov_ctrs[i, 1], marker="+", c="tab:red", label="fov ctr")
#   plot_fov(points[i], thetas[i], fov_deg, ROBOT_RANGE, ax=ax)
# 
# plt.show()
  




# for _ in range(ROBOTS_NUM):
#   detections_ids.append([])

detections_ids = []

for s in range(1, NUM_STEPS+1):
  print(f"*** Step {s} ***")
  fig, ax = plt.subplots(1, 1, figsize=(10,10))
  # mirror points across each edge of the env
  dummy_points = np.zeros((5*ROBOTS_NUM, 2))
  dummy_points[:ROBOTS_NUM, :] = points
  mirrored_points = mirror(points)
  mir_pts = np.array(mirrored_points)
  dummy_points[ROBOTS_NUM:, :] = mir_pts


  # Voronoi partitioning
  vor = Voronoi(dummy_points)

  conv = True
  lim_regions = []
  poly_regions = []

  detected = np.zeros((Y.shape[0]), dtype=bool)
  

  # Simulate cooperative detections inside each robots sensing region
  for i in range(X.shape[0]):
    x_i = X[i, :]
    x_pt = Point(X[i, 0], X[i, 1])
    for j in range(ROBOTS_NUM):
      # robot = vor.points[j]
      robot = points[j]
      # d = np.linalg.norm(robot - x_i)
      # if d <= ROBOT_RANGE: 
      #   detected[i] = True
      if insideFOV(np.append(robot, thetas[j]), x_i, fov_deg, ROBOT_RANGE):
        detected[i] = True
        gammas[i] = 0.0
        times[i] = t_now
        if i not in detections_ids:        # only append points not visited yet
          detections_ids.append(i)
    # if not detected:
    #   gammas[i] += GAMMA_RATE

  # centralized fitting and prediction

  # Define training set
  # X_train, y_train = X[detections_ids], y[detections_ids]
  X_train, y_train, t_train = X[detections_ids], Y[detections_ids], times[detections_ids]
  # print("t_now: ", t_now)
  # print("Times of current detections: ", t_train)
  
  # Concatenate spatial and temporal features
  X_train = np.concatenate((X_train, np.expand_dims(t_train, 1)), 1)

  if 1 not in y_train:
    y_train[np.random.randint(y_train.shape[0])] = 1


  ### Training
  gp.fit(X_train, y_train)

  # Prediction on the whole environment
  times_now = t_now * np.ones((X.shape[0], 1))
  X_st = np.concatenate((X, times_now), 1)
  y_pred = gp.predict(X_st)
  y_prob = gp.predict_proba(X_st)
  probs = y_prob[:, 1]


  # Add forgetting factor
  if TIME_VAR:
    # gammas[detected == 0] += 0.1 * (probs[detected == 0] - 0.5)
    # probs[detected == 0] -= gammas[detected == 0]
    # gammas[detected == 0] += 0.05
    # probs[(detected == 0) & (probs > 0.5)] -= gammas[(detected == 0) & (probs > 0.5)]
    # probs[(detected == 0) & (probs < 0.5)] += gammas[(detected == 0) & (probs < 0.5)]
    probs[detected == 0] = probs_old[detected == 0] - 0.05*(probs_old[detected == 0] - 0.5)
    probs_old = probs

  y_prob_vis = probs
  ax.scatter(X[:, 0], X[:, 1], c=y_prob_vis, cmap="YlOrRd", vmin=0.0, vmax=1.0, zorder=0)
  


  # fig, axs = plt.subplots(2, ROBOTS_NUM//2, figsize=(16,9))
  for it in range(TARGETS_NUM):
    ax.plot(targets_hist[:, it, 0], targets_hist[:, it, 1], c="tab:green")
  ax.scatter(targets[:, 0], targets[:, 1], marker='x', c="tab:green", label="Target")
  
  row = 0
  for idx in range(ROBOTS_NUM):
    if idx >= ROBOTS_NUM/2:
      row = 1
    voronoi_ids = []
    region = vor.point_region[idx]
    poly_vert = []
    for vert in vor.regions[region]:
      v = vor.vertices[vert]
      poly_vert.append(v)
      # plt.scatter(v[0], v[1], c='tab:red')

    poly = Polygon(poly_vert)             # Voronoi cell
    poly_regions.append(poly)
    x,y = poly.exterior.xy
    # plt.plot(x, y, c='tab:orange')
    # robot = np.array([-18.0, -12.0])
    robot = points[idx]
    # plt.scatter(robot[0], robot[1])

    # Intersect with robot range
    step = 0.5
    range_pts = []
    '''
    for th in np.arange(0.0, 2*np.pi, step):
      xi = robot[0] + ROBOT_RANGE * np.cos(th)
      yi = robot[1] + ROBOT_RANGE * np.sin(th)
      pt = Point(xi, yi)
      range_pts.append(pt)
      # plt.plot(xi, yi, c='tab:blue')

    range_poly = Polygon(range_pts)                   # Sensing region
    xc, yc = range_poly.exterior.xy

    lim_region = intersection(poly, range_poly)       # Limited Voronoi cell
    lim_regions.append(lim_region)
    '''
    # centr = np.array([lim_region.centroid.x, lim_region.centroid.y])
    # dist = np.linalg.norm(robot-centr)
    # points[idx, :] = robot + 0.5 * (centr - robot)

    # Calculate values inside Voronoi cell
    for i in range(X.shape[0]):
      x_pt = Point(X[i, 0], X[i, 1])
      if poly.contains(x_pt):                                 # Define points inside voronoi cell
        voronoi_ids.append(i)

    # Calculate centroid
    X_voro = X[voronoi_ids]
    # y_probs_voro = y_prob[voronoi_ids]
    y_probs_voro = probs[voronoi_ids]
    y_probs_voro[y_probs_voro > 0.5] *= 20.0
    A = 0.0
    Cx = 0.0; Cy = 0.0
    for j in range(X_voro.shape[0]):
      A += y_probs_voro[j]
      Cx += X_voro[j, 0] * y_probs_voro[j]
      Cy += X_voro[j, 1] * y_probs_voro[j]
      
    Cx = Cx / A
    Cy = Cy / A

    # Save fig
    # y_prob_vis[np.random.randint(y_prob_vis.shape[0])] = 0.0
    # y_prob_vis[np.random.randint(y_prob_vis.shape[0])] = 1.0
    # axs[row, idx-row*ROBOTS_NUM].scatter(X[:, 0], X[:, 1], c=y_prob_vis, cmap="YlOrRd")
    # axs[row, row*ROBOTS_NUM+i].set_title("Probability")
    if idx==0:
      ax.scatter(robot[0], robot[1], marker=(3, 0, thetas[idx]), label="Robot", c="tab:blue", zorder=10)
      ax.scatter(Cx, Cy, label="Centroid", c="tab:blue", marker="+", zorder=10)
    else:
      ax.scatter(robot[0], robot[1], marker=(3, 0, thetas[idx]), c="tab:blue", zorder=10)
      ax.scatter(Cx, Cy, c="tab:blue", marker="+", zorder=10)
      

    plot_fov(robot, thetas[idx], fov_deg, ROBOT_RANGE, ax=ax)
    # print(f"Centroid: {Cx}, {Cy}")
    x,y = poly.exterior.xy
    ax.plot(x, y, c="tab:red", zorder=10)
    
    # axs[1].scatter(Cx, Cy)
    
    # plt.show()

    # Move robot towards centroid
    centr = np.array([Cx, Cy]).transpose()
    # print(f"Robot: {robot}")
    # print(f"Centroid: {centr}")
    fov_ctr = np.zeros(2)
    fov_ctr[0] = robot[0] + 0.5*ROBOT_RANGE*math.cos(thetas[idx])
    fov_ctr[1] = robot[1] + 0.5*ROBOT_RANGE*math.sin(thetas[idx])
    fov_ctrs[idx] = fov_ctr
    if idx == 0:
      ax.scatter(fov_ctr[0], fov_ctr[1], marker='x', c="tab:purple", label="FoV center", zorder=10)
    else:
      ax.scatter(fov_ctr[0], fov_ctr[1], marker='x', c="tab:purple", zorder=10)

    dist = np.linalg.norm(fov_ctr-centr)
    Kp = 1.0
    vel = centr - fov_ctr
    vel[0] = max(-vmax, min(vmax, vel[0]))
    vel[1] = max(-vmax, min(vmax, vel[1]))
    '''
    th_des = math.atan2(centr[1]-robot[1], centr[0]-robot[0])
    th_diff = th_des - thetas[idx]
    if th_diff > math.pi:
      th_diff -= 2*np.pi
    elif th_diff < -math.pi:
      th_diff += 2*np.pi
    w = max(-wmax, min(wmax, Kp*th_diff))
    '''
    # SFL
    b = 0.5*ROBOT_RANGE
    T = np.array([[math.cos(thetas[idx]), math.sin(thetas[idx])], [-1/b*math.sin(thetas[idx]), 1/b*math.cos(thetas[idx])]])
    # print("T shape: ", T.shape)
    vw = np.matmul(T, vel)
    # print("Final vel shape: ", vw.shape)
    vw[0] = max(-vmax, min(vmax, Kp*vw[0]))         # linear vel
    vw[1] = max(-wmax, min(wmax, Kp*vw[1]))         # angular vel


    points[idx, :] = robot + vw[0]*np.array([math.cos(thetas[idx]), math.sin(thetas[idx])])*dt
    points[idx, 0] = max(0, min(AREA_W, points[idx, 0]))
    points[idx, 1] = max(0, min(AREA_W, points[idx, 1]))
    thetas[idx] += vw[1] * dt
    if thetas[idx] > math.pi:
      thetas[idx] -= 2*np.pi
    elif thetas[idx] < -math.pi:
      thetas[idx] += 2*np.pi

    if dist > 0.1:
      conv = False

  # Update time
  t_now += dt * 0.5
  
  # Move targets
  if MOVING_TARGET:
    for i in range(targets.shape[0]):
      new_x = targets[i, 0] + 0.1*vmax * np.cos(th_targets[i]) * dt
      new_y = targets[i, 1] + 0.1*vmax * np.sin(th_targets[i]) * dt

      # check if outside
      if new_x < 1.0:
        new_x = 1.0
        th_targets[i] = np.pi - th_targets[i] - 0.1*np.pi + 0.2*np.pi*np.random.rand()
      elif new_x > 0.9*AREA_W:
        new_x = 0.9*AREA_W
        th_targets[i] = np.pi - th_targets[i] - 0.1*np.pi + 0.2*np.pi*np.random.rand()

      if new_y < 1.0:
        new_y = 1.0
        th_targets[i] = -th_targets[i] -0.1*np.pi + 0.2*np.pi*np.random.rand()
      elif new_y > 0.9*AREA_W:
        new_y = 0.9*AREA_W
        th_targets[i] = -th_targets[i] - 0.1*np.pi + 0.2*np.pi*np.random.rand()

      targets[i] = [new_x, new_y]
    targets_hist = np.concatenate((targets_hist, np.expand_dims(targets, 0)))

  
  # Set values inside FoVs to 0
  for i in range(X.shape[0]):
    x_i = X[i, :]
    x_pt = Point(X[i, 0], X[i, 1])
    for j in range(ROBOTS_NUM):
      # robot = vor.points[j]
      robot = points[j]
      # d = np.linalg.norm(robot - x_i)
      # if d <= ROBOT_RANGE: 
      #   detected[i] = True
      if insideFOV(np.append(robot, thetas[j]), x_i, fov_deg, ROBOT_RANGE):
        rnd = np.random.rand()
        if rnd < 0.75:
          Y[i] = 0.0

# Update simulated detections
  # Y = np.zeros(GRID_SIZE**2)
  for target in targets:
    target_detected = False
    for rbt in range(ROBOTS_NUM):
      if insideFOV(np.append(points[rbt], thetas[rbt]), target, fov_deg, ROBOT_RANGE):
        target_detected = True
    if target_detected:
      for i in range(0, GRID_SIZE**2, GRID_SIZE):
        # X[i:i+GRID_SIZE, 0] = AREA_W*i/(GRID_SIZE**2)
        for j in range(0, GRID_SIZE):
          # X[i+j, 1] = AREA_W*j/GRID_SIZE
          x_ij = X[i+j, :]
          if np.linalg.norm(target - x_ij) < 1.0:
            rnd = np.random.rand()
            if rnd < DETECTION_PROB:             # only 30% of detections
              Y[i+j] = 1
              # times[i+j] = t_now
    
    
    

  # Save positions for visualization
  # robots_hist = np.vstack((robots_hist, np.expand_dims(points, axis=0)))
  # vis_regions.append(lim_regions)
  # ax.scatter(X[:, 0], X[:, 1], c=y_prob_vis, cmap="YlOrRd", vmin=0.0, vmax=1.0, zorder=0)
  ax.set_xlim([0, AREA_W])
  ax.set_ylim([0, AREA_W])
  plt.legend(loc="upper right")
  figname = f"mvtarg_img{s}.png" if MOVING_TARGET else f"img{s}.png"
  plt.savefig(figpath / figname)
  plt.close()

#   plt.scatter(points[:, 0], points[:, 1])
#   for region in poly_regions:
#     x,y = region.exterior.xy
#     plt.plot(x, y, c="tab:red")
#   plt.show()

  if conv:
    print(f"Converged in {s} iterations")
    break
  # axs[row, s-1-5*row].scatter(points[:, 0], points[:, 1])

plt.figure()
plt.scatter(points[:, 0], points[:, 1])
for region in poly_regions:
  x,y = region.exterior.xy
  plt.plot(x, y, c="tab:red")

finalfigname = "mvtarg_final.png" if MOVING_TARGET else "final.png"
plt.savefig(figpath / finalfigname)

# plt.plot([xmin, xmax], [ymin, ymin])
# plt.plot([xmax, xmax], [ymin, ymax])
# plt.plot([xmax, xmin], [ymax, ymax])
# plt.plot([xmin, xmin], [ymax, ymin])
"""