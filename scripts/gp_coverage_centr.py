import numpy as np
import matplotlib.pyplot as plt
import os

# GP stuff
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression

# Coverage stuff
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely import Polygon, Point, intersection
from tqdm import tqdm
from pathlib import Path



### PARAMETERS
epochs = 100
ROBOTS_NUM = 4
AREA_W = 20.0
vmax = 3.0
path = Path().resolve()



### UTILITY FUCTIONS
def mirror(points):
    mirrored_points = []

    # Define the corners of the square
    # square_corners = [(-0.5*AREA_W, -0.5*AREA_W), (0.5*AREA_W, -0.5*AREA_W), (0.5*AREA_W, 0.5*AREA_W), (-0.5*AREA_W, 0.5*AREA_W)]
    square_corners = [(0.0, 0.0), (AREA_W, 0.0), (AREA_W, AREA_W), (0.0, AREA_W)]

    # Mirror points across each edge of the square
    for edge_start, edge_end in zip(square_corners, square_corners[1:] + [square_corners[0]]):
        edge_vector = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])

        for point in points:
            # Calculate the vector from the edge start to the point
            point_vector = (point[0] - edge_start[0], point[1] - edge_start[1])

            # Calculate the mirrored point by reflecting across the edge
            mirrored_vector = (point_vector[0] - 2 * (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) / (edge_vector[0]**2 + edge_vector[1]**2) * edge_vector[0],
                               point_vector[1] - 2 * (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) / (edge_vector[0]**2 + edge_vector[1]**2) * edge_vector[1])

            # Translate the mirrored vector back to the absolute coordinates
            mirrored_point = (edge_start[0] + mirrored_vector[0], edge_start[1] + mirrored_vector[1])

            # Add the mirrored point to the result list
            mirrored_points.append(mirrored_point)

    return mirrored_points



### MAIN CODE


# Generate target and dataset
targets = np.array(([2.0, 3.0], [15.0, 15.0]))
targets.shape
target = targets[0]

train_ids = []
GRID_SIZE = 100
X = np.zeros((GRID_SIZE**2, 2))
Y = np.zeros(GRID_SIZE**2)
for i in range(0, GRID_SIZE**2, GRID_SIZE):
  X[i:i+GRID_SIZE, 0] = AREA_W*i/(GRID_SIZE**2)
  for j in range(0, GRID_SIZE):
    X[i+j, 1] = AREA_W*j/GRID_SIZE
    x_ij = X[i+j, :]
    for target in targets:
        dist_t = np.linalg.norm(target - x_ij)
        if dist_t < 2.0:
            Y[i+j] = 1
    if x_ij[0] <= 5.0 and x_ij[1] <= 5.0:
      train_ids.append(i+j)


NUM_STEPS = 5
ROBOT_RANGE = 3.0

points = AREA_W * np.random.rand(ROBOTS_NUM, 2)
robots_hist = np.zeros((1, points.shape[0], points.shape[1]))
robots_hist[0, :, :] = points

detections_ids = []
# for _ in range(ROBOTS_NUM):
#   detections_ids.append([])

for s in range(1, NUM_STEPS+1):
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
  fig, axs = plt.subplots(2, ROBOTS_NUM//2, figsize=(16,9))
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
    robot = vor.points[idx]
    # plt.scatter(robot[0], robot[1])

    # Intersect with robot range
    step = 0.5
    range_pts = []
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
    # centr = np.array([lim_region.centroid.x, lim_region.centroid.y])
    # dist = np.linalg.norm(robot-centr)
    # points[idx, :] = robot + 0.5 * (centr - robot)

    # Simulate detections inside sensing region
    for i in range(X.shape[0]):
      x_i = X[i, :]
      x_pt = Point(X[i, 0], X[i, 1])
      d = np.linalg.norm(robot - x_i)
      if d <= ROBOT_RANGE and i not in detections_ids:        # only append points not visited yet
        detections_ids.append(i)
      if poly.contains(x_pt):                                 # Define points inside voronoi cell
        voronoi_ids.append(i)


    # Define training set
    # X_train, y_train = X[detections_ids], y[detections_ids]
    X_train, y_train = X[detections_ids], Y[detections_ids]

    if 1 not in y_train:
      y_train[np.random.randint(y_train.shape[0])] = 1


    ### Training
    gp = GaussianProcessClassifier()
    gp.fit(X_train, y_train)

    # Prediction on the whole environment
    y_pred = gp.predict(X)
    y_prob = gp.predict_proba(X)




    # Calculate centroid
    X_voro = X[voronoi_ids]
    y_probs_voro = y_prob[voronoi_ids]
    A = 0.0
    Cx = 0.0; Cy = 0.0
    for j in range(X_voro.shape[0]):
      A += y_probs_voro[j, 1]
      Cx += X_voro[j, 0] * y_probs_voro[j,1]
      Cy += X_voro[j, 1] * y_probs_voro[j,1]

    Cx = Cx / A
    Cy = Cy / A

    # Save fig
    y_prob_vis = y_prob[:, 0]
    y_prob_vis[np.random.randint(y_prob_vis.shape[0])] = 0.0
    y_prob_vis[np.random.randint(y_prob_vis.shape[0])] = 1.0
    axs[row, idx-row*ROBOTS_NUM].scatter(X[:, 0], X[:, 1], c=y_prob_vis, cmap="gray")
    # axs[row, row*ROBOTS_NUM+i].set_title("Probability")
    axs[row, idx-row*ROBOTS_NUM].scatter(robot[0], robot[1], label="Robot")
    axs[row, idx-row*ROBOTS_NUM].scatter(Cx, Cy, label="Centroid")
    print(f"Centroid: {Cx}, {Cy}")
    x,y = poly.exterior.xy
    axs[row, idx-row*ROBOTS_NUM].plot(x, y, c="tab:red")
    
    # axs[1].scatter(Cx, Cy)
    axs[row, idx-row*ROBOTS_NUM].legend()
    # plt.show()

    # Move robot towards centroid
    centr = np.array([Cx, Cy]).transpose()
    # print(f"Robot: {robot}")
    # print(f"Centroid: {centr}")
    dist = np.linalg.norm(robot-centr)
    vel = 0.8 * (centr - robot)
    vel[0] = max(-vmax, min(vmax, vel[0]))
    vel[1] = max(-vmax, min(vmax, vel[1]))

    points[idx, :] = robot + vel

    if dist > 0.1:
      conv = False

  # Save positions for visualization
  # robots_hist = np.vstack((robots_hist, np.expand_dims(points, axis=0)))
  # vis_regions.append(lim_regions)

  plt.savefig(path / f"pics/img{s}.png")
  plt.show()

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


plt.savefig(path / "pics/final.png")

# plt.plot([xmin, xmax], [ymin, ymin])
# plt.plot([xmax, xmax], [ymin, ymax])
# plt.plot([xmax, xmin], [ymax, ymax])
# plt.plot([xmin, xmin], [ymax, ymin])