
import numpy as np
import os

ROBOTS_NUM = 3
OBSTACLES_NUM = 1
GUI = True
ENV_SIZE = 12.0

# read positions from file
file = os.path.join(os.getcwd(), 'pos.txt')
data = []
with open(str(file), 'r') as f:
    lines = f.readlines()

for l in lines:
    data.append(l)

t = np.zeros((len(data), 3))

for i in range(len(data)):
  data[i] = data[i].replace('\n', '')
  t[i] = tuple(map(float, data[i].split(' ')))

# t =  -0.5*ENV_SIZE + ENV_SIZE*np.random.rand(ROBOTS_NUM, 2)
# th = 2*np.pi*np.random.rand(ROBOTS_NUM, 1)
# t = np.hstack((t, th))


cmd = "roslaunch formation_control flightmare_formation.launch "
for i in range(ROBOTS_NUM):
   cmd += f"x{i}:={t[i,0]}" + " " + f"y{i}:={t[i,1]}" + " " + f"th{i}:={t[i,2]}" + " "

# Target position
cmd += f"xt:={t[ROBOTS_NUM, 0]}" + " " + f"yt:={t[ROBOTS_NUM, 1]}" + " " + f"yawt:=0.785 "


if GUI:
  cmd += "gui:=true"
else:
  cmd += "gui:=false"

## Obstacles node
# cmd += " && roslaunch formation_control obstacles.launch"
# for i in range(OBSTACLES_NUM):
#   cmd += f" x{i}:={t[ROBOTS_NUM + 1 + i, 0]}" + " " + f"y{i}:={t[ROBOTS_NUM + 1 + i, 1]}"

print("Sending command:")
print(cmd)
os.system(cmd)

# print("Sending command:")
# print(cmd)

# # run ros launch file
# os.system(cmd)




