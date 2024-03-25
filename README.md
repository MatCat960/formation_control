
# formation_control

Formation Control for cooperative area coverage by a Multi-Robot System.

## Dependencies

- ROS Noetic
- [osqp-eigen](https://github.com/robotology/osqp-eigen)
- [gmm_msgs](https://github.com/MatCat960/gmm_msgs) - branch `ros1-noetic`
- [gmm_coverage](https://github.com/MatCat960/gmm_coverage) - branch `ros1-noetic`

## How it works

Robots within the MRS are split into clusters for cooperative area coverage. The collectively monitored area results from the intersection of FoVs of robots in a cluster:
`![Cluster's sensing model](pics/area_inters.png)`