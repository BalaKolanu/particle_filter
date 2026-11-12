# particle_filter

## Purpose

ROS 2 Monte Carlo localization for F1TENTH. The node estimates the vehicle pose
from a map, LiDAR scans, odometry, and an initial pose estimate. RangeLibc is
used for accelerated 2D ray casting.

## Use

Install dependencies and build from the workspace root:

~~~bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
~~~

Launch localization:

~~~bash
ros2 launch particle_filter localize_launch.py
~~~

Initialize the filter in RViz with a 2D Pose Estimate.

## Configuration

The main file is:

~~~text
config/localize.yaml
~~~

Important parameters include:

- map_server.map: map name under maps/
- laser_frame and tf_child_frame
- laser-to-base offset parameters
- scan and odometry topics
- particle count, update thresholds, and motion-model noise
- range_method and ray-casting settings

Map YAML and image files belong under maps/. The selected map must match the
frame and resolution used by the LiDAR and SLAM pipeline.

## Interfaces

Inputs:

- LiDAR LaserScan
- odometry
- initial pose estimate

Outputs include:

- localized pose and particle-filter visualization topics
- TF and odometry outputs configured by localize.yaml

The launch also uses nav2_map_server and nav2_lifecycle_manager.

## Theory and further reading

The implementation is Monte Carlo Localization: particles represent pose
hypotheses, the motion model propagates them, and LiDAR likelihoods reweight
them against the map. For deeper study, see the
[original CDDT paper](https://arxiv.org/abs/1705.01167), the
[RangeLibc repository](https://github.com/f1tenth/range_libc), and the
[Nav2 map server documentation](https://docs.nav2.org/configuration/packages/map_server/configuring-map-server.html).
