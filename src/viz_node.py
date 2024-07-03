#!/usr/bin/env python3

import numpy as np
import math

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

rospy.init_node("viz_node")

width = rospy.get_param("~AREA_W", 20.0)
frame_id = rospy.get_param("~frame_id", "odom")
print("Frame id: ", frame_id)

# timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)
bound_pub = rospy.Publisher("boundaries", Marker, queue_size=1)

bound_msg = Marker()
bound_msg.header.frame_id = frame_id
bound_msg.id = 0
bound_msg.type = 4     # LineStrip
bound_msg.action = Marker.ADD

# color
bound_msg.color.r = 1.0
bound_msg.color.a = 1.0

bound_msg.scale.x = 0.1

# position
square_points = [
        Point(-0.5*width, -0.5*width, 0.0),
        Point(0.5*width, -0.5*width, 0.0),
        Point(0.5*width, 0.5*width, 0.0),
        Point(-0.5*width, 0.5*width, 0.0),
        Point(-0.5*width, -0.5*width, 0.0) # Closing the loop
    ]

bound_msg.points.extend(square_points)

while not rospy.is_shutdown():
    bound_pub.publish(bound_msg)

        




    
