#!/usr/bin/env python3

import numpy as np
import math

import rospy
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

VMAX = 0.25

class Target():
    def __init__(self):
        rospy.init_node("target_node")

        self.width = rospy.get_param("~AREA_W", 20.0)
        self.id = rospy.get_param("~TARGET_ID", 0)
        self.dt = rospy.get_param("~dt", 0.2)
        self.frame_id = rospy.get_param("~frame_id", "world")

        self.timer = rospy.Timer(rospy.Duration(self.dt), self.timer_callback)
        self.odomPub = rospy.Publisher("odom", Odometry, queue_size=1)
        
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = self.frame_id
        self.odom_msg.child_frame_id = f"target_{self.id}"

        # start in origin with random theta
        # self.target = np.zeros(3,)
        # self.target[2] = 2*math.pi*np.random.rand()
        self.target = np.ones(3,)

    def timer_callback(self, e):
        th = self.target[2]
        new_x = self.target[0] + VMAX * np.cos(th)*self.dt
        new_y = self.target[1] + VMAX * np.sin(th)*self.dt

        # check if outside
        if new_x < -0.5*self.width + 1:
            new_x = -0.5*self.width + 1
            th = -th - 0.5*np.pi + np.pi*np.random.rand()
        elif new_x > 0.5*self.width - 1:
            new_x = 0.5*self.width - 1
            th = -th - 0.5*np.pi + np.pi*np.random.rand() 
        
        if new_y < -0.5*self.width + 1:
            new_y = -0.5*self.width + 1
            th = -th - 0.5*np.pi + np.pi*np.random.rand()

        elif new_y > 0.5*self.width - 1:
            new_y = 0.5*self.width - 1
            th = -th - 0.5*np.pi + np.pi*np.random.rand()

        # update target
        self.target = np.array([new_x, new_y, th])
        
        # publish ros msg
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, th)
        self.odom_msg.pose.pose.position.x = self.target[0]
        self.odom_msg.pose.pose.position.y = self.target[1]
        self.odom_msg.pose.pose.orientation.x = quaternion[0]
        self.odom_msg.pose.pose.orientation.y = quaternion[1]
        self.odom_msg.pose.pose.orientation.z = quaternion[2]
        self.odom_msg.pose.pose.orientation.w = quaternion[3]
        self.odom_msg.header.stamp = rospy.Time.now()
        self.odomPub.publish(self.odom_msg)

def main(args=None):
    targetNode = Target()    
    rospy.spin()


if __name__ == '__main__':
    main()



    
