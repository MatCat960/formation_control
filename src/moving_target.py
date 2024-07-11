#!/usr/bin/env python3

import numpy as np
import math

import rospy
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

VMAX = 0.25

class Target():
    def __init__(self):
        rospy.init_node("target_node")

        self.width = rospy.get_param("~AREA_W", 20.0)
        self.id = rospy.get_param("~TARGET_ID", 0)
        self.dt = rospy.get_param("~dt", 0.2)
        self.width = rospy.get_param("~AREA_W", 20.0)
        self.ROBOT_RANGE = rospy.get_param("~ROBOT_RANGE", 3.0)
        self.frame_id = rospy.get_param("~frame_id", "world")

        self.timer = rospy.Timer(rospy.Duration(self.dt), self.timer_callback)
        self.odomSub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.velPub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        
        self.vel_msg = Twist()

        # start in origin with random theta
        # self.target = np.zeros(3,)
        # self.target[2] = 2*math.pi*np.random.rand()
        self.target = np.ones(3,)

    def odom_callback(self, msg):
        self.target[0] = msg.pose.pose.position.x
        self.target[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        (roll, pitch, yaw) = euler_from_quaternion(quat)
        self.target[2] = yaw


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

        # move target
        vel = np.array([new_x, new_y]) - self.target[:2]
        b = 0.5 * self.ROBOT_RANGE
        Kp = 0.8
        T = np.array([[math.cos(th), math.sin(th)],
                    [-1/b * math.sin(th), 1/b * math.cos(th)]])
        vel = np.matmul(T, vel)
        v = Kp * vel[0]
        w = Kp * vel[1]
        
        # publish ros msg
        self.vel_msg.linear.x = v * math.cos(th)
        self.vel_msg.linear.y = v * math.sin(th)
        self.vel_msg.angular.z = w
        self.velPub.publish(self.vel_msg)


def main(args=None):
    targetNode = Target()    
    rospy.spin()


if __name__ == '__main__':
    main()



    
