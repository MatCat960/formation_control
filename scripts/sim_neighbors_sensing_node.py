#!/usr/bin/env python3
from collections import deque
from functools import partial

import rclpy
from rclpy.node import Node
from arrc_interfaces.msg import Neighbors
from nav_msgs.msg import Odometry





class MinimalPublisher(Node):

    def __init__(self, robot_nr=10):
        super().__init__('neighbor_manager')
        self.publisher_ = self.create_publisher(Neighbors, 'neighbors_odometry', 1)
        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.subs = []
        self.neighbor_dict = {}
        self.message_queue = deque()  # Queue to store messages and timestamps

        for i in range(robot_nr):
            self.subs.append(self.create_subscription(Odometry, f'/Drone{i+1}/odometry', partial(self.listener_callback, f'/Drone{i+1}'), 10))

    def listener_callback(self, index, msg):
        current_time = self.get_clock().now().nanoseconds / 1e6  # Convert to milliseconds
        self.message_queue.append((index, msg, current_time))
        print(f"Message received from {index}. Queue size: {len(self.message_queue)}")

    def timer_callback(self):
        current_time = self.get_clock().now().nanoseconds / 1e6  # Convert to milliseconds
        msg = Neighbors()

        # Process messages that have been in the queue for at least 500 ms
        while self.message_queue:
            index, odom_msg, timestamp = self.message_queue[0]
            if current_time - timestamp >= 500:  # Check if 500 ms have passed
                self.neighbor_dict[index] = odom_msg
                self.message_queue.popleft()  # Remove processed message
            else:
                break

        # Populate the Neighbors message
        for k, v in self.neighbor_dict.items():
            if k == self.get_namespace():
                continue
            msg.neighbors.append(v)

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
