#!/usr/bin/env/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped

class iTTC(Node):
    def __init__(self):
        super().__init__("ittc_node")

        self.current_velocity = None
        self.angle_min = None
        self.angle_increment = None

        self.odom_subscriber = self.create_subscription(
            Odometry,
            "/ego_racecar/odom",
            self.odom_callback,
            10
        )

        self.scan_subscriber = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10
        )

        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            "/drive",
            10
        )
    
    # Odom return the x and y velocities of the vehicle
    def odom_callback(self, msgs):
        self.current_velocity = np.array([msgs.twist.twist.linear.x, msgs.twist.twist.linear.y])
    
    def scan_callback(self, msgs):
        if self.current_velocity is None:
            self.get_logger().info("Velocity is None, skipping calculations.")
            return
        
        # Find the lidar scans, they are an array of how far the nearest object is at each angle
        ranges = np.array(msgs.ranges)
        # Angle Min = Starting Angle
        self.angle_min = msgs.angle_min
        # Angle Increment is how much the angle increments at each index
        self.angle_increment = msgs.angle_increment
        # Init an empty ittc_array
        ittc_array = np.zeros_like(ranges)

        for i, distance in enumerate(ranges):
            # Calculate the current angle by adding the starting angle by the incrememnt * curr_index
            curr_angle = self.angle_min + (i * self.angle_increment)
            # Create a unit vector using cos for x and sin for y
            curr_direction = np.array([np.cos(curr_angle), np.sin(curr_angle)])

            # We can calculate the relative velocity by dot product of the current_velocity and our direction
            relative_velocity = np.dot(self.current_velocity, curr_direction)

            # Calculate the iTTC array for each angle
            if relative_velocity > 0:
                ittc_array[i] = distance / relative_velocity
            else:
                ittc_array[i] = np.inf
        
        try:
            min_value = np.nanmin(ittc_array)
        except ValueError:
            min_value = np.nan

        # Judge Min Value to determine if a crash is immenent
        # If it is, then publish to Ackermann Msgs with a speed of 0.0
        safety_iTTC = 1.0
        if min_value < safety_iTTC:
            self.get_logger().warn("Minimum iTTC is below threshold, publishing speed 0.0.")
            self.emergency_brake()
    
    def emergency_brake(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.drive_publisher.publish(msg)
        self.get_logger().info("/drive successfully published to 0.0")



def main(args=None):
    rclpy.init()
    ittc_node = iTTC()
    rclpy.spin(ittc_node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()