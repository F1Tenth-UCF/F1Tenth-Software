import rclpy
from rclpy.node import Node
import time
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')
        self.prev_time = time.time()

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # Create subscribers and publishers
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            lidarscan_topic,
            self.scan_callback,
            10
        )

        self.ackermann_publisher = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10
        )

        # Set PID gains
        self.kp = 0.3
        self.kd = 0.05
        self.ki = 0.01

        # Store history
        self.integral = 0
        self.prev_error = 0
        
        # Timer to limit the control frequency
        self.timer = self.create_timer(0.07, self.control_loop)
        
        # Store error and velocity
        self.current_error = 0
        self.velocity = 3

    def get_range(self, range_data, angle):
        """
        Helper to return the corresponding range measurement at a given angle.
        """
        angle_rad = np.deg2rad(angle)
        angle_location = int((angle_rad - range_data.angle_min) / range_data.angle_increment)

        if 0 <= angle_location < len(range_data.ranges):
            curr_range = range_data.ranges[angle_location]
            if not np.isfinite(curr_range):
                return 0.0
            return curr_range
        return 0.0

    def get_error(self, range_data, dist):
        """
        Calculates the error to the wall.
        """
        if np.isfinite(range_data):
            return dist - range_data
        return 0.0
    
    def control_loop(self):
        """
        Function called by the timer to actuate control based on PID.
        """
        self.pid_control(self.current_error, self.velocity)

    def pid_control(self, error, velocity):
        """
        Apply PID control to compute the steering angle and velocity.
        """
        current_time = time.time()
        dt = current_time - self.prev_time

        max_integral = 10  # Example threshold

        # PID calculation
        P = self.kp * error
        self.integral += error * dt
        I = self.ki * self.integral
        derivative = (error - self.prev_error) / dt
        D = self.kd * derivative

        angle = (P + I + D)

        # Adjust velocity based on the steering angle
        if abs(angle) < 5:
            velocity = 1.5
        elif abs(angle) < 10:
            velocity = 1.0
        else:
            velocity = 0.5

        self.get_logger().info(f"Steering Angle: {angle}, Speed: {velocity}")
        
        # Publish the control message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = velocity
        self.ackermann_publisher.publish(drive_msg)

        # Update previous error and time for the next iteration
        self.prev_error = error
        self.prev_time = current_time

    def scan_callback(self, msg):
        """
        Callback for LaserScan messages to calculate the error.
        """
        a_angle = -60
        lookahead = 1

        b_laser = self.get_range(msg, -90)  # Range to the wall at 90 degrees
        a_laser = self.get_range(msg, a_angle)  # Range to the wall at an angle
        theta = np.deg2rad(a_angle + 90)

        # Calculate alpha
        if a_laser * np.sin(theta) != 0:
            alpha_var = np.arctan(((a_laser * np.cos(theta)) - b_laser) / (a_laser * np.sin(theta)))
            dt = b_laser * np.cos(alpha_var)
        else:
            return

        # Project car forward using lookahead distance
        dt = dt + (lookahead * np.sin(alpha_var))

        # Update current error based on the desired distance (e.g., 5 meters)
        self.current_error = self.get_error(dt, 1.1)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    wall_follow_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()