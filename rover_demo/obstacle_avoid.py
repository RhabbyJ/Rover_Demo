import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class GoToGoalNode(Node):
    def __init__(self):
        super().__init__("go_to_goal_node")

        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.latest_scan = None
        self.x = None
        self.y = None
        self.yaw = None
        self.goal_reached_logged = False

        # Goal
        self.goal_x = 4.0
        self.goal_y = 0.0

        # Motion / thresholds
        self.safe_distance = 0.9
        self.goal_tolerance = 0.3
        self.forward_speed = 0.2
        self.turn_speed = 0.5
        self.avoid_forward_speed = 0.08
        self.heading_tolerance = 0.2

        # State machine
        self.state = "GO_TO_GOAL"
        self.avoid_direction = 1.0  # +1 left, -1 right
        self.avoid_steps_remaining = 0
        self.max_avoid_steps = 18  # about 1.8 sec at 10 Hz
        self.clear_steps_remaining = 0

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info(
            f"Go-to-goal node started. Goal = ({self.goal_x}, {self.goal_y})"
        )

    def scan_callback(self, msg):
        self.latest_scan = msg

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

    def quaternion_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def get_min_distance(self, data):
        valid = [d for d in data if not math.isinf(d) and not math.isnan(d)]
        if len(valid) == 0:
            return float("inf")
        return min(valid)

    def get_scan_sectors(self):
        if self.latest_scan is None:
            return None, None, None

        ranges = list(self.latest_scan.ranges)
        n = len(ranges)
        if n == 0:
            return None, None, None

        center = n // 2
        front_width = n // 14
        side_width = n // 5

        front = ranges[max(0, center - front_width) : min(n, center + front_width)]
        left = ranges[
            min(n, center + front_width) : min(n, center + front_width + side_width)
        ]
        right = ranges[
            max(0, center - front_width - side_width) : max(0, center - front_width)
        ]

        front_min = self.get_min_distance(front)
        left_min = self.get_min_distance(left)
        right_min = self.get_min_distance(right)

        return front_min, left_min, right_min

    def control_loop(self):
        if self.x is None or self.y is None or self.yaw is None:
            return

        front_min, left_min, right_min = self.get_scan_sectors()
        if front_min is None:
            return

        cmd = Twist()

        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        distance_to_goal = math.sqrt(dx * dx + dy * dy)

        if distance_to_goal < self.goal_tolerance:
            self.cmd_pub.publish(Twist())
            if not self.goal_reached_logged:
                self.get_logger().info("Goal reached. Stopping.")
                self.goal_reached_logged = True
            return
        else:
            self.goal_reached_logged = False

        goal_heading = math.atan2(dy, dx)
        heading_error = self.normalize_angle(goal_heading - self.yaw)

        # --- STATE MACHINE ---

        if self.state == "GO_TO_GOAL":
            if front_min < self.safe_distance:
                if left_min > right_min:
                    self.avoid_direction = 1.0
                else:
                    self.avoid_direction = -1.0

                self.avoid_steps_remaining = self.max_avoid_steps
                self.state = "AVOID_OBSTACLE"
            else:
                if abs(heading_error) > self.heading_tolerance:
                    cmd.linear.x = 0.0
                    cmd.angular.z = (
                        self.turn_speed if heading_error > 0.0 else -self.turn_speed
                    )
                else:
                    cmd.linear.x = self.forward_speed
                    cmd.angular.z = 0.0

        elif self.state == "AVOID_OBSTACLE":
            cmd.linear.x = self.avoid_forward_speed
            cmd.angular.z = self.avoid_direction * self.turn_speed
            self.avoid_steps_remaining -= 1

            if self.avoid_steps_remaining <= 0 and front_min > (
                self.safe_distance + 0.2
            ):
                # Increased clear steps to drive further past the object
                self.clear_steps_remaining = 35
                self.state = "CLEAR_OBSTACLE"

        elif self.state == "CLEAR_OBSTACLE":
            cmd.linear.x = self.forward_speed
            # Add a slight outward curve to push away from the edge of the obstacle
            cmd.angular.z = self.avoid_direction * 0.15
            self.clear_steps_remaining -= 1

            if front_min < self.safe_distance:
                self.avoid_steps_remaining = self.max_avoid_steps
                self.state = "AVOID_OBSTACLE"
            elif self.clear_steps_remaining <= 0:
                self.state = "RECOVER_TO_GOAL"

        elif self.state == "RECOVER_TO_GOAL":
            if front_min < self.safe_distance:
                self.avoid_steps_remaining = self.max_avoid_steps
                self.state = "AVOID_OBSTACLE"
            else:
                if abs(heading_error) > self.heading_tolerance:
                    # Move forward slightly while turning to ARC around the edge
                    # instead of spinning in place!
                    cmd.linear.x = 0.1
                    cmd.angular.z = (
                        self.turn_speed if heading_error > 0.0 else -self.turn_speed
                    )
                else:
                    self.state = "GO_TO_GOAL"
                    cmd.linear.x = self.forward_speed
                    cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = GoToGoalNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.cmd_pub.publish(Twist())
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
