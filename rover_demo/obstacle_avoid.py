import math
import os
from datetime import datetime

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan


class GoToGoalNode(Node):
    def __init__(self):
        super().__init__("go_to_goal_node")

        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, "/imu", self.imu_callback, 10
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.latest_scan = None
        self.latest_imu = None
        self.x = None
        self.y = None
        self.yaw = None
        self.goal_reached_logged = False
        self.logs_closed = False

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

        # Last command for logging
        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0

        # Create log folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.expanduser(f"~/ros2_ws/logs/run_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.scan_log = open(os.path.join(self.log_dir, "scan_log.txt"), "w", encoding="utf-8", buffering=1)
        self.odom_log = open(os.path.join(self.log_dir, "odom_log.txt"), "w", encoding="utf-8", buffering=1)
        self.imu_log = open(os.path.join(self.log_dir, "imu_log.txt"), "w", encoding="utf-8", buffering=1)
        self.cmd_log = open(os.path.join(self.log_dir, "cmd_vel_log.txt"), "w", encoding="utf-8", buffering=1)
        self.summary_log = open(os.path.join(self.log_dir, "summary_log.txt"), "w", encoding="utf-8", buffering=1)

        self.get_logger().info(
            f"Go-to-goal node started. Goal = ({self.goal_x}, {self.goal_y})"
        )
        self.get_logger().info(f"Logging to: {self.log_dir}")

        self.timer = self.create_timer(0.1, self.control_loop)

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def scan_callback(self, msg):
        self.latest_scan = msg
        if self.logs_closed:
            return

        t = self.now_sec()
        self.scan_log.write(
            f"time={t:.3f}, angle_min={msg.angle_min}, angle_max={msg.angle_max}, "
            f"angle_increment={msg.angle_increment}, ranges={list(msg.ranges)}\n"
        )

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

        if self.logs_closed:
            return

        t = self.now_sec()
        self.odom_log.write(
            f"time={t:.3f}, x={self.x:.4f}, y={self.y:.4f}, yaw={self.yaw:.4f}, "
            f"vx={msg.twist.twist.linear.x:.4f}, wz={msg.twist.twist.angular.z:.4f}\n"
        )

    def imu_callback(self, msg):
        self.latest_imu = msg
        if self.logs_closed:
            return

        t = self.now_sec()
        self.imu_log.write(
            f"time={t:.3f}, "
            f"orientation=({msg.orientation.x}, {msg.orientation.y}, {msg.orientation.z}, {msg.orientation.w}), "
            f"angular_velocity=({msg.angular_velocity.x}, {msg.angular_velocity.y}, {msg.angular_velocity.z}), "
            f"linear_acceleration=({msg.linear_acceleration.x}, {msg.linear_acceleration.y}, {msg.linear_acceleration.z})\n"
        )

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

    def close_logs(self):
        if self.logs_closed:
            return

        for handle in (
            self.scan_log,
            self.odom_log,
            self.imu_log,
            self.cmd_log,
            self.summary_log,
        ):
            handle.close()

        self.logs_closed = True

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
                self.close_logs()
                self.timer.cancel()
            return

        self.goal_reached_logged = False
        goal_heading = math.atan2(dy, dx)
        heading_error = self.normalize_angle(goal_heading - self.yaw)

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
                self.clear_steps_remaining = 35
                self.state = "CLEAR_OBSTACLE"

        elif self.state == "CLEAR_OBSTACLE":
            cmd.linear.x = self.forward_speed
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
                    cmd.linear.x = 0.1
                    cmd.angular.z = (
                        self.turn_speed if heading_error > 0.0 else -self.turn_speed
                    )
                else:
                    self.state = "GO_TO_GOAL"
                    cmd.linear.x = self.forward_speed
                    cmd.angular.z = 0.0

        self.last_cmd_linear = cmd.linear.x
        self.last_cmd_angular = cmd.angular.z

        t = self.now_sec()
        imu_wz = 0.0
        if self.latest_imu is not None:
            imu_wz = self.latest_imu.angular_velocity.z

        self.summary_log.write(
            f"time={t:.3f}, state={self.state}, x={self.x:.4f}, y={self.y:.4f}, "
            f"yaw={self.yaw:.4f}, goal_x={self.goal_x}, goal_y={self.goal_y}, "
            f"dist={distance_to_goal:.4f}, heading_error={heading_error:.4f}, "
            f"front_min={front_min:.4f}, left_min={left_min:.4f}, right_min={right_min:.4f}, "
            f"imu_wz={imu_wz:.4f}, cmd_linear={self.last_cmd_linear:.4f}, cmd_angular={self.last_cmd_angular:.4f}\n"
        )

        self.cmd_log.write(
            f"time={t:.3f}, linear_x={cmd.linear.x:.4f}, angular_z={cmd.angular.z:.4f}, state={self.state}\n"
        )

        self.cmd_pub.publish(cmd)

    def destroy_node(self):
        self.close_logs()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GoToGoalNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.cmd_pub.publish(Twist())
        except Exception:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
