"""Restart the particle filter when its pose output remains too slow."""

from collections import deque
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


class PoseRateWatchdog(Node):
    """Monitor a pose topic and request a supervised particle-filter restart."""

    def __init__(self):
        super().__init__('pose_rate_watchdog')
        self.declare_parameter('pose_topic', '/pf/pose/odom')
        self.declare_parameter('restart_service', '/particle_filter/restart')
        self.declare_parameter('minimum_rate_hz', 30.0)
        self.declare_parameter('failure_duration_sec', 5.0)
        self.declare_parameter('check_period_sec', 0.5)
        self.declare_parameter('autonomy_set_topic', '/safety/autonomy_set')
        self.declare_parameter('autonomy_enabled_topic', '/safety/autonomy_enabled')
        self.declare_parameter('autonomy_disable_guard_sec', 0.5)

        self.pose_topic = str(self.get_parameter('pose_topic').value)
        self.restart_service_name = str(self.get_parameter('restart_service').value)
        self.minimum_rate_hz = float(self.get_parameter('minimum_rate_hz').value)
        self.failure_duration_sec = float(
            self.get_parameter('failure_duration_sec').value
        )
        self.check_period_sec = float(self.get_parameter('check_period_sec').value)
        self.autonomy_set_topic = str(
            self.get_parameter('autonomy_set_topic').value
        )
        self.autonomy_enabled_topic = str(
            self.get_parameter('autonomy_enabled_topic').value
        )
        self.autonomy_disable_guard_sec = float(
            self.get_parameter('autonomy_disable_guard_sec').value
        )

        self.received_times = deque()
        self.first_observation_time = None
        self.restart_requested = False
        self.restart_pending_time = None
        self.restart_client = self.create_client(
            Trigger, self.restart_service_name
        )
        self.autonomy_set_pub = self.create_publisher(
            Bool, self.autonomy_set_topic, 10
        )
        self.autonomy_enabled_pub = self.create_publisher(
            Bool, self.autonomy_enabled_topic, 10
        )
        self.pose_sub = self.create_subscription(
            Odometry, self.pose_topic, self._pose_callback, 10
        )
        self.check_timer = self.create_timer(
            self.check_period_sec, self._check_rate
        )
        self.get_logger().info(
            'Pose watchdog active: topic=%s minimum_rate=%.1f Hz '
            'failure_duration=%.1f s'
            % (self.pose_topic, self.minimum_rate_hz, self.failure_duration_sec)
        )

    def _pose_callback(self, _msg):
        now = time.monotonic()
        if self.restart_requested:
            # The supervised PF may have restarted before the Trigger future
            # completed. Treat the next pose as the beginning of a new window.
            self.received_times.clear()
            self.first_observation_time = now
            self.restart_requested = False
            self.restart_pending_time = None
        if self.first_observation_time is None:
            self.first_observation_time = now
        self.received_times.append(now)
        self._discard_old(now)

    def _discard_old(self, now):
        cutoff = now - self.failure_duration_sec
        while self.received_times and self.received_times[0] < cutoff:
            self.received_times.popleft()

    def _check_rate(self):
        if self.restart_requested:
            return

        now = time.monotonic()
        if self.restart_pending_time is not None:
            self._publish_autonomy_disabled()
            if now - self.restart_pending_time < self.autonomy_disable_guard_sec:
                return
            self._request_particle_filter_restart()
            return

        if self.first_observation_time is None:
            return

        if now - self.first_observation_time < self.failure_duration_sec:
            return

        self._discard_old(now)
        observed_rate = len(self.received_times) / self.failure_duration_sec
        if observed_rate >= self.minimum_rate_hz:
            return

        self.restart_pending_time = now
        self.get_logger().error(
            'Pose rate is %.2f Hz for %.2f s (< %.2f Hz); requesting '
            'particle-filter restart after disabling autonomy.'
            % (observed_rate, self.failure_duration_sec, self.minimum_rate_hz)
        )
        self._publish_autonomy_disabled()

    def _publish_autonomy_disabled(self):
        msg = Bool()
        msg.data = False
        # The setter is the canonical path; publishing the state topic too
        # gives controllers an immediate safety gate during the guard period.
        self.autonomy_set_pub.publish(msg)
        self.autonomy_enabled_pub.publish(msg)

    def _request_particle_filter_restart(self):
        self.restart_requested = True
        if not self.restart_client.wait_for_service(timeout_sec=1.0):
            self.restart_requested = False
            self.restart_pending_time = None
            self.get_logger().error(
                'Particle-filter restart service is unavailable; watchdog '
                'will retry.'
            )
            return
        future = self.restart_client.call_async(Trigger.Request())
        future.add_done_callback(self._restart_response)

    def _restart_response(self, future):
        try:
            response = future.result()
            if not response.success:
                self.restart_requested = False
                self.restart_pending_time = None
                self.get_logger().error(
                    'Particle-filter restart request was rejected: %s'
                    % response.message
                )
        except Exception as exc:
            self.restart_requested = False
            self.restart_pending_time = None
            self.get_logger().error(
                'Particle-filter restart request failed: %s' % exc
            )


def main(args=None):
    rclpy.init(args=args)
    watchdog = PoseRateWatchdog()
    try:
        rclpy.spin(watchdog)
    finally:
        watchdog.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
