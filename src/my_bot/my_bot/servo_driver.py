#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool
import serial
import math

class ServoDriver(Node):

    def __init__(self):
        super().__init__('servo_driver')

        self.declare_parameters(namespace='', parameters=[
            ('serial_port', '/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0'),
            ('baud_rate', 921600),
            ('lag_offset', -2.0)
        ])

        self.create_service(SetBool, 'toggle_servo_mode', self.mode_cb)

        # Servo joint publisher (unchanged)
        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        # Bucket state publisher
        self.bucket_pub = self.create_publisher(String, '/bucket_position', 10)

        port = self.get_parameter('serial_port').value
        baud = self.get_parameter('baud_rate').value
        self.lag_offset = self.get_parameter('lag_offset').value

        try:
            self.ser = serial.Serial(port, baud, timeout=0.01)
            self.get_logger().info(f"Connected to Servo at {port}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect: {e}")

        self.prev_angle = 0.0
        self.last_bucket_state = "UNKNOWN"

        self.timer = self.create_timer(0.01, self.read_serial)

    def read_serial(self):

        if not hasattr(self, 'ser') or not self.ser.is_open:
            return

        while self.ser.in_waiting:

            try:
                line = self.ser.readline().decode().strip()

                if not line:
                    continue

                parts = line.split(',')

                # Expect: timestamp,angle,state
                if len(parts) < 3:
                    continue

                raw_angle = float(parts[1])
                bucket_state = parts[2]

                # ---- Lag Compensation ----
                final_angle = raw_angle

                if raw_angle > self.prev_angle:
                    final_angle += self.lag_offset
                elif raw_angle < self.prev_angle:
                    final_angle -= self.lag_offset

                self.prev_angle = raw_angle

                # ---- Publish Servo Joint ----
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = ['servo_mount_joint']
                msg.position = [math.radians(-final_angle)]

                self.pub.publish(msg)

                # ---- Publish Bucket State if changed ----
                if bucket_state != self.last_bucket_state:

                    bucket_msg = String()
                    bucket_msg.data = bucket_state

                    self.bucket_pub.publish(bucket_msg)

                    self.last_bucket_state = bucket_state

            except Exception as e:
                self.get_logger().warn(f"Serial parse error: {e}")

    def mode_cb(self, request, response):

        if self.ser and self.ser.is_open:

            if request.data:
                self.ser.write(b's')
            else:
                self.ser.write(b'p')

            response.success = True

        return response


def main():
    rclpy.init()
    rclpy.spin(ServoDriver())
    rclpy.shutdown()