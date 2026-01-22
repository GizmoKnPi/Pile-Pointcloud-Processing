#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import serial
import math

class ServoDriver(Node):
    def __init__(self):
        super().__init__('servo_driver')
        self.declare_parameters(namespace='', parameters=[
            ('serial_port', '/dev/ttyUSB1'),
            ('baud_rate', 921600),
            ('lag_offset', -2.0)
        ])
        
        # Publisher for URDF Update
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # Serial Setup
        port = self.get_parameter('serial_port').value
        baud = self.get_parameter('baud_rate').value
        self.lag_offset = self.get_parameter('lag_offset').value
        
        try:
            self.ser = serial.Serial(port, baud, timeout=0.01)
            self.get_logger().info(f"Connected to Servo at {port}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect: {e}")

        self.prev_angle = 0.0
        self.timer = self.create_timer(0.01, self.read_serial)

    def read_serial(self):
        if not hasattr(self, 'ser') or not self.ser.is_open: return

        while self.ser.in_waiting:
            try:
                line = self.ser.readline().decode().strip()
                if not line: continue
                
                parts = line.split(',')
                if len(parts) < 2: continue
                
                raw_angle = float(parts[1])
                
                # --- Lag Compensation ---
                final_angle = raw_angle
                if raw_angle > self.prev_angle:
                    final_angle += self.lag_offset
                elif raw_angle < self.prev_angle:
                    final_angle -= self.lag_offset
                self.prev_angle = raw_angle

                # --- Publish Joint State ---
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = ['servo_mount_joint']
                # Convert Deg to Rad and invert if needed (based on your specific mount)
                msg.position = [math.radians(-final_angle)] 
                self.pub.publish(msg)

            except ValueError:
                pass

def main():
    rclpy.init()
    rclpy.spin(ServoDriver())
    rclpy.shutdown()