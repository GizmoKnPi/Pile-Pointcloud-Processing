#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger
from std_msgs.msg import Int32

import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import laser_geometry.laser_geometry as lg
import sensor_msgs_py.point_cloud2 as pc2

import open3d as o3d
import numpy as np


class ScanAccumulator(Node):

    def __init__(self):
        super().__init__('scan_accumulator')

        # -------- PARAMETERS --------
        self.declare_parameter('buffer_duration', 10.0)
        self.buffer_duration = self.get_parameter('buffer_duration').value

        # -------- SESSION DATA --------
        self.points = []
        self.scan_start_time = None
        self.is_scanning = False

        # -------- ROS SETUP --------
        self.lp = lg.LaserProjection()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_cb,
            10
        )

        self.create_service(
            Trigger,
            'reset_scan_buffer',
            self.reset_cb
        )

        self.create_service(
            Trigger,
            'save_scan_cloud',
            self.save_cb
        )

        self.count_pub = self.create_publisher(
            Int32,
            '/scan_point_count',
            10
        )

        self.get_logger().info("ScanAccumulator ready")

    # --------------------------------------------------
    # RESET SCAN SESSION
    # --------------------------------------------------

    def reset_cb(self, request, response):

        self.get_logger().info("Starting new scan session")

        self.points = []
        self.scan_start_time = self.get_clock().now()
        self.is_scanning = True

        msg = Int32()
        msg.data = 0
        self.count_pub.publish(msg)

        response.success = True
        response.message = "Scan session started"

        return response

    # --------------------------------------------------
    # SCAN CALLBACK
    # --------------------------------------------------

    def scan_cb(self, msg):

        if not self.is_scanning:
            return

        try:

            trans = self.tf_buffer.lookup_transform(
                'base_link',
                msg.header.frame_id,
                msg.header.stamp,
                rclpy.time.Duration(seconds=0.1)
            )

            cloud_laser = self.lp.projectLaser(msg)
            cloud_base = do_transform_cloud(cloud_laser, trans)

            gen = pc2.read_points(
                cloud_base,
                field_names=("x", "y", "z"),
                skip_nans=True
            )

            for p in gen:
                self.points.append([p[0], p[1], p[2]])

            msg = Int32()
            msg.data = len(self.points)
            self.count_pub.publish(msg)

        except Exception:
            pass

    # --------------------------------------------------
    # SAVE CLOUD
    # --------------------------------------------------

    def save_cb(self, request, response):

        num_points = len(self.points)

        self.get_logger().info(f"Saving cloud with {num_points} points")

        if num_points == 0:
            response.success = False
            response.message = "Buffer empty"
            return response

        try:

            pcd = o3d.geometry.PointCloud()

            pcd.points = o3d.utility.Vector3dVector(
                np.array(self.points, dtype=np.float64)
            )

            save_path = "/tmp/raw_scan.pcd"

            o3d.io.write_point_cloud(save_path, pcd)

            self.get_logger().info(f"Saved → {save_path}")

            self.is_scanning = False

            response.success = True
            response.message = f"Saved {num_points} points"

        except Exception as e:

            self.get_logger().error(str(e))

            response.success = False
            response.message = str(e)

        return response


def main():
    rclpy.init()
    rclpy.spin(ScanAccumulator())
    rclpy.shutdown()


if __name__ == '__main__':
    main()