#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from std_srvs.srv import Trigger
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import laser_geometry.laser_geometry as lg
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np

class ScanAccumulator(Node):
    def __init__(self):
        super().__init__('scan_accumulator')
        
        self.accumulated_points = []
        self.is_scanning = True
        self.lp = lg.LaserProjection()
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_service(Trigger, 'save_scan_cloud', self.save_cb)

        self.get_logger().info("Accumulator Ready.")

    def scan_cb(self, msg):
        if not self.is_scanning: return
        
        try:
            # 1. Get Transform
            trans = self.tf_buffer.lookup_transform(
                'base_link', 
                msg.header.frame_id, 
                msg.header.stamp,
                rclpy.time.Duration(seconds=0.1)
            )
            
            # 2. Project & Transform
            cloud_laser = self.lp.projectLaser(msg)
            cloud_base = do_transform_cloud(cloud_laser, trans)
            
            # 3. FIX: Explicitly extract X, Y, Z as standard floats
            # skip_nans=True is important!
            gen = pc2.read_points(cloud_base, field_names=("x", "y", "z"), skip_nans=True)
            
            # Convert the generator to a standard list of lists [[x,y,z], [x,y,z]]
            # This fixes the "numpy.void" error
            batch = [[p[0], p[1], p[2]] for p in gen]
            
            if batch:
                self.accumulated_points.extend(batch)

        except Exception:
            pass

    def save_cb(self, request, response):
        self.is_scanning = False
        num_points = len(self.accumulated_points)
        self.get_logger().info(f"Stopping scan. Processing {num_points} points...")
        
        if num_points == 0:
            response.success = False
            response.message = "No points accumulated!"
            return response

        # 4. Create Open3D Cloud
        try:
            pcd = o3d.geometry.PointCloud()
            # Now this works because accumulated_points is a simple list of floats
            pcd.points = o3d.utility.Vector3dVector(np.array(self.accumulated_points, dtype=np.float64))
            
            save_path = "/tmp/raw_scan.pcd"
            o3d.io.write_point_cloud(save_path, pcd)
            
            response.success = True
            response.message = f"Saved {num_points} pts to {save_path}"
            self.get_logger().info(f"SUCCESS: Saved to {save_path}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save: {e}")
            response.success = False
            response.message = str(e)
        
        # Reset
        self.accumulated_points = []
        self.is_scanning = True
        return response

def main():
    rclpy.init()
    rclpy.spin(ScanAccumulator())
    rclpy.shutdown()