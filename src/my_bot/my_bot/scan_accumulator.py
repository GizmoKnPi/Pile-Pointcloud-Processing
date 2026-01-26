#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import laser_geometry.laser_geometry as lg
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
from collections import deque

class ScanAccumulator(Node):
    def __init__(self):
        super().__init__('scan_accumulator')
        
        # --- PARAMETERS ---
        self.declare_parameter('buffer_duration', 10.0) # Keep last 10 seconds of data
        
        # We use a DEQUE (Double Ended Queue) for efficient "Sliding Window"
        # Format: [ (timestamp, [points...]), (timestamp, [points...]) ]
        self.scan_buffer = deque()
        
        self.is_scanning = True
        self.lp = lg.LaserProjection()
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_service(Trigger, 'save_scan_cloud', self.save_cb)

        duration = self.get_parameter('buffer_duration').value
        self.get_logger().info(f"Accumulator Ready. Buffer Window: {duration} seconds")

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
            
            # 2. Convert & Transform
            cloud_laser = self.lp.projectLaser(msg)
            cloud_base = do_transform_cloud(cloud_laser, trans)
            
            # 3. Extract Points
            gen = pc2.read_points(cloud_base, field_names=("x", "y", "z"), skip_nans=True)
            batch = [[p[0], p[1], p[2]] for p in gen]
            
            if batch:
                # 4. Add to Buffer with TIMESTAMP
                current_time = self.get_clock().now().nanoseconds
                self.scan_buffer.append((current_time, batch))
                
                # 5. PRUNE OLD DATA (The Sliding Window Logic)
                # Convert seconds to nanoseconds (1 sec = 1e9 ns)
                max_age_ns = self.get_parameter('buffer_duration').value * 1e9
                
                # While the oldest packet is too old, throw it away
                while self.scan_buffer:
                    oldest_time, _ = self.scan_buffer[0]
                    if (current_time - oldest_time) > max_age_ns:
                        self.scan_buffer.popleft() # Remove from left (Oldest)
                    else:
                        break # Oldest is fresh enough, stop checking

        except Exception:
            pass

    def save_cb(self, request, response):
        # Flatten the buffer into a single list of points
        all_points = []
        for _, batch in self.scan_buffer:
            all_points.extend(batch)
            
        num_points = len(all_points)
        self.get_logger().info(f"Processing snapshot of last {self.get_parameter('buffer_duration').value}s. Points: {num_points}")
        
        if num_points == 0:
            response.success = False
            response.message = "Buffer is empty!"
            return response

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(all_points, dtype=np.float64))
            
            save_path = "/tmp/raw_scan.pcd"
            o3d.io.write_point_cloud(save_path, pcd)
            
            response.success = True
            response.message = f"Saved {num_points} pts"
            self.get_logger().info(f"Saved to {save_path}")
            
        except Exception as e:
            self.get_logger().error(f"Save failed: {e}")
            response.success = False
            response.message = str(e)
        
        # Note: We do NOT clear the buffer here.
        # This allows you to save again instantly without waiting 10s to fill up again.
        return response

def main():
    rclpy.init()
    rclpy.spin(ScanAccumulator())
    rclpy.shutdown()