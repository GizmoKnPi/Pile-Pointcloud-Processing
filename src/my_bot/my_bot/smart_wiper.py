#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import math
import os

class SmartWiper(Node):
    def __init__(self):
        super().__init__('smart_wiper')
        
        # --- 1. Parameters ---
        self.declare_parameters(namespace='', parameters=[
            ('voxel_size', 0.01),
            ('roi_x_min', 0.0), ('roi_x_max', 3.0),
            ('roi_y_min', -1.0), ('roi_y_max', 1.0),
            ('roi_z_min', -0.5), ('roi_z_max', 1.5),
            ('outlier_neighbors', 20),
            ('outlier_std_ratio', 2.0),
            ('wiper_min_angle', 110.0),
            ('wiper_max_angle', 250.0),
            # NEW: Wall Removal Params
            ('wall_check_enable', True),
            ('wall_distance_threshold', 0.05) # 5cm tolerance for flatness
        ])

        self.get_logger().info("=== Loaded Configuration ===")
        self.get_logger().info(f"Voxel Size: {self.get_parameter('voxel_size').value}")
        self.get_logger().info(f"ROI X: {self.get_parameter('roi_x_min').value} to {self.get_parameter('roi_x_max').value}")
        self.get_logger().info(f"ROI Y: {self.get_parameter('roi_y_min').value} to {self.get_parameter('roi_y_max').value}")
        self.get_logger().info(f"ROI Z: {self.get_parameter('roi_z_min').value} to {self.get_parameter('roi_z_max').value}")
        self.get_logger().info(f"Wiper Angles: {self.get_parameter('wiper_min_angle').value} to {self.get_parameter('wiper_max_angle').value}")
        self.get_logger().info("============================")
        
        # --- 3. Publishers & Services ---
        self.marker_pub = self.create_publisher(MarkerArray, '/scoop_markers', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.pcd_pub = self.create_publisher(PointCloud2, '/cleaned_cloud', 10)
        
        self.create_service(Trigger, 'process_scan', self.process_cb)
        self.get_logger().info("Smart Wiper Ready. Waiting for trigger...")

    def process_cb(self, req, res):
        input_path = "/tmp/raw_scan.pcd"
        if not os.path.exists(input_path):
            res.success = False
            res.message = "No scan file found! Did you save first?"
            self.get_logger().error(res.message)
            return res

        self.get_logger().info("Loading cloud...")
        pcd = o3d.io.read_point_cloud(input_path)
        
        if len(pcd.points) == 0:
            res.success = False
            res.message = "Cloud is empty."
            return res

        # 1. CLEAN (Includes Wall Removal)
        pcd = self.clean_cloud(pcd)
        self.get_logger().info(f"Points after cleaning: {len(pcd.points)}")

        if len(pcd.points) < 10:
            res.success = False
            res.message = "Too few points after cleaning."
            return res
        
        # 2. PUBLISH CLEANED CLOUD
        self.publish_open3d_cloud(pcd)

        # 3. ANALYZE
        center, start, end = self.find_scoop_path(pcd)
        
        # 4. PUBLISH MARKERS
        self.publish_markers(center, start, end)
        self.publish_nav_goal(start, end)
        
        res.success = True
        res.message = "Path Calculated & Cleaned Cloud Published!"
        return res

    def clean_cloud(self, pcd):
        # A. Remove NaNs
        pcd = pcd.remove_non_finite_points()
        
        # B. Voxel Downsample
        voxel = self.get_parameter('voxel_size').value
        pcd = pcd.voxel_down_sample(voxel)
        
        # C. ROI Crop
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(
                self.get_parameter('roi_x_min').value, 
                self.get_parameter('roi_y_min').value, 
                self.get_parameter('roi_z_min').value
            ), 
            max_bound=(
                self.get_parameter('roi_x_max').value, 
                self.get_parameter('roi_y_max').value, 
                self.get_parameter('roi_z_max').value
            ) 
        )
        pcd = pcd.crop(bbox)
        
        # D. Wall Removal (RANSAC) - NEW LOGIC
        if self.get_parameter('wall_check_enable').value and len(pcd.points) > 50:
            pcd = self.remove_walls(pcd)
        
        # E. Outlier Removal (Clean up dust)
        nb = self.get_parameter('outlier_neighbors').value
        std = self.get_parameter('outlier_std_ratio').value
        if len(pcd.points) > nb:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
        
        return pcd

    def remove_walls(self, pcd):
        thresh = self.get_parameter('wall_distance_threshold').value
        
        # Segment the largest plane in the cloud
        plane_model, inliers = pcd.segment_plane(distance_threshold=thresh,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        
        # Plane Equation: ax + by + cz + d = 0
        # Normal Vector: [a, b, c]
        [a, b, c, d] = plane_model
        
        # Check if Vertical:
        # If the plane is a wall, the Normal is horizontal (X or Y aligned).
        # Therefore, 'c' (the Z component) should be small (close to 0).
        # If 'c' is large (close to 1), it's a floor/ceiling.
        
        if abs(c) < 0.2: 
            # It IS a vertical wall (Normal is horizontal)
            self.get_logger().info(f"Vertical Wall Detected (Normal Z={c:.2f}). Removing {len(inliers)} points.")
            # Invert=True keeps everything EXCEPT the wall
            pcd = pcd.select_by_index(inliers, invert=True)
        else:
            self.get_logger().info(f"Largest plane is NOT a wall (Normal Z={c:.2f}). Keeping it.")
            
        return pcd

    def publish_open3d_cloud(self, pcd):
        points = np.asarray(pcd.points)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "base_link"
        cloud_msg = pc2.create_cloud_xyz32(header, points)
        self.pcd_pub.publish(cloud_msg)

    def find_scoop_path(self, pcd):
        points = np.asarray(pcd.points)
        
        # 1. Median Center (The "Heart" of the pile)
        center = np.median(points, axis=0)
        
        min_angle = self.get_parameter('wiper_min_angle').value
        max_angle = self.get_parameter('wiper_max_angle').value
        
        best_angle = 0
        max_length = 0
        
        # We will store the vector (direction) of the best path, not just the point
        best_dir_vec = np.array([1.0, 0.0, 0.0]) 
        
        steps = int(max_angle - min_angle)
        angles = np.linspace(min_angle, max_angle, steps)
        
        # 2. Sweep to find the "Longest Axis" (Shape of the pile)
        for angle_deg in angles:
            angle_rad = np.deg2rad(angle_deg)
            dir_vec = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
            
            vec_from_center = points - center
            projections = np.dot(vec_from_center, dir_vec)
            dist_sq = np.sum(vec_from_center**2, axis=1) - projections**2
            valid_mask = dist_sq < (0.05 ** 2) 
            
            if np.sum(valid_mask) > 5:
                current_length = np.max(projections[valid_mask])
                if current_length > max_length:
                    max_length = current_length
                    best_angle = angle_deg
                    best_dir_vec = dir_vec

        # 3. INTELLIGENT FLIP (The Fix)
        # We have the axis, but we don't know if it points "Front-to-Back" or "Back-to-Front".
        # We calculate both possible start points (Front Edge and Back Edge).
        
        edge_a = center + best_dir_vec * max_length
        edge_b = center - best_dir_vec * max_length
        
        # Robot is always at (0,0,0) in 'base_link'
        dist_a = np.linalg.norm(edge_a) # Distance from Robot to Edge A
        dist_b = np.linalg.norm(edge_b) # Distance from Robot to Edge B
        
        # Pick the point closest to the robot as the "Start"
        if dist_a < dist_b:
            start_pt = edge_a
        else:
            start_pt = edge_b
            
        # End Point is always the Center (we scoop INTO the pile)
        end_pt = center

        # --- NEW: PRINT THE ANGLE ---
        # Calculate the final approach angle based on the decided Start->End direction
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        final_approach_deg = np.degrees(np.arctan2(dy, dx))
        
        self.get_logger().info(f"--------------------------------------------------")
        self.get_logger().info(f"PILE DETECTED:")
        self.get_logger().info(f"   - Axis Alignment: {best_angle:.2f}° (Geometric)")
        self.get_logger().info(f"   - Approach Angle: {final_approach_deg:.2f}° (Robot Frame)")
        self.get_logger().info(f"   - Pile Length:    {max_length:.3f} m")
        self.get_logger().info(f"--------------------------------------------------")
        # ----------------------------

        return center, start_pt, end_pt

    def publish_nav_goal(self, start, end):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = start[0]
        msg.pose.position.y = start[1]
        msg.pose.position.z = 0.0 
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        yaw = math.atan2(dy, dx)
        q = self.quaternion_from_yaw(yaw)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        self.goal_pub.publish(msg)

    def publish_markers(self, center, start, end):
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = "base_link"
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.id = 0
        m.points = [Point(x=start[0], y=start[1], z=start[2]), 
                    Point(x=end[0], y=end[1], z=end[2])]
        m.scale.x = 0.05 
        m.scale.y = 0.1 
        m.color.r = 1.0; m.color.a = 1.0
        ma.markers.append(m)
        s = Marker()
        s.header.frame_id = "base_link"
        s.type = Marker.SPHERE
        s.action = Marker.ADD
        s.id = 1
        s.pose.position.x = center[0]
        s.pose.position.y = center[1]
        s.pose.position.z = center[2]
        s.scale.x = 0.1; s.scale.y = 0.1; s.scale.z = 0.1
        s.color.b = 1.0; s.color.a = 1.0
        ma.markers.append(s)
        self.marker_pub.publish(ma)

    def quaternion_from_yaw(self, yaw):
        return [0.0, 0.0, math.sin(yaw/2), math.cos(yaw/2)]

def main():
    rclpy.init()
    rclpy.spin(SmartWiper())
    rclpy.shutdown()