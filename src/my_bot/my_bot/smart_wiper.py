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
        
        # 1. Pivot Center
        pivot_center = np.median(points, axis=0)
        
        min_angle = self.get_parameter('wiper_min_angle').value
        max_angle = self.get_parameter('wiper_max_angle').value
        
        # SCORE TRACKING
        best_combined_score = -1.0
        best_dir_vec = np.array([1.0, 0.0, 0.0]) 
        best_max_proj = 0.0
        best_min_proj = 0.0
        best_point_count = 0  # Just for logging
        best_length = 0       # Just for logging
        
        # Pre-calculate 2D data
        points_xy = points[:, :2] 
        center_xy = pivot_center[:2]
        
        steps = int(max_angle - min_angle)
        angles = np.linspace(min_angle, max_angle, steps)
        
        # --- SWEEP ---
        for angle_deg in angles:
            angle_rad = np.deg2rad(angle_deg)
            dir_xy = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            
            vec_xy = points_xy - center_xy
            projections = np.dot(vec_xy, dir_xy)
            dist_sq_xy = np.sum(vec_xy**2, axis=1) - projections**2
            
            # Width Check
            valid_mask = dist_sq_xy < (0.05 ** 2) 
            
            count = np.sum(valid_mask)
            
            if count > 5:
                # Calculate Length
                valid_projections = projections[valid_mask]
                curr_min = np.min(valid_projections)
                curr_max = np.max(valid_projections)
                length = curr_max - curr_min
                
                # --- THE WIN-WIN FORMULA ---
                # Score = Length * Mass
                # This rewards paths that are BOTH long AND dense.
                score = length * count
                
                if score > best_combined_score:
                    best_combined_score = score
                    best_dir_vec = np.array([dir_xy[0], dir_xy[1], 0.0])
                    best_min_proj = curr_min
                    best_max_proj = curr_max
                    # Save stats for the log report
                    best_point_count = count
                    best_length = length

        # --- DENSITY PEAK (Blue Ball) ---
        final_dir_xy = best_dir_vec[:2]
        vec_xy = points_xy - center_xy
        projections = np.dot(vec_xy, final_dir_xy)
        dist_sq_xy = np.sum(vec_xy**2, axis=1) - projections**2
        
        valid_mask = dist_sq_xy < (0.05 ** 2)
        valid_projections = projections[valid_mask]
        
        if len(valid_projections) > 0:
            bins = np.arange(best_min_proj, best_max_proj, 0.05) 
            if len(bins) < 2: bins = 2
            hist, bin_edges = np.histogram(valid_projections, bins=bins)
            max_bin_idx = np.argmax(hist)
            peak_proj = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx+1]) / 2.0
            final_center = pivot_center + best_dir_vec * peak_proj
        else:
            final_center = pivot_center

        # --- FLIP ---
        edge_a = pivot_center + best_dir_vec * best_max_proj
        edge_b = pivot_center + best_dir_vec * best_min_proj
        
        if np.linalg.norm(edge_a) < np.linalg.norm(edge_b):
            start_pt = edge_a
        else:
            start_pt = edge_b
        end_pt = final_center

        # Log Result
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        deg = np.degrees(np.arctan2(dy, dx))
        
        self.get_logger().info(f"--------------------------------------------------")
        self.get_logger().info(f"PILE DETECTED (Hybrid Score: Length x Mass):")
        self.get_logger().info(f"   - Approach Angle: {deg:.2f}°")
        self.get_logger().info(f"   - Length:         {best_length:.3f} m")
        self.get_logger().info(f"   - Points (Mass):  {best_point_count}")
        self.get_logger().info(f"   - Winning Score:  {best_combined_score:.1f}")
        self.get_logger().info(f"--------------------------------------------------")

        return final_center, start_pt, end_pt

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