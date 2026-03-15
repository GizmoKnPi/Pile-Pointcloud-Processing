#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger, SetBool
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, Float32 # 🚀 IMPORTED Float32
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

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
            ('roi_x_min', 0.8), ('roi_x_max', 2.0),
            ('roi_y_min', -1.0), ('roi_y_max', 1.0),
            ('roi_z_min', 0.01), ('roi_z_max', 1.5),
            ('outlier_neighbors', 20),
            ('outlier_std_ratio', 2.0),
            ('wiper_min_angle', 110.0),
            ('wiper_max_angle', 250.0),
            ('wall_check_enable', True),
            ('wall_distance_threshold', 0.005), 
            ('standoff_distance', 0.75),
            # 🚀 NEW: Pile measurement parameters
            ('pile_height_threshold', 0.01), # 10cm above ground
            ('bucket_width', 0.45),           # Physical width of the scoop
            ('bucket_depth_max', 4)        # Max distance to drive into pile
        ])

        self.get_logger().info("=== Loaded Configuration ===")
        self.get_logger().info(f"Voxel Size: {self.get_parameter('voxel_size').value}")
        self.get_logger().info(f"ROI Z: {self.get_parameter('roi_z_min').value} to {self.get_parameter('roi_z_max').value}")
        self.get_logger().info(f"Bucket Specs: {self.get_parameter('bucket_width').value}m wide, {self.get_parameter('bucket_depth_max').value}m max depth")
        self.get_logger().info("============================")

        # Clients
        self.servo_client = self.create_client(SetBool, 'toggle_servo_mode')
        self.reset_client = self.create_client(Trigger, 'reset_scan_buffer')

        # Services
        self.create_service(Trigger, 'start_rocking', self.start_rocking_cb)
        self.create_service(Trigger, 'process_scan', self.process_cb)

        # Nav2 services
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/scoop_markers', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.goal_viz_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.pcd_pub = self.create_publisher(PointCloud2, '/cleaned_cloud', 10)
        
        # 🚀 NEW: Publish the final calculated drive-in depth for the BT
        self.depth_pub = self.create_publisher(Float32, '/scoop_depth', 10)
        
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
            res.success = False; res.message = "Cloud is empty."
            return res

        # 1. CLEAN
        pcd = self.clean_cloud(pcd)
        self.get_logger().info(f"Points after cleaning: {len(pcd.points)}")

        if len(pcd.points) < 10:
            res.success = False; res.message = "Too few points after cleaning."
            return res
        
        # 2. PUBLISH CLEANED CLOUD
        self.publish_open3d_cloud(pcd)

        # 3. ANALYZE (🚀 NEW: Now unpacking visible_depth)
        center, start, end, visible_depth = self.find_scoop_path(pcd)
        
        # 🚀 NEW: Publish the depth for the Behavior Tree
        depth_msg = Float32()
        depth_msg.data = float(visible_depth)
        self.depth_pub.publish(depth_msg)

        # 4. PUBLISH MARKERS & NAV GOAL
        bucket_w = self.get_parameter('bucket_width').value
        self.publish_markers(center, start, end, visible_depth, bucket_w)
        self.send_nav_goal(start, end)
        
        # STOP THE SERVO ROCKING
        if self.servo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Process complete. Sending 'p' to stop servo.")
            self.servo_client.call_async(SetBool.Request(data=False))
        
        res.success = True
        res.message = "Path Calculated & Cleaned Cloud Published!"
        return res

    def clean_cloud(self, pcd):
        pcd = pcd.remove_non_finite_points()
        voxel = self.get_parameter('voxel_size').value
        pcd = pcd.voxel_down_sample(voxel)
        
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(self.get_parameter('roi_x_min').value, self.get_parameter('roi_y_min').value, self.get_parameter('roi_z_min').value), 
            max_bound=(self.get_parameter('roi_x_max').value, self.get_parameter('roi_y_max').value, self.get_parameter('roi_z_max').value) 
        )
        pcd = pcd.crop(bbox)
        
        if self.get_parameter('wall_check_enable').value and len(pcd.points) > 50:
            pcd = self.remove_walls(pcd)
        
        nb = self.get_parameter('outlier_neighbors').value
        std = self.get_parameter('outlier_std_ratio').value
        if len(pcd.points) > nb:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
        
        return pcd

    def remove_walls(self, pcd):
        thresh = self.get_parameter('wall_distance_threshold').value
        plane_model, inliers = pcd.segment_plane(distance_threshold=thresh, ransac_n=3, num_iterations=1000)
        [a, b, c, d] = plane_model
        if abs(c) < 0.2: 
            self.get_logger().info(f"Vertical Wall Detected (Normal Z={c:.2f}). Removing {len(inliers)} points.")
            pcd = pcd.select_by_index(inliers, invert=True)
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
        pivot_center = np.median(points, axis=0)
        
        min_angle = self.get_parameter('wiper_min_angle').value
        max_angle = self.get_parameter('wiper_max_angle').value
        
        best_combined_score = -1.0
        best_dir_vec = np.array([1.0, 0.0, 0.0]) 
        best_max_proj = 0.0
        best_min_proj = 0.0
        best_point_count = 0 
        best_length = 0       
        
        points_xy = points[:, :2] 
        center_xy = pivot_center[:2]
        
        steps = int(max_angle - min_angle)
        angles = np.linspace(min_angle, max_angle, steps)
        
        for angle_deg in angles:
            angle_rad = np.deg2rad(angle_deg)
            dir_xy = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            
            vec_xy = points_xy - center_xy
            projections = np.dot(vec_xy, dir_xy)
            dist_sq_xy = np.sum(vec_xy**2, axis=1) - projections**2
            
            valid_mask = dist_sq_xy < (0.05 ** 2) 
            count = np.sum(valid_mask)
            
            if count > 5:
                valid_projections = projections[valid_mask]
                curr_min = np.min(valid_projections)
                curr_max = np.max(valid_projections)
                length = curr_max - curr_min
                score = length * count
                
                if score > best_combined_score:
                    best_combined_score = score
                    best_dir_vec = np.array([dir_xy[0], dir_xy[1], 0.0])
                    best_min_proj = curr_min
                    best_max_proj = curr_max
                    best_point_count = count
                    best_length = length

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

        edge_a = pivot_center + best_dir_vec * best_max_proj
        edge_b = pivot_center + best_dir_vec * best_min_proj
        
        if np.linalg.norm(edge_a) < np.linalg.norm(edge_b):
            start_pt = edge_a
        else:
            start_pt = edge_b
        end_pt = final_center

        # ---------------------------------------------------------
        # 🚀 NEW: MEASURE VISIBLE PILE DEPTH (Fixed Vector Direction)
        # ---------------------------------------------------------
        z_thresh = self.get_parameter('pile_height_threshold').value
        bucket_width = self.get_parameter('bucket_width').value
        
        z_mask = points[:, 2] > z_thresh
        width_mask = dist_sq_xy < ((bucket_width / 2.0) ** 2)
        valid_pile_mask = z_mask & width_mask
        
        # 1. Calculate the explicit "drive-in" vector (start -> end)
        dx_drive = end_pt[0] - start_pt[0]
        dy_drive = end_pt[1] - start_pt[1]
        drive_len = math.sqrt(dx_drive*dx_drive + dy_drive*dy_drive)
        
        if drive_len > 0:
            explicit_drive_dir = np.array([dx_drive / drive_len, dy_drive / drive_len])
        else:
            explicit_drive_dir = np.array([1.0, 0.0]) # Fallback

        if np.any(valid_pile_mask):
            valid_pile_points = points_xy[valid_pile_mask]
            vec_from_start = valid_pile_points - start_pt[:2]
            
            # 2. Project onto our guaranteed inward-pointing vector
            projections_from_start = np.dot(vec_from_start, explicit_drive_dir)
            raw_depth = np.max(projections_from_start)
        else:
            raw_depth = 0.0
            self.get_logger().warn("No points found above Z-threshold in the scoop path!")

        # 3. Double Clamp: Never greater than bucket max, NEVER less than 0.0
        max_allowed = self.get_parameter('bucket_depth_max').value
        visible_depth = max(0.0, min(raw_depth, max_allowed))

        # ---------------------------------------------------------

        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        deg = np.degrees(np.arctan2(dy, dx))
        
        self.get_logger().info(f"--------------------------------------------------")
        self.get_logger().info(f"PILE DETECTED:")
        self.get_logger().info(f"   - Approach Angle: {deg:.2f}°")
        self.get_logger().info(f"   - Raw Pile Depth: {raw_depth:.3f} m")
        self.get_logger().info(f"   - Clamped Scoop:  {visible_depth:.3f} m  <-- Driving this far")
        self.get_logger().info(f"--------------------------------------------------")

        # 🚀 NEW: Returning visible_depth to pass to the publisher
        return final_center, start_pt, end_pt, visible_depth

    def send_nav_goal(self, start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            dir_x = dx / length
            dir_y = dy / length
            standoff = self.get_parameter('standoff_distance').value
            safe_x = start[0] - (dir_x * standoff)
            safe_y = start[1] - (dir_y * standoff)
        else:
            safe_x = start[0]
            safe_y = start[1]

        yaw = math.atan2(dy, dx)
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = safe_x  
        pose.pose.position.y = safe_y  
        pose.pose.position.z = 0.0
        
        q = self.quaternion_from_yaw(yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        try:
            if self.tf_buffer.can_transform("map", "base_link", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
                tf = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
                pose_in_map = do_transform_pose(pose.pose, tf)
            else:
                self.get_logger().error("Transform map->base_link not available!")
                return
        except Exception as e:
            self.get_logger().error(f"TF Error: {e}")
            return

        pose_map = PoseStamped()
        pose_map.header.stamp = self.get_clock().now().to_msg()
        pose_map.header.frame_id = "map"
        pose_map.pose = pose_in_map

        self.goal_viz_pub.publish(pose_map)

        goal = NavigateToPose.Goal()
        goal.pose = pose_map

        self.get_logger().info("Waiting for Nav2 action server...")
        self.nav_client.wait_for_server()
        self.nav_client.send_goal_async(goal)

    # 🚀 NEW: Signature updated to accept visible_depth and bucket_width
    def publish_markers(self, center, start, end, visible_depth, bucket_width):
        ma = MarkerArray()
        
        # 1. The Line (Arrow)
        m = Marker()
        m.header.frame_id = "base_link"
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.id = 0
        m.points = [Point(x=start[0], y=start[1], z=start[2]), Point(x=end[0], y=end[1], z=end[2])]
        m.scale.x = 0.05; m.scale.y = 0.1 
        m.color.r = 1.0; m.color.a = 1.0
        ma.markers.append(m)
        
        # 2. The Target (Sphere)
        s = Marker()
        s.header.frame_id = "base_link"
        s.type = Marker.SPHERE
        s.action = Marker.ADD
        s.id = 1
        s.pose.position.x = center[0]; s.pose.position.y = center[1]; s.pose.position.z = center[2]
        s.scale.x = 0.1; s.scale.y = 0.1; s.scale.z = 0.1
        s.color.b = 1.0; s.color.a = 1.0
        ma.markers.append(s)

        # ---------------------------------------------------------
        # 🚀 NEW: THE SCOOP VOLUME (Green Transparent Cube)
        # ---------------------------------------------------------
        c = Marker()
        c.header.frame_id = "base_link"
        c.type = Marker.CUBE
        c.action = Marker.ADD
        c.id = 2
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            dir_x = dx / length
            dir_y = dy / length
        else:
            dir_x = 1.0; dir_y = 0.0

        c.pose.position.x = start[0] + (dir_x * (visible_depth / 2.0))
        c.pose.position.y = start[1] + (dir_y * (visible_depth / 2.0))
        c.pose.position.z = 0.1 # Float it slightly off the ground
        
        yaw = math.atan2(dy, dx)
        q = self.quaternion_from_yaw(yaw)
        c.pose.orientation.x = q[0]; c.pose.orientation.y = q[1]; c.pose.orientation.z = q[2]; c.pose.orientation.w = q[3]
        
        c.scale.x = float(visible_depth)   # Depth of the bite
        c.scale.y = float(bucket_width)    # Width of your bucket
        c.scale.z = 0.2                    # Arbitrary height for visual effect
        
        c.color.r = 0.0; c.color.g = 1.0; c.color.b = 0.0
        c.color.a = 0.4 # Semi-transparent
        
        ma.markers.append(c)
        # ---------------------------------------------------------

        self.marker_pub.publish(ma)

    def start_rocking_cb(self, req, res):

        self.get_logger().info("Starting new scan cycle")

        # -------- RESET SCAN BUFFER --------
        if self.reset_client.wait_for_service(timeout_sec=1.0):
            reset_req = Trigger.Request()
            
            # 🚀 FIX: Removed the spin_until_future_complete deadlock!
            # We just fire the request asynchronously and keep moving.
            self.reset_client.call_async(reset_req)
            self.get_logger().info("Scan buffer reset requested")
            
        else:
            self.get_logger().warn("reset_scan_buffer service not available")

        # -------- START ROCKING SERVO --------
        if self.servo_client.wait_for_service(timeout_sec=1.0):
            self.servo_client.call_async(SetBool.Request(data=True))
            
            self.get_logger().info("Servo rocking started")
            res.success = True
            res.message = "Servo rocking started"
            
        else:
            # 🚀 FIX: Restored your safety fallback so you know if it fails
            res.success = False
            res.message = "Servo driver service not available"
            self.get_logger().error(res.message)

        return res
    
    def quaternion_from_yaw(self, yaw):
        return [0.0, 0.0, math.sin(yaw/2), math.cos(yaw/2)]

def main():
    rclpy.init()
    rclpy.spin(SmartWiper())
    rclpy.shutdown()

if __name__ == '__main__':
    main()