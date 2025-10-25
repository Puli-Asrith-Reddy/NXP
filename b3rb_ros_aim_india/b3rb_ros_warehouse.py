# Copyright 2025 NXP

# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rclpy.callback_groups import ReentrantCallbackGroup

import math
import time
import numpy as np
import cv2
from typing import Optional, Tuple
import asyncio
import threading
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import BehaviorTreeLog
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from synapse_msgs.msg import Status
from synapse_msgs.msg import WarehouseShelf

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

import tkinter as tk
from tkinter import ttk

QOS_PROFILE_DEFAULT = 10
SERVER_WAIT_TIMEOUT_SEC = 5.0

PROGRESS_TABLE_GUI = True


class WindowProgressTable:
	def __init__(self, root, shelf_count):
		self.root = root
		self.root.title("Shelf Objects & QR Link")
		self.root.attributes("-topmost", True)
	


		self.row_count = 2
		self.col_count = shelf_count

		self.boxes = []
		for row in range(self.row_count):
			row_boxes = []
			for col in range(self.col_count):
				box = tk.Text(root, width=10, height=3, wrap=tk.WORD, borderwidth=1,
					      relief="solid", font=("Helvetica", 14))
				box.insert(tk.END, "NULL")
				box.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
				row_boxes.append(box)
			self.boxes.append(row_boxes)

		# Make the grid layout responsive.
		for row in range(self.row_count):
			self.root.grid_rowconfigure(row, weight=1)
		for col in range(self.col_count):
			self.root.grid_columnconfigure(col, weight=1)

	def change_box_color(self, row, col, color):
		self.boxes[row][col].config(bg=color)

	def change_box_text(self, row, col, text):
		self.boxes[row][col].delete(1.0, tk.END)
		self.boxes[row][col].insert(tk.END, text)

box_app = None
def run_gui(shelf_count):
	global box_app
	root = tk.Tk()
	box_app = WindowProgressTable(root, shelf_count)
	root.mainloop()


class WarehouseExplore(Node):
	""" HELLOOO Initializes warehouse explorer node with the required publishers and subscriptions.

		Returns:
			None
	"""
	def __init__(self):
		super().__init__('warehouse_explore')
		
		self.clusters_to_visit = []
		
		
		self.shelf_id=-1
		self.shelf_idx = None

		#FLAGS 
		self.timer_flag=True
		self.in_shelf_mode = False
		self.flag_for_one_call=True
		self.reached_offset_goal = False 
		self.first_extra_offset=True
		self.no_need = False
		self.exploration_complete = False
		self.all_shelves_visited = False
		self.run_once = True
		
		
		self.diagonal_clusters = []
		self.visited_qr_side = False
		self.visited_qr_side_2=False
		self.heuristic_angle = None

		self.recent_goals = []         		# store last few visited frontier positions
		self.revisit_distance_thresh = 3.0  # meters — minimum distance between new goals
		self.max_recent_goals = 5



		self.allowed_objects = ['banana', 'car', 'clock', 'cup', 'horse', 'potted plant', 'teddy bear', 'zebra']


		self.shelf_number=1

		self.shelf_qr_data=WarehouseShelf()
		
		self.qr_detector = cv2.QRCodeDetector() 

		self.action_client = ActionClient(
			self,
			NavigateToPose,
			'/navigate_to_pose')

		self.subscription_pose = self.create_subscription(
			PoseWithCovarianceStamped,
			'/pose',
			self.pose_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_global_map = self.create_subscription(
			OccupancyGrid,
			'/global_costmap/costmap',
			self.global_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_simple_map = self.create_subscription(
			OccupancyGrid,
			'/map',
			self.simple_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_status = self.create_subscription(
			Status,
			'/cerebri/out/status',
			self.cerebri_status_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_behavior = self.create_subscription(
			BehaviorTreeLog,
			'/behavior_tree_log',
			self.behavior_tree_log_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_shelf_objects = self.create_subscription(
			WarehouseShelf,
			'/shelf_objects',
			self.shelf_objects_callback,
			QOS_PROFILE_DEFAULT)

		# Subscription for camera images.
		self.subscription_camera = self.create_subscription(
			CompressedImage,
			'/camera/image_raw/compressed',
			self.camera_image_callback,
			QOS_PROFILE_DEFAULT)

		self.publisher_joy = self.create_publisher(
			Joy,
			'/cerebri/in/joy',
			QOS_PROFILE_DEFAULT)

		# Publisher for output image (for debug purposes).
		self.publisher_qr_decode = self.create_publisher(
			CompressedImage,
			"/debug_images/qr_code",
			QOS_PROFILE_DEFAULT)

		self.publisher_shelf_data = self.create_publisher(
			WarehouseShelf,
			"/shelf_data",
			QOS_PROFILE_DEFAULT)

		self.declare_parameter('shelf_count', 1)
		self.declare_parameter('initial_angle', 0.0)

		self.shelf_count = \
			self.get_parameter('shelf_count').get_parameter_value().integer_value
		self.initial_angle = \
			self.get_parameter('initial_angle').get_parameter_value().double_value

		# --- Robot State ---
		self.armed = False
		self.logger = self.get_logger()

		# --- Robot Pose ---
		self.pose_curr = PoseWithCovarianceStamped()
		self.buggy_pose_x = 0.0
		self.buggy_pose_y = 0.0
		self.buggy_center = (0.0, 0.0)
		self.world_center = (0.0, 0.0)

		# --- Map Data ---
		self.simple_map_curr = None
		self.global_map_curr = None

		# --- Goal Management ---
		self.xy_goal_tolerance = 0.5
		self.goal_completed = True  # No goal is currently in-progress.
		self.goal_handle_curr = None
		self.cancelling_goal = False
		self.recovery_threshold = 10

		# --- Goal Creation ---
		self._frame_id = "map"

		# --- Exploration Parameters ---
		self.max_step_dist_world_meters = 7.0
		self.min_step_dist_world_meters = 4.0
		self.full_map_explored_count = 0

		# --- QR Code Data ---
		self.qr_code_str = "Empty"
		if PROGRESS_TABLE_GUI:
			self.table_row_count = 0
			self.table_col_count = 0

		# --- Shelf Data ---
		self.shelf_objects_curr = WarehouseShelf()



	def pose_callback(self, message):
		"""Callback function to handle pose updates.

		Args:
			message: ROS2 message containing the current pose of the rover.

		Returns:
			None
		"""
		self.pose_curr = message
		self.buggy_pose_x = message.pose.pose.position.x
		self.buggy_pose_y = message.pose.pose.position.y
		self.buggy_center = (self.buggy_pose_x, self.buggy_pose_y)
		
		if not hasattr(self, 'initial_ref_pose') or self.initial_ref_pose is None:
			self.initial_ref_pose = message  # Store early pose once
			self.get_logger().info(f"✅ Stored initial reference pose at x={message.pose.pose.position.x:.2f}, y={message.pose.pose.position.y:.2f}")




	def simple_map_callback(self, message):
		"""Callback function to handle simple map updates.

		Args:
			message: ROS2 message containing the simple map data.

		Returns:
			None
		"""
		self.simple_map_curr = message
		map_info = self.simple_map_curr.info
		self.world_center = self.get_world_coord_from_map_coord(
			map_info.width / 2, map_info.height / 2, map_info
		)

	def get_clusters_dbscan(self,
        map_data,
        map_info,
        shelf_width   = 1.4,   # metres
        shelf_height  = 0.5,   # metres
        tolerance     = 0.25,  # metres
        plot=False):
		"""
		Detect shelf sized clusters in an occupancy grid using DBSCAN.

		Args
		----
		map_data : 2D numpy array of int8/uint8
			OccupancyGrid.data reshaped to [height, width].
		map_info : nav_msgs/OccupancyGrid.info
			Provides .resolution (m/px) and .origin (geometry_msgs/Pose).
		shelf_width, shelf_height : float
			Expected shelf footprint (metres).  Swap tolerated.
		tolerance : float
			Size tolerance (metres).
		plot : bool
			If True, show a matplotlib figure like the one you posted.

		Returns
		-------
		list[dict]
			One dict per matching cluster:
			{
				'centroid_map'  : (cx_px, cy_px),     # in pixel coords
				'centroid_world': (wx_m, wy_m),       # map frame (metres)
				'width_m'       : cluster_width,      # metres (long side)
				'height_m'      : cluster_height,     # metres (short side)
				'angle_deg'     : rotation_ccw        # degrees
			}
		"""

		res   = map_info.resolution                # m / cell
		origin= map_info.origin                    # geometry_msgs/Pose

		# ------------------------------------------------------------------ #
		# 1. Extract every occupied cell (value 100)
		# ------------------------------------------------------------------ #
		occupied = np.argwhere(map_data == 100)    # [[row, col], ...]
		if occupied.size == 0:
			return []

		# DBSCAN expects (x, y).  Our array is (row=y, col=x).
		coords_px = occupied[:, [1, 0]]            # swap to (x, y)

		# ------------------------------------------------------------------ #
		# 2. DBSCAN clustering (distance in *pixels*)
		#    eps ≈ half the longest shelf side, converted to pixels.
		# ------------------------------------------------------------------ #
		eps_px = int(max(shelf_width, shelf_height) / res / 2.5) or 1
		db     = DBSCAN(eps=eps_px, min_samples=5).fit(coords_px)
		labels = db.labels_

		clusters = []
		show = plot
		if show:
			fig, ax = plt.subplots()
			ax.set_aspect('equal')

		for cid in set(labels):
			if cid == -1:
				continue   # noise

			pts_px = coords_px[labels == cid]              # (N,2)

			# ------------------------------------------------------------------ #
			# 3. Get a minimum‑area rotated box around the points
			# ------------------------------------------------------------------ #
			if pts_px.shape[0] < 5:
				continue
			rect= cv2.minAreaRect(pts_px.astype(np.float32))
			(cx, cy), (w, h), angle = rect                 # cx,cy in px; w,h in px
			width_m, height_m = w * res, h * res

			# Accept either orientation (width≈1.4 & height≈0.5, or vice‑versa)
			good_size = (
				abs(width_m  - shelf_width ) < tolerance and
				abs(height_m - shelf_height) < tolerance
			) or (
				abs(width_m  - shelf_height) < tolerance and
				abs(height_m - shelf_width ) < tolerance
			)
			if not good_size:
				continue

			# World (map frame) metres
			wx = origin.position.x + cx * res
			wy = origin.position.y + cy * res

			clusters.append({
				'centroid_map'  : (cx, cy),
				'centroid_world': (wx, wy),
				'width_m'       : width_m,
				'height_m'      : height_m,
				'angle_deg'     : angle
			})

			# ------------------------------------------------------------------ #
			# 4. Optional live plot
			# ------------------------------------------------------------------ #
			if show:
				box = cv2.boxPoints(rect).astype(int)
				ax.plot(pts_px[:, 0], pts_px[:, 1], '.', label=f'cluster {cid}')
				ax.plot(cx, cy, 'kx')
				ax.add_patch(plt.Polygon(box, closed=True,
										fill=False, edgecolor='red', linewidth=1))
				ax.text(cx, cy, f'{cid}', ha='center', va='center')

		if show:
			ax.invert_yaxis()        # pixel (0,0) at top‑left like an image
			ax.legend()
			plt.title('Shelf clusters detected (DBSCAN)')
			plt.show()

		return clusters


	def global_map_callback(self, message):
		"""Callback function to handle global map updates.
		Prioritizes frontiers near shelves/obstacles but ignores map boundaries.
		"""
		self.global_map_curr = message
		if not self.goal_completed:
			return
		# Record last goal position to prevent revisiting nearby frontiers
		if self.goal_handle_curr:
			goal_pose = self.goal_handle_curr.goal_request.goal.pose.pose.position
			self.recent_goals.append((goal_pose.x, goal_pose.y))
			if len(self.recent_goals) > self.max_recent_goals:
				self.recent_goals.pop(0)  # keep it limited


		height, width = self.global_map_curr.info.height, self.global_map_curr.info.width
		map_info = self.global_map_curr.info
		map_array = np.array(self.global_map_curr.data).reshape((height, width))

		clusters = self.get_clusters_dbscan(map_array, map_info, plot=False)
		shelf_like = [c for c in clusters if
					abs(c['width_m'] - 1.4) < 0.25 or abs(c['height_m'] - 0.5) < 0.25
					or abs(c['width_m'] - 0.5) < 0.25 or abs(c['height_m'] - 1.4) < 0.25]
		
		if self.run_once:
			if len(shelf_like) >= self.shelf_count and not self.exploration_complete:
				self.logger.info(f"📦 Found {len(shelf_like)} shelves after goal — reselecting based on initial angle.")
				self.clusters = shelf_like
				self.exploration_complete = True
				self.no_need = True
				self.in_shelf_mode = True
				self.select_first_shelf_from_initial_angle()
			else:
				self.get_logger().info("all shelves not detected yet")
			self.run_once = False

		# --- Convert 3 m to pixels ---
		boundary_margin = int(2.0 / map_info.resolution)

		# --- Find frontiers ---
		frontiers = self.get_frontiers_for_space_exploration(map_array)
		if not frontiers:
			self.full_map_explored_count += 1
			self.get_logger().info("No more frontiers — switching to shelf clustering mode.")
			self.clusters = self.get_clusters_dbscan(map_array, map_info, plot=False)
			if not self.no_need:
				self.in_shelf_mode = True
				self.select_first_shelf_from_initial_angle()
			return

		# ---------------------------------------------------------------------- #
		# Filter out frontiers too close to map edges (< 3 m)
		# ---------------------------------------------------------------------- #
		filtered_frontiers = []
		for fy, fx in frontiers:
			if (fx < boundary_margin or fy < boundary_margin or
				fx > width - boundary_margin or fy > height - boundary_margin):
				continue  # skip near-boundary cells
			filtered_frontiers.append((fy, fx))

		if not filtered_frontiers:
			self.get_logger().warn("All frontiers near boundary — waiting for new map update.")
			return

		# ---------------------------------------------------------------------- #
		# Choose best frontier based on obstacle density + distance penalty
		# ---------------------------------------------------------------------- #
		best_score = -float('inf')
		best_frontier = None

		for fy, fx in filtered_frontiers:
			fx_world, fy_world = self.get_world_coord_from_map_coord(fx, fy, map_info)
			distance = euclidean((fx_world, fy_world), self.buggy_center)

			# obstacle density
			y0, y1 = max(0, fy - 3), min(map_array.shape[0], fy + 4)
			x0, x1 = max(0, fx - 3), min(map_array.shape[1], fx + 4)
			window = map_array[y0:y1, x0:x1]
			obstacle_density = np.sum(window > 0) / window.size

			# compute penalty for being near a previously visited goal
			penalty = 0.0
			for gx, gy in self.recent_goals:
				dist_prev = euclidean((fx_world, fy_world), (gx, gy))
				if dist_prev < self.revisit_distance_thresh:
					penalty += (self.revisit_distance_thresh - dist_prev) * 3.0  # stronger if closer

			# combine into final score
			score = obstacle_density * 3.0 + 0.25 * distance - penalty

			self.get_logger().debug(
				f"Frontier ({fx:.1f},{fy:.1f}) | obs={obstacle_density:.2f} | dist={distance:.2f} | penalty={penalty:.2f} | score={score:.2f}"
			)

			if score > best_score:
				best_score = score
				self.get_logger().info(f"score : {score} , best score : {best_score}")
				best_frontier = (fy, fx)

		# ---------------------------------------------------------------------- #
		# Send goal if valid frontier found
		# ---------------------------------------------------------------------- #
		if best_frontier and not self.exploration_complete:
			fy, fx = best_frontier
			goal = self.create_goal_from_map_coord(fx, fy, map_info)
			self.send_goal_from_world_pose(goal)
			self.get_logger().info(f"🚀 Exploring frontier near obstacles at ({fx}, {fy})")
		else:
			self.get_logger().warn("⚠️ No valid frontier found within 3 m-safe zone.")

						

	def send_next_shelf_goal(self):

		shelf = self.clusters[self.shelf_idx]
		x, y = shelf['centroid_world']
		angle_deg = shelf['angle_deg']
		width_m = shelf['width_m']
		height_m = shelf['height_m']


		if width_m > height_m:
			face_angle_deg = angle_deg + 90
		else:
			face_angle_deg = angle_deg

		face_angle_deg = face_angle_deg % 360
		
		

		offset_x, offset_y = self.offset_goal_away_from_shelf(x, y, face_angle_deg)
		yaw_rad = self.create_yaw_from_vector(x,y,offset_x,offset_y)

		self.clear_shelf_qr_data()
		
		goal = self.create_goal_from_world_coord(offset_x, offset_y, yaw_rad)
		success = self.send_goal_from_world_pose(goal)

		if success:
			self.logger.info(f" Going to Shelf {self.shelf_idx+ 1} at {offset_x:.2f}, {offset_y:.2f}")
		else:
			self.logger.warn(f" Could not send goal for shelf {self.shelf_idx + 1}.")
			
		
	def offset_goal_away_from_shelf(self,x, y, angle_deg, offset=2.2):
		angle_rad = math.radians(angle_deg)
		# Move backward along the shelf normal (negative direction
		offset_x = x - offset * math.cos(angle_rad)
		offset_y = y - offset * math.sin(angle_rad)
		return offset_x, offset_y
	
	def goal_result_callback(self, future):
		"""
		Callback function executed when the navigation goal reaches a final result.

		Args:
			future (rclpy.Future): check_shelf_after_goalThe future that is result of the navigation action.
		"""
		status = future.result().status
		# NOTE: Refer https://docs.ros2.org/foxy/api/action_msgs/msg/GoalStatus.html.

		if status == GoalStatus.STATUS_SUCCEEDED:
			self.logger.info("Goal completed successfully!")
		else:
			self.logger.warn(f"Goal failed with status: {status}")

		self.goal_completed = True  # Mark goal as completed.
		self.goal_handle_curr = None  # Clear goal handle.

		self.get_logger().info(f"check {self.shelf_qr_data.object_name}")

		if self.all_shelves_visited:
			return


		# 🧠 Re-run clustering dynamically after every goal
		if self.global_map_curr :

			height, width = self.global_map_curr.info.height, self.global_map_curr.info.width
			map_array = np.array(self.global_map_curr.data).reshape((height, width))
			map_info = self.global_map_curr.info

			clusters = self.get_clusters_dbscan(map_array, map_info, plot=False)
			shelf_like = [c for c in clusters if
						abs(c['width_m'] - 1.4) < 0.25 or abs(c['height_m'] - 0.5) < 0.25
						or abs(c['width_m'] - 0.5) < 0.25 or abs(c['height_m'] - 1.4) < 0.25]

			if len(shelf_like) >= self.shelf_count and not self.exploration_complete:
				self.logger.info(f"📦 Found {len(shelf_like)} shelves after goal — reselecting based on initial angle.")
				self.clusters = shelf_like
				self.exploration_complete = True
				self.no_need = True
				self.in_shelf_mode = True
				self.select_first_shelf_from_initial_angle()

			

		#only for first shelf
		if self.in_shelf_mode :
			
			if not self.visited_qr_side:

				if self.first_extra_offset:
					self.clear_shelf_qr_data()

					shelf = self.clusters[self.shelf_idx]
					x, y = shelf['centroid_world']
					angle_deg = shelf['angle_deg']
					width_m = shelf['width_m']
					height_m = shelf['height_m']

					if width_m > height_m:
						face_angle_deg = angle_deg + 90
					else:
						face_angle_deg = angle_deg

					face_angle_deg = face_angle_deg % 360

					# 📍 First offset from shelf
					offset_x, offset_y = self.offset_goal_away_from_shelf(x, y, face_angle_deg)
					yaw_rad = self.create_yaw_from_vector(x, y, offset_x, offset_y)

					# ➕ Extra offset in the same direction (20 cm or whatever you choose)
					extra_offset = 1.0 # meters
					extra_x = offset_x - extra_offset * math.cos(yaw_rad)
					extra_y = offset_y - extra_offset * math.sin(yaw_rad)

					goal = self.create_goal_from_world_coord(extra_x, extra_y, yaw_rad)
					success = self.send_goal_from_world_pose(goal)

					if success:
						self.get_logger().info("extra offset goal sent")				
					self.first_extra_offset = False
					return
				
				else:

					
					self.visited_qr_side = True
					self.get_logger().info("Going to QR side...")
					self.check_shelf_after_goal()
					
				  # Exit early so we don't publish now
			else:		
					self.publisher_shelf_data.publish(self.shelf_qr_data)
					self.get_logger().info("data published")
					
					self.clear_shelf_qr_data()
					self.in_shelf_mode = False
	
					self.select_next_shelf_from_heuristic()
					return

		#other shelves
		if self.shelf_number >1  :
			
			if not self.visited_qr_side_2:

				if self.reached_offset_goal == False : 
					self.get_logger().info("starting 1m offset")

					extra_offset = 1.0 # 100 cm
					extra_x = self.offset_x - extra_offset * math.cos(self.yaw_rad)
					extra_y = self.offset_y - extra_offset * math.sin(self.yaw_rad)

					self.reached_offset_goal = True 
					extra_goal = self.create_goal_from_world_coord(extra_x, extra_y, self.yaw_rad)
					self.send_goal_from_world_pose(extra_goal)		
					return

				else:
					self.get_logger().info("going to the Qr side")
					self.visited_qr_side_2 = True
					self.reached_offset_goal = False
					self.RotateToQr_pub()
					

					return
			else:
				self.visited_qr_side_2=False
				self.publisher_shelf_data.publish(self.shelf_qr_data)
				self.get_logger().info("data published")
				self.clear_shelf_qr_data()

				if self.shelf_number >= len(self.clusters):
					self.exp_complete=True
					self.get_logger().info("🎉 All shelves visited. Exploration complete.")
					self.all_shelves_visited = True
					return

				else:
					self.select_next_shelf_from_heuristic()
						 
				

	def check_shelf_after_goal(self):

		# it checks if the diagonal shelf is the first shelf and if yes then store the objects and move to qr side
		
		
		if self.shelf_qr_data.object_count or self.shelf_qr_data.object_name:
			self.get_logger().info("////////////////////////Object/QR data found. Going to QR side..")

			Shelf = self.clusters[self.shelf_idx]
			x, y = Shelf['centroid_world']
			Angle_deg = Shelf['angle_deg']
			Width_m = Shelf['width_m']
			Height_m = Shelf['height_m']

			if Width_m < Height_m:
				face_angle_deg = Angle_deg + 90
			else:
				face_angle_deg = Angle_deg

			face_angle_deg = face_angle_deg % 360
			
			

			offset_x, offset_y = self.offset_goal_away_from_shelf(x,y, face_angle_deg)
			yaw_rad = self.create_yaw_from_vector(x,y,offset_x,offset_y)

			
			
			
			goal = self.create_goal_from_world_coord(offset_x, offset_y, yaw_rad)
			success = self.send_goal_from_world_pose(goal)
			if success:
				self.get_logger().info("//////////////////////goal to QR side sent")

	def select_first_shelf_from_initial_angle(self):
	
		if self.initial_angle is None:
			self.get_logger().warn("Initial heuristic angle not set.")
			return

		if not hasattr(self, 'initial_ref_pose') or self.initial_ref_pose is None:
			self.get_logger().warn("Initial reference pose not available.")
			return

		x0 = self.initial_ref_pose.pose.pose.position.x
		y0 = self.initial_ref_pose.pose.pose.position.y

		
		best_angle_diff = float('inf')

		for idx, cluster in enumerate(self.clusters):  # only among diagonal clusters
			x1, y1 = cluster['centroid_world']
			dx = x1 - x0
			dy = y1 - y0

			angle_to_cluster = math.degrees(math.atan2(dy, dx)) % 360
			diff = abs(angle_to_cluster - self.initial_angle)
			diff = min(diff, 360 - diff)

			if diff < best_angle_diff:
				best_angle_diff = diff
				self.shelf_idx = idx

		if self.shelf_idx is not None:
			self.get_logger().info(f"🎯 First shelf selected by angle match: idx={self.shelf_idx}, angle diff={best_angle_diff:.2f}")
			self.send_next_shelf_goal()
		else:
			self.get_logger().warn("⚠️ Could not find a matching shelf for initial angle.")


	def extract_heuristic_angle(self, qr_data_str):
		try:
			parts = qr_data_str.split('_')
			self.shelf_id=int(parts[0])
			return float(parts[1])  # second part is angle
		except (IndexError, ValueError):
			return None
		
	def select_next_shelf_from_heuristic(self):
		"""
		Selects and navigates to the next shelf from all detected clusters based on the heuristic angle
		decoded from the QR code.
		"""
		if self.heuristic_angle is None:
			self.get_logger().warn("Heuristic angle not set. Cannot determine next shelf.")
			return

		# Current shelf position
		current_cluster = self.clusters[self.shelf_idx]
			
	
		x0, y0 = current_cluster['centroid_world']

		self.best_idx = None
		best_angle_diff = float('inf')

		# Find the closest shelf in the heuristic direction
		for idx, cluster in enumerate(self.clusters):
			if idx == self.shelf_idx:
				continue  # Skip the current shelf

			x1, y1 = cluster['centroid_world']
			dx = x1 - x0
			dy = y1 - y0

			# Compute angle between current shelf and candidate shelf
			angle_to_candidate = math.degrees(math.atan2(dy, dx)) % 360
			diff = abs(angle_to_candidate - self.heuristic_angle)
			diff = min(diff, 360 - diff)  # Normalize to [0, 180]

			if diff < best_angle_diff:
				best_angle_diff = diff
				self.best_idx = idx

		if self.best_idx is not None:
			self.shelf_idx = self.best_idx
			selected_cluster = self.clusters[self.best_idx]

			x, y = selected_cluster['centroid_world']
			angle_deg = selected_cluster['angle_deg']
			width = selected_cluster['width_m']
			height = selected_cluster['height_m']

			# Determine face angle
			if width > height:
				face_angle_deg = angle_deg + 90
			else:
				face_angle_deg = angle_deg
			face_angle_deg = face_angle_deg % 360

			# Offset to position the robot in front of the shelf
			self.offset_x, self.offset_y = self.offset_goal_away_from_shelf(x, y, face_angle_deg)
			self.yaw_rad = self.create_yaw_from_vector(x, y, self.offset_x, self.offset_y)

			# Send goal to move to the new shelf
			goal = self.create_goal_from_world_coord(self.offset_x, self.offset_y, self.yaw_rad)
			success = self.send_goal_from_world_pose(goal)
			if success:

				self.shelf_number += 1
				self.get_logger().info(f"✅ Sent goal to next shelf at cluster index {self.best_idx} with angle diff {best_angle_diff:.2f}")
				self.get_logger().info(f"Sending goal to shelf at index {idx}, coords=({self.buggy_pose_x }, {self.buggy_pose_y})")

			else:
				self.get_logger().warn("⚠️ Failed to send goal to next shelf.")
		else:
			self.get_logger().warn("⚠️ No matching shelf found based on heuristic angle.")
			

	def RotateToQr_pub(self):

			self.get_logger().info("////////////////////////Object/QR data found. Going to QR side..")

			Shelf = self.clusters[self.best_idx]
			x, y = Shelf['centroid_world']
			Angle_deg = Shelf['angle_deg']
			Width_m = Shelf['width_m']
			Height_m = Shelf['height_m']

			if Width_m < Height_m:
				face_angle_deg = Angle_deg + 90
			else:
				face_angle_deg = Angle_deg

			face_angle_deg = face_angle_deg % 360
			
			

			offset_x, offset_y = self.offset_goal_away_from_shelf(x,y, face_angle_deg)
			yaw_rad = self.create_yaw_from_vector(x,y,offset_x,offset_y)
			
			
			goal = self.create_goal_from_world_coord(offset_x, offset_y, yaw_rad)
			success = self.send_goal_from_world_pose(goal)

			if success:
				self.get_logger().info("//////////////////////goal to QR side sent")


	def camera_image_callback(self, message):
			"""Callback function to handle incoming camera images.

			Args:
				message: ROS2 message of the type sensor_msgs.msg.CompressedImage.

			Returns:
				None
			"""
			
			np_arr = np.frombuffer(message.data, np.uint8)
			image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

			if image is None:
				self.get_logger().warn("Failed to decode image")
				return
			
			self.qr_text, points, smth = self.qr_detector.detectAndDecode(image)

			

			if self.qr_text:

				self.get_logger().info(f"------------QR Code Detected: {self.qr_text}")
				self.shelf_qr_data.qr_decoded=self.qr_text
				self.heuristic_angle=self.extract_heuristic_angle(self.qr_text)



	def shelf_objects_callback(self, message: WarehouseShelf):
			"""Callback function to handle shelf objects updates.

			Args:ws:
			//localhost:8765
				message: ROS2 message containing shelf objects data.

			Returns:
				None


			"""

			if len(self.shelf_qr_data.object_count)<len(message.object_count) or len(self.shelf_qr_data.object_count)== 0:
					
					self.shelf_objects_curr = message
					self.shelf_qr_data.object_count=self.shelf_objects_curr.object_count
					self.shelf_qr_data.object_name=self.shelf_objects_curr.object_name

			

			"""
			* Example for sending WarehouseShelf messages for evaluation.
				shelf_data_message = WarehouseShelf()

				shelf_data_message.object_name = ["car", "clock"]
				shelf_data_message.object_count = [1, 2]
				shelf_data_message.qr_decoded = "test qr string"

				self.publisher_shelf_data.publish(shelf_data_message)

			* Alternatively, you may store the QR for current shelf as self.qr_code_str.
				Then, add it as self.shelf_objects_curr.qr_decoded = self.qr_code_str
				Then, publish as self.publisher_shelf_data.publish(self.shelf_objects_curr)
				This, will publish the current detected objects with the last QR decoded.
			"""

			# Optional code for populating TABLE GUI with detected objects and QR data.
			"""
			if PROGRESS_TABLE_GUI:
				shelf = self.shelf_objects_curr
				obj_str = ""
				for name, count in zip(shelf.object_name, shelf.object_count):
					obj_str += f"{name}: {count}\n"

				box_app.change_box_text(self.table_row_count, self.table_col_count, obj_str)
				box_app.change_box_color(self.table_row_count, self.table_col_count, "cyan")
				self.table_row_count += 1

				box_app.change_box_text(self.table_row_count, self.table_col_count, self.qr_code_str)
				box_app.change_box_color(self.table_row_count, self.table_col_count, "yellow")
				self.table_row_count = 0
				self.table_col_count += 1
				"""
	def clear_shelf_qr_data(self):
		self.shelf_qr_data.object_name = []
		self.shelf_qr_data.object_count = []
		self.shelf_qr_data.qr_decoded = ""

	
	



	def get_frontiers_for_space_exploration(self, map_array):
		"""Identifies frontiers for space exploration.

		Args:
			map_array: 2D numpy array representing the map.

		Returns:
			frontiers: List of tuples representing frontier coordinates.
		"""
		frontiers = []
		for y in range(1, map_array.shape[0] - 1):
			for x in range(1, map_array.shape[1] - 1):
				if map_array[y, x] == -1:  # Unknown space and not visited.
					neighbors_complete = [
						(y, x - 1),
						(y, x + 1),
						(y - 1, x),
						(y + 1, x),
						(y - 1, x - 1),
						(y + 1, x - 1),
						(y - 1, x + 1),
						(y + 1, x + 1)
					]

					near_obstacle = False
					for ny, nx in neighbors_complete:
						if map_array[ny, nx] > 0:  # Obstacles.
							near_obstacle = True
							break
					if not near_obstacle:
						continue

					neighbors_cardinal = [
						(y, x - 1),
						(y, x + 1),
						(y - 1, x),
						(y + 1, x),
					]

					for ny, nx in neighbors_cardinal:
						if map_array[ny, nx] == 0:  # Free space.
							frontiers.append((ny, nx))
							break

		return frontiers



	def publish_debug_image(self, publisher, image):
		"""Publishes images for debugging purposes.

		Args:
			publisher: ROS2 publisher of the type sensor_msgs.msg.CompressedImage.
			image: Image given by an n-dimensional numpy array.

		Returns:
			None
		"""
		if image.size:
			message = CompressedImage()
			_, encoded_data = cv2.imencode('.jpg', image)
			message.format = "jpeg"
			message.data = encoded_data.tobytes()
			publisher.publish(message)



	def cerebri_status_callback(self, message):
		"""Callback function to handle cerebri status updates.

		Args:
			message: ROS2 message containing cerebri status.

		Returns:
			None
		"""
		if message.mode == 3 and message.arming == 2:
			self.armed = True
		else:
			# Initialize and arm the CMD_VEL mode.
			msg = Joy()
			msg.buttons = [0, 1, 0, 0, 0, 0, 0, 1]
			msg.axes = [0.0, 0.0, 0.0, 0.0]
			self.publisher_joy.publish(msg)

	def behavior_tree_log_callback(self, message):
		"""Alternative method for checking goal status.

		Args:
			message: ROS2 message containing behavior tree log.

		Returns:
			None
		"""
		for event in message.event_log:
			if (event.node_name == "FollowPath" and
				event.previous_status == "SUCCESS" and
				event.current_status == "IDLE"):
				# self.goal_completed = True
				# self.goal_handle_curr = None
				pass

	def rover_move_manual_mode(self, speed, turn):
		"""Operates the rover in manual mode by publishing on /cerebri/in/joy.

		Args:
			speed: The speed of the car in float. Range = [-1.0, +1.0];
				   Direction: forward for positive, reverse for negative.
			turn: Steer value of the car in float. Range = [-1.0, +1.0];
				  Direction: left turn for positive, right turn for negative.

		Returns:
			None
		"""
		msg = Joy()
		msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
		msg.axes = [0.0, speed, 0.0, turn]
		self.publisher_joy.publish(msg)



	def cancel_goal_callback(self, future):
		"""
		Callback function executed after a cancellation request is processed.

		Args:
			future (rclpy.Future): The future is the result of the cancellation request.
		"""
		cancel_result = future.result()
		if cancel_result:
			self.logger.info("Goal cancellation successful.")
			self.cancelling_goal = False  # Mark cancellation as completed (success).
			return True
		else:
			self.logger.error("Goal cancellation failed.")
			self.cancelling_goal = False  # Mark cancellation as completed (failed).
			return False

	def cancel_current_goal(self):
		"""Requests cancellation of the currently active navigation goal."""
		if self.goal_handle_curr is not None and not self.cancelling_goal:
			self.cancelling_goal = True  # Mark cancellation in-progress.
			self.logger.info("Requesting cancellation of current goal...")
			cancel_future = self.action_client._cancel_goal_async(self.goal_handle_curr)
			cancel_future.add_done_callback(self.cancel_goal_callback)

		


		

	def goal_response_callback(self, future):
		"""
		Callback function executed after the goal is sent to the action server.

		Args:
			future (rclpy.Future): The future that is server's response to goal request.
		"""
		goal_handle = future.result()
		if not goal_handle.accepted:
			self.logger.warn('Goal rejected :(')
			self.goal_completed = True  # Mark goal as completed (rejected).
			self.goal_handle_curr = None  # Clear goal handle.
		else:
			self.logger.info('Goal accepted :)')
			self.goal_completed = False  # Mark goal as in progress.
			self.goal_handle_curr = goal_handle  # Store goal handle.

			get_result_future = goal_handle.get_result_async()
			get_result_future.add_done_callback(self.goal_result_callback)

	def goal_feedback_callback(self, msg):
		"""
		Callback function to receive feedback from the navigation action.

		Args:
			msg (nav2_msgs.action.NavigateToPose.Feedback): The feedback message.
		"""
		distance_remaining = msg.feedback.distance_remaining
		number_of_recoveries = msg.feedback.number_of_recoveries
		navigation_time = msg.feedback.navigation_time.sec
		estimated_time_remaining = msg.feedback.estimated_time_remaining.sec

		self.logger.debug(f"Recoveries: {number_of_recoveries}, "
				  f"Navigation time: {navigation_time}s, "
				  f"Distance remaining: {distance_remaining:.2f}, "
				  f"Estimated time remaining: {estimated_time_remaining}s")

		if number_of_recoveries > self.recovery_threshold and not self.cancelling_goal:
			self.logger.warn(f"Cancelling. Recoveries = {number_of_recoveries}.")
			self.cancel_current_goal()  # Unblock by discarding the current goal.
		

	def send_goal_from_world_pose(self, goal_pose):
		"""
		Sends a navigation goal to the Nav2 action server.

		Args:
			goal_pose (geometry_msgs.msg.PoseStamped): The goal pose in the world frame.

		Returns:
			bool: True if the goal was successfully sent, False otherwise.
		"""
		if not self.goal_completed or self.goal_handle_curr is not None:
			return False

		self.goal_completed = False  # Starting a new goal.

		goal = NavigateToPose.Goal()
		goal.pose = goal_pose

		if not self.action_client.wait_for_server(timeout_sec=SERVER_WAIT_TIMEOUT_SEC):
			self.logger.error('NavigateToPose action server not available!')
			return False

		# Send goal asynchronously (non-blocking).
		goal_future = self.action_client.send_goal_async(goal, self.goal_feedback_callback)
		goal_future.add_done_callback(self.goal_response_callback)

		return True



	def _get_map_conversion_info(self, map_info) -> Optional[Tuple[float, float]]:
		"""Helper function to get map origin and resolution."""
		if map_info:
			origin = map_info.origin
			resolution = map_info.resolution
			return resolution, origin.position.x, origin.position.y
		else:
			return None

	def get_world_coord_from_map_coord(self, map_x: int, map_y: int, map_info) \
					   -> Tuple[float, float]:
		"""Converts map coordinates to world coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			world_x = (map_x + 0.5) * resolution + origin_x
			world_y = (map_y + 0.5) * resolution + origin_y
			return (world_x, world_y)
		else:
			return (0.0, 0.0)

	def get_map_coord_from_world_coord(self, world_x: float, world_y: float, map_info) \
					   -> Tuple[int, int]:
		"""Converts world coordinates to map coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			map_x = int((world_x - origin_x) / resolution)
			map_y = int((world_y - origin_y) / resolution)
			return (map_x, map_y)
		else:
			return (0, 0)

	def _create_quaternion_from_yaw(self, yaw: float) -> Quaternion:
		"""Helper function to create a Quaternion from a yaw angle."""
		cy = math.cos(yaw * 0.5)
		sy = math.sin(yaw * 0.5)
		q = Quaternion()
		q.x = 0.0
		q.y = 0.0
		q.z = sy
		q.w = cy
		return q

	def create_yaw_from_vector(self, dest_x: float, dest_y: float,
				   source_x: float, source_y: float) -> float:
		"""Calculates the yaw angle from a source to a destination point.
			NOTE: This function is independent of the type of map used.

			Input: World coordinates for destination and source.
			Output: Angle (in radians) with respect to x-axis.
		"""
		delta_x = dest_x - source_x
		delta_y = dest_y - source_y
		yaw = math.atan2(delta_y, delta_x)

		return yaw

	def create_goal_from_world_coord(self, world_x: float, world_y: float,
					 yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from world coordinates.
			NOTE: This function is independent of the type of map used.
		"""
		goal_pose = PoseStamped()
		goal_pose.header.stamp = self.get_clock().now().to_msg()
		goal_pose.header.frame_id = self._frame_id

		goal_pose.pose.position.x = world_x
		goal_pose.pose.position.y = world_y

		if yaw is None and self.pose_curr is not None:
			# Calculate yaw from current position to goal position.
			source_x = self.pose_curr.pose.pose.position.x
			source_y = self.pose_curr.pose.pose.position.y
			yaw = self.create_yaw_from_vector(world_x, world_y, source_x, source_y)
		elif yaw is None:
			yaw = 0.0
		else:  # No processing needed; yaw is supplied by the user.
			pass

		goal_pose.pose.orientation = self._create_quaternion_from_yaw(yaw)

		pose = goal_pose.pose.position
		print(f"Goal created: ({pose.x:.2f}, {pose.y:.2f}, yaw={yaw:.2f})")
		return goal_pose

	def create_goal_from_map_coord(self, map_x: int, map_y: int, map_info,
				       yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from map coordinates."""
		world_x, world_y = self.get_world_coord_from_map_coord(map_x, map_y, map_info)

		return self.create_goal_from_world_coord(world_x, world_y, yaw)


def main(args=None):
	rclpy.init(args=args)

	warehouse_explore = WarehouseExplore()

	if PROGRESS_TABLE_GUI:
		gui_thread = threading.Thread(target=run_gui, args=(warehouse_explore.shelf_count,))
		gui_thread.start()

	rclpy.spin(warehouse_explore)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	warehouse_explore.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
