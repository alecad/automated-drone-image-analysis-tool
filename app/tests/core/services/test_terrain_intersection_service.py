import math

import numpy as np

from app.core.services.TerrainIntersectionService import compute_ground_intersection, DEMProvider


class FlatDEM(DEMProvider):
	def __init__(self, elevation_m: float = 0.0):
		self.elevation_m = elevation_m
	def get_elevation(self, latitude: float, longitude: float):
		return float(self.elevation_m)


def test_nadir_intersection_flat_dem():
	# Camera parameters
	camera_lat = 45.0
	camera_lon = 9.0
	camera_alt_m = 100.0
	image_w = 4000
	image_h = 3000
	focal_mm = 8.8
	sensor_w_mm = 6.4
	sensor_h_mm = 4.8
	yaw = 0.0
	pitch = -90.0  # nadir
	center_x = image_w / 2
	center_y = image_h / 2

	lat, lon, alt = compute_ground_intersection(
		camera_lat, camera_lon, camera_alt_m,
		center_x, center_y,
		image_w, image_h,
		focal_mm, sensor_w_mm, sensor_h_mm,
		yaw, pitch,
		roll_deg=0.0,
		dem_provider=FlatDEM(0.0),
		max_distance_m=1000.0,
		step_initial_m=50.0,
	)

	assert lat is not None and lon is not None
	# With nadir and flat ground, intersection should be directly below
	assert abs(lat - camera_lat) < 1e-10
	assert abs(lon - camera_lon) < 1e-10
	assert abs(alt - 0.0) < 1e-6