import math
from typing import Callable, Optional, Tuple

import numpy as np


class DEMProvider:
	"""Abstract DEM provider interface."""

	def get_elevation(self, latitude: float, longitude: float) -> Optional[float]:
		"""Return elevation in meters (EGM96 geoid approx / MSL) for lat, lon."""
		raise NotImplementedError


class SRTMProvider(DEMProvider):
	"""SRTM GL1/GL3 provider using the lightweight srtm.py library.

	Falls back gracefully if the library or data are unavailable.
	"""

	def __init__(self):
		self._elevation_data = None
		try:
			import srtm
			self._elevation_data = srtm.Srtm1HeightMapCollection()
		except Exception:
			self._elevation_data = None

	def get_elevation(self, latitude: float, longitude: float) -> Optional[float]:
		if self._elevation_data is None:
			return None
		try:
			val = self._elevation_data.get_altitude(latitude, longitude)
			return float(val) if val is not None else None
		except Exception:
			return None


# ---- Geodesy helpers (WGS84) ----
_A = 6378137.0  # semi-major axis (m)
_F = 1 / 298.257223563
_B = _A * (1 - _F)
_E2 = 1 - (_B * _B) / (_A * _A)


def _radii_of_curvature(lat_rad: float) -> Tuple[float, float]:
	"""Return meridian (M) and prime-vertical (N) radii at latitude."""
	sin_lat = math.sin(lat_rad)
	denom = math.sqrt(1 - _E2 * sin_lat * sin_lat)
	N = _A / denom
	M = _A * (1 - _E2) / (denom ** 3)
	return M, N


def _normalize(v: np.ndarray) -> np.ndarray:
	m = np.linalg.norm(v)
	return v / m if m > 0 else v


def _camera_axes_from_yaw_pitch(yaw_deg: float, pitch_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Compute world ENU axes of the camera: right (x), down (y), forward (z).

	- World frame: ENU (east, north, up)
	- Yaw: 0° = North, 90° = East (clockwise from North)
	- Pitch: 0° = horizontal; negative values look down; -90° = nadir
	"""
	yaw = math.radians(yaw_deg or 0.0)
	# Horizontal heading unit vector (ENU)
	h = np.array([math.sin(yaw), math.cos(yaw), 0.0], dtype=float)
	# Right vector (ENU), perpendicular to heading and up
	r = np.array([math.cos(yaw), -math.sin(yaw), 0.0], dtype=float)
	# Down tilt angle (positive for looking downward)
	theta = math.radians(-pitch_deg if pitch_deg is not None else 0.0)
	up = np.array([0.0, 0.0, 1.0], dtype=float)
	# Forward axis (ENU)
	fwd = _normalize(h * math.cos(theta) + (-up) * math.sin(theta))
	# Image "down" axis so that (right, down, forward) is right-handed
	down = _normalize(np.cross(fwd, r))
	# Ensure orthonormal triad
	r = _normalize(r)
	fwd = _normalize(fwd)
	return r, down, fwd


def compute_ground_intersection(
		camera_lat: float,
		camera_lon: float,
		camera_alt_m: float,
		pixel_x: float,
		pixel_y: float,
		image_width_px: float,
		image_height_px: float,
		focal_length_mm: float,
		sensor_w_mm: float,
		sensor_h_mm: float,
		yaw_deg: float,
		pitch_deg: float,
		roll_deg: float = 0.0,
		dem_provider: Optional[DEMProvider] = None,
		max_distance_m: float = 5000.0,
		step_initial_m: float = 50.0,
) -> Optional[Tuple[float, float, float]]:
	"""Intersect camera ray through pixel with terrain DEM.

	Returns (lat, lon, alt_m) at intersection or None if no intersection.
	"""
	# Basic validation
	if any(v is None for v in [camera_lat, camera_lon, camera_alt_m, focal_length_mm, sensor_w_mm, sensor_h_mm, image_width_px, image_height_px]):
		return None

	# Camera ray in camera coordinates (x right, y down, z forward)
	cx = image_width_px / 2.0
	cy = image_height_px / 2.0
	dx_mm = (pixel_x - cx) * (sensor_w_mm / image_width_px)
	dy_mm = (pixel_y - cy) * (sensor_h_mm / image_height_px)
	f_mm = float(focal_length_mm)
	d_cam = _normalize(np.array([dx_mm, dy_mm, f_mm], dtype=float))

	# Map camera axes into world ENU using yaw/pitch (ignore roll unless provided)
	r_world, d_world, f_world = _camera_axes_from_yaw_pitch(yaw_deg or 0.0, pitch_deg or 0.0)
	if roll_deg:
		# Apply roll around forward axis to (r, d)
		phi = math.radians(roll_deg)
		c, s = math.cos(phi), math.sin(phi)
		R = np.stack([
			[c + (1 - c) * f_world[0] * f_world[0], (1 - c) * f_world[0] * f_world[1] - s * f_world[2], (1 - c) * f_world[0] * f_world[2] + s * f_world[1]],
			[(1 - c) * f_world[1] * f_world[0] + s * f_world[2], c + (1 - c) * f_world[1] * f_world[1], (1 - c) * f_world[1] * f_world[2] - s * f_world[0]],
			[(1 - c) * f_world[2] * f_world[0] - s * f_world[1], (1 - c) * f_world[2] * f_world[1] + s * f_world[0], c + (1 - c) * f_world[2] * f_world[2]],
		], axis=0)
		r_world = _normalize(R.dot(r_world))
		d_world = _normalize(R.dot(d_world))

	# Compose world ray direction from camera basis
	d_world_vec = _normalize(d_cam[0] * r_world + d_cam[1] * d_world + d_cam[2] * f_world)

	# If pointing up (or flat), no intersection expected
	if d_world_vec[2] >= -1e-6:
		return None

	# Integrate along the ray in ENU meters, updating geographic coords
	lat = float(camera_lat)
	lon = float(camera_lon)
	h = float(camera_alt_m)
	M, N = _radii_of_curvature(math.radians(lat))

	# Prepare DEM provider
	dem = dem_provider or SRTMProvider()
	# Bracketing for intersection
	prev_lat, prev_lon, prev_h = lat, lon, h
	prev_diff = None

	# Adaptive step
	step = step_initial_m
	total = 0.0
	for _ in range(10000):  # hard cap
		if total > max_distance_m:
			break
		# Advance one step
		de, dn, du = d_world_vec[0] * step, d_world_vec[1] * step, d_world_vec[2] * step
		# Update geodetic (small-displacement approximation)
		lat += (dn / (M + h)) * (180.0 / math.pi)
		# Recompute curvature for new latitude occasionally (optional)
		M, N = _radii_of_curvature(math.radians(lat))
		lon += (de / ((N + h) * math.cos(math.radians(lat)))) * (180.0 / math.pi)
		h += du
		total += step

		# Query DEM
		z_dem = dem.get_elevation(lat, lon)
		if z_dem is None:
			# If DEM not available, keep stepping; return None at end
			prev_lat, prev_lon, prev_h = lat, lon, h
			continue
		# Difference (camera ray height vs DEM)
		diff = h - z_dem
		if diff <= 0:  # crossed the ground
			# Binary search between previous point and current to refine
			low_lat, low_lon, low_h = prev_lat, prev_lon, prev_h
			high_lat, high_lon, high_h = lat, lon, h
			for _ in range(20):
				mid_lat = 0.5 * (low_lat + high_lat)
				mid_lon = 0.5 * (low_lon + high_lon)
				mid_h = 0.5 * (low_h + high_h)
				z_mid = dem.get_elevation(mid_lat, mid_lon)
				if z_mid is None:
					break
				if mid_h - z_mid > 0:
					low_lat, low_lon, low_h = mid_lat, mid_lon, mid_h
				else:
					high_lat, high_lon, high_h = mid_lat, mid_lon, mid_h
			return high_lat, high_lon, dem.get_elevation(high_lat, high_lon) or high_h

		prev_lat, prev_lon, prev_h = lat, lon, h
		prev_diff = diff

	return None