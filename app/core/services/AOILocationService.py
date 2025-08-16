import math
from typing import Optional, Tuple

import piexif
from pyproj import Geod

from core.services.ImageService import ImageService
from helpers.LocationInfo import LocationInfo
from .DEMService import DEMService


class AOILocationService:
    """Compute geographic coordinates of an AOI by intersecting image rays with a DEM."""

    def __init__(self, dem_directory: str):
        self.dem_service = DEMService(dem_directory)
        self.geod = Geod(ellps="WGS84")

    def compute_aoi_coordinates(self, image_path: str, pixel_x: float, pixel_y: float) -> Optional[Tuple[float, float, float]]:
        """Return (lat, lon, elevation) of the AOI center or None if unavailable."""
        image_service = ImageService(image_path)
        yaw, pitch = image_service.get_gimbal_orientation()
        altitude = image_service.get_asl_altitude('m')
        hfov = image_service.get_camera_hfov()
        gps = LocationInfo.get_gps(exif_data=image_service.exif_data)

        if None in (yaw, pitch, altitude, hfov) or not gps:
            return None

        width = image_service.exif_data['Exif'][piexif.ExifIFD.PixelXDimension]
        height = image_service.exif_data['Exif'][piexif.ExifIFD.PixelYDimension]

        vfov = 2 * math.atan(math.tan(math.radians(hfov / 2)) * height / width)
        fx = width / (2 * math.tan(math.radians(hfov / 2)))
        fy = height / (2 * math.tan(vfov / 2))
        cx, cy = width / 2.0, height / 2.0

        x_off = pixel_x - cx
        y_off = pixel_y - cy
        angle_x = math.degrees(math.atan2(x_off, fx))
        angle_y = math.degrees(math.atan2(y_off, fy))

        yaw_prime = yaw + angle_x
        pitch_prime = pitch - angle_y
        pitch_rad = math.radians(pitch_prime)

        lat0, lon0 = gps['latitude'], gps['longitude']
        alt0 = altitude

        # Step along the ray until we hit terrain or reach 5 km
        step = 10.0
        max_dist = 5000.0
        dist = 0.0
        prev_lat = prev_lon = prev_alt_ray = prev_dem = None
        azimuth = yaw_prime
        while dist <= max_dist:
            dist += step
            lon, lat, _ = self.geod.fwd(lon0, lat0, azimuth, dist)
            dem = self.dem_service.get_elevation(lat, lon)
            if dem is None:
                continue
            alt_ray = alt0 + dist * math.tan(pitch_rad)
            if alt_ray <= dem:
                # binary refine between previous and current distances
                left = dist - step
                right = dist
                for _ in range(10):
                    mid = (left + right) / 2
                    lon, lat, _ = self.geod.fwd(lon0, lat0, azimuth, mid)
                    dem_mid = self.dem_service.get_elevation(lat, lon)
                    alt_mid = alt0 + mid * math.tan(pitch_rad)
                    if dem_mid is None:
                        left = mid
                        continue
                    if alt_mid > dem_mid:
                        left = mid
                    else:
                        right = mid
                        dist = mid
                        dem = dem_mid
                return lat, lon, dem
            prev_lat, prev_lon, prev_alt_ray, prev_dem = lat, lon, alt_ray, dem
        return None
