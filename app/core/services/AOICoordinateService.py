"""Projection of image pixels to geographic coordinates using a DEM."""

import math
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import piexif
from pyproj import Transformer

from core.services.ImageService import ImageService
from core.services.DEMService import DEMService
from helpers.LocationInfo import LocationInfo
from helpers.PickleHelper import PickleHelper


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


class AOICoordinateService:
    """Estimate geographic coordinates for pixels in a drone image."""

    def __init__(self, dem_dir: str):
        self.dem_service = DEMService(dem_dir)
        self._to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
        self._from_ecef = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)

    # ------------------------------------------------------------------
    def _camera_altitude(self, xmp: dict, lon: float, lat: float) -> float:
        """Resolve camera altitude in metres above mean sea level."""

        alt = None
        for key in ("AbsoluteAltitude", "Absolute Altitude"):
            try:
                alt = float(xmp.get(key))
                break
            except (TypeError, ValueError):
                continue
        if alt is not None:
            return alt

        try:
            rel = float(xmp.get("RelativeAltitude"))
        except (TypeError, ValueError):
            rel = 0.0
        try:
            ground = self.dem_service.elevation_at(lon, lat)
        except ValueError:
            ground = 0.0
        return ground + rel

    # ------------------------------------------------------------------
    def _intrinsics(self, exif: dict) -> CameraIntrinsics:
        width = exif['Exif'][piexif.ExifIFD.PixelXDimension]
        height = exif['Exif'][piexif.ExifIFD.PixelYDimension]
        focal = exif['Exif'][piexif.ExifIFD.FocalLength]
        f_mm = focal[0] / focal[1]

        info = PickleHelper.get_drone_sensor_info()
        model = exif['0th'][piexif.ImageIFD.Model].decode('utf-8').strip()
        row = info[(info['Model (Exif)'].str.contains(model)) &
                   (info['Image Source (XMP)'] == 'WideCamera')]
        if row.empty:
            row = info[info['Model (Exif)'].str.contains(model)]
        sensor_w = float(row['sensor_w'].iloc[0])
        sensor_h = float(row['sensor_h'].iloc[0])

        fx = (width / sensor_w) * f_mm
        fy = (height / sensor_h) * f_mm
        cx = width / 2
        cy = height / 2
        return CameraIntrinsics(fx, fy, cx, cy)

    # ------------------------------------------------------------------
    def pixel_to_view(self, image_path: str, x: int, y: int) -> Optional[Dict[str, float]]:
        """Project a pixel onto the terrain and return view information."""

        image_service = ImageService(image_path)
        exif = image_service.exif_data
        xmp = image_service.xmp_data
        gps = LocationInfo.get_gps(exif_data=exif)
        if not gps:
            return None

        lat0 = gps['latitude']
        lon0 = gps['longitude']
        alt0 = self._camera_altitude(xmp, lon0, lat0)

        intr = self._intrinsics(exif)
        width = exif['Exif'][piexif.ExifIFD.PixelXDimension]

        # x is measured from the right edge
        x = width - x
        xn = (x - intr.cx) / intr.fx
        yn = (y - intr.cy) / intr.fy

        yaw0 = math.radians(float(xmp.get('GimbalYawDegree', 0)))
        pitch0 = math.radians(float(xmp.get('GimbalPitchDegree', 0)))

        ang_x = math.atan(xn)
        ang_y = math.atan(yn)
        yaw = yaw0 + ang_x
        pitch = pitch0 - ang_y

        # Direction in local ENU coordinates
        d_e = math.cos(pitch) * math.sin(yaw)
        d_n = math.cos(pitch) * math.cos(yaw)
        d_u = math.sin(pitch)
        dir_enu = np.array([d_e, d_n, d_u])

        # ENU -> ECEF rotation
        lat0_rad = math.radians(lat0)
        lon0_rad = math.radians(lon0)
        slat, clat = math.sin(lat0_rad), math.cos(lat0_rad)
        slon, clon = math.sin(lon0_rad), math.cos(lon0_rad)
        enu2ecef = np.array([
            [-slon, -slat * clon, clat * clon],
            [clon, -slat * slon, clat * slon],
            [0.0, clat, slat],
        ])

        dir_ecef = enu2ecef @ dir_enu
        origin_ecef = np.array(self._to_ecef.transform(lon0, lat0, alt0))

        step = 1.0
        max_dist = 5000.0
        prev = None
        for dist in np.arange(0.0, max_dist + step, step):
            ecef_pt = origin_ecef + dir_ecef * dist
            lon, lat, alt = self._from_ecef.transform(*ecef_pt)
            try:
                dem_alt = self.dem_service.elevation_at(lon, lat)
            except ValueError:
                prev = (dist, lon, lat, alt, None)
                continue

            if prev and alt <= dem_alt:
                d_low, lon_low, lat_low, alt_low, _ = prev
                d_high = dist
                lon_high, lat_high = lon, lat
                for _ in range(15):
                    d_mid = (d_low + d_high) / 2
                    ecef_mid = origin_ecef + dir_ecef * d_mid
                    lon_mid, lat_mid, alt_mid = self._from_ecef.transform(*ecef_mid)
                    dem_mid = self.dem_service.elevation_at(lon_mid, lat_mid)
                    if alt_mid > dem_mid:
                        d_low = d_mid
                    else:
                        d_high = d_mid
                        lon_high, lat_high, dem_alt = lon_mid, lat_mid, dem_mid

                heading = (math.degrees(yaw) + 360) % 360
                tilt = max(0.0, min(90.0, 90.0 + math.degrees(pitch)))
                return {
                    'lat': lat_high,
                    'lon': lon_high,
                    'heading': heading,
                    'tilt': tilt,
                    'range': d_high,
                    'alt': alt0,
                    'target_alt': dem_alt,
                }
            prev = (dist, lon, lat, alt, dem_alt)

        return None

    # ------------------------------------------------------------------
    def pixel_to_latlon(self, image_path: str, x: int, y: int):
        view = self.pixel_to_view(image_path, x, y)
        if view:
            return view['lat'], view['lon']
        return None

