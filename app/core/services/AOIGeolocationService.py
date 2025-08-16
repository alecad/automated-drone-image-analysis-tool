import math
import os
from typing import Optional, Tuple

import rasterio
from rasterio.errors import RasterioIOError
import utm

from core.services.ImageService import ImageService
from helpers.LocationInfo import LocationInfo


class AOIGeolocationService:
    """Compute ground coordinates for Areas of Interest using DEM data."""

    def __init__(self, dem_folder: str):
        self.dem_folder = dem_folder
        self._cache = {}

    def _get_dataset(self, lat: float, lon: float):
        for path, ds in self._cache.items():
            if ds.bounds.left <= lon <= ds.bounds.right and ds.bounds.bottom <= lat <= ds.bounds.top:
                return ds
        if not os.path.isdir(self.dem_folder):
            return None
        for fname in os.listdir(self.dem_folder):
            if not fname.lower().endswith('.tif'):
                continue
            fpath = os.path.join(self.dem_folder, fname)
            try:
                ds = rasterio.open(fpath)
            except RasterioIOError:
                continue
            if ds.bounds.left <= lon <= ds.bounds.right and ds.bounds.bottom <= lat <= ds.bounds.top:
                self._cache[fpath] = ds
                return ds
            ds.close()
        return None

    def _sample_elevation(self, lat: float, lon: float) -> Optional[float]:
        ds = self._get_dataset(lat, lon)
        if ds is None:
            return None
        try:
            for val in ds.sample([(lon, lat)]):
                return float(val[0])
        except Exception:
            return None

    def compute_aoi_coordinate(self, image_path: str, pixel: Tuple[int, int]) -> Optional[Tuple[float, float, float]]:
        service = ImageService(image_path)
        gps = LocationInfo.get_gps(exif_data=service.exif_data)
        if not gps:
            return None
        lat0 = gps['latitude']
        lon0 = gps['longitude']
        alt0 = service.get_asl_altitude('m') or 0
        yaw, pitch = service.get_gimbal_orientation()
        hfov = service.get_camera_hfov()
        if None in (yaw, pitch, hfov):
            return None
        image_height, image_width, _ = service.img_array.shape
        vfov = 2 * math.atan(math.tan(math.radians(hfov)/2) * (image_height / image_width))
        fx = (image_width/2) / math.tan(math.radians(hfov)/2)
        fy = (image_height/2) / math.tan(vfov/2)
        cx = image_width/2
        cy = image_height/2
        x = (pixel[0] - cx) / fx
        y = (pixel[1] - cy) / fy
        x_angle = math.atan(x)
        y_angle = math.atan(y)
        az = math.radians(yaw) + x_angle
        el = math.radians(pitch) - y_angle
        dx = math.cos(el) * math.sin(az)
        dy = math.cos(el) * math.cos(az)
        dz = math.sin(el)
        e0, n0, zone, letter = utm.from_latlon(lat0, lon0)
        step = 10.0
        max_range = 10000.0
        t = 0.0
        prev_e, prev_n, prev_alt = e0, n0, alt0
        prev_dem = self._sample_elevation(lat0, lon0)
        while t <= max_range:
            t += step
            e = e0 + dx * t
            n = n0 + dy * t
            alt = alt0 + dz * t
            lat, lon = utm.to_latlon(e, n, zone, letter)
            dem_alt = self._sample_elevation(lat, lon)
            if dem_alt is None:
                prev_e, prev_n, prev_alt = e, n, alt
                prev_dem = dem_alt
                continue
            if alt <= dem_alt:
                low = t - step
                high = t
                for _ in range(10):
                    mid = (low + high) / 2
                    e_m = e0 + dx * mid
                    n_m = n0 + dy * mid
                    alt_m = alt0 + dz * mid
                    lat_m, lon_m = utm.to_latlon(e_m, n_m, zone, letter)
                    dem_m = self._sample_elevation(lat_m, lon_m)
                    if dem_m is None:
                        low = mid
                        continue
                    if alt_m > dem_m:
                        low = mid
                    else:
                        high = mid
                t = high
                e = e0 + dx * t
                n = n0 + dy * t
                lat, lon = utm.to_latlon(e, n, zone, letter)
                dem_alt = self._sample_elevation(lat, lon) or alt
                return lat, lon, dem_alt
            prev_e, prev_n, prev_alt = e, n, alt
            prev_dem = dem_alt
        return None
