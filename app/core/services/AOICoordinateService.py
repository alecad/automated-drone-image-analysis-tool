import math
import numpy as np
import piexif
import rasterio

from core.services.ImageService import ImageService
from helpers.LocationInfo import LocationInfo
from helpers.PickleHelper import PickleHelper


class AOICoordinateService:
    """Estimate geographic coordinates for pixels in a drone image.

    The service projects a pixel from an image onto the ground surface
    using the camera orientation and a DEM (Digital Elevation Model).
    """

    EARTH_RADIUS = 6378137.0  # meters

    def __init__(self, dem_path):
        self.dem = rasterio.open(dem_path)
        self.elevation = self.dem.read(1)

    def pixel_to_latlon(self, image_path, x, y):
        """Return (lat, lon) for a pixel coordinate.

        Args:
            image_path (str): path to the source image.
            x (int): pixel x coordinate (origin at left).
            y (int): pixel y coordinate (origin at top).

        Returns:
            tuple[float, float] | None: latitude and longitude if the
                projection intersects the DEM, otherwise ``None``.
        """
        image_service = ImageService(image_path)
        exif = image_service.exif_data
        xmp = image_service.xmp_data
        gps = LocationInfo.get_gps(exif_data=exif)
        if not gps:
            return None

        lat0 = gps['latitude']
        lon0 = gps['longitude']
        try:
            alt0 = float(xmp.get('AbsoluteAltitude'))
        except (TypeError, ValueError):
            alt0 = 0.0

        width = exif['Exif'][piexif.ExifIFD.PixelXDimension]
        height = exif['Exif'][piexif.ExifIFD.PixelYDimension]
        focal = exif['Exif'][piexif.ExifIFD.FocalLength]
        f_mm = focal[0] / focal[1]

        info = PickleHelper.get_drone_sensor_info()
        model = exif['0th'][piexif.ImageIFD.Model].decode('utf-8').strip()
        row = info[(info['Model (Exif)'].str.contains(model)) & (info['Image Source (XMP)'] == 'WideCamera')]
        if row.empty:
            row = info[info['Model (Exif)'].str.contains(model)]
        sensor_w = float(row['sensor_w'].iloc[0])
        sensor_h = float(row['sensor_h'].iloc[0])

        fx = (width / sensor_w) * f_mm
        fy = (height / sensor_h) * f_mm
        cx = width / 2
        cy = height / 2

        ang_x = math.atan((x - cx) / fx)
        ang_y = math.atan((y - cy) / fy)

        yaw = math.radians(float(xmp.get('GimbalYawDegree', 0))) + ang_x
        pitch = math.radians(float(xmp.get('GimbalPitchDegree', 0))) + ang_y

        d_e = math.cos(pitch) * math.sin(yaw)
        d_n = math.cos(pitch) * math.cos(yaw)
        d_u = math.sin(pitch)

        step = 5
        max_dist = 5000
        prev = None

        for d in np.arange(0, max_dist + step, step):
            e = d * d_e
            n = d * d_n
            u = d * d_u

            lat = lat0 + (n / self.EARTH_RADIUS) * 180 / math.pi
            lon = lon0 + (e / (self.EARTH_RADIUS * math.cos(math.radians(lat0)))) * 180 / math.pi
            alt = alt0 + u

            try:
                r, c = self.dem.index(lon, lat)
                dem_alt = float(self.elevation[r, c])
            except Exception:
                prev = (d, lat, lon, alt, None)
                continue

            if prev and alt <= dem_alt:
                d0, lat0p, lon0p, alt0p, dem0 = prev
                if dem0 is None:
                    return lat, lon
                frac = (alt0p - dem0) / ((alt0p - dem0) - (alt - dem_alt))
                lat = lat0p + (lat - lat0p) * frac
                lon = lon0p + (lon - lon0p) * frac
                return lat, lon
            prev = (d, lat, lon, alt, dem_alt)

        return None
