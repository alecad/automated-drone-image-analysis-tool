import os
from typing import Optional

import rasterio


class DEMService:
    """Service to query elevations from pre-downloaded DEM tiles.

    The service lazily loads GeoTIFF files from a configured directory and
    retrieves elevation values for geographic coordinates (lat, lon)."""

    def __init__(self, dem_directory: str):
        self.dem_directory = dem_directory
        self._datasets = []

    def _ensure_datasets_loaded(self) -> None:
        if self._datasets or not self.dem_directory:
            return
        if not os.path.isdir(self.dem_directory):
            return
        for name in os.listdir(self.dem_directory):
            if not name.lower().endswith('.tif'):
                continue
            path = os.path.join(self.dem_directory, name)
            try:
                ds = rasterio.open(path)
                self._datasets.append(ds)
            except Exception:
                # Skip unreadable tiles
                continue

    def _get_dataset(self, lat: float, lon: float):
        self._ensure_datasets_loaded()
        for ds in self._datasets:
            bounds = ds.bounds
            if bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top:
                return ds
        return None

    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """Return elevation in meters for the given latitude and longitude."""
        ds = self._get_dataset(lat, lon)
        if ds is None:
            return None
        try:
            row, col = ds.index(lon, lat)
            data = ds.read(1)
            value = data[row, col]
            if ds.nodata is not None and value == ds.nodata:
                return None
            return float(value)
        except Exception:
            return None
