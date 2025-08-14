from pathlib import Path

import rasterio


class DEMService:
    """Load and query elevation data from a directory of GeoTIFF files."""

    def __init__(self, dem_dir):
        self.datasets = []
        dem_dir = Path(dem_dir)
        for path in dem_dir.glob('*.tif'):
            ds = rasterio.open(path)
            self.datasets.append((ds, ds.read(1)))

    def elevation_at(self, lon, lat):
        """Return elevation at the given lon/lat using the first DEM covering the point."""
        for ds, arr in self.datasets:
            if ds.bounds.left <= lon <= ds.bounds.right and ds.bounds.bottom <= lat <= ds.bounds.top:
                r, c = ds.index(lon, lat)
                if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                    return float(arr[r, c])
        raise ValueError('Coordinate outside DEM coverage')
