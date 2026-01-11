from shapely.geometry import Polygon
import numpy as np
from shapely.prepared import prep
import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image, ImageChops


def grid_bounds(geom, nx, ny):
    minx, miny, maxx, maxy = geom.bounds
    gx, gy = np.linspace(minx,maxx,nx+1), np.linspace(miny,maxy,ny+1)
    grid = []
    for i in range(nx):
        for j in range(ny):
            poly_ij = Polygon([[gx[i],gy[j]],[gx[i],gy[j+1]],[gx[i+1],gy[j+1]],[gx[i+1],gy[j]]])
            grid.append( poly_ij )
    return grid


def partition(geom, nx, ny):
    grid = prep(geom)
    # grid = list(filter(grid.intersects, grid_bounds(geom, nx, ny)))
    return grid_bounds(geom, nx, ny)

crs = "EPSG:4326"
path = '../data/shapefile/nyc_boroughs.shp'
nyc = gpd.read_file(path)
# Transform to WGS84 to match lat/lon coordinates
nyc = nyc.to_crs("EPSG:4326")
geom = nyc.dissolve()['geometry'].values[0]

grid = partition(geom, 50, 70)  # Adjust grid dimensions for NYC (more vertical)

fig, ax = plt.subplots(figsize=(8, 10))
nyc.plot(ax=ax, edgecolor='black', aspect=None)
gpd.GeoSeries(grid, crs=crs).boundary.plot(ax=ax, edgecolor='red', aspect=None)
plt.title("Windows for Conv-LSTM's - NYC")
# hide ticks
plt.xticks([]),plt.yticks([])
plt.show()
