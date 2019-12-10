from pathlib import Path
import shapefile
from shapely.geometry import Polygon
import numpy as np

# Paths
top_level_path = Path(__file__).parents[1]
asgsdir = str(top_level_path) + '/ASGS/SA2/2011/'

# Read shape file
filename = asgsdir + 'SA2_2011_AUST.shp'
sf = shapefile.Reader(filename)
count_shapes = len(list(sf.iterShapes()))

# SA2 centroids
region_centroids = np.zeros(shape=(count_shapes+1,2))
count = 0
for shape in list(sf.iterShapes()):

    polygon = Polygon(shape.points)
    center = polygon.centroid
    try:
        region_centroids[count] = [center.x, center.y]
    except:
        print('Could not access point')

    count = count + 1

