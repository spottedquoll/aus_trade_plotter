from pathlib import Path
import shapefile
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
import pyexcel as pe

# Paths
top_level_path = Path(__file__).parents[1]
asgsdir = str(top_level_path) + '/ASGS/'

# Get SA2 meta information
sa2_meta_store = []
records = pe.iget_records(file_name=asgsdir + '/sa2/' + 'SA2_2016_AUST.xlsx')
for row in records:
    sa2_meta_store.append(row)

# Read shape file
filename = asgsdir + '/sa2/SA2_2016_AUST.shp'
sf = shapefile.Reader(filename)
count_shapes = len(list(sf.iterShapes()))

# Set up plot
plt.figure()
ax = plt.axes() # add the axes
ax.set_aspect('equal')

plt.figure()
ax = plt.axes()
ax.set_aspect('equal')

shape_ex = sf.shape(5)  # could break if selected shape has multiple polygons.

# build the polygon from exterior points
polygon = Polygon(shape_ex.points)
patch = PolygonPatch(polygon, facecolor=[0, 0, 0.5], edgecolor=[0, 0, 0], alpha=0.7, zorder=2)
ax.add_patch(patch)

# use bbox (bounding box) to set plot limits
plt.xlim(shape_ex.bbox[0], shape_ex.bbox[2])
plt.ylim(shape_ex.bbox[1], shape_ex.bbox[3])

plt.show()