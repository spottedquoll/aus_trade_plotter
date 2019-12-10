from pathlib import Path
import shapefile
from shapely.geometry import Polygon
import numpy as np
import pyexcel as pe
from trade_allocation.spatial import haversine
import xlsxwriter
import os


# Paths
top_level_path = Path(__file__).parents[3]
asgsdir = str(top_level_path) + '/ASGS/SA2/2011/'

# Read shape file
filename = asgsdir + 'SA2_2011_AUST.shp'
sf = shapefile.Reader(filename)
count_shapes = len(list(sf.iterShapes()))

# SA2 centroids
region_centroids = np.zeros(shape=(count_shapes,2))
count = 0
for shape in list(sf.iterShapes()):

    polygon = Polygon(shape.points)
    center = polygon.centroid
    try:
        region_centroids[count] = [center.x, center.y]
    except:
        pass  # some SA2s are not physical places...

    count = count + 1

# Get port locations
port_locations = []
records = pe.iget_records(file_name=str(top_level_path) + '/ABS/' + 'Australian_port_locations.xlsx')
for row in records:
    port_locations.append(row)

# Calculate distances
distances = np.zeros(shape=(len(region_centroids), len(port_locations)))

row_counter = 0
for i in region_centroids:

    origin_lat = i[1]
    origin_lon = i[0]

    col_counter = 0

    if i[0] == 0 and i[1] == 0:
        stop = 1
    else:
        for j in port_locations:
            dest_lat = j['Latitude']
            dest_lon = j['Longitude']
            dist = haversine(origin_lon, origin_lat, dest_lon, dest_lat)
            distances[row_counter][col_counter] = dist

            col_counter = col_counter + 1

    row_counter = row_counter + 1

# Export distance matrix
workdir = os.path.dirname(os.path.realpath(__file__))
book = xlsxwriter.Workbook(workdir + '/port_region_distances.xlsx')
sheet1 = book.add_worksheet('sheet1')

for i, l in enumerate(distances):
    for j, col in enumerate(l):
        sheet1.write(i, j, col)

book.close()