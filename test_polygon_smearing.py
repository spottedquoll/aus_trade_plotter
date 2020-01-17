import os
import h5py
import numpy as np
from spatial import read_shape_files, read_sa2_meta
import geopandas as gp

print('Build adjaceny matrix plotter')

# Get SA2 meta information and shape files
asgs_path = os.environ['asgs_dir'] + 'SA2/2011/'

sa2_meta_store = read_sa2_meta(asgs_path + 'SA2_2011_AUST.xlsx')
all_shapes = read_shape_files(asgs_path + 'SA2_2011_AUST.shp')

n_sa2s = len(sa2_meta_store)
assert len(all_shapes) == n_sa2s, 'SA2 meta version is different to the shape file version'

adjacency = []

df = gp.read_file(asgs_path + 'SA2_2011_AUST.shp') # open file
df["NEIGHBORS"] = None

for index, row in df.iterrows():
    print(str(index))
    if row['geometry'] is not None:
        neighbors = df[df.geometry.touches(row['geometry'])]
        new_row = np.zeros(n_sa2s)
        new_row[neighbors.index.values] = 1
        adjacency.append(new_row)

#
# for index in sub_regions:
#
#     shape = all_shapes[int(index) - 1]
#     polygon_parts = len(shape.parts)
#
#     try:
#         colour_rgb = colour_scaling.get_rgb(colour_scale_data[count])
#     except:
#         raise ValueError('Could not retrieve RBG values')
#
#     if polygon_parts == 1:
#         polygon = Polygon(shape.points)
#
#         try:
#             patch = PolygonPatch(polygon, fc=(colour_rgb[0], colour_rgb[1], colour_rgb[2], 0.7), ec="none")
#             ax.add_patch(patch)
#         except:
#             raise ValueError('Could not build patch')
#
#     elif polygon_parts > 1:
#         for ip in range(polygon_parts):  # loop over parts, plot separately
#             i0 = shape.parts[ip]
#             if ip < polygon_parts - 1:
#                 i1 = shape.parts[ip + 1] - 1
#             else:
#                 i1 = len(shape.points)
#
#             polygon = Polygon(shape.points[i0:i1 + 1])
#
#             patch = PolygonPatch(polygon, fc=(colour_rgb[0], colour_rgb[1], colour_rgb[2], 0.7), ec=polygon_edge)
#             ax.add_patch(patch)