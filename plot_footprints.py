import os
import h5py
import numpy as np
from spatial import read_shape_files, read_sa2_meta
from lib import colour_polygons_by_vector
from utils import flatten_list, is_empty, clean_string
from pyexcel_xlsx import get_data

#  Create save directory
save_dir = os.environ['save_dir']
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# Get SA2 meta information and shape files
asgs_path = os.environ['asgs_dir'] + 'SA2/2011/'

sa2_meta_store = read_sa2_meta(asgs_path + 'SA2_2011_AUST.xlsx')
all_shapes = read_shape_files(asgs_path + 'SA2_2011_AUST.shp')

if len(all_shapes) != len(sa2_meta_store):
    raise ValueError('SA2 meta version is different to the shape file version')

# Read footprint results
birds_dir = os.environ['birds_dir']
filename = birds_dir + '/results/' + 'footprints.h5'
f = h5py.File(filename, 'r')

footprints_by_subregion = list(f['footprints_usa_fd_aus_subregions'])

# Relate the base regions to the SA2 regions
subnat_region_legend = flatten_list(list(f['subnat_region_legend']))

# Fix orientation of footprints data (no. of subregions in the SA2 list matches no. of footprints regions)
footprints_by_subregion = np.rot90(footprints_by_subregion, k=1)
assert max(subnat_region_legend) == footprints_by_subregion[0].shape

# Read the bird names
data = get_data(birds_dir + '/results/' + 'birds_satellite_labels.xlsx')
bird_labels = flatten_list(data['Sheet1'])

assert len(footprints_by_subregion) == len(bird_labels)

# Figure bounding box [x_min, x_max, y_min, y_max]
aus_bounding_box = [110, 155, -45, -5]

# For each bird plot the threats intensity over Aus
for i, b in enumerate(bird_labels):

    bird_name = bird_labels[i]

    # The footprints for this bird
    footprint_set = footprints_by_subregion[i]

    # Check there is a nonzero footprint for this bird
    if sum(footprint_set) != 0:

        print('Plotting ' + bird_name)

        # Make data structure to plot
        intensity_by_sa2 = []

        # Link each sub-national footprint value with member SA2s
        for j, val in enumerate(subnat_region_legend):
            parent_base_region = int(subnat_region_legend[j])
            intensity_by_sa2.append(footprint_set[parent_base_region-1])

        assert not is_empty(intensity_by_sa2)
        assert sum(intensity_by_sa2) > 0

        # Send to plotter
        save_fname = save_dir + '/' + clean_string(bird_name, [' ', "'"], '_', case='lower') + '.png'
        regions_to_plot = list(range(1, 2215))
        colour_polygons_by_vector(intensity_by_sa2, all_shapes, regions_to_plot, save_fname,
                                  bounding_box=aus_bounding_box)

    print('Skipping ' + bird_name)


# Some base regions have no impacts

print('Finished plotting bird footprints')