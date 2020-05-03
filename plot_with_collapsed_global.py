import os
import h5py
import numpy as np
import shutil
from spatial import read_shape_files, read_sa2_meta
from lib import colour_polygons_by_vector, make_save_name, make_sa2_adjacency, smear_adjacent_values
from utils import flatten_list, is_empty, is_within_tolerance
from pyexcel_xlsx import get_data
import geopandas as gp

print('Starting footprint plotter')

# Options
clean_save_dir = False  # Deletes previous results!
draw_frame = False  # Frame around plot
colour_palette = 'BuPu'
plot_background_colour = 'gainsboro'
colour_normalisation = 'symlog'  # 'linear'  'symlog'
figure_quality = 500  # dpi
common_colour_scaling = True
polygon_smearing = False
normalise_set_totals = False
log_scale_z = False
filter_birds = False
scaler = 1  # 50
color_bar = True
scaling_limits = (0, 10)

# Check directories
print('Checking directories')

birds_dir = os.environ['birds_dir']
assert os.path.isdir(birds_dir) is True

asgs_path = os.environ['asgs_dir'] + 'SA2/2011/'
assert os.path.isdir(asgs_path) is True

# Create output directory
save_dir = os.environ['save_dir']
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
elif clean_save_dir:
    shutil.rmtree(save_dir)
    os.mkdir(save_dir)

print('Set save path to ' + save_dir.replace(birds_dir, '/.../'))

# Read SA2 meta
sa2_meta_store = read_sa2_meta(asgs_path + 'SA2_2011_AUST.xlsx')
all_shapes = read_shape_files(asgs_path + 'SA2_2011_AUST.shp')

n_sa2s = len(sa2_meta_store)
assert len(all_shapes) == n_sa2s, 'SA2 meta version is different to the shape file version'
regions_to_plot = list(range(1, n_sa2s + 1))

# Read footprint results
filename = birds_dir + 'results/sa2_h5s/' + 'footprints_remake.h5'

print('Reading ' + filename)

f = h5py.File(filename, 'r')

# Relate the base regions to the SA2 regions
f_legend = h5py.File(birds_dir + 'results/' + 'footprints.h5', 'r')
subnat_region_legend = flatten_list(list(f_legend['subnat_region_legend']))
assert n_sa2s == len(subnat_region_legend)
base_regions = list(set(subnat_region_legend))

base_region_state_key = read_sa2_meta(birds_dir + '/results/' + 'base_region_state_member_labels.csv')

# Read the bird names
data = get_data(birds_dir + '/results/' + 'birds_satellite_labels.xlsx')
bird_labels = flatten_list(data['Sheet1'])

assert f['nested_footprints_sa2'].shape[1] == len(bird_labels)  # labels dim matches data

# Figure bounding box [x_min, x_max, y_min, y_max]
aus_bounding_box = [112, 155, -45, -9]

# Bird filtering by name
filter_birds_list = ['Buff-breasted Button-quail', "Coxen's Fig-Parrot", 'Golden-shouldered Parrot'
                     , 'Fleurieu Peninsula Southern Emu-wren', 'Southern Star Finch'
                     , 'Fleurieu Peninsula Southern Emu-wren', 'Southern Black-throated Finch ']

assert all(elem in bird_labels for elem in filter_birds_list)

fields_to_plot = ['global_footprints_sa2', 'nested_footprints_sa2', 'raw_satellite_sa2'] # , 'nested_footprints_qld_sa2'

global_total = sum(flatten_list(list(f['global_footprints_sa2'])))
print('Global total is ' + '{:.2f}'.format(global_total))

if polygon_smearing:

    df = gp.read_file(asgs_path + 'SA2_2011_AUST.shp')  # open file
    adjacency_matrix = make_sa2_adjacency(df, n_sa2s)
    assert len(adjacency_matrix) == n_sa2s and len(adjacency_matrix[0]) == n_sa2s

# Start plotting
for field_name in fields_to_plot:

    print('Plotting ' + field_name)

    # Unpack footprints from store
    if field_name not in list(f):
        print('Field does not exist in results')
    else:
        footprints_by_subregion = list(f[field_name])

        dataset_total = sum(sum(footprints_by_subregion))
        print('Dataset total: ' + str('{:.2f}'.format(dataset_total)))

        # Fix orientation of footprints data
        footprints_by_subregion = np.transpose(footprints_by_subregion)
        assert len(subnat_region_legend) == footprints_by_subregion[0].shape[0]
        assert len(footprints_by_subregion) == len(bird_labels)

        intensity_by_sa2 = np.zeros(n_sa2s)  # Make data structure to plot

        # Get the footprints for each bird
        for i, b in enumerate(bird_labels):
            if not filter_birds or (filter_birds and b not in filter_birds_list):
                intensity_by_sa2 = intensity_by_sa2 + footprints_by_subregion[i]

        # Tests
        assert not is_empty(intensity_by_sa2)
        assert sum(intensity_by_sa2) > 0

        if not filter_birds:  # sum does not change
            assert is_within_tolerance(sum(intensity_by_sa2), dataset_total, 0.001)

        if polygon_smearing:
            intensity_by_sa2 = smear_adjacent_values(intensity_by_sa2, adjacency_matrix, mixing=0.33)

        if normalise_set_totals and '_global' not in field_name:
            intensity_by_sa2 = intensity_by_sa2 * global_total/dataset_total  #* 1.6

        if not all(intensity_by_sa2 > 0):
            intensity_by_sa2 = intensity_by_sa2 + (-1*min(intensity_by_sa2))

        if log_scale_z:
            intensity_by_sa2 = np.log(intensity_by_sa2)  # + 1
        else:
            intensity_by_sa2 = intensity_by_sa2 * scaler

        assert all(intensity_by_sa2 >= 0)

        print('Final plotted total: ' + str('{:.2f}'.format(sum(intensity_by_sa2)))
              + ', max in set: ' + str('{:.2f}'.format(max(intensity_by_sa2))))

        # Make filename
        description = 'sa2'
        if filter_birds:
            description = description + '_fb'
        if log_scale_z:
            description = description + '_lsz'
        if scaler > 1:
            description = description + '_sc' + str(int(scaler))

        save_fname = make_save_name(save_dir, description, field_name, colour_palette, colour_normalisation,
                                    colour_option=common_colour_scaling, smear_option=polygon_smearing)

        if 'global' not in save_fname:
            save_fname = save_fname.replace('_fd', '_nested_mrio_')

        # Send to plotter
        colour_polygons_by_vector(intensity_by_sa2, all_shapes, regions_to_plot, save_fname
                                  , bounding_box=aus_bounding_box, colour_map=colour_palette, attach_colorbar=color_bar
                                  , plot_background=plot_background_colour, show_frame=draw_frame
                                  , normalisation_str=colour_normalisation, quality=figure_quality
                                  , colour_min_max=scaling_limits)