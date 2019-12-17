import os
import h5py
import numpy as np
from spatial import read_shape_files, read_sa2_meta
from lib import colour_polygons_by_vector
from utils import flatten_list, is_empty, clean_string, is_within_tolerance
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

# Relate the base regions to the SA2 regions
subnat_region_legend = flatten_list(list(f['subnat_region_legend']))
n_sa2s = len(subnat_region_legend)

# Read the bird names
data = get_data(birds_dir + '/results/' + 'birds_satellite_labels.xlsx')
bird_labels = flatten_list(data['Sheet1'])

# Figure bounding box [x_min, x_max, y_min, y_max]
aus_bounding_box = [110, 155, -45, -5]

# Footprint dimensions
fd_segments = ['USA', 'DEU', 'JPN', 'CHN']
fd_products = ['all_products', 'aus_products', 'aus_agri_products', 'all_agri_products']

# Options
draw_frame = False  # Frame around plot
divide_intensity_among_sa2s = True  # divide total threat intensity among member SA2s in base region
plot_all_birds = True
plot_separate_birds = False
colourings = 'Purples'  # 'PuRd'  # 'plasma'
plot_background_colour = 'gainsboro'
colour_scaling = ['symlog', 'linear']
figure_quality = 900  # dpi
common_colour_scaling = True  # Use same min-max colour scales for all countries

# Precalculate the value ranges for each product set, to allow common colour scaling (combined over countries)
limits_of_product_sets = {}
for product in fd_products:
    limits_of_product_sets[product] = {}
    combine = []
    for fd_seg in fd_segments:
        field_name = 'footprints_fdseg_' + fd_seg + '_' + product
        footprints_by_subregion = list(f[field_name])
        combine.append(flatten_list(footprints_by_subregion))
    limits_of_product_sets[product]['min'] = min(flatten_list(combine))
    limits_of_product_sets[product]['max'] = max(flatten_list(combine))

# Loop over countries
for fd_seg in fd_segments:

    print('Using ' + fd_seg + ' final demand')

    for product in fd_products:

        # Unpack footprints from store
        field_name = 'footprints_fdseg_' + fd_seg + '_' + product
        footprints_by_subregion = list(f[field_name])

        # Fix orientation of footprints data (no. of subregions in the SA2 list matches no. of footprints regions)
        footprints_by_subregion = np.rot90(footprints_by_subregion, k=1)

        assert max(subnat_region_legend) == footprints_by_subregion[0].shape
        assert len(footprints_by_subregion) == len(bird_labels)

        # Set colour scaling to use same min-max for all figures
        if common_colour_scaling:
            scaling_limits = (limits_of_product_sets[product]['min'] * 0.95,
                              limits_of_product_sets[product]['max'] * 1.05)
        else:
            scaling_limits = None

        # Plot the threats intensity over Aus (All birds same figure)
        if plot_all_birds is True:

            print('Plotting all birds same figure ' + field_name)

            intensity_by_sa2 = np.zeros(n_sa2s)  # Make data structure to plot

            for i, b in enumerate(bird_labels):

                # The footprints for this bird
                footprint_set = footprints_by_subregion[i]

                # Check there is a nonzero footprint for this bird
                if sum(footprint_set) != 0:

                    # Link each sub-national footprint value with member SA2s
                    for j, val in enumerate(subnat_region_legend):
                        parent_base_region = int(subnat_region_legend[j])
                        intensity_value = footprint_set[parent_base_region-1]
                        if intensity_value > 0:
                            if divide_intensity_among_sa2s is False:
                                intensity_by_sa2[j] = intensity_by_sa2[j] + intensity_value
                            else:
                                matches = len([p for p in subnat_region_legend if p == parent_base_region])
                                intensity_by_sa2[j] = intensity_by_sa2[j] + intensity_value/matches

            assert not is_empty(intensity_by_sa2)
            if divide_intensity_among_sa2s is True:
                assert is_within_tolerance(sum(intensity_by_sa2), sum(sum(footprints_by_subregion)), 0.001)

            regions_to_plot = list(range(1, n_sa2s + 1))

            for cs in colour_scaling:

                # Send to plotter
                description = field_name.lower().replace('footprints_', '')
                save_fname = save_dir + '/' + 'all_birds_' + description + '_' + colourings.lower() + '_' + cs
                if common_colour_scaling:
                    save_fname = save_fname + '_comcol'
                save_fname = save_fname + '.png'

                colour_polygons_by_vector(intensity_by_sa2, all_shapes, regions_to_plot, save_fname,
                                          bounding_box=aus_bounding_box, colour_map=colourings,
                                          plot_background=plot_background_colour, show_frame=draw_frame
                                          , normalisation=cs, quality=figure_quality, colour_min_max=scaling_limits)

        if plot_separate_birds is True:

            # For each bird plot the threats intensity over Aus (Plot each bird in separate figure)
            print('Plotting birds on separate figures')

            for i, b in enumerate(bird_labels):

                bird_name = bird_labels[i]

                # The footprints for this bird
                footprint_set = footprints_by_subregion[i]

                # Check there is a nonzero footprint for this bird
                if sum(footprint_set) != 0:

                    print('Plotting ' + bird_name + ' for ' + fd_seg + ' final demand')

                    intensity_by_sa2 = np.zeros(n_sa2s)  # Make data structure to plot

                    # Link each sub-national footprint value with member SA2s
                    for j, val in enumerate(subnat_region_legend):
                        parent_base_region = int(subnat_region_legend[j])
                        intensity_by_sa2[j] = footprint_set[parent_base_region-1]

                    assert not is_empty(intensity_by_sa2)
                    assert sum(intensity_by_sa2) > 0

                    regions_to_plot = list(range(1, n_sa2s + 1))

                    for cs in colour_scaling:

                        # Send to plotter
                        clean_bird_name = clean_string(bird_name, [' ', "'"], '_', case='lower')
                        save_fname = (save_dir + '/' + clean_bird_name + '_' + field_name.lower() + '_'
                                               + colourings.lower() + '_' + cs)
                        if common_colour_scaling:
                            save_fname = save_fname + '_comcol'
                        save_fname = save_fname + '.png'

                        colour_polygons_by_vector(intensity_by_sa2, all_shapes, regions_to_plot, save_fname
                                                                  , bounding_box=aus_bounding_box, colour_map=colourings
                                                                  , plot_background=plot_background_colour
                                                                  , show_frame=draw_frame, normalisation=cs
                                                                  , quality=figure_quality
                                                                  , colour_min_max=scaling_limits)
                else:
                    print('Skipping ' + bird_name)

# Some base regions have no impacts

print('.')
print('Finished')
