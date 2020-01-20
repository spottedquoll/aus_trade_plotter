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

# Get SA2 meta information and shape files
asgs_path = os.environ['asgs_dir'] + 'SA2/2011/'

sa2_meta_store = read_sa2_meta(asgs_path + 'SA2_2011_AUST.xlsx')
all_shapes = read_shape_files(asgs_path + 'SA2_2011_AUST.shp')

n_sa2s = len(sa2_meta_store)
assert len(all_shapes) == n_sa2s, 'SA2 meta version is different to the shape file version'

# Read footprint results
birds_dir = os.environ['birds_dir']
filename = birds_dir + '/results/' + 'footprints.h5'
f = h5py.File(filename, 'r')

# Relate the base regions to the SA2 regions
subnat_region_legend = flatten_list(list(f['subnat_region_legend']))
assert n_sa2s == len(subnat_region_legend)
base_regions = list(set(subnat_region_legend))

base_region_state_key = read_sa2_meta(birds_dir + '/results/' + 'base_region_state_member_labels.csv')

# Read the bird names
data = get_data(birds_dir + '/results/' + 'birds_satellite_labels.xlsx')
bird_labels = flatten_list(data['Sheet1'])

assert f[list(f)[0]].shape[1] == len(bird_labels)  # labels dim matches data

# Figure bounding box [x_min, x_max, y_min, y_max]
aus_bounding_box = [112, 155, -45, -9]

# Footprint dimensions
fd_segments = ['ALL', 'USA']  # , 'JPN', 'DEU', 'GBR', 'DEU', 'CHN', IND, FRA
fd_products = ['aus_agri_products', 'all_products', 'aus_products']  # , 'aus_products', 'all_agri_products'
stressors = 'threatsall'  # ['threatsall', 'threatsbeef']

# Options
clean_save_dir = False  # Deletes previous results!
regions_to_plot = list(range(1, n_sa2s + 1))
draw_frame = False  # Frame around plot
divide_intensity_among_sa2s = True  # divide total threat intensity among member SA2s in base region
colour_palette = 'BuPu'  # 'RdPu'  # 'Purples'  # 'PuRd'  # 'plasma' # BuPu
plot_background_colour = 'gainsboro'
colour_normalisation = 'symlog'  # 'symlog', 'log', 'linear'
figure_quality = 600  # dpi
common_colour_scaling = True
results_offset = 4  # 1
force_scaling = 'all_products'  # 'all_products' or None
apply_polygon_smearing = True
smearing_opts = [True, False]

plot_all_birds = True  # All bird footprints, driven by a country, on the one figure
plot_all_birds_at_sa2 = True  # All bird footprints, driven by a country, on the one figure, at SA2 resolution
plot_separate_birds = False  # Each bird and country pair plotted on a new figure
plot_subregion_fd = False
plot_custom_region_groups = False
plot_custom_bird_groups = False
plot_global_mrio_footprints = False

# Custom regions [[NSW, Qld], Qld only, NT and WA]
custom_regions = [[1, 3, 7, 12, 13, 14, 15, 16, 17, 18, 19]
                  , [3, 13, 14, 16, 17, 19]
                  , [5, 8, 9, 10, 11, 20, 21]]

custom_region_labels = ['NSW&Qld', 'Qld', 'NT&WA']

# Alligator Rivers Yellow Chat, Noisy Scrub-bird, Southern Black-throated Finch, Southern Cassowary
custom_bird_subset = [0, 10, 11, 12, 13]

# Output directory
print('Creating save directory')
save_dir = os.environ['save_dir']
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
elif clean_save_dir:
    shutil.rmtree(save_dir)
    os.mkdir(save_dir)

# Precalculate the value ranges for each product set, to allow common colour scaling (combined over countries)
#   Sum over all birds
if common_colour_scaling:

    print('Creating limits for max-min colour scaling')

    limits_of_product_sets = {}
    for product in fd_products:

        limits_of_product_sets[product] = {}
        combine = []

        for fd_seg in fd_segments:
            field_name = 'footprints_fdseg' + fd_seg + '_' + product + '_' + stressors
            footprints_by_subregion = list(f[field_name])
            combine.append(flatten_list(footprints_by_subregion))

        limits_of_product_sets[product]['min'] = min(flatten_list(combine))
        limits_of_product_sets[product]['max'] = max(flatten_list(combine))
        assert limits_of_product_sets[product]['min'] >= 0

    #   Individual birds
    limits_of_product_sets_by_bird = {}
    for product in fd_products:

        limits_of_product_sets_by_bird[product] = {}

        for i, b in enumerate(bird_labels):

            bird_name = bird_labels[i]
            limits_of_product_sets_by_bird[product][bird_name] = {}
            combine = []

            for fd_seg in fd_segments:

                field_name = 'footprints_fdseg' + fd_seg + '_' + product + '_' + stressors
                footprints_by_subregion = list(f[field_name])
                footprints_by_subregion = np.transpose(footprints_by_subregion)
                combine.append(footprints_by_subregion[i])

            limits_of_product_sets_by_bird[product][bird_name]['min'] = min(flatten_list(combine))
            limits_of_product_sets_by_bird[product][bird_name]['max'] = max(flatten_list(combine))
            assert limits_of_product_sets_by_bird[product][bird_name]['min'] >= 0

    #   Limits for base region fd segment case (common over all birds and fd_segments)
    limits_of_product_sets_by_bird_br_segs = {}
    product = 'aus_agri_products'
    combine = []
    for fd_seg in fd_segments:
        for k in base_regions:

            # Unpack footprints from store
            field_name = 'footprints_fdseg' + fd_seg + '_' + product + '_' + str(int(k)) + '_' + stressors
            footprints_by_subregion = list(f[field_name])
            combine.append(flatten_list(footprints_by_subregion))

    limits_of_product_sets_by_bird_br_segs['min'] = min(flatten_list(combine))
    limits_of_product_sets_by_bird_br_segs['max'] = max(flatten_list(combine))
    assert limits_of_product_sets_by_bird_br_segs['min'] >= 0
else:
    limits_of_product_sets = []

# Test input data
test_field = 'footprints_fdsegUSA_aus_products' + '_' + stressors
test_bird = 'Alligator Rivers Yellow Chat'
if test_field in list(f) and test_bird in bird_labels:

    footprints_by_subregion = list(f[test_field])
    footprints_by_subregion = np.transpose(footprints_by_subregion)
    assert len(footprints_by_subregion) == len(bird_labels)

    test_set = footprints_by_subregion[bird_labels.index(test_bird)]
    assert np.nonzero(test_set)[0][0] == 20 - 1  # corresponding to base region 20 (will fail if regagg changes)

if apply_polygon_smearing:
    print('Building SA2 adjacency matrix')

    df = gp.read_file(asgs_path + 'SA2_2011_AUST.shp')  # open file
    adjacency_matrix = make_sa2_adjacency(df, n_sa2s)
    assert len(adjacency_matrix) == n_sa2s and len(adjacency_matrix[0]) == n_sa2s

# Loop over countries
for fd_seg in fd_segments:

    print('Using ' + fd_seg + ' final demand')

    for product in fd_products:

        # Unpack footprints from store
        field_name = 'footprints_fdseg' + fd_seg + '_' + product + '_' + stressors
        footprints_by_subregion = list(f[field_name])

        # Fix orientation of footprints data (no. of subregions in the SA2 list matches no. of footprints regions)
        footprints_by_subregion = np.transpose(footprints_by_subregion)

        assert max(subnat_region_legend) == footprints_by_subregion[0].shape
        assert len(footprints_by_subregion) == len(bird_labels)

        # Plot the threats intensity over Aus (All birds same figure)
        if plot_all_birds is True:

            # Colour scaling can be set to use same min-max for all figures
            if common_colour_scaling:
                scaling_limits = (limits_of_product_sets[product]['min'] * 0.95,
                                  limits_of_product_sets[product]['max'] * 1.05)
            else:
                scaling_limits = None

            intensity_by_sa2 = np.zeros(n_sa2s)  # Make data structure to plot

            for i, b in enumerate(bird_labels):

                # The footprints for this bird
                footprint_set = footprints_by_subregion[i]

                # Check there is a nonzero footprint for this bird
                if sum(footprint_set) != 0:

                    # Link each sub-national footprint value with member SA2s
                    for j, val in enumerate(subnat_region_legend):
                        parent_region = int(subnat_region_legend[j])
                        intensity_value = footprint_set[parent_region - 1]
                        if intensity_value > 0:
                            if divide_intensity_among_sa2s is False:
                                intensity_by_sa2[j] = intensity_by_sa2[j] + intensity_value
                            else:
                                n_matches = len([p for p in subnat_region_legend if p == parent_region])
                                intensity_by_sa2[j] = intensity_by_sa2[j] + intensity_value / n_matches

            assert not is_empty(intensity_by_sa2)

            if divide_intensity_among_sa2s is True:
                assert is_within_tolerance(sum(intensity_by_sa2), sum(sum(footprints_by_subregion)), 0.001)

            # Send to plotter
            save_fname = make_save_name(save_dir, 'all_birds', field_name, colour_palette, cs,
                                        colour_option=common_colour_scaling)

            colour_polygons_by_vector(intensity_by_sa2, all_shapes, regions_to_plot, save_fname
                                      , bounding_box=aus_bounding_box, colour_map=colour_palette
                                      , plot_background=plot_background_colour, show_frame=draw_frame
                                      , normalisation=cs, quality=figure_quality, colour_min_max=scaling_limits)

            # For each bird plot the threats intensity over Aus (Plot each bird in separate figure)
            if plot_separate_birds is True:
                for i, b in enumerate(bird_labels):

                    bird_name = bird_labels[i]
                    footprint_set = footprints_by_subregion[i]

                    # Set colour scaling to use same min-max for all figures
                    if common_colour_scaling:
                        scaling_limits = (limits_of_product_sets_by_bird[product][bird_name]['min'] * 0.95,
                                          limits_of_product_sets_by_bird[product][bird_name]['max'] * 1.05)
                    else:
                        scaling_limits = None

                    # Check there is a nonzero footprint for this bird
                    if sum(footprint_set) != 0:

                        intensity_by_sa2 = np.zeros(n_sa2s)  # Make data structure to plot

                        # Link each sub-national footprint value with member SA2s
                        for j, val in enumerate(subnat_region_legend):
                            parent_region = int(subnat_region_legend[j])
                            intensity_by_sa2[j] = footprint_set[parent_region - 1]

                        assert not is_empty(intensity_by_sa2)
                        assert sum(intensity_by_sa2) > 0

                        # Send to plotter
                        save_fname = make_save_name(save_dir, bird_name, field_name, colour_palette, cs,
                                                    colour_option=common_colour_scaling)

                        colour_polygons_by_vector(intensity_by_sa2, all_shapes, regions_to_plot, save_fname
                                                  , bounding_box=aus_bounding_box
                                                  , colour_map=colour_palette, quality=figure_quality
                                                  , plot_background=plot_background_colour
                                                  , show_frame=draw_frame, normalisation=cs
                                                  , colour_min_max=scaling_limits)

            if plot_subregion_fd is True and product == 'aus_agri_products':

                # Set colour scaling to use same min-max for all figures
                # Scaling is common between birds, since multiple birds are shown together on different figures
                if common_colour_scaling:
                    scaling_limits = (limits_of_product_sets_by_bird_br_segs['min'] * 0.95,
                                      limits_of_product_sets_by_bird_br_segs['max'] * 1.05)
                else:
                    scaling_limits = None

                for k in base_regions:

                    if (fd_seg == 'USA' and k in [1, 3, 13, 14, 15, 16, 17, 18, 19, 20, 21]) \
                       or ((fd_seg == 'JPN' or fd_seg == 'DEU') and k in [20, 21]):

                        # Unpack footprints from store
                        field_name = 'footprints_fdseg_' + fd_seg + '_' + product + '_' + str(int(k))
                        footprints_by_subregion_and_br = list(f[field_name])

                        # Fix orientation of footprints data
                        footprints_by_subregion_and_br = np.transpose(footprints_by_subregion_and_br)

                        assert max(subnat_region_legend) == footprints_by_subregion_and_br[0].shape
                        assert len(footprints_by_subregion_and_br) == len(bird_labels)

                        for i, b in enumerate(bird_labels):

                            bird_name = bird_labels[i]
                            footprint_set = footprints_by_subregion_and_br[i]

                            # Check there is a nonzero footprint for this bird
                            if sum(footprint_set) != 0:

                                intensity_by_sa2 = np.zeros(n_sa2s)  # Make data structure to plot

                                # Link each sub-national footprint value with member SA2s
                                for j, val in enumerate(subnat_region_legend):
                                    parent_region = int(subnat_region_legend[j])
                                    intensity_by_sa2[j] = footprint_set[parent_region - 1]

                                assert not is_empty(intensity_by_sa2)
                                assert sum(intensity_by_sa2) > 0

                                # Send to plotter
                                save_fname = make_save_name(save_dir, bird_name, field_name, colour_palette, cs,
                                                            colour_option=common_colour_scaling)

                                colour_polygons_by_vector(intensity_by_sa2, all_shapes, regions_to_plot, save_fname
                                                          , bounding_box=aus_bounding_box
                                                          , colour_map=colour_palette, quality=figure_quality
                                                          , plot_background=plot_background_colour
                                                          , show_frame=draw_frame, normalisation=cs
                                                          , colour_min_max=scaling_limits)

        if plot_custom_region_groups and product == 'aus_agri_products':

            # Set colour scaling to use same min-max for all figures
            # Scaling is common between birds, since multiple birds are shown together but in different figures
            if common_colour_scaling:
                scaling_limits = (limits_of_product_sets_by_bird_br_segs['min'] * 0.95,
                                  limits_of_product_sets_by_bird_br_segs['max'] * 1.05)
            else:
                scaling_limits = None

            for i, b in enumerate(bird_labels):

                bird_name = bird_labels[i]

                for z, base_regions_subset in enumerate(custom_regions):

                    #  Store to collect footprints for all regions in set, for this bird and fd segment
                    intensity_by_sa2 = np.zeros(n_sa2s)

                    for k in base_regions_subset:

                        # Unpack footprints from store
                        field_name = 'footprints_fdseg' + fd_seg + '_' + product + '_' + str(int(k)) \
                                     + '_' + stressors
                        footprints_by_subregion_and_br = list(f[field_name])

                        # Fix orientation of footprints data
                        footprints_by_subregion_and_br = np.transpose(footprints_by_subregion_and_br)

                        assert max(subnat_region_legend) == footprints_by_subregion_and_br[0].shape
                        assert len(footprints_by_subregion_and_br) == len(bird_labels)

                        intensity_value = footprints_by_subregion_and_br[i][k-1]

                        # Check there is a nonzero footprint for this bird
                        if intensity_value > 0:

                            # Link each sub-national footprint value with member SA2s
                            n_matches = len([p for p in subnat_region_legend if p == k])
                            for j, val in enumerate(subnat_region_legend):

                                parent_region = int(subnat_region_legend[j])

                                if parent_region == k:
                                    if divide_intensity_among_sa2s is False:
                                        intensity_by_sa2[j] = intensity_by_sa2[j] + intensity_value
                                    else:
                                        intensity_by_sa2[j] = intensity_by_sa2[j] + intensity_value / n_matches

                            assert not is_empty(intensity_by_sa2)
                            assert sum(intensity_by_sa2) > 0

                    if max(intensity_by_sa2) != min(intensity_by_sa2):

                        if results_offset != 1:
                            intensity_by_sa2 = intensity_by_sa2 * results_offset

                        if max(intensity_by_sa2) > scaling_limits[1]:
                            print('Check scaling!')

                        # Send to plotter
                        save_fname = make_save_name(save_dir, bird_name + '_' + custom_region_labels[z],
                                                    field_name, colour_palette, cs,
                                                    colour_option=common_colour_scaling)

                        colour_polygons_by_vector(intensity_by_sa2, all_shapes, regions_to_plot, save_fname
                                                  , bounding_box=aus_bounding_box
                                                  , colour_map=colour_palette, quality=figure_quality
                                                  , plot_background=plot_background_colour
                                                  , show_frame=draw_frame, normalisation=cs
                                                  , colour_min_max=scaling_limits)

        if plot_custom_bird_groups and product == 'aus_agri_products':

            # Set colour scaling to use same min-max for all figures
            # Scaling is common between birds, since multiple birds are shown together but in different figures
            if common_colour_scaling:
                scaling_limits = (limits_of_product_sets_by_bird_br_segs['min'] * 0.95,
                                  limits_of_product_sets_by_bird_br_segs['max'] * 1.05)
            else:
                scaling_limits = None

            for z, base_regions_subset in enumerate(custom_regions):  # Each grouping of base regions

                #  Store to collect footprints for all regions in set, for this bird and fd segment
                intensity_by_sa2 = np.zeros(n_sa2s)

                for k in base_regions_subset:  # Each base region in the set

                    # Unpack footprints from store
                    field_name = 'footprints_fdseg' + fd_seg + '_' + product + '_' + str(int(k)) \
                                 + '_' + stressors
                    footprints_by_subregion_and_br = list(f[field_name])

                    # Fix orientation of footprints data
                    footprints_by_subregion_and_br = np.transpose(footprints_by_subregion_and_br)

                    assert max(subnat_region_legend) == footprints_by_subregion_and_br[0].shape
                    assert len(footprints_by_subregion_and_br) == len(bird_labels)

                    for i in custom_bird_subset:

                        intensity_value = footprints_by_subregion_and_br[i][k-1]
                        if intensity_value > 0:

                            # Link each sub-national footprint value with member SA2s
                            n_matches = len([p for p in subnat_region_legend if p == k])
                            for j, val in enumerate(subnat_region_legend):

                                parent_region = int(subnat_region_legend[j])

                                if parent_region == k:
                                    if divide_intensity_among_sa2s is False:
                                        intensity_by_sa2[j] = intensity_by_sa2[j] + intensity_value
                                    else:
                                        intensity_by_sa2[j] = intensity_by_sa2[j] + intensity_value / n_matches

                            assert not is_empty(intensity_by_sa2)
                            assert sum(intensity_by_sa2) > 0

                if max(intensity_by_sa2) != min(intensity_by_sa2):

                    if results_offset != 1:
                        intensity_by_sa2 = intensity_by_sa2 * results_offset

                    if max(intensity_by_sa2) > scaling_limits[1]:
                        print('Check scaling!')

                    # Send to plotter
                    save_fname = make_save_name(save_dir, 'custom_birds_' + custom_region_labels[z],
                                                field_name, colour_palette, cs,
                                                colour_option=common_colour_scaling)

                    colour_polygons_by_vector(intensity_by_sa2, all_shapes, regions_to_plot, save_fname
                                              , bounding_box=aus_bounding_box
                                              , colour_map=colour_palette, quality=figure_quality
                                              , plot_background=plot_background_colour
                                              , show_frame=draw_frame, normalisation=cs
                                              , colour_min_max=scaling_limits)

# Plot the threats intensity over Aus for each SA2 (All birds same figure)
if plot_all_birds_at_sa2 is True:
    print('Plotting footprints at SA2')

    for fd_seg in fd_segments:
        for product in fd_products:
            for polygon_smearing in smearing_opts:

                # Unpack footprints from store
                field_name = 'footprints_fdseg' + fd_seg + '_' + product + '_' + stressors + '_sa2'
                footprints_by_subregion = list(f[field_name])

                # Fix orientation of footprints data
                footprints_by_subregion = np.transpose(footprints_by_subregion)
                assert len(subnat_region_legend) == footprints_by_subregion[0].shape[0]
                assert len(footprints_by_subregion) == len(bird_labels)

                # Colour scaling can be set to use same min-max for all figures
                if common_colour_scaling:
                    if force_scaling is not None:
                        scaling_limits = (limits_of_product_sets[force_scaling]['min'] * 0.95,
                                          limits_of_product_sets[force_scaling]['max'] * 1.05)
                    else:
                        scaling_limits = (limits_of_product_sets[product]['min'] * 0.95,
                                          limits_of_product_sets[product]['max'] * 1.05)
                else:
                    scaling_limits = None

                intensity_by_sa2 = np.zeros(n_sa2s)  # Make data structure to plot

                # Get the footprints for each bird
                for i, b in enumerate(bird_labels):
                    intensity_by_sa2 = intensity_by_sa2 + footprints_by_subregion[i]

                assert not is_empty(intensity_by_sa2)
                assert sum(intensity_by_sa2) > 0

                if polygon_smearing:
                    intensity_by_sa2 = smear_adjacent_values(intensity_by_sa2, adjacency_matrix, mixing=0.33)

                # Send to plotter
                description = 'all_birds_sa2'
                save_fname = make_save_name(save_dir, description, field_name, colour_palette, cs,
                                            colour_option=common_colour_scaling, smear_option=polygon_smearing)

                colour_polygons_by_vector(intensity_by_sa2, all_shapes, regions_to_plot, save_fname,
                                          bounding_box=aus_bounding_box, colour_map=colour_palette,
                                          plot_background=plot_background_colour, show_frame=draw_frame
                                          , normalisation=cs, quality=figure_quality
                                          , colour_min_max=scaling_limits)

if plot_global_mrio_footprints is True:

    print('Plotting footprints from global MRIO at SA2')

    for fd_seg in fd_segments:
        for product in fd_products:
            for polygon_smearing in smearing_opts:

                # Unpack footprints from store
                field_name = 'footprints_global_mrio_' + fd_seg + '_' + product + '_' + stressors + '_sa2'
                footprints_by_subregion = list(f[field_name])

                # Fix orientation of footprints data
                footprints_by_subregion = np.transpose(footprints_by_subregion)
                assert len(subnat_region_legend) == footprints_by_subregion[0].shape[0]
                assert len(footprints_by_subregion) == len(bird_labels)

                # Colour scaling can be set to use same min-max for all figures
                if common_colour_scaling:
                    if force_scaling is not None:
                        scaling_limits = (limits_of_product_sets[force_scaling]['min'] * 0.95,
                                          limits_of_product_sets[force_scaling]['max'] * 1.05)
                    else:
                        scaling_limits = (limits_of_product_sets[product]['min'] * 0.95,
                                          limits_of_product_sets[product]['max'] * 1.05)
                else:
                    scaling_limits = None

                intensity_by_sa2 = np.zeros(n_sa2s)  # Make data structure to plot

                # Get the footprints for each bird
                for i, b in enumerate(bird_labels):
                    intensity_by_sa2 = intensity_by_sa2 + footprints_by_subregion[i]

                assert not is_empty(intensity_by_sa2)
                assert sum(intensity_by_sa2) > 0

                if polygon_smearing:
                    intensity_by_sa2 = smear_adjacent_values(intensity_by_sa2, adjacency_matrix, mixing=0.25
                                                             , maintain_sum=True)

                # Send to plotter
                description = 'all_birds_sa2'
                save_fname = make_save_name(save_dir, description, field_name, colour_palette, cs,
                                            colour_option=common_colour_scaling, smear_option=polygon_smearing)

                colour_polygons_by_vector(intensity_by_sa2, all_shapes, regions_to_plot, save_fname
                                          , bounding_box=aus_bounding_box, colour_map=colour_palette
                                          , plot_background=plot_background_colour, show_frame=draw_frame
                                          , normalisation=cs, quality=figure_quality
                                          , colour_min_max=scaling_limits)

print('.')
print('Finished')
