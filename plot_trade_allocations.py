from spatial import read_shape_files, read_sa2_meta
import h5py
from lib import colour_polygons_by_vector, collate_weights
import pyexcel as pe
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
import itertools
import numpy as np
from quoll.utils import get_local_volume
import os

# Directories
volume = get_local_volume()
work_path = volume + 'more_projects/emily1/Projects/2018_nesting_paper/visualise_allocation/'
asgs_path = work_path + 'ASGS/SA2/2011/'
input_path = work_path + 'allocation_results/'
results_path = work_path + 'finished_figures/'

#    Check save directories exist
if not os.path.isdir(results_path):
    os.mkdir(results_path)

# Get SA2 meta information and shape files
sa2_meta_store = read_sa2_meta(asgs_path + 'SA2_2011_AUST.xlsx')
all_shapes = read_shape_files(asgs_path + 'SA2_2011_AUST.shp')

if len(all_shapes) != len(sa2_meta_store):
    raise ValueError('SA2 meta version is different to the shape file version')

# SA2 member lists
#   QLD
filename = input_path + 'state_sa2_members_qld.h5'
f = h5py.File(filename, 'r')
qld_sa2_members = list(f['sa2_members'])  # Qld
aus_sa2s = list(range(1, 2215))  # Aus

# Port names and locations
port_locations = []
records = pe.iget_records(file_name=str(work_path) + '/ABS/' + 'Australian_port_locations.xlsx')
for row in records:
    port_locations.append(row)

# Figure bounding box [x_min, x_max, y_min, y_max]
aus_bounds = [110, 155, -45, -5]
qld_bounds = [136, 155, -30, -8]

# Options
allocations = ['production', 'gravity', 'combined']
trade_directions = ['exports']  # 'imports',
colour_maps = ['plasma']  # , 'Greys', 'hot', 'viridis', 'BuPu', 'PuRd' 'plasma_r', 'Greys', 'Greys_r',
colour_scalings = ['linear', 'symlog']  # None, 'symlog' # colour normalisation
colour_bar = [True]  # True, False
discrete_colours = [15, None]  # 5, False
sectors = ['beef']  # , 'beef', 'meat'
years = ['2009']  # '2000','2005',
hard_norm = [None, 'minmax', 'log']  #  True, normalise data prior to plotting
common_colour_scale = [False, True]
fix_combined_scaling = True

plot_trade_regions = [{'aus_region': 'qld', 'port': 'Brisbane', 'sa2s': qld_sa2_members, 'bounds': qld_bounds}
                      , {'aus_region': 'aus', 'port': None, 'sa2s': aus_sa2s, 'bounds': aus_bounds}]

iterable = list(itertools.product(years, trade_directions, common_colour_scale, colour_scalings
                                  , colour_maps, colour_bar, discrete_colours, sectors, hard_norm
                                  , plot_trade_regions))

# Plot maps
for item in iterable:

    # Unpack
    y, trade_direction, ccs, cn, clr_map, cb, dc, commodity, hn, trade_regions = (item[0], item[1], item[2]
                                                                                         , item[3], item[4], item[5]
                                                                                         , item[6], item[7], item[8]
                                                                                         , item[9])

    collated_weights = collate_weights(y, allocations, input_path, trade_direction, port_locations, commodity
                                       , results_path, trade_regions['aus_region'], port_name=trade_regions['port'])

    # Add new combined measure
    if fix_combined_scaling:
        collated_weights['combinedavg'] = list(collated_weights[['production', 'gravity']].mean(axis=1))

    columns = list(collated_weights)

    # Data scaling
    if hn is not None:
        if hn is 'minmax':
            scaler = MinMaxScaler()
        elif hn is 'log':
            scaler = FunctionTransformer(np.log10, validate=True)
        else:
            raise ValueError('Unknown scaler requested')
        collated_weights[columns] = scaler.fit_transform(collated_weights[columns].values)

    # Common colour scaling for all figures
    if ccs is True:
        scaling_limits = (collated_weights.min().min()*0.95, collated_weights.max().max()*1.05)
    else:
        scaling_limits = None

    # Check save sub-directories exist
    results_path_subdir = results_path + trade_regions['aus_region'] + '/' + y + '/'
    if not os.path.isdir(results_path_subdir):
        os.mkdir(results_path_subdir)

    # Plot each type of trade allocation
    for c in columns:

        # Make description and filename
        port_name = trade_regions['port']
        if port_name is None:
            port_name = 'allports'
        description = (trade_direction + '_' + trade_regions['aus_region'] + '_' + port_name + '_' + commodity
                       + '_' + c + '_cs' + str(cn) + '_' + clr_map + '_cb' + str(cb) + '_dc' + str(dc)
                       + '_fsc' + str(ccs) + '_hn' + str(hn) + '_' + y).lower()

        save_fname = results_path_subdir + description + '.png'

        print('Plotting: ' + description)
        data = list(collated_weights[c])

        # Send to plotter
        colour_polygons_by_vector(data, all_shapes, trade_regions['sa2s'], save_fname
                                  , bounding_box=trade_regions['bounds'], normalisation=cn, colour_map=clr_map
                                  , attach_colorbar=cb, discrete_bins=dc, colour_min_max=scaling_limits)

print('All finished')
