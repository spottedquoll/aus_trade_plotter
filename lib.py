import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
import numpy as np
from numpy import genfromtxt
from utils import clean_string, is_empty
import pandas


class MplColorHelper:

    def __init__(self, cmap_name, min_val, max_val, normalisation='linear', discrete_bins=None):

        self.cmap_name = cmap_name

        if discrete_bins is None:
            self.cmap = plt.get_cmap(cmap_name)
        else:
            self.cmap = plt.get_cmap(cmap_name, discrete_bins)

        if normalisation is not None:
            if normalisation is 'linear':
                self.norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
            elif normalisation is 'log':
                self.norm = mpl.colors.LogNorm(vmin=min_val, vmax=max_val)
            elif normalisation is 'symlog':
                self.norm = mpl.colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=min_val, vmax=max_val)
            else:
                raise ValueError('Unknown normalisation method')

            self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        else:
            self.scalarMap = cm.ScalarMappable(norm=None, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def colour_polygons_by_vector(colour_scale_data, all_shapes, sub_regions, save_file_name, bounding_box=None
                              , normalisation='linear', colour_map='plasma', attach_colorbar=False, discrete_bins=None
                              , colour_min_max=None, polygon_edge='none', plot_background=None, show_frame=True
                              , quality=900):

    # Determine colour scaling from data or use exogenous data
    if colour_min_max is None:
        min_val = np.min(colour_scale_data)
        max_val = np.max(colour_scale_data)
    else:
        min_val = colour_min_max[0]
        max_val = colour_min_max[1]

    colour_scaling = MplColorHelper(colour_map, min_val, max_val, normalisation=normalisation
                                    , discrete_bins=discrete_bins)

    if len(colour_scale_data) != len(sub_regions):
        raise ValueError('Data dim does not match number of polygons.')

    count = 0

    # Set up plot
    fig = plt.figure(facecolor=plot_background)
    ax = plt.axes()
    ax.set_aspect('equal')

    # Read and plot all the shape files
    for index in sub_regions:

        shape = all_shapes[int(index) - 1]
        polygon_parts = len(shape.parts)

        try:
            colour_rgb = colour_scaling.get_rgb(colour_scale_data[count])
        except:
            raise ValueError('Could not retrieve RBG values')

        if polygon_parts == 1:
            polygon = Polygon(shape.points)

            try:
                patch = PolygonPatch(polygon, fc=(colour_rgb[0], colour_rgb[1], colour_rgb[2], 0.7), ec="none")
                ax.add_patch(patch)
            except:
                raise ValueError('Could not build patch')

        elif polygon_parts > 1:
            for ip in range(polygon_parts):  # loop over parts, plot separately
                i0 = shape.parts[ip]
                if ip < polygon_parts - 1:
                    i1 = shape.parts[ip + 1] - 1
                else:
                    i1 = len(shape.points)

                polygon = Polygon(shape.points[i0:i1 + 1])

                patch = PolygonPatch(polygon, fc=(colour_rgb[0], colour_rgb[1], colour_rgb[2], 0.7), ec=polygon_edge)
                ax.add_patch(patch)

        count = count + 1

    # Plot limits
    if bounding_box is not None:
        plt.xlim(bounding_box[0], bounding_box[1])
        plt.ylim(bounding_box[2], bounding_box[3])

    # Frame around plot (cannot simply turn axes off, as this interfers with plot background colour
    if show_frame is False:
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

    # Figure background colour
    if plot_background is not None:
        ax.set_facecolor(plot_background)

    if attach_colorbar:
        sm = plt.cm.ScalarMappable(cmap=getattr(plt.cm, colour_map), norm=plt.Normalize(vmin=min_val, vmax=max_val))
        sm._A = []
        plt.colorbar(sm)

    plt.savefig(save_file_name, dpi=quality, bbox_inches='tight')
    plt.clf()
    plt.close("all")


def collate_weights(year, allocations, input_path, trade_direction, port_locations, commodity, results_path, aus_region
                    , port_name=None):

    all_weights = None

    for z in allocations:

        # Get the trade allocation
        trade_data = genfromtxt(input_path + z + '_' + trade_direction + '_' + commodity + '_domestic_flows_'
                                + aus_region + '_' + year + '.csv', delimiter=',')

        # Find the appropriate port locations
        port_locations_pd = pandas.DataFrame(port_locations)
        if port_name is not None:
            matching_ports = port_locations_pd.index[port_locations_pd['Port Name'] == port_name].tolist()

            assert(len(matching_ports) > 0)
            assert (len(matching_ports) <= len(trade_data))

            # Squash the totals by ...?
            a = np.array(trade_data)
            region_totals = None
            for m in matching_ports:
                if region_totals is None:
                    region_totals = a[:, m]
                else:
                    region_totals = region_totals + a[:, m]
        else:
            if trade_data.ndim > 1:
                region_totals = np.sum(trade_data, axis=1)  # sum over the port dimension
            else:
                region_totals = trade_data

        all_weights = pandas.concat([all_weights, pandas.DataFrame(region_totals, columns=[z])], axis=1)

    # Save to xlsx
    if port_name is None:
        port_name = 'allports'
    all_weights.to_excel(results_path + 'collate_weights_' + trade_direction + '_' + port_name + '_' + aus_region
                         + '_' + year + '.xlsx', header=True, index=False)

    return all_weights


def get_port_index(port_name, port_locations):

    port_locations_pd = pandas.DataFrame(port_locations)
    matching_ports = port_locations_pd.index[port_locations_pd['Port Name'] == port_name].tolist()
    if len(matching_ports) == 0:
        raise ValueError('Port could not be found')

    return matching_ports


def make_save_name(save_dir, prefix, field_name, colour_name, normalisation, colour_option=None):

    prefix = clean_string(prefix, [' ', "'"], '_', case='lower')
    field_name = field_name.replace('fdseg', 'fd')
    description = field_name.lower().replace('footprints_', '')

    save_fname = save_dir + '/' + prefix + '_' + description + '_' + colour_name.lower() + '_' + normalisation

    if colour_option is not None and colour_option is True:
        save_fname = save_fname + '_comcol'

    save_fname = save_fname + '.png'
    return save_fname


def make_sa2_adjacency(gp_df, n_sa2s):

    adjacency = []

    for index, row in gp_df.iterrows():
        print(str(index))
        if row['geometry'] is not None:
            neighbors = gp_df[gp_df.geometry.touches(row['geometry'])]
            new_row = np.zeros(n_sa2s)
            new_row[neighbors.index.values] = 1
            adjacency.append(new_row)

    return adjacency


def smear_adjacent_values(values, adjacency_matrix, rounds=None):

    if rounds is None:
        rounds = 1

    assert len(values) == len(adjacency_matrix)
    assert len(values) == len(adjacency_matrix[0])

    values_smeared = np.zeros(len(values))

    for idx, val in enumerate(values):
        neighbors = np.nonzero(adjacency_matrix[idx])
        if is_empty(neighbors):
            values_smeared = values[idx]
        else:
            neighboring_vals = values[neighbors]
            smeared = np.mean(np.append(neighboring_vals, values[idx]))
            values_smeared[idx] = smeared

    return values_smeared
