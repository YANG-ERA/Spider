#import squidpy as sq
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import squidpy as sq
import matplotlib.pyplot as plt
import scipy
import random 
import matplotlib
import seaborn as sns
import math
#import time
from scipy.special import softmax
from scipy.optimize import minimize
from numpy.random import uniform
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import plotly.graph_objects as go
from matplotlib import cm, colors


import numba
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


def get_ct_sample(Num_celltype=None, Num_sample=None, prior=None):
    if prior is None:
        prior = np.ones(Num_celltype) / Num_celltype
    else:
        assert len(
            prior) == Num_celltype, "please check your prior length, that must be equal to the number of celltype"
        assert sum(prior) <= 1, "Sum of prior is out of range!"
        assert 1 - sum(prior) < 1e-3, "Sum of prior is equal to 1"

    Num_ct_sample = np.zeros(Num_celltype, dtype=np.int32)
    Num_ct_sample[0:(Num_celltype - 1)] = (
                prior[0:(Num_celltype - 1)] * Num_sample + 0.5 * np.ones(Num_celltype - 1)).astype(int)
    Num_ct_sample[Num_celltype - 1] = Num_sample - np.sum(Num_ct_sample[0:(Num_celltype - 1)])

    # Check for any 0 elements in Num_ct_sample
    zero_elements = np.where(Num_ct_sample == 0)[0]
    if zero_elements.size > 0:
        # Find the index of the largest element in Num_ct_sample
        max_index = np.argmax(Num_ct_sample)
        # Distribute one sample from the largest element to each of the 0 elements
        Num_ct_sample[zero_elements] = 1
        Num_ct_sample[max_index] -= len(zero_elements)
    return Num_ct_sample

def addictive_freq(n_c):
    target_freq = np.ones((n_c,n_c))
    for i in np.arange(n_c):
        target_freq[i,i] = 4*(n_c-1)
    
    return target_freq/np.sum(target_freq,axis=1,keepdims=True)

def exclusive_freq(n_c):
    target_freq = np.ones((n_c,n_c))
    for i in np.arange(n_c):
        target_freq[i,i] = 3*(n_c-1)
        if i%2 == 1:
            target_freq[i-1,i] = 3*(n_c-1)
            target_freq[i,i-1] = 3*(n_c-1)
    
    return target_freq/np.sum(target_freq,axis=1,keepdims=True)

def stripe_freq(n_c):
    target_freq = np.ones((n_c,n_c))
    for i in np.arange(n_c):
        target_freq[i,i] = 3*(n_c-1)
        if i>0:
            target_freq[i-1,i] = 3*(n_c-1)
    
    return target_freq/np.sum(target_freq,axis=1,keepdims=True)

## 把squidpy的参数全部拿过来

def get_spaital_network(Num_sample = None, spatial = None,n_neighs = None,radius = None,coord_type = "grid",n_rings = 2,set_diag = False):
    spatial_adata = ad.AnnData(np.empty((Num_sample,1),dtype="float32"))
    spatial_adata.obsm["spatial"] = spatial
    sq.gr.spatial_neighbors(spatial_adata, n_rings=n_rings, coord_type=coord_type, n_neighs=n_neighs, radius=radius,set_diag =set_diag)
    sn = spatial_adata.obsp["spatial_connectivities"]
    
    return sn

def init_ct(Num_celltype = None,Num_ct_sample = None, seed = None):
    if seed is not None:
        np.random.seed(seed)
    init_assign = np.repeat(np.arange(Num_celltype), Num_ct_sample)
    np.random.shuffle(init_assign)
    return init_assign

# def get_onehot_ct(init_assign = None):
#     label_encoder = LabelEncoder()
#     integer_encoded = label_encoder.fit_transform(init_assign)
#     onehot_encoder = OneHotEncoder(sparse_output=False)
#     integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#     onehot_ct = onehot_encoder.fit_transform(integer_encoded)
#     return onehot_ct.astype(np.float32)

def get_trans(adata = None, ct = None,n_neighs=8):
    sn = get_spaital_network(Num_sample=adata.obs.shape[0],
                         spatial=adata.obsm["spatial"], coord_type = "generic",
                         n_neighs=n_neighs)
    onehot_ct = get_onehot_ct(init_assign=ct)
    nb_count = np.array(sn * onehot_ct, dtype=np.float32)
    target_trans = get_nb_freq(nb_count=nb_count, onehot_ct=onehot_ct)
    return target_trans

@numba.jit("float32[:, ::1](float32[:, ::1], float32[:, ::1])")
def get_nb_freq( nb_count = None, onehot_ct = None):
#     nb_freq = onehot_ct.T @ nb_count
    nb_freq = np.dot(onehot_ct.T, nb_count)
    res = nb_freq/nb_freq.sum(axis = 1).reshape(onehot_ct.shape[1],-1)
    return res


def swap_ct(celltype_assignment = None, Num_celltype = None,swap_num = None):
    swap_cluster = np.random.choice(np.arange(Num_celltype),2,replace = False)
    swap_i_index = np.random.choice(np.where(celltype_assignment==swap_cluster[0])[0],swap_num,replace = False)
    swap_j_index = np.random.choice(np.where(celltype_assignment==swap_cluster[1])[0],swap_num,replace = False)
    
    return (swap_i_index,swap_cluster[0]), (swap_j_index,swap_cluster[1])
    
def get_swap_nb_count(  nb_count = None,swap_i = None, swap_j = None,sn = None):
    
    swap_i_nb_index = sn[swap_i[0]].indices
    nb_count[swap_i_nb_index,swap_i[1]] -= 1
    nb_count[swap_i_nb_index,swap_j[1]] += 1
    
    swap_j_nb_index = sn[swap_j[0]].indices
    nb_count[swap_j_nb_index,swap_j[1]] -= 1
    nb_count[swap_j_nb_index,swap_i[1]] += 1
    
    return nb_count

def plot_3d_cell_types(cell_locations, cell_labels, 
                       title="3D Cell Type Visualization",
                       marker_size=5,
                       opacity=1,
                       interactive=True,
                       save_path=None,
                       auto_open=True):
    """
    3D visualization of cell type distribution using the same color mapping as 2D slices
    
    Parameters:
    ----------
    cell_locations : ndarray (N x 3)
        Array of 3D cell coordinates
    cell_labels : array-like (N,)
        Cell type labels (numeric or string)
    title : str
        Plot title
    marker_size : int
        Point size (default increased to 8)
    opacity : float
        Transparency (0-1)
    interactive : bool
        Whether to generate interactive plot using plotly
    save_path : str, optional
        Path to save HTML file (if None, won't save)
    auto_open : bool
        Whether to automatically open the saved HTML file
    
    Returns:
    -------
    matplotlib Figure or plotly Figure object
    """
    unique_labels = np.unique(cell_labels)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if interactive:
        fig = go.Figure()
        
        for i, label in enumerate(unique_labels):
            mask = cell_labels == label
            color_idx = i % len(default_colors)
            color_rgb = colors.to_rgb(default_colors[color_idx])
            color_hex = colors.rgb2hex(color_rgb)
            
            fig.add_trace(go.Scatter3d(
                x=cell_locations[mask, 0],
                y=cell_locations[mask, 1],
                z=cell_locations[mask, 2],
                mode='markers',
                name=str(label),
                marker=dict(
                    size=marker_size,
                    color=color_hex,
                    opacity=opacity
                ),
                hovertext=f"Type: {label}"
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            legend_title="Cell Types"
        )
        
        # Save HTML if path is provided
        if save_path is not None:
            fig.write_html(
                save_path,
                include_plotlyjs='cdn',
                full_html=True,
                auto_open=auto_open
            )
        
        return fig
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, label in enumerate(unique_labels):
            mask = cell_labels == label
            color_idx = i % len(default_colors)
            ax.scatter(
                cell_locations[mask, 0],
                cell_locations[mask, 1],
                cell_locations[mask, 2],
                color=default_colors[color_idx],
                s=marker_size,
                alpha=opacity,
                label=str(label)
            )
        
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(title="Cell Types", bbox_to_anchor=(1.05, 1))
        
        return fig

import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
import os

def slice_anndata_by_z(adata, z_key='spatial', z_bins=10):
    """
    Split h5ad file based on Z-axis coordinates
    
    Parameters:
    ----------
    adata : AnnData
        Original spatial transcriptomics data
    z_key : str
        Key name in obsm containing 3D coordinates (default: 'spatial')
    z_bins : int or array-like
        Number of slices or custom Z-axis boundaries
    
    Returns:
    -------
    list of AnnData
        List of sliced AnnData objects
    """
    # Check if 3D coordinates exist
    if z_key not in adata.obsm:
        raise ValueError(f"3D coordinates not found in obsm['{z_key}']")
    
    # Get Z-axis coordinates
    z_coords = adata.obsm[z_key][:, 2]
    
    # Determine slice boundaries
    if isinstance(z_bins, int):
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        bin_edges = np.linspace(z_min, z_max, z_bins+1)
    else:
        bin_edges = np.asarray(z_bins)
    
    sliced_adatas = []
    for i in range(len(bin_edges)-1):
        z_low, z_high = bin_edges[i], bin_edges[i+1]
        
        # Get cell indices for current slice
        cell_idx = (z_coords >= z_low) & (z_coords < z_high)
        
        # Create subset AnnData
        adata_slice = adata[cell_idx, :].copy()
        
        # Add slice metadata
        adata_slice.uns['slice_info'] = {
            'slice_id': i,
            'z_range': (float(z_low), float(z_high)),
            'n_cells': adata_slice.n_obs
        }
        adata_slice.obs["x"] = adata_slice.obsm["spatial"][:, 0]
        adata_slice.obs["y"] = adata_slice.obsm["spatial"][:, 1]
        
        sliced_adatas.append(adata_slice)
        print(f"Created slice {i} (Z={z_low:.1f}-{z_high:.1f}) with {adata_slice.n_obs} cells")
    
    # Create summary information
    summary = {
        'n_slices': len(sliced_adatas),
        'z_ranges': [adata.uns['slice_info']['z_range'] for adata in sliced_adatas],
        'n_cells': [adata.n_obs for adata in sliced_adatas]
    }
    
    return sliced_adatas

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def plot_z_slices(slices, output_path=None, figsize_per_slice=5, dpi=300):
    """
    Visualize multiple z-slices with cell type information
    
    Parameters:
    ----------
    slices : list of AnnData
        List of sliced AnnData objects (output from slice_anndata_by_z)
    output_path : str, optional
        Path to save the figure (if None, won't save)
    figsize_per_slice : int
        Base size for each subplot (default: 5)
    dpi : int
        Figure resolution (default: 300)
    
    Returns:
    -------
    matplotlib Figure object
    """
    # Collect all unique cell types across slices
    all_celltypes = []
    for adata_slice in slices:
        all_celltypes.extend(adata_slice.obs['celltype'].unique())
    unique_celltypes = sorted(set(all_celltypes))
    
    # Create color mapping using default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {ct: color for ct, color in zip(unique_celltypes, colors)}
    
    # Create figure with appropriate size
    n_slices = len(slices)
    fig, axes = plt.subplots(1, n_slices, 
                           figsize=(figsize_per_slice*n_slices, figsize_per_slice), 
                           dpi=dpi)
    
    # If only one slice, make axes iterable
    if n_slices == 1:
        axes = [axes]
    
    # Plot each slice
    for i, (adata_slice, ax) in enumerate(zip(slices, axes)):
        # Get spatial coordinates or use random if missing
        if 'spatial' in adata_slice.obsm:
            spatial_data = adata_slice.obsm['spatial']
            x = spatial_data[:, 0]
            y = spatial_data[:, 1]
        else:
            print(f"Warning: Slice {i} missing spatial coordinates. Using pseudo-coordinates.")
            x = np.random.rand(adata_slice.n_obs) * 10
            y = np.random.rand(adata_slice.n_obs) * 10
        
        celltypes = adata_slice.obs['celltype']
        
        # Plot each cell type
        for ct in np.unique(celltypes):
            mask = celltypes == ct
            color = color_map[ct] if ct in color_map else 'gray'
            
            ax.scatter(
                x[mask], y[mask],
                s=50,
                label=ct,
                c=[color] * np.sum(mask),
                alpha=1
            )
        
        # Set title and labels
        z_range = adata_slice.uns.get('slice_info', {}).get('z_range', [0, 0])
        ax.set_title(f"Slice {i}\n(Z={z_range[0]:.1f}-{z_range[1]:.1f})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Create unified legend
    handles, labels = [], []
    for ct in unique_celltypes:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color_map[ct], markersize=10))
        labels.append(ct)
    
    axes[-1].legend(handles, labels,
                   bbox_to_anchor=(1.05, 1),
                   title='Cell Types',
                   fontsize=8)
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, 
                   format='pdf', 
                   bbox_inches='tight',
                   dpi=dpi,
                   transparent=False)
    
    plt.show()
    return fig



def get_onehot_ct(init_assign = None):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(init_assign)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_ct = onehot_encoder.fit_transform(integer_encoded)
    return onehot_ct.astype(np.float32)

def move_list(lst, k):
    ## k>0 left move
    ## k<0 right move
    return lst[k:]+lst[:k]

def layer_cell_level_sim(Num_sample = None, Num_celltype = 4,
                   image_width = None, image_height = None,
                  scenarios_mode = None):

    full_sample = image_width * image_height
    a, b = np.meshgrid(np.arange(image_width),
                       np.arange(image_height),
                       indexing='ij')
    ## Num_celltype is equal to Num_layer
    layer_label = np.repeat(np.arange(Num_celltype),full_sample/Num_celltype)  
    ct_label_init = np.zeros((full_sample))
    sc_loc = np.vstack((a.reshape(-1), b.reshape(-1),layer_label,ct_label_init)).T
    rand_index = np.random.choice(np.arange(full_sample),Num_sample,replace = False)
   
    sim_sc_loc = sc_loc[rand_index]
    
   
    for i in range(Num_celltype):
        sc_layer_idx = np.where(sim_sc_loc[:, 2] == i)[0]
        sim_sc_loc[sc_layer_idx, 3] = np.argmax(
            np.random.multinomial(n=1,
                                  pvals=move_list(scenarios_mode, k=-i),
                                  size=len(sc_layer_idx)), 1)  
        
    return sim_sc_loc
    
def layer_spot_level_sim(spot_diameter=None, image_width=None, image_height=None,
                         sim_sc_loc=None, merge=False, gap=0):
    grid_row = int(image_width / spot_diameter)
    grid_col = int(image_height / spot_diameter)
    grid_num = grid_row * grid_col
    points_x_indices = (sim_sc_loc[:, 0] / spot_diameter).astype(int)
    points_y_indices = (sim_sc_loc[:, 1] / spot_diameter).astype(int)
    
    # Spot ID for each cell
    cell_spot_id = (points_x_indices * grid_col + points_y_indices).astype(int)
    # Cell ID
    cell_id = np.arange(cell_spot_id.size)
    
    # Replace np.where, for each grid obtain cell indices for each spot (rows: spots, columns: cell_id)
    spot_cell_idx_matrix = csr_matrix(
        (np.ones(cell_spot_id.size), (cell_spot_id, cell_id)),
        shape=(grid_num, cell_spot_id.size))
    
    # One-hot encoding for cell layer annotations
    cell_layer_onehot = get_onehot_ct(sim_sc_loc[:, 2])
    
    # Should pay attention to these spots which do not contain cells
    # Get cell type counts for each spot
    spot_layer = spot_cell_idx_matrix * cell_layer_onehot
    
    # Extract spots without masked cells
    mask = np.all(spot_layer == 0, axis=1)
    
    # Note: there exist spots with all zeros
    spot_layer = np.argmax(spot_layer, axis=1)
    
    # Take center points of grid as spot coordinates
    a, b = np.meshgrid(np.arange(grid_row),
                   np.arange(grid_col),
                   indexing='ij')
    spot_pixel_row = (a.reshape(-1) + 0.5) * spot_diameter
    spot_pixel_col = (b.reshape(-1) + 0.5) * spot_diameter
    
    spot_loc = np.vstack((spot_pixel_row, spot_pixel_col, spot_layer)).T
    subspot_loc = spot_loc[~mask]  # Remove spots without cells (author considers these grids as subspots)
                                   # If needing something like 10X ST, requires merging 9 grids

    if merge is True:
        windows_row = 3
        windows_col = 3
        gap = gap

        # Edge cases don't have 3 * 3 grids, should we set gap?
        spot_row_num = int(grid_row / (windows_row + gap))
        spot_col_num = int(grid_col / (windows_col + gap))

        # lra and lrb are center grids of 9 grids, actually coordinates of bottom-left point of center grid
        lra = a[np.arange(1, spot_row_num * (windows_row + gap), (windows_row + gap)
                  ), :len(np.arange(1, spot_col_num * (windows_col + gap), (windows_col + gap)))].reshape(-1)
        lrb = b[:len(np.arange(1, spot_row_num * (windows_row + gap), (windows_row + gap))),
            np.arange(1, spot_col_num * (windows_col + gap), (windows_col + gap))].reshape(-1)   
        
        # Center grid indices
        lrspot_layer_idx = lra * grid_col + lrb   
        
        # Get neighbors of center points, i.e., 9 grids including itself
        sn = get_spaital_network(Num_sample=spot_loc.shape[0], spatial=spot_loc[:, :2],
                         n_neighs=8, coord_type="grid", n_rings=1, set_diag=True)
        
        spot_layer_onehot = get_onehot_ct(spot_loc[:, 2])
        
        # Get the most frequent layer label among the 9 merged spot layers
        merged_spot_layer = np.argmax(sn[lrspot_layer_idx] * spot_layer_onehot, axis=1)
        merged_spot_pixel_row = (lra.reshape(-1) + 0.5) * spot_diameter
        merged_spot_pixel_col = (lrb.reshape(-1) + 0.5) * spot_diameter
        merged_spot_loc = np.vstack((merged_spot_pixel_row, 
                                   merged_spot_pixel_col,
                                   merged_spot_layer)).T
        return subspot_loc, merged_spot_loc
    else:
        return subspot_loc 

def muti_circle(row=10, col=10, Num_celltype = 4):
    layer_label = np.zeros(row*col,dtype = np.int32)
    #n_spots = row*col
    layer_width_a = round(row/(2*Num_celltype))
    layer_width_b = round(col/(2*Num_celltype))
    a, b = np.meshgrid(range(row),
                    range(col),
                    indexing='ij')
    a = a.reshape(-1)
    b = b.reshape(-1)
    for i in range(1,Num_celltype):
        layer_index = np.intersect1d(np.where((a>=(layer_width_a*i)) & (a<(row-(layer_width_a*i))))[0],np.where((b>=(layer_width_b*i)) & (b<(col-(layer_width_b*i))))[0])
        layer_label[layer_index] = i
        
    return a, b, layer_label     


def muti_square(row=10, col=10, Num_celltype = 4):
    layer_label = np.zeros(row*col,dtype = np.int32)
    #n_spots = row*col
    bound_a = round(row/2)
    bound_b = round(col/2)
    a, b = np.meshgrid(range(row),
                    range(col),
                    indexing='ij')
    a = a.reshape(-1)
    b = b.reshape(-1)
    layer_index1 = np.intersect1d(np.where((a>=bound_a))[0],np.where((b< bound_b))[0])
    layer_index2 = np.intersect1d(np.where((a< bound_a))[0],np.where((b>=bound_b))[0])
    layer_index3 = np.intersect1d(np.where((a>=bound_a))[0],np.where((b>=bound_b))[0])
    layer_label[layer_index1] = 1
    layer_label[layer_index2] = 2
    layer_label[layer_index3] = 3
        
    return a, b, layer_label

def naive_cell_level_sim(Num_sample = None, Num_celltype = 4,
                image_width = None, image_height = None,
                scenarios_mode =None,method = "SpatialPCA"):
    '''
    method ={ 'SpatialPCA','muti_circle','muti_square'}
    '''
    full_sample = image_width * image_height
    a, b = np.meshgrid(np.arange(image_width),
                        np.arange(image_height),
                        indexing='ij')     
    if method == 'SpatialPCA':

        ## Num_celltype is equal to Num_layer
        layer_label = np.repeat(np.arange(Num_celltype),full_sample/Num_celltype)
        
    if method == 'muti_circle':
        _, _, layer_label = muti_circle(row=image_width, col=image_height, Num_celltype = Num_celltype)
    if method == 'muti_square':
        _, _, layer_label = muti_square(row=image_width, col=image_height, Num_celltype = Num_celltype)
    
    
    ct_label_init = np.zeros((full_sample))
    sc_loc = np.vstack((a.reshape(-1), b.reshape(-1),layer_label,ct_label_init)).T
    rand_index = np.random.choice(np.arange(full_sample),Num_sample,replace = False)
   
    sim_sc_loc = sc_loc[rand_index]
    
    
    for i in range(Num_celltype):
        sc_layer_idx = np.where(sim_sc_loc[:, 2] == i)[0]
        sim_sc_loc[sc_layer_idx, 3] = np.argmax(
                                np.random.multinomial(n=1,
                                  pvals=move_list(scenarios_mode, k=-i),
                                  size=len(sc_layer_idx)), 1)  
        
    return sim_sc_loc    

