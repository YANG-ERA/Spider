
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
from .utils import *
from .Annealing import *
from .simulate_10X import *
import numba
from scipy.sparse import csr_matrix

def get_sim_cell_level_expr(celltype_assignment=None, adata=None,
                           Num_celltype=None, Num_ct_sample=None,
                           match_list=None, ct_key=None):
    """
    Generate simulated cell-level expression data by sampling from reference single-cell data.
    
    For each simulated cell, assigns an index from reference single-cell data,
    ensuring cells from the same group sample from the same cell type in the reference.
    
    Parameters:
    ----------
    celltype_assignment : array-like
        Array specifying the cell type assignment for each simulated cell
    adata : AnnData
        Reference single-cell AnnData object containing expression data
    Num_celltype : int
        Number of distinct cell types to simulate
    Num_ct_sample : list[int]
        Number of cells to sample for each cell type
    match_list : list
        List mapping simulated cell type indices to reference cell type labels
    celltype_key : str
        Key in adata.obs containing reference cell type labels
    
    Returns:
    -------
    AnnData
        Simulated cell-level expression data with same variables as input adata
    
    Raises:
    ------
    AssertionError
        If reference adata is not provided
    """
    assert adata is not None, (
        "Please provide a reference AnnData object. "
        "Alternatively, a single-cell data simulation strategy is available as an option."
    )

    idx_list = celltype_assignment.copy()
    for i in range(Num_celltype):
        # Find cells of specified type in reference data
        ct_idx = np.where(adata.obs[ct_key] == match_list[i])[0]
        # Sample cells from reference to match simulation requirements
        sim_cell_idx = np.random.choice(
            ct_idx, 
            size=Num_ct_sample[i], 
            replace=True
        )

        idx_list[celltype_assignment == i] = sim_cell_idx

    sim_cell_expr = adata[idx_list]
    return sim_cell_expr    

def get_sim_spot_level_expr(spot_num=None, spot_row=None, spot_col=None,
                            Num_sample=None, spot_diameter=None, image_width=None,
                            image_height=None, celltype_assignment=None, coord_type=None,
                            cell_spatial=None, sim_cell_expr=None, gap=0,
                            spot_generate_type="square", cell_coord_type="grid",
                            grid_row=None, grid_col=None):
    if spot_generate_type == "square" and cell_coord_type == "generic":
        # Grid partitioning for random points, not grid merging

        grid_row = np.ceil(image_width / spot_diameter).astype(int)
        grid_col = np.ceil(image_height / spot_diameter).astype(int)
        grid_num = grid_row * grid_col
        points_x_indices = (cell_spatial[:, 0] / spot_diameter).astype(int)
        points_y_indices = (cell_spatial[:, 1] / spot_diameter).astype(int)

        # Spot ID for each cell
        cell_spot_id = (points_x_indices * grid_col + points_y_indices).astype(int)
        # Cell ID
        cell_id = np.arange(cell_spot_id.size)

        # Replace np.where, for each grid obtain cell indices for each spot (rows: spots, columns: cell_id)
        spot_cell_idx_matrix = csr_matrix(
            (np.ones(cell_spot_id.size), (cell_spot_id, cell_id)),
            shape=(grid_num, cell_spot_id.size))

        spot_expr = spot_cell_idx_matrix * sim_cell_expr.X

        onehot_ct = get_onehot_ct(init_assign=celltype_assignment)

        # Note: some spots may contain no cells
        spot_ct_count = spot_cell_idx_matrix * onehot_ct

        # Take center points of grid as spot coordinates
        a, b = np.meshgrid(np.arange(grid_row),
                       np.arange(grid_col),
                       indexing='ij')
        spot_pixel_row = (a.reshape(-1) + 0.5) * spot_diameter
        spot_pixel_col = (b.reshape(-1) + 0.5) * spot_diameter

        spot_loc = np.vstack((spot_pixel_row, spot_pixel_col)).T

        return spot_expr, spot_loc, spot_cell_idx_matrix, spot_ct_count
    
    if spot_generate_type == "circle" and cell_coord_type == "generic":
        # Define diameter and gap

        grid_row = int(image_width / gap)
        grid_col = int(image_height / gap)
        grid_num = grid_row * grid_col

        # Take center points of grid as spot coordinates
        a, b = np.meshgrid(np.arange(grid_row),
                       np.arange(grid_col),
                       indexing='ij')
        spot_pixel_row = (a.reshape(-1) + 0.5) * gap
        spot_pixel_col = (b.reshape(-1) + 0.5) * gap

        spot_loc = np.vstack((spot_pixel_row, spot_pixel_col)).T

        cellandspot_loc = np.vstack((cell_location, spot_loc))

        # Get neighbors of center points
        sn = get_spaital_network(Num_sample=cellandspot_loc.shape[0], spatial=cellandspot_loc,
                         coord_type="generic", n_rings=1, set_diag=False,
                                 radius=spot_diameter/2)

        spot_cell_idx_matrix = sn[Num_sample:, Num_sample]

        spot_expr = spot_cell_idx_matrix * sim_cell_expr.X

        onehot_ct = get_onehot_ct(init_assign=celltype_assignment)

        # Note: some spots may contain no cells
        spot_ct_count = spot_cell_idx_matrix * onehot_ct

        return spot_expr, spot_loc, spot_cell_idx_matrix 
    
    if spot_generate_type == "square" and cell_coord_type == "grid":
        # Also includes grid merging
        if grid_row is None and grid_col is None:
            grid_row = np.sqrt(Num_sample).astype(int)
            grid_col = grid_row
        # Grid merging: we merge nine grids similar to ST, while setting gap between spots
        windows_row = 3
        windows_col = 3
        gap = gap
        spot_diameter = 1

        # Edge cases don't have 3 * 3 grids, should we set gap?
        spot_row_num = int(grid_row / (windows_row + gap))
        spot_col_num = int(grid_col / (windows_col + gap))

        a, b = np.meshgrid(np.arange(grid_row),
                   np.arange(grid_col),
                   indexing='ij')
        # lra and lrb are center grids of 9 grids, actually coordinates of bottom-left point of center grid
        lra = a[np.arange(1, spot_row_num * (windows_row + gap), (windows_row + gap)
                  ), :len(np.arange(1, spot_col_num * (windows_col + gap), (windows_col + gap)))].reshape(-1)
        lrb = b[:len(np.arange(1, spot_row_num * (windows_row + gap), (windows_row + gap))),
            np.arange(1, spot_col_num * (windows_col + gap), (windows_col + gap))].reshape(-1)   

        # Center grid indices
        lrspot_layer_idx = lra * grid_col + lrb   

        # Get neighbors of center points, i.e., 9 grids including itself
        sn = get_spaital_network(Num_sample=cell_spatial.shape[0], spatial=cell_spatial,
                         n_neighs=8, coord_type="grid", n_rings=1, set_diag=True)

        spot_cell_idx_matrix = sn[lrspot_layer_idx] 
        spot_expr = spot_cell_idx_matrix * sim_cell_expr.X

        onehot_ct = get_onehot_ct(init_assign=celltype_assignment)

        # Note: some spots may contain no cells
        spot_ct_count = spot_cell_idx_matrix * onehot_ct

        spot_pixel_row = (lra.reshape(-1) + 0.5) * spot_diameter
        spot_pixel_col = (lrb.reshape(-1) + 0.5) * spot_diameter
        spot_loc = np.vstack((spot_pixel_row, spot_pixel_col)).T 

        return spot_expr, spot_loc, spot_cell_idx_matrix    
    
    if custom_spot_loc is not None:
        # Also supports specified spot coordinates

        # must be numpy array with shape (n*2)
        # custom_spot_loc and spot_diameter
        spot_loc = custom_spot_loc

        cellandspot_loc = np.vstack((cell_location, spot_loc))
        sn = get_spaital_network(Num_sample=cellandspot_loc.shape[0], spatial=cellandspot_loc,
                         coord_type="generic", n_rings=1, set_diag=False,
                                 radius=spot_diameter/2)

        spot_cell_idx_matrix = sn[cell_location.shape[0]:, cell_location.shape[0]]

        spot_expr = spot_cell_idx_matrix * sim_cell_expr.X

        onehot_ct = get_onehot_ct(init_assign=celltype_assignment)

        # Note: some spots may contain no cells
        spot_ct_count = spot_cell_idx_matrix * onehot_ct

        return spot_expr, spot_loc, spot_cell_idx_matrix

def generate_spot_level_data(spider_adata, save_path, spot_diameter=0.5, gap=0,
                            spot_generate_type="square", coord_type="generic"):
    """
    Generate spot-level spatial transcriptomics data from single-cell data using Spider.
    
    Parameters:
    ----------
    spider_adata : AnnData
        Input single-cell resolution AnnData object containing:
        - obsm['spatial']: cell coordinates
        - obs['label']: cell type assignments
        - var: gene information
    save_path : str
        Directory path to save output h5ad file
    spot_diameter : float, optional
        Diameter of simulated spots (default: 0.5)
    gap : float, optional
        Gap between spots (default: 0)
    spot_generate_type : str, optional
        Spot generation pattern ("square" or other supported types) (default: "square")
    coord_type : str, optional
        Coordinate system type (default: "generic")
    
    Returns:
    -------
    AnnData
        Spot-level AnnData object containing:
        - obsm['spatial']: spot coordinates
        - uns['W']: cell type counts per spot
    """
    # Normalize coordinates to start at (0,0)
    x_min = np.min(spider_adata.obsm["spatial"][:, 0])
    y_min = np.min(spider_adata.obsm["spatial"][:, 1])
    spider_adata.obsm["spatial"][:, 0] -= x_min
    spider_adata.obsm["spatial"][:, 1] -= y_min
    
    # Generate spot-level data
    spot_expr, spot_loc, _, spot_ct_count = get_sim_spot_level_expr(
        spot_num=None,
        spot_row=None,
        spot_col=None,
        Num_sample=spider_adata.shape[0],
        spot_diameter=spot_diameter,
        image_width=np.max(spider_adata.obsm["spatial"][:, 0]),
        image_height=np.max(spider_adata.obsm["spatial"][:, 1]),
        celltype_assignment=spider_adata.obs.label,
        coord_type=coord_type,
        cell_spatial=spider_adata.obsm["spatial"],
        sim_cell_expr=spider_adata,
        gap=gap,
        spot_generate_type=spot_generate_type,
        cell_coord_type="generic",
        grid_row=None,
        grid_col=None
    )
    
    # Filter empty spots and restore original coordinates
    mask = np.all(spot_ct_count == 0, axis=1)
    spot_loc[:, 0] += x_min
    spot_loc[:, 1] += y_min
    
    # Create AnnData object
    adata = ad.AnnData(spot_expr)
    adata.obsm["spatial"] = spot_loc
    adata.var = spider_adata.var
    adata.obs.index = [f'spot{j}' for j in range(spot_expr.shape[0])]
    adata = adata[~mask]
    adata.uns["W"] = spot_ct_count[~mask]
    
    # Save results
    adata.write_h5ad(f"{save_path}")
    
    return adata         
