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
from .utils import *
from .Annealing import *
from .enhance import *
from scipy.sparse import coo_matrix,csr_matrix
from joblib import Parallel, delayed

def get_mesh_counts(locations = None, grid_row = None, grid_col = None,
                   image_width = None, image_height = None):
    
    grid_num = grid_row *grid_col
    
    if image_width is None and image_height is None:
        
        image_width  = np.max(locations[:,0]).round().astype(int)
        image_height = np.max(locations[:,1]).round().astype(int)
 
    dx = image_width / grid_row  
    dy = image_height / grid_col
    points_x_indices = np.clip((locations[:, 0] / dx).astype(int), 0, grid_row - 1)
    points_y_indices = np.clip((locations[:, 1] / dy).astype(int), 0, grid_col - 1)
   
#     print("cell location sort by meshgrid")
#     points_sortidex = np.lexsort([points_y_indices,points_x_indices])
   
#     res = csr_matrix((np.ones_like(points_x_indices, dtype=np.int), 
#                         (points_x_indices,points_y_indices )), 
#                         shape=(grid_row ,grid_col)).A
#     counts = res.reshape(-1)

    ## 细胞的spot编号
    cell_spot_id = (points_x_indices*grid_col + points_y_indices).astype(int)
    
    ## 细胞的编号
    cell_id = np.arange(cell_spot_id.size)
    assert cell_spot_id.max() < grid_num, "存在越界的网格索引"
    ##  替换np.where 每一个grid  obtain 每一个spot的cell的index 行cell_id 列spot 
    cell_spot_idx_matrix = csr_matrix(
        (np.ones(cell_spot_id.size), (cell_id,cell_spot_id)),
        shape=(cell_spot_id.size,grid_num))
    
   
    a, b = np.meshgrid(np.arange(grid_row),
                   np.arange(grid_col),
                   indexing='ij')
    
    grid_row_loc = a.reshape(-1) * dx
    grid_col_loc = b.reshape(-1) * dy
    
    return  cell_spot_idx_matrix, grid_row_loc, grid_col_loc, image_width, image_height


def simulate_10X(cell_num=None,
                 Num_celltype=None,
                 prior=None,
                 target_trans=None,
                 Num_ct_sample=None,
                 spot_radius=None,
                 spot_min=None,
                 spot_max=None,
                 image_width=None,
                 image_height=None,
                 cell_location=None,
                 tol=2e-2,
                 T=1e-2,
                 loop_times=None,
                 smallsample_max_iter=80000,
                 bigsample_max_iter=10000,
                 ref=None):
    '''
    Problems: 1. this function, you must give `image_width` and `image_height`, but 
                 function `get_mesh_counts` write the situation that these paras are given
                 by yourself.
              2. How to control `spot_min` and `spot_max`
    '''

    # we will compute meshgrid nums to determine how many
    ## 用户从HE学出细胞位置，通过cell_location给定
    ## 也可以基于细胞数 随机设定
    ## 判断 cell_num是否为空，补
    ## 判断cell_num与cell_location同时存在,num = cell_location.shape[0]
    if ref is not None:
        cell_num = ref.shape[0]
        cell_location = ref.obsm["spatial"]
        Num_ct_sample = np.bincount(ref.obs.label)
        randcelltype = init_ct(Num_celltype=len(Num_ct_sample),
                               Num_ct_sample=Num_ct_sample)
        Num_celltype = len(Num_ct_sample)
        if cell_num <= 10000:
            loop_times = 1
        else:
            grid_row = np.sqrt(cell_num).astype(int)
            grid_col = grid_row
            grid_num = grid_row * grid_col
            cell_spot_idx_matrix, grid_row_loc, grid_col_loc, image_width, image_height = get_mesh_counts(
                locations=cell_location,
                grid_row=grid_row,
                grid_col=grid_col,
                image_width=image_width,
                image_height=image_height)

            # cell_location = cell_location[points_sortidex, :]

            print("enhance resolution based on meshgrid")
            ## 注意此时grid_row *grid_col != grid_spatial
            grid_spatial = np.vstack((grid_row_loc, grid_col_loc)).T

            celltype_assignment, _, loop_times = enhance_loop(
                Num_sample=grid_num,
                Num_celltype=Num_celltype,
                prior=prior,
                target_trans=target_trans,
                original_grid=grid_spatial,
                grid_row=grid_row,
                grid_col=grid_col,
                loop_times=None,
                windows_row_list=None,
                windows_col_list=None,
                swap_num_list=None,
                tol_list=None,
                T=1e-5,
                smallsample_max_iter=smallsample_max_iter,
                bigsample_max_iter=bigsample_max_iter)

            print(
                "cell type labels of the grid are inversely assigned to cells, and adjust ratio."
            )

            ## 得到网格的细胞类型的onehot
            grid_ct_onehot = get_onehot_ct(init_assign=celltype_assignment.reshape(-1))
            randcelltype = np.argmax(cell_spot_idx_matrix * grid_ct_onehot, axis=1)
            # randcelltype = np.repeat(celltype_assignment, counts)

            ##  adjust ratio
            # Num_ct_sample = get_ct_sample(Num_celltype=Num_celltype,
            #                               Num_sample=cell_num,
            #                               prior=prior)
            randcelltype = mutate(celltype_assignment=randcelltype,
                                  Num_ct_sample=Num_ct_sample)

    else:
        if cell_location is not None:
            cell_num = cell_location.shape[0]
        else:
            points_x = np.random.uniform(0, image_width, cell_num)
            points_y = np.random.uniform(0, image_height, cell_num)
            cell_location = np.vstack((points_x, points_y)).T
        if cell_num <= 10000:

            ##  adjust ratio
            if Num_ct_sample is None:
                Num_ct_sample = get_ct_sample(Num_celltype=Num_celltype,
                                              Num_sample=cell_num,
                                              prior=prior)

            randcelltype = init_ct(Num_celltype = Num_celltype,
                                   Num_ct_sample = Num_ct_sample )

            loop_times = 1

        else:
            grid_row = np.sqrt(cell_num).astype(int)
            grid_col = grid_row
            grid_num = grid_row * grid_col
            cell_spot_idx_matrix, grid_row_loc, grid_col_loc, image_width, image_height  = get_mesh_counts(
                locations=cell_location,
                grid_row=grid_row,
                grid_col=grid_col,
                image_width=image_width,
                image_height=image_height)

           # cell_location = cell_location[points_sortidex, :]

            print("enhance resolution based on meshgrid")
            ## 注意此时grid_row *grid_col != grid_spatial
            grid_spatial = np.vstack((grid_row_loc, grid_col_loc)).T

            celltype_assignment, _, loop_times = enhance_loop(
                Num_sample=grid_num,
                Num_celltype=Num_celltype,
                prior=prior,
                target_trans=target_trans,
                original_grid=grid_spatial,
                grid_row=grid_row,
                grid_col=grid_col,
                loop_times=None,
                windows_row_list=None,
                windows_col_list=None,
                swap_num_list=None,
                tol_list=None,
                T=1e-5,
                smallsample_max_iter=smallsample_max_iter,
                bigsample_max_iter=bigsample_max_iter)

            print(
                "cell type labels of the grid are inversely assigned to cells, and adjust ratio."
            )

            ## 得到网格的细胞类型的onehot
            grid_ct_onehot = get_onehot_ct(init_assign = celltype_assignment.reshape(-1) )
            randcelltype = np.argmax(cell_spot_idx_matrix * grid_ct_onehot,axis = 1)
            #randcelltype = np.repeat(celltype_assignment, counts)

            ##  adjust ratio
            Num_ct_sample = get_ct_sample(Num_celltype=Num_celltype,
                                          Num_sample=cell_num,
                                          prior=prior)
            randcelltype = mutate(celltype_assignment=randcelltype,
                                  Num_ct_sample=Num_ct_sample)
    if cell_num <= 8:
        n_neighs = np.int(np.ceil(cell_num/2))
    else:
        n_neighs = 8
    randsn = get_spaital_network(Num_sample=cell_num,
                                 spatial=cell_location, coord_type = "generic",
                                 n_neighs=n_neighs)

    onehot_ct = get_onehot_ct(init_assign=randcelltype)
    nb_count = np.array(randsn * onehot_ct, dtype=np.float32)

    swap_num = 2 ** (loop_times-1)
    print("swap_num: ", swap_num)
    celltype_assignment = STsim(Num_sample=cell_num,
                             Num_celltype=Num_celltype,
                             celltype_assignment=randcelltype,
                             target_trans=target_trans,
                             T=T,
                             chain_len=100,
                             error=1e2,
                             onehot_ct=onehot_ct,
                             tol=tol,
                             decay=0.5,
                             nb_count=nb_count,
                             sn=randsn,
                             swap_num=swap_num,
                             smallsample_max_iter=smallsample_max_iter,
                             bigsample_max_iter=bigsample_max_iter)
    

    return celltype_assignment, cell_location


def extract_loc(img = None,
                layer = None,
                library_id = None,
                channel =  0,
                ):
    '''
    Extract cell locations from an image. 
    
    用squidpy做完细胞分割后，利用该函数可以实现获取细胞像素坐标，之前的图像处理流程参考:
    https://squidpy.readthedocs.io/en/latest/auto_examples/image/compute_segment_hne.html
    Parameters
    ----------
    %(img_container)s
    %(img_layer)s
    layer
        Image layer in ``img`` that should be processed. If `None` and only 1 layer is present, it will be selected.
    %(library_id)s
        If `None`, all Z-dimensions are processed at once, treating the image as a 3D volume.
    channel
        Channel index to use for extracting. If `None`, use all channels.
    Returns
    -------
    Return an array with cell locations extracted from image. the frist column represents the
    y_coord of cell in the image, the second column represents the x_coord of cell in the image.
    '''
    layer = img._get_layer(layer)
    library_id = img._get_library_ids(library_id)
    if channel is None and img[layer].shape[-1] > 1:
        raise ValueError("Watershed segmentation does not work with multiple channels.")
        
    segments_matrix = np.array(img["segmented_watershed"])
    segments_matrix =  coo_matrix(segments_matrix.reshape(segments_matrix.shape[0:2]))
    segments_matrix = pd.DataFrame({"pixel_y":segments_matrix.row,
                             "pixel_x":segments_matrix.col,
                              "segment_label":segments_matrix.data})
    cell_location = segments_matrix.groupby(segments_matrix["segment_label"]).mean().values 
    print(f"Number of cells in image: {cell_location.shape[0]}")
    return cell_location


def get_mesh_counts_3d(locations=None, grid_x=None, grid_y=None, grid_z=None,
                      image_width=None, image_height=None, image_depth=None):
    """
    Modified version for 3D spatial coordinates
    Returns cell-spot index matrix and grid locations in 3D space
    """
    grid_num = grid_x * grid_y * grid_z
    
    if image_width is None and image_height is None and image_depth is None:
        image_width = np.max(locations[:,0]).round().astype(int)
        image_height = np.max(locations[:,1]).round().astype(int)
        image_depth = np.max(locations[:,2]).round().astype(int)
 
    dx = image_width / grid_x  
    dy = image_height / grid_y
    dz = image_depth / grid_z
    
    # Calculate 3D grid indices for each cell
    points_x_indices = (locations[:,0] / dx).astype(int)
    points_y_indices = (locations[:,1] / dy).astype(int)
    points_z_indices = (locations[:,2] / dz).astype(int)
    
    # Calculate unique spot ID for each cell in 3D grid
    cell_spot_id = (points_x_indices * grid_y * grid_z + 
                    points_y_indices * grid_z + 
                    points_z_indices).astype(int)
    
    cell_id = np.arange(cell_spot_id.size)
    
    # Create sparse matrix mapping cells to spots
    cell_spot_idx_matrix = csr_matrix(
        (np.ones(cell_spot_id.size), (cell_id, cell_spot_id)),
        shape=(cell_spot_id.size, grid_num))
    
    # Generate 3D grid coordinates
    a, b, c = np.meshgrid(np.arange(grid_x),
                          np.arange(grid_y),
                          np.arange(grid_z),
                          indexing='ij')
    
    grid_x_loc = a.reshape(-1) * dx
    grid_y_loc = b.reshape(-1) * dy
    grid_z_loc = c.reshape(-1) * dz
    
    return cell_spot_idx_matrix, grid_x_loc, grid_y_loc, grid_z_loc, image_width, image_height, image_depth


def simulate_10X_3d(cell_num=None,
                   Num_celltype=None,
                   prior=None,
                   target_trans=None,
                   Num_ct_sample=None,
                   image_width=None,
                   image_height=None,
                   image_depth=None,
                   cell_location=None,
                   tol=2e-2,
                   T=1e-2,
                   loop_times=None,
                   smallsample_max_iter=80000,
                   bigsample_max_iter=10000,
                   ref=None):
    """
    3D version of spatial simulation
    """
    if ref is not None:
        cell_num = ref.shape[0]
        cell_location = ref.obsm["spatial_3d"]  # Assuming 3D coordinates in obsm
        Num_ct_sample = np.bincount(ref.obs.label)
        randcelltype = init_ct(Num_celltype=len(Num_ct_sample),
                             Num_ct_sample=Num_ct_sample)
        Num_celltype = len(Num_ct_sample)
        
        if cell_num <= 10000:
            loop_times = 1
        else:
            # For 3D grid, we use cube root for initial grid dimensions
            grid_dim = round(cell_num ** (1/3))
            grid_x = grid_dim
            grid_y = grid_dim
            grid_z = grid_dim
            grid_num = grid_x * grid_y * grid_z
            
            cell_spot_idx_matrix, grid_x_loc, grid_y_loc, grid_z_loc, image_width, image_height, image_depth = get_mesh_counts_3d(
                locations=cell_location,
                grid_x=grid_x,
                grid_y=grid_y,
                grid_z=grid_z,
                image_width=image_width,
                image_height=image_height,
                image_depth=image_depth)

            grid_spatial = np.vstack((grid_x_loc, grid_y_loc, grid_z_loc)).T

            celltype_assignment, _, loop_times = enhance_loop(
                Num_sample=grid_num,
                Num_celltype=Num_celltype,
                prior=prior,
                target_trans=target_trans,
                original_grid=grid_spatial,
                grid_row=grid_x,  # Using grid_x for row dimension
                grid_col=grid_y,  # Using grid_y for col dimension
                # Need to modify enhance_loop to handle 3D if needed
                loop_times=None,
                windows_row_list=None,
                windows_col_list=None,
                swap_num_list=None,
                tol_list=None,
                T=1e-5,
                smallsample_max_iter=smallsample_max_iter,
                bigsample_max_iter=bigsample_max_iter)

            grid_ct_onehot = get_onehot_ct(init_assign=celltype_assignment.reshape(-1))
            randcelltype = np.argmax(cell_spot_idx_matrix * grid_ct_onehot, axis=1)
            randcelltype = mutate(celltype_assignment=randcelltype,
                                Num_ct_sample=Num_ct_sample)
    else:
        if cell_location is not None:
            cell_num = cell_location.shape[0]
        else:
            # Generate random 3D coordinates
            points_x = np.random.uniform(0, image_width, cell_num)
            points_y = np.random.uniform(0, image_height, cell_num)
            points_z = np.random.uniform(0, image_depth, cell_num)
            cell_location = np.vstack((points_x, points_y, points_z)).T
            
        if cell_num <= 10000:
            if Num_ct_sample is None:
                Num_ct_sample = get_ct_sample(Num_celltype=Num_celltype,
                                            Num_sample=cell_num,
                                            prior=prior)

            randcelltype = init_ct(Num_celltype=Num_celltype,
                                 Num_ct_sample=Num_ct_sample)
            loop_times = 1
        else:
            grid_dim = round(cell_num ** (1/3))
            grid_x = grid_dim
            grid_y = grid_dim
            grid_z = grid_dim
            grid_num = grid_x * grid_y * grid_z
            
            cell_spot_idx_matrix, grid_x_loc, grid_y_loc, grid_z_loc, image_width, image_height, image_depth = get_mesh_counts_3d(
                locations=cell_location,
                grid_x=grid_x,
                grid_y=grid_y,
                grid_z=grid_z,
                image_width=image_width,
                image_height=image_height,
                image_depth=image_depth)

            grid_spatial = np.vstack((grid_x_loc, grid_y_loc, grid_z_loc)).T

            celltype_assignment, _, loop_times = enhance_loop(
                Num_sample=grid_num,
                Num_celltype=Num_celltype,
                prior=prior,
                target_trans=target_trans,
                original_grid=grid_spatial,
                grid_row=grid_x,
                grid_col=grid_y,
                loop_times=None,
                windows_row_list=None,
                windows_col_list=None,
                swap_num_list=None,
                tol_list=None,
                T=1e-5,
                smallsample_max_iter=smallsample_max_iter,
                bigsample_max_iter=bigsample_max_iter)

            grid_ct_onehot = get_onehot_ct(init_assign=celltype_assignment.reshape(-1))
            randcelltype = np.argmax(cell_spot_idx_matrix * grid_ct_onehot, axis=1)
            
            if Num_ct_sample is None:
                Num_ct_sample = get_ct_sample(Num_celltype=Num_celltype,
                                            Num_sample=cell_num,
                                            prior=prior)
            randcelltype = mutate(celltype_assignment=randcelltype,
                                Num_ct_sample=Num_ct_sample)

    # For 3D spatial network, we need to modify get_spaital_network or use a 3D-aware method
    if cell_num <= 8:
        n_neighs = np.int(np.ceil(cell_num/2))
    else:
        n_neighs = 8
        
    randsn = get_spaital_network(Num_sample=cell_num,
                               spatial=cell_location, 
                               coord_type="generic",
                               n_neighs=n_neighs)

    onehot_ct = get_onehot_ct(init_assign=randcelltype)
    nb_count = np.array(randsn * onehot_ct, dtype=np.float32)

    swap_num = 2 ** (loop_times-1)
    print("swap_num: ", swap_num)
    
    celltype_assignment = STsim(Num_sample=cell_num,
                             Num_celltype=Num_celltype,
                             celltype_assignment=randcelltype,
                             target_trans=target_trans,
                             T=T,
                             chain_len=100,
                             error=1e2,
                             onehot_ct=onehot_ct,
                             tol=tol,
                             decay=0.5,
                             nb_count=nb_count,
                             sn=randsn,
                             swap_num=swap_num,
                             smallsample_max_iter=smallsample_max_iter,
                             bigsample_max_iter=bigsample_max_iter)
    
    return celltype_assignment, cell_location


def extract_loc_3d(img=None,
                  layer=None,
                  library_id=None,
                  channel=0):
    """
    3D version of cell location extraction
    Extracts cell centroids from 3D segmentation data
    """
    layer = img._get_layer(layer)
    library_id = img._get_library_ids(library_id)
    
    if channel is None and img[layer].shape[-1] > 1:
        raise ValueError("Watershed segmentation does not work with multiple channels.")
        
    segments_matrix = np.array(img["segmented_watershed"])
    
    # For 3D data, we need to handle z-dimension
    if len(segments_matrix.shape) == 3:  # 2D + channels
        # Handle as 2D case (original behavior)
        segments_matrix = coo_matrix(segments_matrix.reshape(segments_matrix.shape[0:2]))
        segments_matrix = pd.DataFrame({
            "pixel_y": segments_matrix.row,
            "pixel_x": segments_matrix.col,
            "segment_label": segments_matrix.data
        })
    elif len(segments_matrix.shape) == 4:  # 3D data
        # Convert 3D segmentation to sparse format
        segments_matrix = coo_matrix(
            segments_matrix.reshape(
                segments_matrix.shape[0] * segments_matrix.shape[1] * segments_matrix.shape[2],
                1
            )
        )
        # Calculate 3D coordinates from linear indices
        z_coords = segments_matrix.row // (segments_matrix.shape[1] * segments_matrix.shape[2])
        remainder = segments_matrix.row % (segments_matrix.shape[1] * segments_matrix.shape[2])
        y_coords = remainder // segments_matrix.shape[2]
        x_coords = remainder % segments_matrix.shape[2]
        
        segments_matrix = pd.DataFrame({
            "pixel_z": z_coords,
            "pixel_y": y_coords,
            "pixel_x": x_coords,
            "segment_label": segments_matrix.data
        })
    else:
        raise ValueError("Unsupported image dimensionality")
    
    # Group by segment label and calculate centroids
    cell_location = segments_matrix.groupby(segments_matrix["segment_label"]).mean().values 
    print(f"Number of cells in image: {cell_location.shape[0]}")
    return cell_location




def generate_in_grid_positions(n_cells, grid_bounds, dx, dy):
    """
    Generate random positions within grid boundaries
    
    Parameters:
        n_cells: Number of cells to generate
        grid_bounds: Grid boundaries (x_min, x_max, y_min, y_max)
        dx: Grid width
        dy: Grid height
    
    Returns:
        Array of randomly generated positions within grid (n_cells, 2)
    """
    x_min, x_max, y_min, y_max = grid_bounds
    # Ensure positions stay within grid boundaries
    new_x = np.random.uniform(x_min, min(x_min + dx, x_max), n_cells)
    new_y = np.random.uniform(y_min, min(y_min + dy, y_max), n_cells)
    return np.column_stack((new_x, new_y))

def PSA_worker(i, cell_spot_idx_matrix, cell_location, adata, perturb, 
               grid_row_loc, grid_col_loc, dx, dy, grid_row, grid_col):
    """
    Modified worker function supporting position perturbation
    """
    patch_cell_idx = np.where(cell_spot_idx_matrix.getcol(i).toarray() != 0)[0]
    patch_cell_id = adata.obs.label[patch_cell_idx]
    
    # Calculate current grid boundaries
    row_idx = i // grid_col
    col_idx = i % grid_col
    x_min = row_idx * dx
    x_max = (row_idx + 1) * dx
    y_min = col_idx * dy
    y_max = (col_idx + 1) * dy
    grid_bounds = (x_min, x_max, y_min, y_max)
    
    # Get original positions or generate random positions within grid
    if perturb and len(patch_cell_idx) > 0:
        patch_cell_loc = generate_in_grid_positions(
            len(patch_cell_idx), grid_bounds, dx, dy
        )
    else:
        patch_cell_loc = cell_location[patch_cell_idx]
    
    if len(patch_cell_idx) == 0:
        return None, None
    elif len(np.unique(patch_cell_id)) == 1:
        return [int(x) for x in patch_cell_id.values.tolist()], patch_cell_loc
    else:
        # Adaptively determine neighborhood size
        n_cells = len(patch_cell_idx)
        n_neighs = np.int(np.ceil(n_cells/2)) if n_cells <= 8 else 8
        
        # Build spatial network
        sn = get_spaital_network(
            Num_sample=n_cells,
            spatial=patch_cell_loc, 
            coord_type="generic",
            n_neighs=n_neighs
        )

        # Prepare simulation parameters
        patch_ct = np.sort(np.unique(patch_cell_id))
        mapping_dict = dict(zip(patch_ct, range(len(patch_ct))))
        patch_cell_id_map = [mapping_dict[x] for x in patch_cell_id]

        onehot_ct = get_onehot_ct(init_assign=patch_cell_id_map)
        nb_count = np.array(sn * onehot_ct, dtype=np.float32)
        target_trans = get_nb_freq(nb_count=nb_count, onehot_ct=onehot_ct)

        Num_ct_sample = np.bincount(patch_cell_id_map)
        prior = Num_ct_sample/np.sum(Num_ct_sample)
        
        # Execute simulation
        res, _ = simulate_10X(
            cell_num=n_cells,
            Num_celltype=target_trans.shape[0],
            prior=prior,
            target_trans=target_trans,
            Num_ct_sample=Num_ct_sample,
            spot_radius=None,
            spot_min=None,
            spot_max=None,
            image_width=None,
            image_height=None,
            cell_location=patch_cell_loc,
            tol=2e-2,
            T=1e-2,
            loop_times=None,
            smallsample_max_iter=1000,
            bigsample_max_iter=10000,
            ref=None
        )
    
        # Map back to original cell types
        mapping_dict_rev = dict(zip(range(len(patch_ct)), patch_ct))
        res = [mapping_dict_rev[x] for x in res]

        return res, patch_cell_loc

def PSA(grid_row, grid_col, adata, ct_key="label", seed=2024, perturb=False):
    """
    Modified PSA function supporting position perturbation
    
    Parameters:
        grid_row, grid_col: Number of grid rows and columns
        adata: AnnData object containing spatial coordinates and cell types
        ct_key: Key name for cell type column
        seed: Random seed
        perturb: Whether to generate random positions within grid
    """
    np.random.seed(seed)
    grid_num = grid_row * grid_col
    
    # Get original coordinates and grid information
    cell_location = adata.obsm["spatial"]
    cell_spot_idx_matrix, grid_row_loc, grid_col_loc, image_width, image_height = get_mesh_counts(
        locations=cell_location,
        grid_row=grid_row,
        grid_col=grid_col
    )
    
    # Calculate grid dimensions
    dx = image_width / grid_row
    dy = image_height / grid_col
    
    # Prepare result storage
    all_simulated_labels = adata.obs[ct_key].copy()
    all_perturbed_locations = cell_location.copy()
    
    # Parallel processing for each grid
    parallel_obj = Parallel(n_jobs=4, verbose=10, backend='loky')
    results = parallel_obj(
        delayed(PSA_worker)(
            i, 
            cell_spot_idx_matrix, 
            cell_location, 
            adata, 
            perturb,
            grid_row_loc, 
            grid_col_loc, 
            dx, 
            dy,
            grid_row,
            grid_col
        ) 
        for i in range(grid_num)
    )
    
    # Consolidate results
    for i in range(grid_num):
        result = results[i]
        if result is None:
            continue
            
        simulated_labels, perturbed_locs = result
        patch_cell_idx = np.where(cell_spot_idx_matrix.getcol(i).toarray() != 0)[0]
        
        if simulated_labels is not None and len(patch_cell_idx) > 0:
            all_simulated_labels[patch_cell_idx] = simulated_labels
        if perturbed_locs is not None and len(patch_cell_idx) > 0:
            all_perturbed_locations[patch_cell_idx] = perturbed_locs
    
    # Update adata object
    spider_adata = adata.copy()
    spider_adata.obs["spider_simu"] = all_simulated_labels
    spider_adata.obsm["spatial_perturbed"] = all_perturbed_locations
    
    return spider_adata    