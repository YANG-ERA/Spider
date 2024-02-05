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


def get_mesh_counts(locations = None, grid_row = None, grid_col = None,
                   image_width = None, image_height = None):
    
    grid_num = grid_row *grid_col
    
    if image_width is None and image_height is None:
        
        image_width  = np.max(locations[:,0]).round().astype(int)
        image_height = np.max(locations[:,1]).round().astype(int)
 
    dx = image_width / grid_row  
    dy = image_height / grid_col
    points_x_indices = (locations[:,0] / dx).astype(int)
    points_y_indices = (locations[:,1] / dy).astype(int)
    
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

    randsn = get_spaital_network(Num_sample=cell_num,
                                 spatial=cell_location, coord_type = "generic",
                                 n_neighs=8)

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