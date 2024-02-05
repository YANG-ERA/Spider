
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


def get_sim_cell_level_expr(celltype_assignment=None, adata=None,
                            Num_celltype=None, Num_ct_sample=None,
                            match_list=None, ct_key=None):
    ## 给每一个估计的细胞，分配一种单细胞里面的细胞index，
    ## 前提是同一组细胞的index对应到单细胞数据里面是同一种细胞类型
    assert adata is not None, print(
        "please input an adata about singlecell, if you don't want to find it in your folder," +
        "here we We provide a single-cell data simulation strategy for you to choose")

    idx_list = celltype_assignment.copy()
    for i in range(Num_celltype):
        ## 找到单细胞中的指定细胞类型的细胞
        ct_idx = np.where(adata.obs[ct_key] == match_list[i])[0]
        ## 抽取指定模拟该细胞类型数量的细胞，得到在单细胞里面的idx
        sim_cell_idx = np.random.choice(ct_idx, size=Num_ct_sample[i], replace=True)

        idx_list[np.where(celltype_assignment == i)[0]] = sim_cell_idx

    sim_cell_expr = adata[idx_list]
    # sim_cell_expr.obs["sim_celltype"] = res
    return sim_cell_expr
    

def get_sim_spot_level_expr( spot_num = None, spot_row = None, spot_col = None,
                            Num_sample = None, spot_diameter = None, image_width = None,
                            image_height = None,celltype_assignment = None,coord_type = None,
                            cell_spatial = None,sim_cell_expr = None,gap = 0,
                           spot_generate_type = "square",cell_coord_type = "grid",
                           grid_row = None, grid_col = None):
    if spot_generate_type == "square" and cell_coord_type == "generic":
        ## 随机点的网格划分，不是网格merge 

        grid_row = np.ceil(image_width / spot_diameter).astype(int)
        grid_col = np.ceil(image_height / spot_diameter).astype(int)
        grid_num = grid_row * grid_col
        points_x_indices = (cell_spatial[:, 0] / spot_diameter).astype(int)
        points_y_indices = (cell_spatial[:, 1] / spot_diameter).astype(int)

        ## 细胞的spot编号
        cell_spot_id = (points_x_indices*grid_col + points_y_indices).astype(int)
        ## 细胞的编号
        cell_id = np.arange(cell_spot_id.size)


        ##  替换np.where 每一个grid  obtain 每一个spot的cell的index 行spot 列cell_id
        spot_cell_idx_matrix = csr_matrix(
            (np.ones(cell_spot_id.size), (cell_spot_id, cell_id)),
            shape=(grid_num, cell_spot_id.size))

        spot_expr = spot_cell_idx_matrix * sim_cell_expr.X

        onehot_ct = get_onehot_ct(init_assign=celltype_assignment)

        ## 注意有些spot里面可能没有cell
        spot_ct_count = spot_cell_idx_matrix * onehot_ct

        ## 取网格的中心点作为spot坐标
        a, b = np.meshgrid(np.arange(grid_row),
                       np.arange(grid_col),
                       indexing='ij')
        spot_pixel_row = (a.reshape(-1)+0.5)*spot_diameter
        spot_pixel_col = (b.reshape(-1)+0.5)*spot_diameter

        spot_loc =  np.vstack((spot_pixel_row, spot_pixel_col)).T

        return spot_expr, spot_loc, spot_cell_idx_matrix, spot_ct_count 
    
    if spot_generate_type == "circle" and cell_coord_type == "generic":
  
     ####### 规定直径 and gap

        grid_row = int(image_width / gap)
        grid_col = int(image_height / gap)
        grid_num = grid_row * grid_col

        ## 取网格的中心点作为spot坐标
        a, b = np.meshgrid(np.arange(grid_row),
                       np.arange(grid_col),
                       indexing='ij')
        spot_pixel_row = (a.reshape(-1)+0.5)*gap
        spot_pixel_col = (b.reshape(-1)+0.5)*gap

        spot_loc =  np.vstack((spot_pixel_row, spot_pixel_col)).T

        cellandspot_loc =  np.vstack((cell_location,spot_loc))

        ## get中心点的邻居
        sn = get_spaital_network(Num_sample=cellandspot_loc.shape[0],spatial=cellandspot_loc,
                         coord_type = "generic",n_rings = 1,set_diag=False,
                                 radius = spot_diameter/2)

        spot_cell_idx_matrix = sn[Num_sample:,Num_sample]

        spot_expr = spot_cell_idx_matrix * sim_cell_expr.X

        onehot_ct = get_onehot_ct(init_assign=celltype_assignment)

        ## 注意有些spot里面可能没有cell
        spot_ct_count = spot_cell_idx_matrix * onehot_ct

        return spot_expr, spot_loc, spot_cell_idx_matrix 
    
    if spot_generate_type == "square" and cell_coord_type == "grid":

      ##### 还有网格merge 
        if grid_row is None and grid_col is None:
            grid_row = np.sqrt(Num_sample).astype(int)
            grid_col = grid_row
        ## 网格merge,我们取九个格子merge 类似ST 同时设置spot之间的gap
        windows_row = 3
        windows_col = 3
        gap = gap
        spot_diameter = 1

        ## 边缘没有3*3的网格,设不设置gap?
        spot_row_num = int(grid_row/(windows_row+gap))
        spot_col_num = int(grid_col/(windows_col+gap))

        a, b = np.meshgrid(np.arange(grid_row),
                   np.arange(grid_col),
                   indexing='ij')
        ## lra 与lrb是9个网格的中心网格 实际是中心网格左下角的点的坐标
        lra = a[np.arange(1, spot_row_num*(windows_row+gap), (windows_row+gap)
                  ), :len(np.arange(1, spot_col_num*(windows_col+gap), (windows_col+gap)))].reshape(-1)
        lrb = b[:len(np.arange(1, spot_row_num*(windows_row+gap), (windows_row+gap))),
            np.arange(1, spot_col_num*(windows_col+gap), (windows_col+gap))].reshape(-1)   

        ## 中心网格标号
        lrspot_layer_idx = lra*grid_col+lrb   

        ## get中心点的邻居，即9个网格 包括自己，
        sn = get_spaital_network(Num_sample=cell_spatial.shape[0],spatial=cell_spatial,
                         n_neighs=8,coord_type = "grid",n_rings = 1,set_diag=True)

        spot_cell_idx_matrix = sn[lrspot_layer_idx] 
        spot_expr = spot_cell_idx_matrix * sim_cell_expr.X

        onehot_ct = get_onehot_ct(init_assign=celltype_assignment)

        ## 注意有些spot里面可能没有cell
        spot_ct_count = spot_cell_idx_matrix * onehot_ct

        spot_pixel_row = (lra.reshape(-1)+0.5)*spot_diameter
        spot_pixel_col = (lrb.reshape(-1)+0.5)*spot_diameter
        spot_loc =  np.vstack((spot_pixel_row, spot_pixel_col)).T 


        return spot_expr, spot_loc, spot_cell_idx_matrix    
    
    if custom_spot_loc is not None:
        ## 以及指定spot坐标set  

        ## must numpy shape(n*2)
        ## custom_spot_loc and spot_diameter
        spot_loc = custom_spot_loc

        cellandspot_loc =  np.vstack((cell_location,spot_loc ))
        sn = get_spaital_network(Num_sample=cellandspot_loc.shape[0],spatial=cellandspot_loc,
                         coord_type = "generic",n_rings = 1,set_diag=False,
                                 radius = spot_diameter/2)

        spot_cell_idx_matrix = sn[cell_location.shape[0]:,cell_location.shape[0]]

        spot_expr = spot_cell_idx_matrix * sim_cell_expr.X

        onehot_ct = get_onehot_ct(init_assign=celltype_assignment)

        ## 注意有些spot里面可能没有cell
        spot_ct_count = spot_cell_idx_matrix * onehot_ct

        return spot_expr, spot_loc, spot_cell_idx_matrix 
