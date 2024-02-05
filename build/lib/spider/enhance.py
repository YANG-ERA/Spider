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

import numba




def mutate_cell(celltype_assignment = None, mutate_ct = None, mutate_toct = None,times = None):
    
    mutate_index = np.random.choice(np.where(celltype_assignment==mutate_ct)[0],size = times, replace=False )
    
    celltype_assignment[mutate_index] = mutate_toct
    
    return celltype_assignment
    

def mutate(celltype_assignment=None, Num_ct_sample=None):
    compare = np.bincount(celltype_assignment) - Num_ct_sample
    while compare.any(): 
        print("np.bincount(celltype_assignment):  ",np.bincount(celltype_assignment))
        print("Num_ct_sample: ",Num_ct_sample)
        print("compare: ",compare)
        mutate_ct = np.where(compare > 0)[0][0]
        mutate_toct = np.where(compare < 0)[0][0]
        celltype_assignment = mutate_cell(
            celltype_assignment=celltype_assignment,
            mutate_ct=mutate_ct,
            mutate_toct=mutate_toct,
            times=np.min([compare[mutate_ct], -compare[mutate_toct]]))
        compare = np.bincount(celltype_assignment) - Num_ct_sample
    return celltype_assignment

def gen_lowpixel_grid(Num_sample=None,
                       original_grid = None,
                      grid_row=None,
                      grid_col=None,
                      windows_row=None,
                      windows_col=None):
    '''
    Attention:
        original_grid is grid location in original resolution (n*2)
        when `original_grid` is not None, you must input grid_row and grid_col 
            grid_row : the number of grid by row
            grid_col : the number of grid by col
        If you don't have original_grid, you must input Num_sample
        Here, cells are arranged into a square grid according to the sample size,
        and the length of its sides is `np.sqrt(Num_sample).astype(int)`
    
    '''
    if grid_row is None and grid_col is None:
        grid_row = np.sqrt(Num_sample).astype(int)
        grid_col = grid_row
        
    a, b = np.meshgrid(np.arange(grid_row),
                       np.arange(grid_col),
                       indexing='ij')

    lra = a[np.arange(0, grid_row, windows_row
                      ), :len(np.arange(0, grid_col, windows_col))].reshape(-1)
    lrb = b[:len(np.arange(0, grid_row, windows_row)),
            np.arange(0, grid_col, windows_col)].reshape(-1)
    grid_row_num, grid_col_num = grid_row, grid_col
    lr_spatial_idx = np.vstack((lra, lrb)).T
    
    if original_grid is None:
        lr_spatial = lr_spatial_idx.copy() 
    else:
        original_grid_x = original_grid[:,0].reshape(grid_row,grid_col)
        original_grid_y = original_grid[:,1].reshape(grid_row,grid_col)
        lr_spatial_x = original_grid_x[lra,lrb].reshape(-1)
        lr_spatial_y = original_grid_y[lra,lrb].reshape(-1)
        lr_spatial = np.vstack((lr_spatial_x, lr_spatial_y)).T
    return lr_spatial_idx, lr_spatial, grid_row_num, grid_col_num



def enhance_res(Num_sample=None,
                original_grid=None,
                grid_row=None,
                grid_col=None,
                windows_row=None,
                windows_col=None,
                target_trans=None,
                Num_celltype=None,
                prior=None,
                T=None,
                lrct_assignment=None,
                swap_num=None,
                tol=None,
                smallsample_max_iter=None,
                bigsample_max_iter=None):
    '''
      如果只给定 Num_sample 那么默认是正方形网格，如果给定了grid_row and grid_col，则按照
      该方式生成。要注意Num_sample = grid_row * grid_col
      windows_row: 低分辨率下，划分网格的长度
      windows_col: 低分辨率下，划分网格的宽度
      T: 退火的初始温度
      lrct_assignment: 低分辨率下的网格点的细胞类型分配，注意是一个shape为grid_row,grid_col的矩阵
    '''

    ##选取低分率下的网格代表元
    lr_spatial_idx, lr_spatial, grid_row_num, grid_col_num = gen_lowpixel_grid(
	 original_grid=original_grid,
        Num_sample=Num_sample,
        grid_row=grid_row,
        grid_col=grid_col,
        windows_row=windows_row,
        windows_col=windows_col)

    
    if lrct_assignment is None:
        Num_ct_sample = get_ct_sample(Num_celltype=Num_celltype,
                                      Num_sample=lr_spatial.shape[0],
                                      prior=prior)
        celltype_assignment = init_ct(Num_celltype=Num_celltype,
                                      Num_ct_sample=Num_ct_sample)
        #print("lrct_None_Num_ct_sample:  ",Num_ct_sample)

    else:
        celltype_assignment = lrct_assignment[lr_spatial_idx[:, 0],
                                              lr_spatial_idx[:, 1]].reshape(-1)

    ## 要能整除！
    
    ##最后一次的时候，要考虑此时的ct比例，要与prior一致，所以要mutate
    if windows_row == 1 and windows_col == 1:
        Num_ct_sample = get_ct_sample(Num_celltype=Num_celltype,
                                      Num_sample=Num_sample,
                                      prior=prior)
        celltype_assignment = mutate(celltype_assignment=celltype_assignment,
                                     Num_ct_sample=Num_ct_sample)

    sn = get_spaital_network(Num_sample=lr_spatial.shape[0],
                             spatial=lr_spatial,
                             n_neighs=8)
    onehot_ct = get_onehot_ct(init_assign=celltype_assignment)
    nb_count = np.array(sn * onehot_ct, dtype=np.float32)

    resct_assignment = STsim(Num_sample=lr_spatial.shape[0],
                             Num_celltype=Num_celltype,
                             celltype_assignment=celltype_assignment,
                             target_trans=target_trans,
                             T=T,
                             chain_len=100,
                             error=1e2,
                             onehot_ct=onehot_ct,
                             tol=tol,
                             decay=0.5,
                             nb_count=nb_count,
                             sn=sn,
                             swap_num=swap_num,
                             smallsample_max_iter=smallsample_max_iter,
                             bigsample_max_iter=bigsample_max_iter)
    print("grid_row:",grid_row_num)
    print("windows_row:",windows_row)

    ct_grid = np.array(resct_assignment).reshape(
        len(np.arange(0, grid_row_num, windows_row)),
        len(np.arange(0, grid_col_num, windows_col)))
    ct_grid = np.repeat(np.repeat(ct_grid, windows_row, axis=0),
                        windows_col,
                        axis=1)

    return ct_grid, lr_spatial

def enhance_loop(Num_sample=None,
                 Num_celltype=None,
                 prior=None,
                 target_trans=None,
                 original_grid=None,
                 celltype_assignment = None,
                 grid_row=None,
                 grid_col=None,
                 loop_times=None,
                 windows_row_list=None,
                 windows_col_list=None,
                 swap_num_list=None,
                 tol_list=None,
                 T=1e-5,
                 smallsample_max_iter=80000,
                 bigsample_max_iter=10000):
    '''
    Attention: 我们可以基于样本量推荐一个loop_times,
                小样本许多参数是固定的，you can see `STsim` function
                验证用户填写的list是否符合规则
                
    
    '''
    if Num_sample <=10000: loop_times  = 1
    if loop_times is None:
        loop_times = np.round(np.log2(Num_sample / 1000) / 2).astype(int) + 1

    ##需要注意 不同分辨率窗口的row 与col 彼此之间要整除
    if windows_col_list is None and windows_row_list is None:
        windows_row_list = [2**i for i in range(loop_times)]
        windows_row_list.reverse()
        windows_col_list = [2**i for i in range(loop_times)]
        windows_col_list.reverse()

    ## 默认小于10000，swap_num = 1，并且要与windows对应
    if swap_num_list is None:
        swap_num_list = [2**i for i in range(1, loop_times + 1)]

    if tol_list is None:
        tol_list = [2e-2] * loop_times
    
   # celltype_assignment = None
    for i in np.arange(loop_times):
        celltype_assignment, lr_spatial = enhance_res(
                Num_sample=Num_sample,
                original_grid=original_grid,
                grid_row=grid_row,
                grid_col=grid_col,
                windows_row=windows_row_list[i],
                windows_col=windows_col_list[i],
                target_trans=target_trans,
                Num_celltype=Num_celltype,
                prior=prior,
                T=1e-5,
                lrct_assignment=celltype_assignment,
                swap_num=swap_num_list[i],
                tol=tol_list[i],
                smallsample_max_iter=smallsample_max_iter,
                bigsample_max_iter=bigsample_max_iter)
    return celltype_assignment, lr_spatial, loop_times