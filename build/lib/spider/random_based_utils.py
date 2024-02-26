import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import squidpy as sq

import scipy
import random 
import seaborn as sns
import math
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix

import os
import os.path as osp
import sys
from collections import Counter

import numba
import multiprocessing as mp
from functools import partial
import torch as t
import torch.distributions as dists
import argparse as arp
from typing import Dict,Callable,List


###############################################
#RCTD
def assign_random_cellnum(cellnum,celtypenum):
    node = list(np.random.choice(np.arange(cellnum),(celtypenum-1),replace = True))  #改为True
    node.append(cellnum)
    node.sort() #n个节点排序
    b=np.zeros(celtypenum,dtype="int32")
    for i in range(celtypenum):
        if i == 0:
            b[i] = node[i]
        else:
            b[i] = node[i]-node[i-1]      
    return b

def generat_grid(row=50, col=50):
    x = np.linspace(0,100,row)
    y = np.linspace(0,100,col)
    x1, x2 = np.meshgrid(x, y)
    spatial = np.array([x1.ravel(), x2.ravel()]).T  # coordinate of x and y
    return x1,x2, spatial

def RCTD_naive(scdata, row=50, col=50, Min = 5, Max = 10, ctkey = 'celltype', maxctnum = 3):
    '''
    scdata: single-cell data
    Num_sample: the number of spot. can be sqrt.
    Min/Max: min/max cell number in each spot
    ctkey: celltype key in adata
    maxctnum: celltype number in each spot
    '''
    #spatial coor
    x1,x2,spatial = generat_grid(row, col) #空间坐标
    
    Num_sample=row * col  #spot数
    Num_pool = scdata.X.shape[0]  #单细胞数 (修改)
    Num_gene = scdata.X.shape[1]  #gene数 （修改）
    celltypes = scdata.obs[ctkey] #各cell的ct
    #Allcelltype = np.array(scdata.obs.CellType.cat.categories) #提取celltype name
    Allcelltype = scdata.obs[ctkey].unique() #提取celltype name
    Num_celltype = Allcelltype.shape[0] #celltype数
    #onehot
    onehot_encoder = OneHotEncoder(sparse_output=True)
    integer_encoded = np.array(celltypes).reshape(len(celltypes), 1)
    cell2celltype = onehot_encoder.fit_transform(integer_encoded)
    
    #assigncelltospot
    
    cellnum = np.random.randint(Min, Max,Num_sample)
    ####
    if Max == 1:
        cellnum = np.ones(Num_sample)
    eachcellnum = np.zeros((Num_sample,maxctnum),dtype = "int32")
    indices = []

    for i in range(Num_sample):
        
        if Max == 1:
            eachcellnum[i,np.random.randint(0,maxctnum,1)] = 1
        else:
            eachcellnum[i,:] = assign_random_cellnum(cellnum[i], maxctnum)
        if Num_celltype>maxctnum:
            celltype = np.random.choice(Allcelltype,maxctnum,replace = True) #抽3个不重复ct  修改为True
        else:
            celltype = Allcelltype
        for j in range(min(maxctnum,celltype.shape[0])):
            choicepool = np.where((scdata.obs[ctkey] == celltype[j]))[0]  #3个ct对应cell的序号
            tempchoice = np.random.choice(choicepool, eachcellnum[i,j],replace = True)
            indices = np.concatenate((indices,tempchoice))

    indptr = np.concatenate(([0],np.cumsum(cellnum)))
    elements = np.ones(indices.shape[0])
    # 矩阵中第i行非零元素的列号为indices[indptr[i]:indptr[i+1]]，相应的值为elements[indptr[i]:indptr[i+1]]
    spots2cell = csr_matrix((elements, indices, indptr), shape=(Num_sample, Num_pool))
    # 比例矩阵
    W = spots2cell @ cell2celltype
    # 每个spot的表达矩阵
    exp_spots2gene = spots2cell @ scdata.X
    return W, exp_spots2gene,spatial,spots2cell

@numba.jit(nopython=True, parallel=True)
def get_bin_edges(a, bins):
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in numba.prange(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges

# Modified from https://github.com/numba/numba-examples
@numba.jit(nopython=True, parallel=False)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_max = bin_edges[-1]
    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin
    bin = np.searchsorted(bin_edges, x)-1
    if bin < 0 or bin >= n:
        return None
    else:
        return bin
    
# Modified from https://github.com/numba/numba-examples
@numba.jit(nopython=True, parallel=False)
def numba_histogram(a, bin_edges):
    hist = np.zeros((bin_edges.shape[0] - 1,), dtype=np.intp)
    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1
    return hist, bin_edges

# Modified from https://github.com/numba/numba-examples
# Modified from https://rdrr.io/bioc/scRecover/src/R/countsSampling.R
# Downsample cell reads to a fraction
@numba.jit(nopython=True, parallel=True)
def downsample_cell(cell_counts,max_counts):
    readsGet = np.sort(np.random.choice(np.arange(np.sum(cell_counts)), np.intp(max_counts), replace=False))
    cumCounts = np.concatenate((np.array([0]),np.cumsum(cell_counts)))
    counts_new = numba_histogram(readsGet,cumCounts)[0]
    counts_new = counts_new.astype(np.float32)
    return counts_new

def downsample_matrix_by_cell(matrix,max_counts,n_cpus=None):
    if n_cpus is not None:
        with mp.Pool(16) as p:
            matrix_ds = p.map(partial(downsample_cell,max_counts), matrix)
    else:
        matrix_ds = [downsample_cell(i,max_counts) if i.sum() > max_counts else i for i in matrix ]
    return np.array(matrix_ds)

def generat_grid2(locations = [50,50]):
    x = np.linspace(0,100,locations[0])
    y = np.linspace(0,100,locations[1])
    x1, x2 = np.meshgrid(x, y)
    spatial = np.array([x1.ravel(), x2.ravel()]).T  # coordinate of x and y
    return x1,x2, spatial
#################################################
#STRIDE
def STRIDE_naive(scdata,celltype_keys,n_spots,min_cells=2,max_cells=10):
    '''
    params:
    scdata: AnnData
    celltype_keys: .obs['celltype_keys']
    n_spots: n*n
    '''
    n=np.sqrt(n_spots).astype(int)
    x1,x2,spatial = generat_grid2([n,n])
    n_cells = scdata.X.shape[0]
    celltypes = scdata.obs[celltype_keys]
    # 将每个细胞的细胞类型转为onehot
    n_celltypes = celltypes.unique().shape[0]
    onehot_encoder = OneHotEncoder(sparse_output=True)
    integer_encoded = np.array(celltypes).reshape(len(celltypes), 1)
    cell2celltype = onehot_encoder.fit_transform(integer_encoded)

    #############
    # 每个spot的包含的细胞
    n_spots_cell = np.random.randint(min_cells,max_cells,n_spots)
    indptr = np.concatenate(([0],np.cumsum(n_spots_cell)))
    indices = np.random.choice(n_cells,indptr[-1], replace=True)  #改成True
    elements = np.ones(indices.shape[0])
    # 矩阵中第i行非零元素的列号为indices[indptr[i]:indptr[i+1]]，相应的值为elements[indptr[i]:indptr[i+1]]
    spots2cell = csr_matrix((elements, indices, indptr), shape=(n_spots, n_cells))
    # 比例矩阵
    W = spots2cell @ cell2celltype
    # 每个spot的表达矩阵
    exp_spots2gene = spots2cell @ scdata.X #X 改成 scdata.X
    return W, exp_spots2gene,spatial
########################################################
#stereoscope
def _assemble_spot(cnt : np.ndarray,
                  labels : np.ndarray,
                  alpha : float = 1.0,
                  fraction : float = 1.0,  #修改处
                  bounds : List[int] = [10,30],
                  )->Dict[str,t.Tensor]:

    """Assemble single spot

    generates one synthetic ST-spot
    from provided single cell data

    Parameter:
    ---------
    cnt : np.ndarray
        single cell count data [n_cells x n_genes]
    labels : np.ndarray
        single cell annotations [n_cells]
    alpha : float
        dirichlet distribution
        concentration value
    fraction : float
        fraction of transcripts from each cell
        being observed in ST-spot

    Returns:
    -------
    Dictionary with expression data,
    proportion values and number of
    cells from each type at every
    spot

    """

    # sample between 10 to 30 cells to be present
    # at spot
    n_cells = dists.uniform.Uniform(low = bounds[0],
                                    high = bounds[1]).sample().round().type(t.int)
    n_cells = 1   ####修改
    # get unique labels found in single cell data
    uni_labs, uni_counts = np.unique(labels,
                                     return_counts = True)

    # make sure sufficient number
    # of cells are present within
    # all cell types
    assert np.all(uni_counts >=  30), \
            "Insufficient number of cells"

    # get number of different
    # cell types present
    n_labels = uni_labs.shape[0]

    # sample number of types to
    # be present at current spot
    n_types = dists.uniform.Uniform(low = 1,
                                    high =  n_labels).sample()

    n_types = n_types.round().type(t.int)
    n_types = 1    #####修改
    # select which types to include
    pick_types = t.randperm(n_labels)[0:n_types]
    # pick at least one cell for spot
    members = t.zeros(n_labels).type(t.float)
    while members.sum() < 1:
        # draw proportion values from probability simplex
        member_props = dists.Dirichlet(concentration = alpha * t.ones(n_types)).sample()
        # get integer number of cells based on proportions
        members[pick_types] = (n_cells * member_props).round()

    # get proportion of each type
    props = members / members.sum()
    # convert to ints
    members = members.type(t.int)
    # get number of cells from each cell type

    # generate spot expression data
    spot_expr = t.zeros(cnt.shape[1]).type(t.float32)

    for z in range(n_types):
        # get indices of selected type
        idx = np.where(labels == uni_labs[pick_types[z]])[0]
        # pick random cells from type
        np.random.shuffle(idx)
        idx = idx[0:members[pick_types[z]]]
        # add fraction of transcripts to spot expression
        spot_expr +=  t.tensor((cnt[idx,:]*fraction).sum(axis = 0).round().astype(np.float32))


    return {'expr':spot_expr,
            'proportions':props,
            'members': members,
           }

def assemble_data_set(cnt : pd.DataFrame,
                      labels : pd.DataFrame,
                      n_spots : int,
                      n_genes : int,
                      n_cell_range : List[int],
                      assemble_fun : Callable = _assemble_spot,
                     )-> Dict[str,pd.DataFrame]:

    """Assemble Synthetic ST Data Set

    Assemble synthetic ST Data Set from
    a provided single cell data set

    Parameters:
    ----------
    cnt : pd.DataFrame
        single cell count data
    labels : pd.DataFrame
        single cell annotations
    n_spots : int
        number of spots to generate
    n_genes : int
        number of gens to include
    assemble_fun : Callable
        function to assemble single spot

    """

    # get labels
    ##labels = labels.loc[:,'bio_celltype']
    labels = labels.loc[:,labels.columns[0]]
    # make sure number of genes does not
    # exceed number of genes present
    n_genes = np.min((cnt.shape[1],n_genes))
    # select top expressed genes
    keep_genes = np.argsort(cnt.sum(axis=0))[::-1]
    keep_genes = keep_genes[0:n_genes]
    cnt = cnt.iloc[:,keep_genes]

    # get unique labels
    uni_labels = np.unique(labels.values)
    n_labels = uni_labels.shape[0]

    # prepare matrices
    st_cnt = np.zeros((n_spots,cnt.shape[1]))
    st_prop = np.zeros((n_spots,n_labels))
    st_memb = np.zeros((n_spots,n_labels))

    #np.random.seed(1337)
    #t.manual_seed(1337)
    # generate one spot at a time
    for spot in range(n_spots):
        spot_data = assemble_fun(cnt.values,
                                 labels.values,
                                 bounds = n_cell_range,
                                 )

        st_cnt[spot,:] = spot_data['expr']
        st_prop[spot,:] = spot_data['proportions']
        st_memb[spot,:] =  spot_data['members']

        index = pd.Index(['Spotx' + str(x + 1) for \
                          x in range(n_spots) ])

    # convert to pandas DataFrames
    st_cnt = pd.DataFrame(st_cnt,
                          index = index,
                          columns = cnt.columns,
                         )

    st_prop = pd.DataFrame(st_prop,
                           index = index,
                           columns = uni_labels,
                          )
    st_memb = pd.DataFrame(st_memb,
                           index = index,
                           columns = uni_labels,
                           )


    return {'counts':st_cnt,
            'proportions':st_prop,
            'members':st_memb}


def stereoscope_naive(sc_cnt_pth = None, sc_lbl_pth = None, out_dir = None, data = None):
    n_spots = len(data)
    n_genes = data.X.shape[1]
    # read data
    sc_cnt = pd.read_csv(sc_cnt_pth,
                         #sep = '\t',
                         index_col = 0,
                         header = 0)

    sc_lbl = pd.read_csv(sc_lbl_pth,
                         #sep = '\t',
                         index_col = 0,
                         header = 0)
    # match count and label data
    inter = sc_lbl.index
    labels = sc_lbl.iloc[:,0].values
    # get unique labels
    uni_labs, uni_counts = np.unique(labels, return_counts = True)
    # only keep types with more than 2*30 cells
    keep_types = uni_counts > 30*2
    keep_cells = np.isin(labels, uni_labs[keep_types])
    labels = labels[keep_cells]
    sc_cnt = sc_cnt.iloc[keep_cells,:]
    sc_lbl = sc_lbl.iloc[keep_cells,:]
    uni_labs, uni_counts = np.unique(labels, return_counts = True)
    n_types = uni_labs.shape[0]
    assert np.all(uni_counts > 2), \
                "Only one cell in types"

    # get member indices for each set
    idx_generation = []
    idx_validation = []
    #np.random.seed(1337)
    for z in range(n_types):
        tmp_idx = np.where(labels == uni_labs[z])[0]
        n_generation = int(round(tmp_idx.shape[0] / 2 ))

        idx_generation += tmp_idx[0:n_generation].tolist()
        idx_validation += tmp_idx[n_generation::].tolist()
    idx_generation.sort()
    idx_validation.sort()
    # make sure no members overlap between sets
    assert len(set(idx_generation).intersection(set(idx_validation))) == 0, \
            "validation and genreation set are not orthogonal"
    # assemble sets from indices
    cnt_validation = sc_cnt.iloc[idx_validation,:]
    cnt_generation = sc_cnt.iloc[idx_generation,:]

    lbl_validation = sc_lbl.iloc[idx_validation,:]
    lbl_generation = sc_lbl.iloc[idx_generation,:]

    # save sets
    cnt_validation.to_csv(osp.join(out_dir,'.'.join(['validation',osp.basename(sc_cnt_pth)])),
                          #sep = '\t',
                          header = True,
                          index = True,
                          index_label = 'cell',)

    cnt_generation.to_csv(osp.join(out_dir,'.'.join(['generation',osp.basename(sc_cnt_pth)])),
                          #sep = '\t',
                          header = True,
                          index = True,
                          index_label = 'cell')

    lbl_validation.to_csv(osp.join(out_dir,'.'.join(['validation',osp.basename(sc_lbl_pth)])),
                          #sep = '\t',
                          header = True,
                          index = True,
                          index_label = 'cell')

    lbl_generation.to_csv(osp.join(out_dir,'.'.join(['generation',osp.basename(sc_lbl_pth)])),
                          #sep = '\t',
                          header = True,
                          index = True,
                          index_label = 'cell')
    
    sc_cnt = pd.read_csv(sc_cnt_pth,
                     #sep = '\t',
                     index_col = 0,
                     header = 0)

    sc_lbl = pd.read_csv(sc_lbl_pth,
                     #sep = '\t',
                     index_col = 0,
                     header = 0)
    assembled_set = assemble_data_set(sc_cnt,
                                  sc_lbl,
                                  n_spots = n_spots,
                                  n_genes = n_genes,
                                  n_cell_range = [0,1],
                                  assemble_fun = _assemble_spot,
                                  )
    for k,v in assembled_set.items() :
        out_pth = osp.join(out_dir, '.'.join([k,'st_synth','tsv']))
        v.to_csv(out_pth,
                    #sep = '\t',
                    index = True,
                    header = True,
                   )
    return None		
