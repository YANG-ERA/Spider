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


import numba
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
def get_ct_sample(Num_celltype = None, Num_sample = None, prior = None):
    if prior is None:
        prior = np.ones(Num_celltype)/Num_celltype
    else:
        assert len(prior) == Num_celltype, "please check your prior length, that must be equal to the number of celltype"
        assert sum(prior)<= 1, "Sum of prior is out of range!"
        assert 1-sum(prior) <1e-3,"Sum of prior is equal to 1"
    
    Num_ct_sample = np.zeros(Num_celltype,dtype = np.int32)
    Num_ct_sample[0:(Num_celltype-1)] = (prior[0:(Num_celltype-1)]*Num_sample+0.5*np.ones(Num_celltype-1)).astype(int)
    Num_ct_sample[Num_celltype-1] = Num_sample-np.sum(Num_ct_sample[0:(Num_celltype-1)])
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

def get_onehot_ct(init_assign = None):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(init_assign)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_ct = onehot_encoder.fit_transform(integer_encoded)
    return onehot_ct.astype(np.float32)

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



