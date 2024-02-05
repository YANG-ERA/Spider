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

def STsim(Num_sample=None,
          Num_celltype = None,
          celltype_assignment=None,
          target_trans = None,
          T=1000,
          chain_len=100,
          error=1e2,
          tol=2e-2,
          decay=0.5,
          onehot_ct=None,
          nb_count=None,
          sn=None,
          swap_num=None,
          smallsample_max_iter=None,
          bigsample_max_iter=None):
    '''
        smallsample: <10000 and swap_num = 1
        bigsample: >=10000
    '''

    error_record = []
    iter_n = 0
    print("Refine cell type using Metropolis–Hastings algorithm.")
    print("Sample num"+":{}".format(Num_sample) )


    nb_freq = get_nb_freq(nb_count=nb_count, onehot_ct=onehot_ct)
    dist_before = sum(scipy.stats.entropy(target_trans, nb_freq))

    if Num_sample <= 10000:
        swap_num = 1
        max_iter = smallsample_max_iter
        criterion3 = True
        same_errlen = 250
        while error > tol and iter_n < max_iter and criterion3:

            for i in range(chain_len):

                before_ct_assign, before_onehot_ct, before_nb_count = celltype_assignment.copy(), onehot_ct.copy(), nb_count.copy()

                swap_i, swap_j = swap_ct(
                    celltype_assignment=celltype_assignment,
                    Num_celltype=Num_celltype,
                    swap_num=swap_num)

                celltype_assignment[swap_i[0]], celltype_assignment[
                    swap_j[0]] = celltype_assignment[
                        swap_j[0]], celltype_assignment[swap_i[0]]
                onehot_ct[swap_i[0]], onehot_ct[swap_j[0]] = onehot_ct[
                    swap_j[0]], onehot_ct[swap_i[0]]

                nb_count = get_swap_nb_count(nb_count=nb_count,
                                             swap_i=swap_i,
                                             swap_j=swap_j,
                                             sn=sn)
                # nb_count = sn * onehot_ct
                nb_freq = get_nb_freq(nb_count=nb_count, onehot_ct=onehot_ct)

                dist_after = sum(scipy.stats.entropy(target_trans, nb_freq))

                if 0 > (dist_before - dist_after):
                    #Reject the swap, so swap back.
                    p = math.exp((dist_before - dist_after) / T)
                    r = np.random.uniform(low=0, high=1)
                    if p < r:
                        celltype_assignment, nb_count, onehot_ct = before_ct_assign, before_nb_count, before_onehot_ct
                    else:
                        dist_before = dist_after
                else:
                    dist_before = dist_after
                iter_n += 1
        #     print(dist_before - dist_after)
            T *= decay
            if iter_n % 100 == 0:
                error = np.linalg.norm(nb_freq - target_trans)
                error_record.append(error)
            if iter_n % 1000 ==0:
                print("%5d iteration, error %.3f" % (iter_n, error))

            if len(error_record) > same_errlen:
                if len(set(error_record[-same_errlen:])) == 1:
                    criterion3 = False
    else:
        max_iter = bigsample_max_iter
        criterion3 = True
        while error > tol and iter_n < max_iter:

            for i in range(chain_len):

                #dist_before = sum(scipy.stats.entropy(target_trans,nb_freq))

                before_ct_assign, before_onehot_ct, before_nb_count = celltype_assignment.copy(), onehot_ct.copy(), nb_count.copy()

                swap_i, swap_j = swap_ct(
                    celltype_assignment=celltype_assignment,
                    Num_celltype=Num_celltype,
                    swap_num=swap_num)

                celltype_assignment[swap_i[0]], celltype_assignment[
                    swap_j[0]] = celltype_assignment[
                        swap_j[0]], celltype_assignment[swap_i[0]]
                onehot_ct[swap_i[0]], onehot_ct[swap_j[0]] = onehot_ct[
                    swap_j[0]], onehot_ct[swap_i[0]]

                nb_count = get_swap_nb_count(nb_count=nb_count,
                                             swap_i=swap_i,
                                             swap_j=swap_j,
                                             sn=sn)
                # nb_count = sn * onehot_ct
                nb_freq = get_nb_freq(nb_count=nb_count, onehot_ct=onehot_ct)

                dist_after = sum(scipy.stats.entropy(target_trans, nb_freq))

                if 0 > (dist_before - dist_after):
                    p = math.exp((dist_before - dist_after) / T)
                    r = np.random.uniform(low=0, high=1)
                    if p < r:
                        celltype_assignment, nb_count, onehot_ct = before_ct_assign, before_nb_count, before_onehot_ct
                    else:
                        dist_before = dist_after
                else:
                    dist_before = dist_after
                iter_n += 1
        #     print(dist_before - dist_after)
            T *= decay
            if iter_n % 500 == 0:
                error = np.linalg.norm(nb_freq - target_trans)
                print("%5d iteration, error %.3f" % (iter_n, error))

    return celltype_assignment