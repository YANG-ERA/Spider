from .solver import AnnealingConfig, solve_cell_types

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

    max_iter = smallsample_max_iter if Num_sample <= 10000 else bigsample_max_iter
    if max_iter is None:
        max_iter = 80000 if Num_sample <= 10000 else 10000

    config = AnnealingConfig(
        max_iter=max_iter,
        temperature=T,
        cooling=decay,
        tol=tol,
        record_every=500 if Num_sample > 10000 else 250,
        verbose=True,
    )
    result = solve_cell_types(
        adjacency=sn,
        target_transition=target_trans,
        n_celltypes=Num_celltype,
        initial_labels=celltype_assignment,
        config=config,
    )
    return result.labels
