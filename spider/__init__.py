"""Spider: simulation tools for spatial transcriptomics data.

The package uses lazy loading at the top level so ``import spider`` stays fast
and does not require optional backends such as torch, scanpy, squidpy, napari,
or cellpose unless the corresponding feature is used.
"""

from __future__ import annotations

from importlib import import_module

__version__ = "1.2.0"

_LAZY_ATTRS = {
    "AnnealingConfig": "solver",
    "AnnealingResult": "solver",
    "DataLoader": "data_op",
    "KL_divergence": "data_op",
    "PSA": "simulate_10X",
    "PSA_worker": "simulate_10X",
    "RCTD_naive": "random_based_utils",
    "STRIDE_naive": "random_based_utils",
    "STsim": "Annealing",
    "Simulator": "joint_simulator",
    "SimDataLoader": "joint_simulator",
    "SpatialSimulation": "api",
    "addictive_freq": "core",
    "assemble_data_set": "random_based_utils",
    "attractive_freq": "core",
    "compute_bin": "random_based_utils",
    "continuous_multinomial": "random_generator",
    "downsample_cell": "random_based_utils",
    "downsample_matrix_by_cell": "random_based_utils",
    "embedding_reduce": "data_op",
    "enhance_loop": "enhance",
    "enhance_res": "enhance",
    "exclusive_freq": "core",
    "estimate_parameters": "solver",
    "estimate_parameters_by_zone": "solver",
    "estimate_parameters_from_adata": "api",
    "extract_loc": "simulate_10X",
    "extract_loc_3d": "simulate_10X",
    "generate_spot_level_data": "sim_expr",
    "generat_grid": "random_based_utils",
    "generat_grid2": "random_based_utils",
    "gen_lowpixel_grid": "enhance",
    "get_adjacency": "data_op",
    "get_adjacency_knearest": "data_op",
    "get_bin_edges": "random_based_utils",
    "get_ct_sample": "core",
    "get_gene_prior": "joint_simulator",
    "get_knearest_distance": "data_op",
    "get_mesh_counts": "simulate_10X",
    "get_mesh_counts_3d": "simulate_10X",
    "get_nb_freq": "core",
    "get_neighbourhood_count": "data_op",
    "get_nf_prior": "joint_simulator",
    "get_onehot_ct": "core",
    "get_sim_cell_level_expr": "sim_expr",
    "get_sim_spot_level_expr": "sim_naive",
    "get_spaital_network": "neighbors",
    "get_spatial_network": "neighbors",
    "get_swap_nb_count": "core",
    "get_trans": "utils",
    "init_ct": "core",
    "layer_cell_level_sim": "utils",
    "layer_spot_level_sim": "utils",
    "load_loader": "data_op",
    "load_simulation": "joint_simulator",
    "make_transition_matrix": "api",
    "mutate": "enhance",
    "mutate_cell": "enhance",
    "muti_circle": "utils",
    "muti_square": "utils",
    "multinomial_wrapper": "random_generator",
    "naive_cell_level_sim": "utils",
    "numba_histogram": "random_based_utils",
    "one_hot_vector": "data_op",
    "pca_reduce": "data_op",
    "plot_3d_cell_types": "utils",
    "plot_z_slices": "utils",
    "run_scsim": "run_scsim",
    "save_df": "run_scsim",
    "save_loader": "data_op",
    "save_simulation": "joint_simulator",
    "save_smfish": "data_op",
    "scsim": "scsim",
    "sim_naive_cell": "sim_naive",
    "sim_naive_spot": "sim_naive",
    "sim_naive_spot_splatter": "sim_naive",
    "simulate_cells": "api",
    "simulate_10X": "simulate_10X",
    "simulate_10X_3d": "simulate_10X",
    "slice_anndata_by_z": "utils",
    "solve_cell_types": "solver",
    "solve_cell_types_by_zone": "solver",
    "stereoscope_naive": "random_based_utils",
    "stripe_freq": "core",
    "swap_ct": "core",
    "tag2int": "data_op",
    "transition_frequency": "core",
    "tsne_reduce": "data_op",
    "valid_neighbourhood_frequency": "opt",
    "validate_proportions": "core",
    "validate_transition_matrix": "core",
}

__all__ = sorted(_LAZY_ATTRS) + ["__version__"]


def __getattr__(name):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module 'spider' has no attribute {name!r}")
    module = import_module(f".{_LAZY_ATTRS[name]}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
