"""Spatial neighbor graph construction for Spider.

The original implementation delegated this job to Squidpy. For core
simulation, a lightweight sklearn/scipy implementation is enough and keeps
Squidpy available as an optional image/spatial-analysis dependency.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def get_spatial_network(
    Num_sample=None,
    spatial=None,
    n_neighs=None,
    radius=None,
    coord_type="grid",
    n_rings=2,
    set_diag=False,
    metric="euclidean",
):
    """Construct a sparse adjacency matrix from spatial coordinates."""

    if spatial is None:
        raise ValueError("spatial coordinates are required.")

    coords = np.asarray(spatial, dtype=float)
    if coords.ndim != 2:
        raise ValueError("spatial must be a 2D array of coordinates.")

    n_obs = coords.shape[0] if Num_sample is None else int(Num_sample)
    if coords.shape[0] != n_obs:
        raise ValueError("Num_sample must match spatial.shape[0].")
    if n_obs == 0:
        return csr_matrix((0, 0), dtype=np.float32)

    if radius is not None:
        model = NearestNeighbors(radius=radius, metric=metric)
        model.fit(coords)
        adjacency = model.radius_neighbors_graph(coords, mode="connectivity")
    else:
        if n_neighs is None:
            n_neighs = max(1, 8 * int(n_rings or 1)) if coord_type == "grid" else 8
        n_neighbors = min(int(n_neighs) + 1, n_obs)
        model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        model.fit(coords)
        adjacency = model.kneighbors_graph(coords, mode="connectivity")

    adjacency = adjacency.tolil()
    adjacency.setdiag(1 if set_diag else 0)
    adjacency = adjacency.tocsr()
    adjacency.eliminate_zeros()
    return adjacency.astype(np.float32)


def get_spaital_network(*args, **kwargs):
    """Backward-compatible alias for the historical misspelled function name."""

    return get_spatial_network(*args, **kwargs)
