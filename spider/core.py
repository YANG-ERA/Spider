"""Core utilities for Spider's transition-matrix simulation model.

This module intentionally depends only on NumPy so it can be imported from the
top-level package without pulling in heavy spatial, plotting, or deep-learning
stacks. Higher-level modules can build on these helpers while keeping optional
methods optional.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def validate_proportions(prior: Iterable[float] | None, n_celltypes: int) -> np.ndarray:
    """Return a normalized cell-type proportion vector after validation."""

    if n_celltypes is None or n_celltypes <= 0:
        raise ValueError("n_celltypes must be a positive integer.")

    if prior is None:
        return np.full(n_celltypes, 1.0 / n_celltypes, dtype=float)

    prior_arr = np.asarray(prior, dtype=float)
    if prior_arr.ndim != 1 or prior_arr.size != n_celltypes:
        raise ValueError("prior must be a 1D array with length equal to n_celltypes.")
    if np.any(prior_arr < 0):
        raise ValueError("prior must not contain negative values.")

    total = prior_arr.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("prior must sum to a positive finite value.")
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError("prior must sum to 1.")

    return prior_arr


def validate_transition_matrix(matrix: Iterable[Iterable[float]], n_celltypes: int | None = None) -> np.ndarray:
    """Validate and return a row-stochastic transition matrix."""

    matrix_arr = np.asarray(matrix, dtype=float)
    if matrix_arr.ndim != 2 or matrix_arr.shape[0] != matrix_arr.shape[1]:
        raise ValueError("transition matrix must be a square 2D array.")
    if n_celltypes is not None and matrix_arr.shape != (n_celltypes, n_celltypes):
        raise ValueError("transition matrix shape must match n_celltypes.")
    if np.any(matrix_arr < 0):
        raise ValueError("transition matrix must not contain negative values.")

    row_sums = matrix_arr.sum(axis=1)
    if np.any(row_sums <= 0) or not np.all(np.isfinite(row_sums)):
        raise ValueError("each transition matrix row must have a positive finite sum.")
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("transition matrix rows must sum to 1.")

    return matrix_arr


def get_ct_sample(Num_celltype=None, Num_sample=None, prior=None):
    """Allocate an integer cell count for each type from prior proportions.

    The function preserves the historical Spider API name and argument spelling.
    It guarantees that counts sum to ``Num_sample`` and, when feasible, avoids
    zero-count cell types because the annealing swap step requires every type to
    be represented.
    """

    if Num_sample is None or Num_sample < 0:
        raise ValueError("Num_sample must be a non-negative integer.")
    prior_arr = validate_proportions(prior, Num_celltype)

    raw = prior_arr * int(Num_sample)
    counts = np.floor(raw).astype(np.int32)
    remainder = int(Num_sample) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        counts[order[:remainder]] += 1

    if Num_sample >= Num_celltype:
        zero_idx = np.where(counts == 0)[0]
        for idx in zero_idx:
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                break
            counts[donor] -= 1
            counts[idx] += 1

    return counts


def attractive_freq(n_c):
    """Predefined attractive pattern transition matrix."""

    target_freq = np.ones((n_c, n_c), dtype=float)
    for i in np.arange(n_c):
        target_freq[i, i] = 4 * (n_c - 1)
    return target_freq / np.sum(target_freq, axis=1, keepdims=True)


def addictive_freq(n_c):
    """Backward-compatible alias for the historical misspelled API."""

    return attractive_freq(n_c)


def exclusive_freq(n_c):
    """Predefined repulsive/exclusive pattern transition matrix."""

    target_freq = np.ones((n_c, n_c), dtype=float)
    for i in np.arange(n_c):
        target_freq[i, i] = 3 * (n_c - 1)
        if i % 2 == 1:
            target_freq[i - 1, i] = 3 * (n_c - 1)
            target_freq[i, i - 1] = 3 * (n_c - 1)
    return target_freq / np.sum(target_freq, axis=1, keepdims=True)


def stripe_freq(n_c):
    """Predefined layered/stripe pattern transition matrix."""

    target_freq = np.ones((n_c, n_c), dtype=float)
    for i in np.arange(n_c):
        target_freq[i, i] = 3 * (n_c - 1)
        if i > 0:
            target_freq[i - 1, i] = 3 * (n_c - 1)
    return target_freq / np.sum(target_freq, axis=1, keepdims=True)


def init_ct(Num_celltype=None, Num_ct_sample=None, seed=None):
    """Initialize a shuffled cell-type assignment with fixed type counts."""

    if Num_celltype is None:
        raise ValueError("Num_celltype is required.")
    counts = np.asarray(Num_ct_sample, dtype=np.int64)
    if counts.ndim != 1 or counts.size != Num_celltype:
        raise ValueError("Num_ct_sample must be a 1D array with length Num_celltype.")
    if np.any(counts < 0):
        raise ValueError("Num_ct_sample must not contain negative counts.")

    rng = np.random.default_rng(seed)
    init_assign = np.repeat(np.arange(Num_celltype), counts)
    rng.shuffle(init_assign)
    return init_assign


def get_onehot_ct(init_assign=None):
    """Return a dense one-hot matrix for a cell-type assignment vector."""

    labels = np.asarray(init_assign)
    if labels.ndim != 1:
        raise ValueError("init_assign must be a 1D array-like object.")
    if labels.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    unique_labels, encoded = np.unique(labels, return_inverse=True)
    onehot_ct = np.zeros((labels.size, unique_labels.size), dtype=np.float32)
    onehot_ct[np.arange(labels.size), encoded] = 1.0
    return onehot_ct


def get_nb_freq(nb_count=None, onehot_ct=None):
    """Compute row-normalized transition frequencies from neighbor counts."""

    nb_count_arr = np.asarray(nb_count, dtype=np.float32)
    onehot_arr = np.asarray(onehot_ct, dtype=np.float32)
    nb_freq = np.dot(onehot_arr.T, nb_count_arr)
    row_sums = nb_freq.sum(axis=1, keepdims=True)
    return np.divide(nb_freq, row_sums, out=np.zeros_like(nb_freq), where=row_sums != 0)


def transition_frequency(labels, adjacency):
    """Compute the empirical transition matrix for labels on an adjacency graph."""

    onehot_ct = get_onehot_ct(labels)
    nb_count = adjacency @ onehot_ct
    return get_nb_freq(nb_count=nb_count, onehot_ct=onehot_ct)


def swap_ct(celltype_assignment=None, Num_celltype=None, swap_num=None):
    """Sample two represented cell types and cell indices to swap."""

    if swap_num is None or swap_num <= 0:
        raise ValueError("swap_num must be a positive integer.")
    assignment = np.asarray(celltype_assignment)
    represented = [ct for ct in range(Num_celltype) if np.any(assignment == ct)]
    if len(represented) < 2:
        raise ValueError("at least two represented cell types are required for swapping.")

    swap_cluster = np.random.choice(np.asarray(represented), 2, replace=False)
    swap_i_pool = np.where(assignment == swap_cluster[0])[0]
    swap_j_pool = np.where(assignment == swap_cluster[1])[0]
    if swap_i_pool.size < swap_num or swap_j_pool.size < swap_num:
        raise ValueError("swap_num exceeds available cells in one selected cell type.")

    swap_i_index = np.random.choice(swap_i_pool, swap_num, replace=False)
    swap_j_index = np.random.choice(swap_j_pool, swap_num, replace=False)
    return (swap_i_index, swap_cluster[0]), (swap_j_index, swap_cluster[1])


def get_swap_nb_count(nb_count=None, swap_i=None, swap_j=None, sn=None):
    """Update neighbor-count matrix after swapping two groups of labels."""

    for idx in swap_i[0]:
        swap_i_nb_index = sn[idx].indices
        nb_count[swap_i_nb_index, swap_i[1]] -= 1
        nb_count[swap_i_nb_index, swap_j[1]] += 1

    for idx in swap_j[0]:
        swap_j_nb_index = sn[idx].indices
        nb_count[swap_j_nb_index, swap_j[1]] -= 1
        nb_count[swap_j_nb_index, swap_i[1]] += 1

    return nb_count
