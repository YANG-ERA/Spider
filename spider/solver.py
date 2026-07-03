"""Incremental solvers for Spider's cell-type assignment problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .core import get_ct_sample, init_ct, validate_transition_matrix


@dataclass
class AnnealingConfig:
    """Configuration for incremental simulated annealing."""

    max_iter: int = 50_000
    temperature: float = 0.05
    cooling: float = 0.995
    min_temperature: float = 1e-6
    tol: float = 2e-2
    patience: int = 50
    record_every: int = 250
    metric: str = "frobenius"
    random_state: int | None = None
    verbose: bool = False


@dataclass
class AnnealingResult:
    """Result returned by the incremental annealing solver."""

    labels: np.ndarray
    loss: float
    transition: np.ndarray
    counts: np.ndarray
    history: list[dict[str, float]]
    n_iter: int
    accepted: int
    converged: bool

    @property
    def acceptance_rate(self) -> float:
        return 0.0 if self.n_iter == 0 else self.accepted / self.n_iter


@dataclass
class EstimatedParameters:
    """Cell-type proportions and transition matrix estimated from labels."""

    prior: np.ndarray
    transition: np.ndarray
    labels: np.ndarray
    celltype_names: np.ndarray


def _adjacency_to_neighbors(adjacency) -> list[np.ndarray]:
    if isinstance(adjacency, (list, tuple)):
        neighbors = [np.asarray(x, dtype=np.int64) for x in adjacency]
        if all(x.ndim == 1 for x in neighbors):
            return neighbors

    if hasattr(adjacency, "indptr") and hasattr(adjacency, "indices"):
        indptr = np.asarray(adjacency.indptr)
        indices = np.asarray(adjacency.indices)
        return [indices[indptr[i] : indptr[i + 1]].astype(np.int64, copy=False) for i in range(len(indptr) - 1)]

    arr = np.asarray(adjacency)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("adjacency must be a square dense array or scipy sparse CSR-like matrix.")
    return [np.flatnonzero(arr[i]).astype(np.int64, copy=False) for i in range(arr.shape[0])]


def _incoming_neighbors(out_neighbors: Sequence[np.ndarray], n_cells: int) -> list[np.ndarray]:
    incoming = [[] for _ in range(n_cells)]
    for src, targets in enumerate(out_neighbors):
        for dst in targets:
            incoming[int(dst)].append(src)
    return [np.asarray(x, dtype=np.int64) for x in incoming]


def _transition_counts(labels: np.ndarray, out_neighbors: Sequence[np.ndarray], n_celltypes: int) -> np.ndarray:
    counts = np.zeros((n_celltypes, n_celltypes), dtype=np.float64)
    for src, targets in enumerate(out_neighbors):
        src_label = labels[src]
        if len(targets):
            target_labels = labels[targets]
            counts[src_label] += np.bincount(target_labels, minlength=n_celltypes)
    return counts


def _normalize_counts(counts: np.ndarray) -> np.ndarray:
    row_sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=np.float64), where=row_sums != 0)


def _loss(counts: np.ndarray, target: np.ndarray, metric: str) -> float:
    observed = _normalize_counts(counts)
    if metric == "frobenius":
        return float(np.linalg.norm(observed - target))
    if metric == "sq_frobenius":
        diff = observed - target
        return float(np.sum(diff * diff))
    if metric == "kl":
        eps = 1e-12
        obs = np.clip(observed, eps, 1.0)
        tgt = np.clip(target, eps, 1.0)
        return float(np.sum(tgt * np.log(tgt / obs)))
    raise ValueError("metric must be one of {'frobenius', 'sq_frobenius', 'kl'}.")


def _affected_edges(
    i: int,
    j: int,
    out_neighbors: Sequence[np.ndarray],
    in_neighbors: Sequence[np.ndarray],
) -> list[tuple[int, int]]:
    edges = set()
    for dst in out_neighbors[i]:
        edges.add((i, int(dst)))
    for dst in out_neighbors[j]:
        edges.add((j, int(dst)))
    for src in in_neighbors[i]:
        edges.add((int(src), i))
    for src in in_neighbors[j]:
        edges.add((int(src), j))
    return list(edges)


def _proposal_counts(
    counts: np.ndarray,
    labels: np.ndarray,
    i: int,
    j: int,
    out_neighbors: Sequence[np.ndarray],
    in_neighbors: Sequence[np.ndarray],
) -> np.ndarray:
    new_counts = counts.copy()
    edges = _affected_edges(i, j, out_neighbors, in_neighbors)
    for src, dst in edges:
        new_counts[labels[src], labels[dst]] -= 1

    li, lj = labels[i], labels[j]
    labels[i], labels[j] = lj, li
    for src, dst in edges:
        new_counts[labels[src], labels[dst]] += 1
    labels[i], labels[j] = li, lj
    return new_counts


def _member_index(labels: np.ndarray, n_celltypes: int):
    members = [np.flatnonzero(labels == k).astype(np.int64).tolist() for k in range(n_celltypes)]
    positions = np.empty(labels.size, dtype=np.int64)
    for label, cells in enumerate(members):
        for pos, cell in enumerate(cells):
            positions[cell] = pos
    return members, positions


def _sample_swap(members: list[list[int]], rng: np.random.Generator) -> tuple[int, int, int, int]:
    represented = np.asarray([idx for idx, cells in enumerate(members) if cells], dtype=np.int64)
    if represented.size < 2:
        raise ValueError("at least two represented cell types are required.")
    a, b = rng.choice(represented, size=2, replace=False)
    pos_i = int(rng.integers(len(members[int(a)])))
    pos_j = int(rng.integers(len(members[int(b)])))
    return int(a), int(b), pos_i, pos_j


def _commit_swap(
    labels: np.ndarray,
    members: list[list[int]],
    positions: np.ndarray,
    a: int,
    b: int,
    pos_i: int,
    pos_j: int,
) -> tuple[int, int]:
    i = members[a][pos_i]
    j = members[b][pos_j]
    members[a][pos_i] = j
    members[b][pos_j] = i
    positions[i] = pos_j
    positions[j] = pos_i
    labels[i], labels[j] = labels[j], labels[i]
    return i, j


def solve_cell_types(
    adjacency,
    target_transition,
    n_celltypes: int | None = None,
    prior: Iterable[float] | None = None,
    n_cells: int | None = None,
    initial_labels: Iterable[int] | None = None,
    config: AnnealingConfig | None = None,
) -> AnnealingResult:
    """Assign cell types so neighborhood transitions approach a target matrix.

    The solver preserves exact cell-type counts by proposing label swaps rather
    than independent reassignments. Unlike the legacy solver, each proposal only
    updates transition counts for affected incoming/outgoing edges.
    """

    cfg = config or AnnealingConfig()
    target = validate_transition_matrix(target_transition, n_celltypes)
    if n_celltypes is None:
        n_celltypes = target.shape[0]

    out_neighbors = _adjacency_to_neighbors(adjacency)
    inferred_n = len(out_neighbors)
    if n_cells is None:
        n_cells = inferred_n
    if n_cells != inferred_n:
        raise ValueError("n_cells must match adjacency size.")

    rng = np.random.default_rng(cfg.random_state)
    if initial_labels is None:
        counts_by_type = get_ct_sample(Num_celltype=n_celltypes, Num_sample=n_cells, prior=prior)
        labels = init_ct(Num_celltype=n_celltypes, Num_ct_sample=counts_by_type, seed=cfg.random_state).astype(np.int64)
    else:
        labels = np.asarray(initial_labels, dtype=np.int64).copy()
        if labels.ndim != 1 or labels.size != n_cells:
            raise ValueError("initial_labels must be a 1D array with one label per cell.")
        if labels.min(initial=0) < 0 or labels.max(initial=0) >= n_celltypes:
            raise ValueError("initial_labels contain labels outside [0, n_celltypes).")

    in_neighbors = _incoming_neighbors(out_neighbors, n_cells)
    counts = _transition_counts(labels, out_neighbors, n_celltypes)
    current_loss = _loss(counts, target, cfg.metric)
    best_labels = labels.copy()
    best_counts = counts.copy()
    best_loss = current_loss
    members, positions = _member_index(labels, n_celltypes)

    temp = float(cfg.temperature)
    accepted = 0
    stale_records = 0
    converged = best_loss <= cfg.tol
    history: list[dict[str, float]] = [
        {"iteration": 0.0, "loss": current_loss, "best_loss": best_loss, "temperature": temp}
    ]

    for iteration in range(1, int(cfg.max_iter) + 1):
        a, b, pos_i, pos_j = _sample_swap(members, rng)
        i = members[a][pos_i]
        j = members[b][pos_j]
        proposed_counts = _proposal_counts(counts, labels, i, j, out_neighbors, in_neighbors)
        proposed_loss = _loss(proposed_counts, target, cfg.metric)
        delta = proposed_loss - current_loss

        if delta <= 0 or rng.random() < np.exp(-delta / max(temp, cfg.min_temperature)):
            _commit_swap(labels, members, positions, a, b, pos_i, pos_j)
            counts = proposed_counts
            current_loss = proposed_loss
            accepted += 1
            if current_loss < best_loss:
                best_loss = current_loss
                best_labels = labels.copy()
                best_counts = counts.copy()
                stale_records = 0

        temp = max(temp * cfg.cooling, cfg.min_temperature)

        if best_loss <= cfg.tol:
            converged = True
            history.append(
                {
                    "iteration": float(iteration),
                    "loss": current_loss,
                    "best_loss": best_loss,
                    "temperature": temp,
                }
            )
            break

        if cfg.record_every and iteration % cfg.record_every == 0:
            history.append(
                {
                    "iteration": float(iteration),
                    "loss": current_loss,
                    "best_loss": best_loss,
                    "temperature": temp,
                }
            )
            stale_records += 1
            if cfg.verbose:
                print(f"{iteration:7d} iteration, loss {current_loss:.4f}, best {best_loss:.4f}")
            if cfg.patience and stale_records >= cfg.patience:
                break

    return AnnealingResult(
        labels=best_labels,
        loss=best_loss,
        transition=_normalize_counts(best_counts),
        counts=best_counts,
        history=history,
        n_iter=iteration if "iteration" in locals() else 0,
        accepted=accepted,
        converged=converged,
    )


def solve_cell_types_by_zone(
    adjacency,
    zone_labels,
    target_transitions,
    n_celltypes: int,
    priors=None,
    initial_labels: Iterable[int] | None = None,
    config: AnnealingConfig | None = None,
) -> dict[str, object]:
    """Run the incremental solver separately inside spatial zones.

    This implements a spatially varying transition model: each zone can carry
    its own target transition matrix, addressing cases where boundary dynamics
    or tissue compartments should not share one global ``P``.
    """

    zones = np.asarray(zone_labels)
    out_neighbors = _adjacency_to_neighbors(adjacency)
    if zones.ndim != 1 or zones.size != len(out_neighbors):
        raise ValueError("zone_labels must contain one zone label per cell.")

    all_labels = np.full(zones.size, -1, dtype=np.int64)
    initial_arr = None if initial_labels is None else np.asarray(initial_labels, dtype=np.int64)
    zone_results = {}
    unique_zones = np.unique(zones)

    for zone in unique_zones:
        global_idx = np.flatnonzero(zones == zone)
        local_lookup = {int(cell): pos for pos, cell in enumerate(global_idx)}
        local_neighbors = []
        for cell in global_idx:
            local_neighbors.append(
                np.asarray(
                    [local_lookup[int(dst)] for dst in out_neighbors[int(cell)] if int(dst) in local_lookup],
                    dtype=np.int64,
                )
            )

        if isinstance(target_transitions, dict):
            target = target_transitions[zone]
        else:
            target = target_transitions

        if priors is None:
            prior = None
        elif isinstance(priors, dict):
            prior = priors[zone]
        else:
            prior = priors

        local_initial = None if initial_arr is None else initial_arr[global_idx]
        result = solve_cell_types(
            adjacency=local_neighbors,
            target_transition=target,
            n_celltypes=n_celltypes,
            prior=prior,
            n_cells=global_idx.size,
            initial_labels=local_initial,
            config=config,
        )
        all_labels[global_idx] = result.labels
        zone_results[zone] = result

    return {"labels": all_labels, "zone_results": zone_results}


def estimate_parameters(labels, adjacency, n_celltypes: int | None = None) -> EstimatedParameters:
    """Estimate prior proportions and transition matrix from labeled spatial data."""

    raw_labels = np.asarray(labels)
    names, encoded = np.unique(raw_labels, return_inverse=True)
    if n_celltypes is None:
        n_celltypes = names.size
    out_neighbors = _adjacency_to_neighbors(adjacency)
    if len(out_neighbors) != encoded.size:
        raise ValueError("adjacency size must match labels length.")

    counts = np.bincount(encoded, minlength=n_celltypes).astype(float)
    prior = counts / counts.sum()
    transition_counts = _transition_counts(encoded, out_neighbors, n_celltypes)
    transition = _normalize_counts(transition_counts)
    return EstimatedParameters(prior=prior, transition=transition, labels=encoded, celltype_names=names)


def estimate_parameters_by_zone(labels, adjacency, zone_labels) -> dict[object, EstimatedParameters]:
    """Estimate proportions and transition matrices independently by zone."""

    zones = np.asarray(zone_labels)
    raw_labels = np.asarray(labels)
    if zones.ndim != 1 or zones.size != raw_labels.size:
        raise ValueError("zone_labels must contain one zone label per cell.")

    out_neighbors = _adjacency_to_neighbors(adjacency)
    estimates = {}
    for zone in np.unique(zones):
        global_idx = np.flatnonzero(zones == zone)
        local_lookup = {int(cell): pos for pos, cell in enumerate(global_idx)}
        local_neighbors = []
        for cell in global_idx:
            local_neighbors.append(
                np.asarray(
                    [local_lookup[int(dst)] for dst in out_neighbors[int(cell)] if int(dst) in local_lookup],
                    dtype=np.int64,
                )
            )
        estimates[zone] = estimate_parameters(raw_labels[global_idx], local_neighbors)
    return estimates
