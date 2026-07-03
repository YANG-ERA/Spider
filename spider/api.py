"""User-facing simulation API for Spider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .core import validate_proportions, validate_transition_matrix
from .solver import (
    AnnealingConfig,
    AnnealingResult,
    estimate_parameters,
    estimate_parameters_by_zone,
    solve_cell_types,
    solve_cell_types_by_zone,
)


@dataclass
class SpatialSimulation:
    """Container returned by :func:`simulate_cells`."""

    coordinates: np.ndarray
    labels: np.ndarray
    target_transition: np.ndarray
    observed_transition: np.ndarray
    prior: np.ndarray
    loss: float
    history: list[dict[str, float]]
    adjacency: object | None = None
    zone_results: dict[object, AnnealingResult] | None = None

    def summary(self) -> dict[str, object]:
        return {
            "n_cells": int(self.labels.size),
            "n_celltypes": int(self.target_transition.shape[0]),
            "loss": float(self.loss),
            "prior": self.prior.tolist(),
            "observed_transition": self.observed_transition.tolist(),
        }

    def to_anndata(self, label_key: str = "spider_label", spatial_key: str = "spatial"):
        """Convert labels and coordinates to an AnnData object."""

        try:
            import anndata as ad
            import pandas as pd
        except ImportError as exc:
            raise ImportError("to_anndata requires anndata and pandas.") from exc

        adata = ad.AnnData(np.empty((self.labels.size, 0)))
        adata.obs = pd.DataFrame({label_key: self.labels.astype(str)})
        adata.obsm[spatial_key] = self.coordinates
        adata.uns["spider"] = self.summary()
        return adata


def make_transition_matrix(
    pattern: str,
    n_celltypes: int,
    strength: float = 0.8,
    groups: Iterable[Iterable[int]] | None = None,
) -> np.ndarray:
    """Create a row-stochastic transition matrix from a named pattern."""

    if n_celltypes <= 0:
        raise ValueError("n_celltypes must be positive.")
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1.")

    pattern = pattern.lower()
    if n_celltypes == 1:
        return np.ones((1, 1), dtype=float)

    base = np.full((n_celltypes, n_celltypes), (1 - strength) / (n_celltypes - 1), dtype=float)

    if pattern in {"attractive", "clustered"}:
        np.fill_diagonal(base, strength)
    elif pattern in {"mixed", "random"}:
        base = np.full((n_celltypes, n_celltypes), 1.0 / n_celltypes, dtype=float)
    elif pattern in {"layered", "stripe", "gyrus"}:
        base = np.full((n_celltypes, n_celltypes), (1 - strength) / max(1, n_celltypes - 1), dtype=float)
        for i in range(n_celltypes):
            base[i, i] = strength
            if i > 0:
                base[i, i - 1] += (1 - strength) / 2
            if i < n_celltypes - 1:
                base[i, i + 1] += (1 - strength) / 2
        base = base / base.sum(axis=1, keepdims=True)
    elif pattern in {"repulsive", "exclusive"}:
        base = np.full((n_celltypes, n_celltypes), 1.0 / max(1, n_celltypes - 1), dtype=float)
        np.fill_diagonal(base, 0.0)
        if groups is not None:
            base *= 1 - strength
            for group in groups:
                group = list(group)
                for i in group:
                    for j in group:
                        if i != j:
                            base[i, j] += strength / max(1, len(group) - 1)
            base = base / base.sum(axis=1, keepdims=True)
    else:
        raise ValueError("pattern must be attractive, mixed, layered, gyrus, repulsive, or exclusive.")

    return validate_transition_matrix(base, n_celltypes)


def generate_coordinates(
    n_cells: int,
    dimensions: int = 2,
    plate_shape: Iterable[float] | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate uniform 2D or 3D cell coordinates."""

    if dimensions not in {2, 3}:
        raise ValueError("dimensions must be 2 or 3.")
    shape = np.ones(dimensions, dtype=float) if plate_shape is None else np.asarray(list(plate_shape), dtype=float)
    if shape.size != dimensions:
        raise ValueError("plate_shape length must match dimensions.")
    if np.any(shape <= 0):
        raise ValueError("plate_shape values must be positive.")

    rng = np.random.default_rng(random_state)
    return rng.random((int(n_cells), dimensions)) * shape


def simulate_cells(
    n_cells: int,
    n_celltypes: int | None = None,
    prior: Iterable[float] | None = None,
    transition: Iterable[Iterable[float]] | None = None,
    pattern: str = "attractive",
    coordinates: np.ndarray | None = None,
    adjacency=None,
    dimensions: int = 2,
    plate_shape: Iterable[float] | None = None,
    n_neighbors: int = 8,
    radius: float | None = None,
    metric: str = "euclidean",
    zone_labels: Iterable[object] | None = None,
    zone_transitions: dict[object, Iterable[Iterable[float]]] | None = None,
    config: AnnealingConfig | None = None,
    random_state: int | None = None,
) -> SpatialSimulation:
    """Simulate cell-type labels on spatial coordinates.

    This is the recommended high-level entry point for new code. It generates
    coordinates when needed, builds a spatial neighbor graph, solves the
    transition-matching assignment problem, and returns diagnostics.
    """

    if n_celltypes is None:
        if transition is not None:
            n_celltypes = np.asarray(transition).shape[0]
        elif prior is not None:
            n_celltypes = len(list(prior))
        else:
            raise ValueError("n_celltypes is required unless prior or transition is provided.")

    prior_arr = validate_proportions(prior, n_celltypes)
    target = make_transition_matrix(pattern, n_celltypes) if transition is None else validate_transition_matrix(transition, n_celltypes)

    if coordinates is None:
        coordinates = generate_coordinates(n_cells, dimensions=dimensions, plate_shape=plate_shape, random_state=random_state)
    else:
        coordinates = np.asarray(coordinates, dtype=float)
        n_cells = coordinates.shape[0]

    if adjacency is None:
        from .neighbors import get_spatial_network

        adjacency = get_spatial_network(
            Num_sample=n_cells,
            spatial=coordinates,
            n_neighs=n_neighbors,
            radius=radius,
            coord_type="generic",
            metric=metric,
        )

    cfg = config or AnnealingConfig(random_state=random_state)
    if zone_labels is None:
        result = solve_cell_types(
            adjacency=adjacency,
            target_transition=target,
            n_celltypes=n_celltypes,
            prior=prior_arr,
            n_cells=n_cells,
            config=cfg,
        )
        return SpatialSimulation(
            coordinates=coordinates,
            labels=result.labels,
            target_transition=target,
            observed_transition=result.transition,
            prior=prior_arr,
            loss=result.loss,
            history=result.history,
            adjacency=adjacency,
        )

    transition_by_zone = (
        {zone: validate_transition_matrix(matrix, n_celltypes) for zone, matrix in zone_transitions.items()}
        if zone_transitions is not None
        else target
    )
    zoned = solve_cell_types_by_zone(
        adjacency=adjacency,
        zone_labels=zone_labels,
        target_transitions=transition_by_zone,
        n_celltypes=n_celltypes,
        priors=prior_arr,
        config=cfg,
    )
    labels = zoned["labels"]
    estimated = estimate_parameters(labels, adjacency, n_celltypes=n_celltypes)
    losses = [res.loss for res in zoned["zone_results"].values()]
    return SpatialSimulation(
        coordinates=coordinates,
        labels=labels,
        target_transition=target,
        observed_transition=estimated.transition,
        prior=prior_arr,
        loss=float(np.mean(losses)) if losses else float("nan"),
        history=[],
        adjacency=adjacency,
        zone_results=zoned["zone_results"],
    )


def estimate_parameters_from_adata(
    adata,
    label_key: str,
    spatial_key: str = "spatial",
    n_neighbors: int = 8,
    radius: float | None = None,
    zone_key: str | None = None,
):
    """Estimate Spider priors and transition matrices from an AnnData object."""

    if label_key not in adata.obs:
        raise KeyError(f"{label_key!r} was not found in adata.obs.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"{spatial_key!r} was not found in adata.obsm.")

    from .neighbors import get_spatial_network

    adjacency = get_spatial_network(
        Num_sample=adata.n_obs,
        spatial=adata.obsm[spatial_key],
        n_neighs=n_neighbors,
        radius=radius,
        coord_type="generic",
    )
    labels = adata.obs[label_key].to_numpy()
    if zone_key is None:
        return estimate_parameters(labels, adjacency)
    if zone_key not in adata.obs:
        raise KeyError(f"{zone_key!r} was not found in adata.obs.")
    return estimate_parameters_by_zone(labels, adjacency, adata.obs[zone_key].to_numpy())
