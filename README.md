# Spider

Spider is a flexible and unified framework for simulating spatial transcriptomics (ST) data. It represents cellular spatial organization with two interpretable inputs: cell-type proportions and a transition probability matrix between adjacent cells. Spider then assigns cell types on a spatial neighbor graph with a batched simulated annealing strategy and can aggregate single-cell profiles into spot-level ST data.

## Publication

Yang J, Wei N, Qu Y, et al. Spider: a flexible and unified framework for simulating spatial transcriptomics data. *Bioinformatics*. 2026;42(1):btaf562. <https://doi.org/10.1093/bioinformatics/btaf562>

## Key Features

- Reference-free simulation driven by cell-type proportions and transition matrices.
- Predefined spatial patterns including attractive, repulsive, layered, and gyrus-like structures.
- Batched simulated annealing for scalable cell-type assignment.
- Cell-level simulation with optional spot-level aggregation.
- Optional compatibility helpers for benchmark-style simulators such as RCTD, STRIDE, stereoscope, and FICT-style workflows.
- Optional image and interactive workflows for histology-derived coordinates and custom domain structures.

## Installation

Install the core package:

```bash
pip install st-spider
```

Install optional feature groups only when needed:

```bash
pip install "st-spider[plot]"
pip install "st-spider[spatial]"
pip install "st-spider[image]"
pip install "st-spider[torch]"
```

The core installation avoids heavy optional dependencies such as `torch`, `scanpy`, `squidpy`, `napari`, and `cellpose`. This keeps the main simulator easier to install while preserving advanced workflows for users who need them.

## Quick Start

```python
import spider

prior = [0.6, 0.3, 0.1]
target = spider.make_transition_matrix("attractive", n_celltypes=3, strength=0.8)

sim = spider.simulate_cells(
    n_cells=1000,
    n_celltypes=3,
    prior=prior,
    transition=target,
    plate_shape=(100, 100),
    random_state=1,
)

print(sim.summary())
adata = sim.to_anndata()
```

See the tutorials at <https://spider-analyses.readthedocs.io/en/latest/>.

## 3D Cell Simulation

`simulate_10X_3d` supports more than 10,000 cells using a three-axis coarse-to-fine grid. It returns one cell-type label and one `(x, y, z)` coordinate per cell. The final annealing step uses a 3D nearest-neighbor graph.

```python
import numpy as np
import spider

prior = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
target = spider.make_transition_matrix("mixed", n_celltypes=5)

cell_types, locations = spider.simulate_10X_3d(
    cell_num=20000,
    Num_celltype=5,
    prior=prior,
    target_trans=target,
    image_width=1000,
    image_height=1000,
    image_depth=1000,
)
assert locations.shape == (20000, 3)
```

## Estimating Parameters From Real Data

Spider can estimate cell-type proportions and neighborhood transition matrices from annotated spatial data:

```python
params = spider.estimate_parameters_from_adata(
    adata,
    label_key="celltype",
    spatial_key="spatial",
    n_neighbors=8,
)

sim = spider.simulate_cells(
    n_cells=adata.n_obs,
    prior=params.prior,
    transition=params.transition,
    coordinates=adata.obsm["spatial"],
    random_state=1,
)
```

## Spatially Varying Patterns

For tissues where a single global transition matrix is too restrictive, pass zone-specific targets:

```python
zone_transitions = {
    "tumor_core": spider.make_transition_matrix("attractive", 3, strength=0.85),
    "invasion_front": spider.make_transition_matrix("mixed", 3),
}

sim = spider.simulate_cells(
    n_cells=adata.n_obs,
    n_celltypes=3,
    prior=[0.5, 0.3, 0.2],
    coordinates=adata.obsm["spatial"],
    zone_labels=adata.obs["region"].to_numpy(),
    zone_transitions=zone_transitions,
    random_state=1,
)
```

## Development

Install development dependencies:

```bash
pip install -e ".[dev,plot,spatial]"
pytest
```

Build release artifacts outside version control:

```bash
python -m build
```
