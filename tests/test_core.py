import numpy as np

import spider


def test_import_spider_is_lightweight():
    assert spider.__version__ == "1.2.0"


def test_count_allocation_preserves_total_and_types():
    counts = spider.get_ct_sample(Num_celltype=3, Num_sample=10, prior=[0.5, 0.3, 0.2])

    assert counts.sum() == 10
    assert counts.shape == (3,)
    assert np.all(counts > 0)


def test_transition_templates_are_row_stochastic():
    for factory in (
        spider.attractive_freq,
        spider.exclusive_freq,
        spider.stripe_freq,
        lambda n: spider.make_transition_matrix("layered", n),
    ):
        matrix = factory(4)

        assert matrix.shape == (4, 4)
        assert np.all(matrix >= 0)
        assert np.allclose(matrix.sum(axis=1), 1.0)


def test_onehot_and_transition_frequency():
    labels = np.array([0, 0, 1, 1])
    adjacency = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.float32,
    )

    onehot = spider.get_onehot_ct(labels)
    transition = spider.transition_frequency(labels, adjacency)

    assert onehot.shape == (4, 2)
    assert transition.shape == (2, 2)
    assert np.allclose(transition.sum(axis=1), 1.0)


def test_incremental_solver_preserves_requested_counts():
    adjacency = np.ones((8, 8), dtype=np.float32) - np.eye(8, dtype=np.float32)
    config = spider.AnnealingConfig(max_iter=200, record_every=50, random_state=7)
    result = spider.solve_cell_types(
        adjacency=adjacency,
        target_transition=spider.make_transition_matrix("mixed", 2),
        n_celltypes=2,
        prior=[0.25, 0.75],
        config=config,
    )

    assert result.labels.shape == (8,)
    assert np.bincount(result.labels, minlength=2).tolist() == [2, 6]
    assert result.transition.shape == (2, 2)
    assert np.allclose(result.transition, spider.transition_frequency(result.labels, adjacency))
    assert result.loss >= 0


def test_simulate_cells_with_dense_adjacency():
    coordinates = np.column_stack((np.arange(6), np.zeros(6)))
    adjacency = np.zeros((6, 6), dtype=np.float32)
    for i in range(5):
        adjacency[i, i + 1] = 1
        adjacency[i + 1, i] = 1

    result = spider.simulate_cells(
        n_cells=6,
        n_celltypes=2,
        prior=[0.5, 0.5],
        pattern="mixed",
        coordinates=coordinates,
        adjacency=adjacency,
        config=spider.AnnealingConfig(max_iter=100, random_state=11),
    )

    assert result.coordinates.shape == (6, 2)
    assert result.labels.shape == (6,)
    assert result.summary()["n_celltypes"] == 2


def test_zone_specific_solver_path():
    adjacency = np.ones((6, 6), dtype=np.float32) - np.eye(6, dtype=np.float32)
    zones = np.array(["left", "left", "left", "right", "right", "right"])
    zoned = spider.solve_cell_types_by_zone(
        adjacency=adjacency,
        zone_labels=zones,
        target_transitions={
            "left": spider.make_transition_matrix("mixed", 2),
            "right": spider.make_transition_matrix("attractive", 2),
        },
        n_celltypes=2,
        priors=[1 / 3, 2 / 3],
        config=spider.AnnealingConfig(max_iter=50, random_state=3),
    )

    assert zoned["labels"].shape == (6,)
    assert set(zoned["zone_results"]) == {"left", "right"}
