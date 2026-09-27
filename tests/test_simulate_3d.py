import numpy as np

import spider
from spider.enhance import enhance_loop
from spider.simulate_10X import get_mesh_counts_3d


def test_3d_grid_uses_all_axes_and_clips_upper_bounds():
    locations = np.array([[0, 0, 0], [10, 20, 30]], dtype=float)
    mapping, x, y, z, *_ = get_mesh_counts_3d(
        locations=locations,
        grid_x=2,
        grid_y=3,
        grid_z=4,
        image_width=10,
        image_height=20,
        image_depth=30,
    )

    assert mapping.shape == (2, 24)
    assert mapping.indices.tolist() == [0, 23]
    assert np.column_stack((x, y, z)).shape == (24, 3)


def test_enhance_loop_handles_rectangular_3d_grid(monkeypatch):
    monkeypatch.setattr("spider.enhance.STsim", lambda **kwargs: kwargs["celltype_assignment"])
    grid = np.stack(
        np.meshgrid(np.arange(3), np.arange(2), np.arange(5), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    labels, sampled, loops = enhance_loop(
        Num_sample=30,
        Num_celltype=2,
        prior=[0.6, 0.4],
        target_trans=np.full((2, 2), 0.5),
        original_grid=grid,
        grid_row=3,
        grid_col=2,
        grid_depth=5,
        loop_times=2,
        windows_row_list=[2, 1],
        windows_col_list=[2, 1],
        windows_depth_list=[3, 1],
        swap_num_list=[2, 1],
        tol_list=[0.02, 0.02],
    )

    assert loops == 2
    assert labels.shape == (3, 2, 5)
    assert sampled.shape == (30, 3)
    assert np.array_equal(sampled, grid)
    assert np.bincount(labels.ravel(), minlength=2).tolist() == [18, 12]


def test_simulate_10x_3d_above_10000_cells():
    prior = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    transition = np.full((5, 5), 0.2)

    for n_cells in (10000, 10001, 20000):
        labels, locations = spider.simulate_10X_3d(
            cell_num=n_cells,
            Num_celltype=5,
            prior=prior,
            target_trans=transition,
            image_width=1000,
            image_height=1000,
            image_depth=1000,
            smallsample_max_iter=1,
            bigsample_max_iter=1,
        )

        expected = spider.get_ct_sample(Num_celltype=5, Num_sample=n_cells, prior=prior)
        assert labels.shape == (n_cells,)
        assert locations.shape == (n_cells, 3)
        assert np.array_equal(np.bincount(labels, minlength=5), expected)
