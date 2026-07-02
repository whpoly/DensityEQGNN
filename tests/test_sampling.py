from density_eqgnn.data.sampling import fractional_grid, random_probe_indices


def test_fractional_grid_centers():
    grid = fractional_grid((2, 2, 1))
    assert grid.shape == (4, 3)
    assert (grid[:, 0] > 0).all()
    assert (grid[:, 0] < 1).all()


def test_random_probe_indices_all_when_none():
    idx = random_probe_indices(5, None, seed=0)
    assert idx.tolist() == [0, 1, 2, 3, 4]

