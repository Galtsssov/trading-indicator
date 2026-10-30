import numpy as np

from qlab import stats


def test_cliffs_delta_extremes():
    a, b = np.array([0.0, 1, 2]), np.array([3.0, 4, 5])
    assert stats.cliffs_delta(a, b) == 1.0
    assert stats.cliffs_delta(b, a) == -1.0
    assert stats.cliffs_delta(a, a) == 0.0


def test_benjamini_hochberg_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    adj = stats.benjamini_hochberg(p)
    np.testing.assert_allclose(adj[:4], [0.02, 0.04, 0.04, 0.02])
    assert np.isnan(adj[4])


def test_false_positive_rate_under_null_is_near_alpha():
    """Без эффекта p < 0.05 должно выпадать примерно в 5% случаев, а CI — чаще всего накрывать 0."""
    rng = np.random.default_rng(0)
    days = np.repeat(np.arange(100), 10)
    hits, covers = 0, 0
    for _ in range(60):
        a, b = rng.choice([-1.0, 1.0], 1000), rng.choice([-1.0, 1.0], 1000)
        c = stats.compare(a, b, days, days, n_boot=300, n_perm=300, rng=rng)
        assert c.primary_test.startswith("permutation")  # дискретное распределение → не t-тест
        hits += c.p_primary < 0.05
        covers += c.delta_ci[0] < 0 < c.delta_ci[1]
    assert hits <= 9          # ≤ 15% при ожидаемых 5%
    assert covers >= 51       # ≥ 85% при ожидаемых 95%


def test_real_effect_is_detected():
    rng = np.random.default_rng(1)
    a = rng.choice([-1.0, 1.0], 3000, p=[0.55, 0.45])
    b = rng.choice([-1.0, 1.0], 3000, p=[0.45, 0.55])
    days = np.repeat(np.arange(300), 10)
    c = stats.compare(a, b, days, days, n_boot=500, n_perm=500, rng=rng)
    assert c.p_primary < 0.01 and c.delta_ci[0] > 0
