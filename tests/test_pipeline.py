"""Сквозной тест на случайном блуждании. Санити-тест движка: без закономерности в данных
сделки из контроля должны давать ожидание ≈ −издержки, а гипотеза — не подтверждаться."""
import numpy as np

from qlab import report, study
from qlab.config import Config

from conftest import random_funding, random_oi, random_walk_m1


def test_end_to_end_on_random_walk(tmp_path):
    cfg = Config.load("configs/mvp.toml")
    cfg.raw["study"].update(bootstrap=200, permutations=200, depth_grid=[0.10, 0.20])
    m1 = {s: random_walk_m1(30 * 1440, seed=i) for i, s in enumerate(cfg.symbols)}
    inp = study.Inputs(m1=m1, oi={s: random_oi(m1[s], 10 + i) for i, s in enumerate(cfg.symbols)},
                       funding={s: random_funding(m1[s]) for s in cfg.symbols})
    results, events = study.run(cfg, inp)
    assert len(results) == 3 * 2 * 3  # монеты × ТФ × (2 depth + диагностика)
    summary = study.overall(results, cfg)
    assert not summary["supported"]
    ctrl = events[events["group"] == "control"]
    for (sym, tf), g in ctrl.groupby(["symbol", "tf"]):
        gross = g["res_1r"].mean()
        se = g["res_1r"].std() / np.sqrt(len(g))
        assert abs(gross) < 4 * se + 0.02, f"{sym} {tf}: валовое ожидание контроля на блуждании {gross:+.3f}"
        assert np.allclose(g["res_1r_net"], g["res_1r"] - g["cost_r"])
    path = report.build(cfg, [], results, events, summary,
                        {"config_hash": "x", "data_hash": "x", "git_commit": "x"}, tmp_path)
    text = path.read_text()
    assert "НЕ ПОДДЕРЖАНА" in text and (tmp_path / "figures" / "robustness.png").exists()
