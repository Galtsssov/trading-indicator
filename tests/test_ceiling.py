"""Прибор Information Ceiling на случайном блуждании: информации нет → вердикт «недостаточно»;
перемешанные метки дают AUC ≈ 0.5; встроенный эффект обнаруживается (AUC растёт, колонка — первая по SHAP)."""
import pytest

pytest.importorskip("lightgbm")

from qlab import ceiling, ceiling_report  # noqa: E402
from qlab.cache import Data  # noqa: E402
from qlab.factory import load_all  # noqa: E402

from conftest import make_config, write_clean  # noqa: E402


def test_ceiling_instrument_on_random_walk(tmp_path):
    cfg = make_config(tmp_path)
    write_clean(tmp_path / "data", ["ETHUSDT", "SOLUSDT", "BTCUSDT"], days=150)
    cfg.raw["splits"] = {"training": ["2024-01-01", "2024-04-01"], "validation": ["2024-04-01", "2024-05-25"],
                         "holdout": ["2024-06-01", "2100-01-01"]}
    cfg.raw["ceiling"].update(symbols=["ETHUSDT"], timeframes=["15m"], horizon_bars=12)
    feats = [n for n in load_all() if not n.startswith("cx_eth")][::3]
    r = ceiling.study(cfg, Data(cfg), "ETHUSDT", "15m", feats, tmp_path / "res", quick=True)
    c = r["configs"]["r2"]
    s = c["sanity"]
    assert abs(s["shuffled_auc"] - 0.5) < 0.03
    assert s["planted_shap_rank"] == 1 and s["planted_auc"] > s["planted_auc_without_effect_model"] + 0.05
    assert abs(c["models"]["lightgbm"]["auc"]["long"][0] - 0.5) < 0.06       # на блуждании информации нет
    assert set(c["ablation"]) >= {"OHLC", "OI", "Funding", "Minute", "Cross Asset", "Regime"}
    assert len(c["learning_curve"]) == 5 and set(c["importance"]) >= {"gain", "shap", "permutation", "mutual_info"}
    path = ceiling_report.build(ceiling_report.load(tmp_path / "res"), cfg.section("ceiling"), tmp_path / "rep")
    assert "НЕДОСТАТОЧНО" in path.read_text()
