"""Сквозные тесты платформы на синтетике: кэш, движок гипотез, holdout, discovery.
Санити: на случайном блуждании discovery не должен находить устойчивых закономерностей."""
import pytest

from qlab import discovery, engine, splits
from qlab.cache import Data
from qlab.hypothesis import Hypothesis
from qlab.registry import Registry, RegistryError

from conftest import make_config, write_clean
from test_registry import copy_h

SYMS = ["ETHUSDT", "SOLUSDT", "BTCUSDT"]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("platform")
    cfg = make_config(tmp)
    write_clean(tmp / "data", SYMS, days=62)
    return tmp, cfg, Data(cfg)


def test_feature_cache_is_reused(env):
    tmp, cfg, data = env
    b1 = data.bars("ETHUSDT", "15m")
    files = list((tmp / "data" / "cache").glob("ETHUSDT_15m_*.parquet"))
    assert len(files) == 1
    mtime = files[0].stat().st_mtime_ns
    b2 = data.bars("ETHUSDT", "15m")
    assert files[0].stat().st_mtime_ns == mtime and len(b1) == len(b2)
    cfg.raw["features"]["volume"]["window"] = 24          # другие параметры → другой ключ
    data.bars("ETHUSDT", "15m")
    cfg.raw["features"]["volume"]["window"] = 48
    assert len(list((tmp / "data" / "cache").glob("ETHUSDT_15m_*.parquet"))) == 1  # старый удалён


def test_engine_validation_and_holdout_lock(env, tmp_path):
    _, cfg, data = env
    p = copy_h(tmp_path, "hypotheses/H002-sweep-continuation.yaml", timeframes=["15m"], clean_after=None)
    raw_crit = Hypothesis.load(p).criteria
    reg = Registry(tmp_path / "r.json")
    h = Hypothesis.load(p)
    reg.register(h)
    with pytest.raises(splits.HoldoutLocked):
        engine.run(cfg, data, h, reg, "holdout", out_root=tmp_path / "checks")
    passed, report = engine.run(cfg, data, h, reg, "validation", out_root=tmp_path / "checks")
    text = report.read_text()
    assert "random_entry" in text and "Режимы рынка" in text and report.with_suffix(".png").exists()
    assert not passed and reg.items["H002"]["status"] == "Rejected"  # на блуждании после издержек edge нет
    assert reg.items["H002"]["checks"][0]["split"] == "validation"
    assert raw_crit["min_events"] == 300


def test_discovery_finds_nothing_on_random_walk(env, tmp_path):
    _, cfg, data = env
    df = discovery.scan(cfg, data)
    assert len(df) > 100
    assert set(df["regime"]) >= {"all", "vol", "btc", "risk"}  # trend на 1 месяце не набирает min_days
    rob = discovery.robust(df)
    assert rob.empty, f"ложные устойчивые находки на случайном блуждании:\n{rob}"
    assert df["significant"].mean() < 0.01
    reg = Registry(tmp_path / "r.json")
    out = discovery.write_report(cfg, df, rob, discovery.candidates(rob, 10), [], tmp_path / "disc")
    assert "training" in out.read_text()


def test_discovery_finds_planted_effect(tmp_path):
    """Положительный контроль: подмешиваем закономерность и требуем, чтобы поиск её нашёл и зарегистрировал."""
    import numpy as np
    from conftest import random_funding, random_oi, random_walk_m1

    cfg = make_config(tmp_path)
    cfg.raw["splits"]["training"] = ["2024-01-01", "2024-03-15"]
    cfg.raw["dataset"]["timeframes"] = ["15m"]
    cfg.raw["discovery"].update(min_days=30, min_events=30)
    for i, s in enumerate(SYMS):
        m = random_walk_m1(80 * 1440, seed=300 + i)
        ret = np.diff(np.log(m["close"].to_numpy()), prepend=np.log(100.0))
        for day in range(80):
            t = day * 1440 + 14 * 60 + 30                  # 09:30 Нью-Йорк зимой = 14:30 UTC
            sign = np.sign(ret[t:t + 15].sum())
            ret[t + 15:t + 135] += sign * 0.0002           # 2 часа дрейфа по направлению первой свечи
        close = 100 * np.exp(np.cumsum(ret))
        open_ = np.r_[100.0, close[:-1]]
        m = m.assign(open=open_, close=close, high=np.maximum(open_, close) * 1.0003,
                     low=np.minimum(open_, close) * 0.9997)
        for ds, df in (("klines_1m", m), ("metrics", random_oi(m, 1)), ("funding", random_funding(m))):
            p = tmp_path / "data" / "clean" / ds
            p.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p / f"{s}.parquet")
    df = discovery.scan(cfg, Data(cfg))
    rob = discovery.robust(df)
    hit = rob[(rob["spec"] == "session_open:ny") & (rob["regime"] == "all")]
    assert not hit.empty and (hit["sign"] > 0).all(), "подмешанный эффект не найден"
    reg = Registry(tmp_path / "r.json")
    cand = discovery.candidates(rob, 10)
    hs = discovery.register_candidates(cfg, reg, cand, "test", root=tmp_path / "disc")
    assert any(h.event["type"] == "session_open" and h.direction == "with" for h in hs)
    assert all(reg.items[h.id]["status"] == "Testing" for h in hs)


def test_factory_analysis_end_to_end(env, tmp_path):
    from qlab.factory import analysis
    _, cfg, data = env
    report = analysis.run(cfg, data, tmp_path / "factory", max_rows=5000)
    text = report.read_text()
    assert "Дубли" in text and "Кластеры" in text
    assert (tmp_path / "factory" / "core_features.json").exists()
    assert len(list(data.cache_dir.glob("factory_ETHUSDT_5m_*.parquet"))) == 1


def test_hypothesis_built_from_factory_features(env, tmp_path):
    """Гипотеза собирается из готовых признаков фабрики — без нового кода."""
    _, cfg, data = env
    p = copy_h(tmp_path, "hypotheses/H002-sweep-continuation.yaml", id="H950", timeframes=["15m"], clean_after=None,
               filters=["mo_ret_3 > 0 and en_permutation_24 < 0.99", "of_taker_share_6 > -0.5"])
    reg = Registry(tmp_path / "r.json")
    h = Hypothesis.load(p)
    reg.register(h)
    _, report = engine.run(cfg, data, h, reg, "validation", out_root=tmp_path / "checks")
    assert "mo_ret_3" in report.read_text()
