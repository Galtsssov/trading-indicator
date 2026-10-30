"""Information Ceiling Study (этап 3): сколько информации извлекается из текущего набора данных.

Модели здесь — измерительные приборы, не стратегии. Правило решения — docs/stage-3-preregistration.md.
Только training → validation; holdout не открывается (splits.window без allow_holdout).

Для одной пары (монета, ТФ):
1. Метки: на каждом баре — лонг и шорт; стоп 1R, цель 2R, 1R = r_atr·ATR, выход через horizon баров;
   путь по минуткам (outcomes.compute). y = 1, если цель раньше стопа. Результат в R после издержек.
2. Признаки — ядро Feature Factory. Training делится на fit и inner (последние 20 %, ранняя остановка)
   с зазором в горизонт метки; validation — только для измерения.
3. Основной прибор LightGBM + 5 других моделей; топ-k сигналов в сутки; калибровка; важность признаков
   (gain, SHAP, permutation, mutual information); абляция групп; кривые обучения; проверки исправности.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from qlab import outcomes, splits
from qlab.cache import Data
from qlab.config import TF_MINUTES, Config
from qlab.factory import catalog

log = logging.getLogger(__name__)
DAY_NS = 86_400_000_000_000

# ---------- группы признаков для абляции ----------
CROSS = {"BTC Context", "ETH Context", "SOL Context", "Relative Strength"}
VOLATILITY = {"Volatility", "Compression", "Expansion"}
VOLUME = {"Volume", "Order Flow Proxy", "Liquidity"}
TIME = {"Time", "Calendar", "Session"}


def group_of(name: str, category: str) -> str:
    if name.startswith("po_oi") or name == "cf_flow_state":
        return "OI"
    if name.startswith("po_funding"):
        return "Funding"
    if name.startswith(("po_ls", "po_taker_ls")):
        return "Long/Short"
    if name.startswith("mi_") or name == "zs_intrabar_rv":
        return "Minute"
    if name.startswith("vw_") or name in ("au_poc_dist", "au_value_width"):
        return "VWAP"
    if category == "Entropy":
        return "Entropy"
    if category in CROSS:
        return "Cross Asset"
    if category == "Regime" or name == "cf_vol_state":
        return "Regime"
    if category in VOLATILITY:
        return "ATR / Volatility"
    if category in VOLUME or name.startswith("pc_volume") or name == "zs_trades_288":
        return "Volume / Flow"
    if category in TIME:
        return "Time"
    return "OHLC"


def feature_groups(names: list[str]) -> dict[str, list[str]]:
    cat = catalog().set_index("name")["category"]
    out: dict[str, list[str]] = {}
    for n in names:
        out.setdefault(group_of(n, cat.get(n, "")), []).append(n)
    return out


# ---------- данные ----------

@dataclass
class Dataset:
    X: pd.DataFrame                 # признаки (float32)
    y: dict[str, np.ndarray]        # "long" / "short": 1 — цель раньше стопа
    res: dict[str, np.ndarray]      # результат сделки в R после издержек
    res_gross: dict[str, np.ndarray]
    cost_r: np.ndarray
    part: np.ndarray                # "fit" / "inner" / "val"
    ts: pd.Series
    bar: np.ndarray

    def idx(self, part: str) -> np.ndarray:
        return np.flatnonzero(self.part == part)


def build_dataset(cfg: Config, data: Data, symbol: str, tf: str, r_atr: float, features: list[str]) -> Dataset:
    c = cfg.section("ceiling")
    tf_min = TF_MINUTES[tf]
    horizon = pd.Timedelta(minutes=c["horizon_bars"] * tf_min)
    bars = data.bars(symbol, tf)
    fac = data.factory(symbol, tf, features)
    tr_start, tr_end = splits.window(cfg, "training")
    va_start, va_end = splits.window(cfg, "validation")
    base = bars.assign(atr_ref=bars["atr"].shift(1))
    ok = base["complete"] & base["atr_ref"].gt(0)
    tr = ok & (base["ts"] >= tr_start) & (base["available_at"] + horizon <= tr_end)
    va = ok & (base["ts"] >= va_start) & (base["available_at"] + horizon <= va_end)
    rows = base[tr | va]
    op = outcomes.OutcomeParams(r_atr=r_atr, targets=(c["target_r"],), horizon_minutes=c["horizon_bars"] * tf_min,
                                cost_pct_round_trip=cfg.cost_pct_round_trip)
    m1 = data.m1(symbol)
    y, res, gross = {}, {}, {}
    valid = np.ones(len(rows), dtype=bool)
    cost_r = None
    tgt = f"{c['target_r']:g}"
    for name, d in (("long", 1), ("short", -1)):
        ev = rows[["ts", "available_at", "atr_ref"]].assign(direction=d).reset_index(drop=True)
        o = outcomes.compute(ev, m1, op)
        valid &= o["exclude_reason"].isna().to_numpy()
        y[name] = (o[f"res_{tgt}r"].to_numpy() == c["target_r"]).astype(np.int8)
        res[name] = o[f"res_{tgt}r_net"].to_numpy()
        gross[name] = o[f"res_{tgt}r"].to_numpy()
        cost_r = o["cost_r"].to_numpy()
    keep = np.flatnonzero(valid)
    bar = rows.index.to_numpy()[keep]
    is_val = va.loc[rows.index].to_numpy()[keep]
    part = np.where(is_val, "val", "fit").astype(object)
    tr_idx = np.flatnonzero(~is_val)
    cut = tr_idx[int(len(tr_idx) * (1 - c["inner_valid_frac"]))]
    part[(~is_val) & (np.arange(len(keep)) >= cut)] = "inner"
    gap = c["horizon_bars"]                                    # зазор: метки fit не заходят в inner
    part[(~is_val) & (np.arange(len(keep)) >= cut - gap) & (np.arange(len(keep)) < cut)] = "gap"
    X = fac.iloc[bar][features].reset_index(drop=True)
    fit_rows = X.iloc[np.flatnonzero(part == "fit")]
    X = X[[f for f in features if fit_rows[f].nunique(dropna=True) > 1]]   # константы на fit: информации нет, HistGB падает
    return Dataset(X, {k: v[keep] for k, v in y.items()}, {k: v[keep] for k, v in res.items()},
                   {k: v[keep] for k, v in gross.items()}, cost_r[keep], part,
                   bars["ts"].iloc[bar].reset_index(drop=True), bar)


# ---------- модели ----------

def make_model(name: str, seed: int, n_estimators: int | None = None):
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=n_estimators or 1000, learning_rate=0.05, num_leaves=63,
                                  min_child_samples=300, subsample=0.7, subsample_freq=1, colsample_bytree=0.5,
                                  reg_lambda=1.0, n_jobs=-1, random_state=seed, verbose=-1)
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.7,
                                 colsample_bytree=0.5, min_child_weight=100, tree_method="hist",
                                 eval_metric="logloss", early_stopping_rounds=50, n_jobs=-1, random_state=seed)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=1000, learning_rate=0.08, depth=6, od_type="Iter", od_wait=50,
                                  verbose=0, thread_count=-1, random_seed=seed)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=300, max_features=0.3,
                                      max_samples=0.1, n_jobs=-1, random_state=seed)
    if name == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=300,
                                              early_stopping=True, validation_fraction=0.1, n_iter_no_change=40,
                                              random_state=seed)
    if name == "mlp":
        from sklearn.impute import SimpleImputer
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import QuantileTransformer
        return make_pipeline(SimpleImputer(strategy="median"),
                             QuantileTransformer(output_distribution="normal", subsample=100_000, random_state=seed),
                             MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=1024, max_iter=40,
                                           early_stopping=True, n_iter_no_change=5, random_state=seed))
    raise ValueError(name)


MLP_MAX_ROWS = 200_000


def fit_predict(name: str, X: pd.DataFrame, y: np.ndarray, fit: np.ndarray, inner: np.ndarray, pred: np.ndarray,
                seed: int, n_estimators: int | None = None):
    """Обучить на fit (ранняя остановка по inner, где применимо) и вернуть (модель, вероятности на pred)."""
    m = make_model(name, seed, n_estimators)
    Xf, yf = X.iloc[fit], y[fit]
    if name == "lightgbm":
        import lightgbm as lgb
        if n_estimators:
            m.fit(Xf, yf)
        else:
            m.fit(Xf, yf, eval_set=[(X.iloc[inner], y[inner])], callbacks=[lgb.early_stopping(50, verbose=False)])
    elif name in ("xgboost", "catboost"):
        m.fit(Xf, yf, eval_set=[(X.iloc[inner], y[inner])], **({"verbose": False} if name == "xgboost" else {}))
    elif name == "mlp":
        sub = np.random.default_rng(seed).choice(len(fit), min(len(fit), MLP_MAX_ROWS), replace=False)
        m.fit(Xf.iloc[sub], yf[sub])
    else:
        m.fit(Xf, yf)
    return m, m.predict_proba(X.iloc[pred])[:, 1]


# ---------- метрики ----------

def day_codes(ts: pd.Series) -> np.ndarray:
    return ts.dt.floor("D").dt.as_unit("ns").astype("int64").to_numpy() // DAY_NS


def auc_ci(y: np.ndarray, p: np.ndarray, days: np.ndarray, b: int, rng) -> tuple[float, float, float]:
    auc = roc_auc_score(y, p)
    u = np.unique(days)
    pos = {d: i for i, d in enumerate(u)}
    groups = np.array([pos[d] for d in days])
    order = np.argsort(groups)
    starts = np.searchsorted(groups[order], np.arange(len(u)))
    ends = np.append(starts[1:], len(order))
    vals = []
    for _ in range(b):
        pick = rng.integers(0, len(u), len(u))
        idx = np.concatenate([order[starts[k]:ends[k]] for k in pick])
        if 0 < y[idx].sum() < len(idx):
            vals.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def top_k_per_day(ts: pd.Series, bar: np.ndarray, scores: dict[str, np.ndarray], k: int, spacing: int) -> pd.DataFrame:
    """Лучшие k сигналов в сутки по вероятности (оба направления), не ближе spacing баров друг к другу."""
    frames = [pd.DataFrame({"i": np.arange(len(bar)), "bar": bar, "day": ts.dt.floor("D").to_numpy(),
                            "dir": d, "p": s}) for d, s in scores.items()]
    cand = pd.concat(frames, ignore_index=True).sort_values(["day", "p"], ascending=[True, False])
    picks = []
    for _, g in cand.groupby("day", sort=True):
        chosen: list[int] = []
        for row in g.itertuples(index=False):
            if all(abs(row.bar - b) >= spacing for b in chosen):
                picks.append(row)
                chosen.append(row.bar)
                if len(chosen) == k:
                    break
    return pd.DataFrame(picks)


def picks_stats(picks: pd.DataFrame, y: dict, res: dict, b: int, alpha_one_sided: float, rng) -> dict:
    if picks.empty:
        return {"n": 0}
    win = np.array([y[d][i] for d, i in zip(picks["dir"], picks["i"])], dtype=float)
    r = np.array([res[d][i] for d, i in zip(picks["dir"], picks["i"])], dtype=float)
    days = pd.factorize(picks["day"])[0]
    nd = days.max() + 1
    w_sum, r_sum, cnt = (np.bincount(days, weights=v, minlength=nd) for v in (win, r, np.ones_like(r)))
    draw = rng.integers(0, nd, size=(b, nd))
    wr_b = w_sum[draw].sum(1) / cnt[draw].sum(1)
    ev_b = r_sum[draw].sum(1) / cnt[draw].sum(1)
    q = 100 * alpha_one_sided
    return {"n": int(len(win)), "days": int(nd), "wr": float(win.mean()), "ev": float(r.mean()),
            "wr_lo": float(np.percentile(wr_b, q)), "ev_lo": float(np.percentile(ev_b, q)),
            "wr_ci95": [float(x) for x in np.percentile(wr_b, [2.5, 97.5])],
            "ev_ci95": [float(x) for x in np.percentile(ev_b, [2.5, 97.5])]}


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    mean_p = np.array([p[idx == k].mean() for k in range(bins)])
    rate = np.array([y[idx == k].mean() for k in range(bins)])
    n = np.array([(idx == k).sum() for k in range(bins)])
    return {"mean_p": mean_p.tolist(), "observed": rate.tolist(), "n": n.tolist(),
            "ece": float(np.sum(n * np.abs(mean_p - rate)) / n.sum()), "brier": float(np.mean((p - y) ** 2)),
            "base_rate": float(y.mean())}


def precision_at(y: np.ndarray, p: np.ndarray, res: np.ndarray) -> list[dict]:
    order = np.argsort(-p)
    out = []
    for frac in (0.001, 0.005, 0.01, 0.05, 0.1):
        top = order[: max(1, int(len(p) * frac))]
        out.append({"top_frac": frac, "n": len(top), "precision": float(y[top].mean()),
                    "recall": float(y[top].sum() / max(y.sum(), 1)), "ev": float(res[top].mean())})
    return out


# ---------- исследование одной пары ----------

def study(cfg: Config, data: Data, symbol: str, tf: str, features: list[str], out: Path,
          quick: bool = False) -> dict:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)      # sklearn: колонки без значений в начале истории
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # пустые децили калибровки на малых выборках
    c = cfg.section("ceiling")
    seed = cfg.section("stats")["seed"]
    rng = np.random.default_rng(seed)
    b_auc, b_top = (100, 400) if quick else (300, c["bootstrap"])
    alpha1 = c["alpha"] / (len(c["symbols"]) * len(c["timeframes"]) * len(c["top_k"]))
    result: dict = {"symbol": symbol, "tf": tf, "n_features": len(features), "configs": {}}
    t0 = time.time()
    for ri, r_atr in enumerate(c["r_atr"]):
        primary = ri == 0
        ds = build_dataset(cfg, data, symbol, tf, r_atr, features)
        fit, inner, val = ds.idx("fit"), ds.idx("inner"), ds.idx("val")
        days_val = day_codes(ds.ts.iloc[val])
        cfg_res: dict = {"r_atr": r_atr, "rows": {"fit": len(fit), "inner": len(inner), "val": len(val)},
                         "cost_r_mean": float(ds.cost_r[val].mean()),
                         "breakeven_wr": float((1 + ds.cost_r[val].mean()) / 3),
                         "base_rate": {d: float(ds.y[d][val].mean()) for d in ds.y}, "models": {}}
        y_val = {d: ds.y[d][val] for d in ds.y}
        res_val = {d: ds.res[d][val] for d in ds.res}
        best_iter = {}
        models = c["models"] if primary and not quick else [c["primary_model"]] if not quick else ["lightgbm", "mlp"]
        for name in models:
            t = time.time()
            p = {}
            for d in ("long", "short"):
                m, p[d] = fit_predict(name, ds.X, ds.y[d], fit, inner, val, seed)
                if name == "lightgbm":
                    best_iter[d] = int(getattr(m, "best_iteration_", None) or m.n_estimators)
                    if primary and d == "long":
                        lgbm_long = m
            mres = {"auc": {d: auc_ci(y_val[d], p[d], days_val, b_auc, rng) for d in p},
                    "calibration": {d: calibration(y_val[d], p[d]) for d in p},
                    "precision_at": {d: precision_at(y_val[d], p[d], res_val[d]) for d in p},
                    "top_k": {str(k): picks_stats(top_k_per_day(ds.ts.iloc[val].reset_index(drop=True), ds.bar[val], p, k,
                                                                c["min_spacing_bars"]), y_val, res_val, b_top, alpha1, rng)
                              for k in c["top_k"]},
                    "seconds": round(time.time() - t, 1)}
            cfg_res["models"][name] = mres
            if name == c["primary_model"]:
                p_primary = p
            log.info("%s %s r=%g %s: AUC long %.4f short %.4f (%.0fs)", symbol, tf, r_atr, name,
                     mres["auc"]["long"][0], mres["auc"]["short"][0], mres["seconds"])
        rnd = {}
        for k in c["top_k"]:
            rp = {d: rng.random(len(val)) for d in ("long", "short")}
            rnd[str(k)] = picks_stats(top_k_per_day(ds.ts.iloc[val].reset_index(drop=True), ds.bar[val], rp, k,
                                                    c["min_spacing_bars"]), y_val, res_val, b_top, alpha1, rng)
        cfg_res["random_picks"] = rnd
        if primary:
            cfg_res["features_used"] = list(ds.X.columns)
            cfg_res.update(_diagnostics(cfg, ds, list(ds.X.columns), fit, inner, val, best_iter, lgbm_long, p_primary, seed, rng,
                                        quick))
        result["configs"][f"r{r_atr:g}"] = cfg_res
    result["seconds"] = round(time.time() - t0)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{symbol}_{tf}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float))
    return result


def _auc_mean(ds: Dataset, X: pd.DataFrame, fit, inner, val, seed, n_iter, y_override=None) -> tuple[float, dict]:
    p, aucs = {}, {}
    for d in ("long", "short"):
        y = ds.y[d] if y_override is None else y_override[d]
        _, p[d] = fit_predict("lightgbm", X, y, fit, inner, val, seed, n_estimators=n_iter[d])
        aucs[d] = roc_auc_score(y[val], p[d])
    return float(np.mean(list(aucs.values()))), p


def _diagnostics(cfg, ds: Dataset, features, fit, inner, val, best_iter, lgbm_long, p_main, seed, rng, quick) -> dict:
    """Важность, абляция, кривые обучения, проверки исправности — основным прибором."""
    from sklearn.feature_selection import mutual_info_classif
    c = cfg.section("ceiling")
    out: dict = {}
    yv = {d: ds.y[d][val] for d in ds.y}
    base_auc = float(np.mean([roc_auc_score(yv[d], p_main[d]) for d in yv]))
    out["base_auc_mean"] = base_auc

    # --- важность признаков (направление long) ---
    t = time.time()
    booster = lgbm_long.booster_
    gain = pd.Series(booster.feature_importance("gain"), index=features)
    sub = rng.choice(val, min(len(val), 5000 if quick else 20_000), replace=False)
    contrib = booster.predict(ds.X.iloc[sub], pred_contrib=True)[:, :-1]
    shap = pd.Series(np.abs(contrib).mean(0), index=features)
    ps = rng.choice(val, min(len(val), 5000 if quick else 30_000), replace=False)
    y_ps = ds.y["long"][ps]
    Xp = ds.X.iloc[ps].copy()
    ref = roc_auc_score(y_ps, lgbm_long.predict_proba(Xp)[:, 1])
    perm = {}
    for f in gain.sort_values(ascending=False).index[: 20 if quick else 60]:
        saved = Xp[f].to_numpy().copy()
        Xp[f] = rng.permutation(saved)
        perm[f] = ref - roc_auc_score(y_ps, lgbm_long.predict_proba(Xp)[:, 1])
        Xp[f] = saved
    mi_idx = rng.choice(fit, min(len(fit), 5000 if quick else 30_000), replace=False)
    Xm = ds.X.iloc[mi_idx].fillna(ds.X.iloc[mi_idx].median())
    mi = pd.Series(mutual_info_classif(Xm.fillna(0), ds.y["long"][mi_idx], random_state=seed), index=features)
    out["importance"] = {"gain": gain.to_dict(), "shap": shap.to_dict(), "permutation": perm, "mutual_info": mi.to_dict(),
                         "seconds": round(time.time() - t)}

    # --- абляция групп ---
    groups = feature_groups(features)
    abl = {}
    for g, cols in groups.items():
        keep = [f for f in features if f not in cols]
        auc, _ = _auc_mean(ds, ds.X[keep], fit, inner, val, seed, best_iter)
        abl[g] = {"n_features": len(cols), "auc": auc, "delta": auc - base_auc}
        log.info("абляция −%s (%d): AUC %.4f (Δ %+.4f)", g, len(cols), auc, auc - base_auc)
    out["ablation"] = abl

    # --- кривые обучения: последние доли fit (ближе к validation) ---
    lc = []
    for frac in c["learning_curve"]:
        part = fit[-max(1000, int(len(fit) * frac)):]
        p_tr, aucs = {}, {}
        for d in ("long", "short"):
            m, p = fit_predict("lightgbm", ds.X, ds.y[d], part, inner, val, seed)
            aucs[d] = roc_auc_score(ds.y[d][val], p)
            tr_s = part[-min(len(part), 50_000):]
            p_tr[d] = roc_auc_score(ds.y[d][tr_s], m.predict_proba(ds.X.iloc[tr_s])[:, 1])
        lc.append({"fraction": frac, "rows": int(len(part)), "val_auc": float(np.mean(list(aucs.values()))),
                   "train_auc": float(np.mean(list(p_tr.values())))})
        log.info("кривая обучения %.4f: %d строк, AUC val %.4f train %.4f", frac, len(part), lc[-1]["val_auc"], lc[-1]["train_auc"])
    out["learning_curve"] = lc

    # --- проверки исправности ---
    sanity = {}
    shuffled = {d: ds.y[d].copy() for d in ds.y}
    for d in shuffled:
        shuffled[d][fit] = rng.permutation(shuffled[d][fit])
        shuffled[d][inner] = rng.permutation(shuffled[d][inner])
    sanity["shuffled_auc"], _ = _auc_mean(ds, ds.X, fit, inner, val, seed, best_iter, shuffled)
    z = rng.normal(size=len(ds.X)).astype(np.float32)
    planted = {d: np.where((z > 1.5) & (rng.random(len(z)) < 0.5), 1, ds.y[d]).astype(np.int8) for d in ds.y}
    Xz = ds.X.assign(planted_signal=z)
    m, pz = fit_predict("lightgbm", Xz, planted["long"], fit, inner, val, seed)   # своя ранняя остановка
    sh = np.abs(m.booster_.predict(Xz.iloc[sub], pred_contrib=True)[:, :-1]).mean(0)
    sanity["planted_auc"] = float(roc_auc_score(planted["long"][val], pz))
    sanity["planted_auc_without_effect_model"] = float(roc_auc_score(planted["long"][val], p_main["long"]))
    sanity["planted_shap_rank"] = int((sh > sh[-1]).sum() + 1)
    noise = pd.DataFrame(rng.normal(size=(len(ds.X), c["random_features"])).astype(np.float32),
                         columns=[f"noise_{i}" for i in range(c["random_features"])])
    Xn = pd.concat([ds.X, noise], axis=1)
    m, pn = fit_predict("lightgbm", Xn, ds.y["long"], fit, inner, val, seed)
    shn = pd.Series(np.abs(m.booster_.predict(Xn.iloc[sub], pred_contrib=True)[:, :-1]).mean(0), index=Xn.columns)
    rank = shn.rank(ascending=False)
    sanity["noise_best_shap_rank"] = int(rank[noise.columns].min())
    sanity["noise_auc"] = float(roc_auc_score(ds.y["long"][val], pn))
    sanity["passed"] = bool(abs(sanity["shuffled_auc"] - 0.5) <= 0.01 and sanity["planted_shap_rank"] == 1
                            and sanity["planted_auc"] > sanity["planted_auc_without_effect_model"] + 0.02
                            and sanity["noise_best_shap_rank"] > 50)
    log.info("исправность: %s", sanity)
    out["sanity"] = sanity
    return out
