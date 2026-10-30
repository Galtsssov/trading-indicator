"""Microstructure Proxy (из минуток внутри бара) и Entropy (упорядоченность движения)."""
import numpy as np
import pandas as pd

from qlab.factory import family, grid, rolling_z, safe_div

MI = ("Microstructure Proxy", "MI")
IB = ("ib_rv", "ib_path", "ib_sign_changes", "ib_max_abs_r", "ib_last_vol_share", "ib_max_vol_share", "n_minutes")
family(*MI, "mi_intrabar_rv", None, formula="sqrt(Σ r_1m²) внутри бара / (a/c)", kind="непрерывный", unit="ATR",
       deps=("ib_rv", "atr"), cost="O(n)", mechanism="Реализованная волатильность по минуткам: буря внутри бара видна, даже если свеча маленькая",
       measures="внутрибаровая волатильность в ATR", why="Информация, которой нет в OHLC бара", limits="Минутные данные шумны",
       fn=lambda x: x.col("ib_rv") / x.a_pct)
family(*MI, "mi_signature_{W}", grid(W=(24, 288)), formula="Σ_W ib_rv² / Σ_W r²", kind="непрерывный", unit="отношение",
       deps=("ib_rv", "close"), cost="O(n)", mechanism="Отношение дисперсий разных масштабов: > 1 — возвратность внутри баров, < 1 — моментум",
       measures="«signature plot»: минутная дисперсия против дисперсии баров за {W}", why="Персистентность внутри бара",
       limits="Чувствительна к отскоку bid/ask",
       fn=lambda x, W: safe_div((x.col("ib_rv") ** 2).rolling(W, min_periods=W).sum(), (x.r ** 2).rolling(W, min_periods=W).sum()))
family(*MI, "mi_path_efficiency", None, formula="|ln(c/o)| / Σ |r_1m| внутри бара", kind="непрерывный", unit="доля [0,1]",
       deps=("ib_path", "open", "close"), cost="O(n)", mechanism="Прямое движение внутри бара — односторонний поток",
       measures="эффективность пути цены внутри бара", why="Различает рывок и болтанку при одинаковой свече",
       limits="Первая минута включает переход от прошлого бара", fn=lambda x: safe_div(np.log(x.c / x.o).abs(), x.col("ib_path")))
family(*MI, "mi_path_efficiency_24", None, formula="mean_24(mi_path_efficiency)", kind="непрерывный", unit="доля [0,1]",
       deps=("ib_path", "open", "close"), cost="O(n)", mechanism="Устойчивость одностороннего потока",
       measures="средняя эффективность пути за 24 бара", why="Режим внутрибаровой направленности", limits="—",
       fn=lambda x: safe_div(np.log(x.c / x.o).abs(), x.col("ib_path")).rolling(24, min_periods=24).mean())
family(*MI, "mi_sign_change_rate", None, formula="ib_sign_changes / (n_minutes − 1)", kind="непрерывный", unit="доля [0,1]",
       deps=("ib_sign_changes", "n_minutes"), cost="O(n)", mechanism="Частые смены направления минуток — маркетмейкинг, баланс",
       measures="частота смен направления минутных движений внутри бара", why="Баланс против одностороннего потока", limits="—",
       fn=lambda x: safe_div(x.col("ib_sign_changes"), x.col("n_minutes") - 1))
family(*MI, "mi_jumpiness", None, formula="max |r_1m| / sqrt(Σ r_1m²)", kind="непрерывный", unit="доля [0,1]",
       deps=("ib_max_abs_r", "ib_rv"), cost="O(n)", mechanism="Одна минута сделала бар — скачок (ордер-удар, ликвидация)",
       measures="доля внутрибарового движения в одной минуте", why="Скачок против плавного хода", limits="—",
       fn=lambda x: safe_div(x.col("ib_max_abs_r"), x.col("ib_rv")))
family(*MI, "mi_last_minute_volume", None, formula="объём последней минуты / объём бара", kind="непрерывный", unit="доля [0,1]",
       deps=("ib_last_vol_share",), cost="O(n)", mechanism="Активность у закрытия бара (боты по закрытию, стопы)",
       measures="доля объёма бара в последней минуте", why="Поток «на закрытие»", limits="—", fn=lambda x: x.col("ib_last_vol_share"))
family(*MI, "mi_max_minute_volume", None, formula="max объёма минуты / объём бара", kind="непрерывный", unit="доля [0,1]",
       deps=("ib_max_vol_share",), cost="O(n)", mechanism="Объём бара сосредоточен в одной минуте — событие", measures="концентрация объёма внутри бара",
       why="Событие против равномерной торговли", limits="—", fn=lambda x: x.col("ib_max_vol_share"))
family(*MI, "mi_impact_per_volume_{W}", grid(W=(24, 288)), formula="mean_W( |r| / (v / mean_288 v) ) / std_288(r)",
       kind="непрерывный", unit="σ на нормированный объём", deps=("close", "volume"), cost="O(n)",
       mechanism="Сколько цены сдвигает типичная единица объёма (аналог лямбды Кайла)", measures="ценовой импакт относительно объёма за {W}",
       why="Тонкий рынок двигается легче", limits="Смешан с информационностью потока",
       fn=lambda x, W: safe_div(x.r.abs(), safe_div(x.v, x.v.rolling(288, min_periods=288).mean())).rolling(W, min_periods=W).mean() / x.rv(288))
family(*MI, "mi_trade_size_z", None, formula="z_288( ln(quote_volume / trades) )", kind="непрерывный", unit="σ",
       deps=("quote_volume", "trades"), cost="O(n)", mechanism="Крупные средние сделки — крупные участники",
       measures="средний размер сделки в σ", why="Присутствие крупных", limits="—",
       fn=lambda x: rolling_z(np.log(safe_div(x.col("quote_volume"), x.col("trades"))), 288))
family(*MI, "mi_data_gap_minutes", None, formula="ТФ в минутах − n_minutes (минуток нет в источнике)", kind="счётчик", unit="минуты",
       deps=("n_minutes",), cost="O(n)", mechanism="Простой биржи или дыра в данных", measures="сколько минут бара отсутствует",
       why="Отсечь артефакты данных", limits="Не про рынок, а про качество данных", fn=lambda x: x.tf_minutes - x.col("n_minutes"))

EN = ("Entropy", "EN")


def _bin_entropy(p):
    p = p.clip(1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


family(*EN, "en_sign_{W}", grid(W=(24, 96)), formula="H(доля растущих баров за W), бит", kind="непрерывный", unit="бит [0,1]",
       deps=("close",), cost="O(n)", mechanism="Предсказуемость направления: 1 — монета, < 1 — перекос", measures="энтропия знаков доходностей за {W}",
       why="Мера случайности последовательности", limits="Не видит порядок",
       fn=lambda x, W: _bin_entropy((x.r > 0).astype(float).rolling(W, min_periods=W).mean()))


def _perm_entropy(x, W):
    c = x.c.to_numpy()
    a, b, d = c[:-2], c[1:-1], c[2:]
    code = np.full(len(c), -1)
    pats = np.select([(a < b) & (b < d), (a < d) & (d <= b), (d <= a) & (a < b), (b <= a) & (a < d), (b < d) & (d <= a)],
                     [0, 1, 2, 3, 4], 5)
    code[2:] = pats
    h = np.zeros(len(c))
    for k in range(6):
        p = pd.Series((code == k).astype(float), index=x.b.index).rolling(W, min_periods=W).mean()
        h = h - (p * np.log(p.where(p > 0))).fillna(0).to_numpy()
    out = pd.Series(h / np.log(6), index=x.b.index)
    return out.where(pd.Series(np.arange(len(c)) >= W + 1, index=x.b.index))


family(*EN, "en_permutation_{W}", grid(W=(24, 96, 288)), formula="−Σ p_k ln p_k / ln 6 по 6 порядковым узорам трёх закрытий за W",
       kind="непрерывный", unit="доля [0,1]", deps=("close",), cost="O(n)",
       mechanism="Упорядоченность локальной формы пути (Bandt–Pompe, 2002)", measures="перестановочная энтропия порядка 3 за {W}",
       why="Улавливает структуру, которую не видят моменты распределения", limits="Порядок 3 — грубо", fn=_perm_entropy)
family(*EN, "en_volume_{W}", grid(W=(24, 96)), formula="(ln S − Σ v ln v / S) / ln W, S = Σ_W v", kind="непрерывный",
       unit="доля [0,1]", deps=("volume",), cost="O(n)", mechanism="Равномерный объём против сосредоточенного",
       measures="энтропия распределения объёма по барам окна {W}", why="Сосредоточенность активности", limits="—",
       fn=lambda x, W: (np.log(x.v.rolling(W, min_periods=W).sum())
                        - safe_div((x.v * np.log(x.v.where(x.v > 0))).rolling(W, min_periods=W).sum(),
                                   x.v.rolling(W, min_periods=W).sum())) / np.log(W))
family(*EN, "en_abs_return_{W}", grid(W=(24, 96)), formula="энтропия долей |r| по барам окна W / ln W", kind="непрерывный",
       unit="доля [0,1]", deps=("close",), cost="O(n)", mechanism="Движение размазано по барам или сделано одним рывком",
       measures="энтропия распределения движения по барам окна {W}", why="Скачки против диффузии", limits="—",
       fn=lambda x, W: (lambda ar: (np.log(ar.rolling(W, min_periods=W).sum())
                                    - safe_div((ar * np.log(ar.where(ar > 0))).rolling(W, min_periods=W).sum(),
                                               ar.rolling(W, min_periods=W).sum())) / np.log(W))(x.r.abs()))
