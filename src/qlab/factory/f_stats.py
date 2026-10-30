"""Statistical, Rolling Statistics, Percentiles, Z-Score: форма распределения доходностей и место текущего бара в нём."""
import numpy as np
import pandas as pd

from qlab.factory import family, grid, rolling_pctl, rolling_z, safe_div

ST = ("Statistical", "ST")
family(*ST, "st_skew_{W}", grid(W=(24, 96, 288)), formula="skew_W(r)", kind="непрерывный", unit="безразмерно", deps=("close",),
       cost="O(n)", mechanism="Асимметрия доходностей: перекос рисков в одну сторону", measures="асимметрия доходностей за {W} баров",
       why="Хвостовой риск направленно", limits="Очень чувствительна к выбросам",
       fn=lambda x, W: x.r.rolling(W, min_periods=W).skew())
family(*ST, "st_kurt_{W}", grid(W=(24, 96, 288)), formula="эксцесс_W(r)", kind="непрерывный", unit="безразмерно", deps=("close",),
       cost="O(n)", mechanism="Толстые хвосты — скачки, каскады", measures="эксцесс доходностей за {W} баров",
       why="Вероятность экстремальных баров", limits="Чувствителен к выбросам",
       fn=lambda x, W: x.r.rolling(W, min_periods=W).kurt())
family(*ST, "st_autocorr1_{W}", grid(W=(24, 96, 288)), formula="corr_W(r_t, r_(t−1))", kind="непрерывный", unit="[−1,1]",
       deps=("close",), cost="O(n)", mechanism="Отрицательная — возврат (отскок спреда, ликвидность), положительная — моментум",
       measures="автокорреляция доходностей с лагом 1 за {W} баров", why="Текущая персистентность на коротком горизонте",
       limits="Шумна; отскок bid/ask смещает в минус", fn=lambda x, W: x.r.rolling(W, min_periods=W).corr(x.r.shift()))
family(*ST, "st_autocorr2_288", None, formula="corr_288(r_t, r_(t−2))", kind="непрерывный", unit="[−1,1]", deps=("close",),
       cost="O(n)", mechanism="Память второго порядка", measures="автокорреляция с лагом 2", why="Структура памяти",
       limits="Шумна", fn=lambda x: x.r.rolling(288, min_periods=288).corr(x.r.shift(2)))
family(*ST, "st_sign_changes_{W}", grid(W=(24, 96)), formula="mean_W[sign r_t ≠ sign r_(t−1)]", kind="непрерывный",
       unit="доля [0,1]", deps=("close",), cost="O(n)", mechanism="Частые смены направления — флэт/шум (тест серий)",
       measures="доля смен знака доходности за {W} баров", why="Непараметрическая мера персистентности", limits="Игнорирует размер",
       fn=lambda x, W: (np.sign(x.r) != np.sign(x.r.shift())).astype(float).rolling(W, min_periods=W).mean())
family(*ST, "st_mean_median_96", None, formula="(mean_96 r − median_96 r) / std_96 r", kind="непрерывный", unit="σ",
       deps=("close",), cost="O(n log W)", mechanism="Устойчивая к выбросам асимметрия", measures="разница среднего и медианы доходностей",
       why="Асимметрия без чувствительности к одному бару", limits="—",
       fn=lambda x: (x.r.rolling(96, min_periods=96).mean() - x.r.rolling(96, min_periods=96).median()) / x.rv(96))

RS = ("Rolling Statistics", "RL")
family(*RS, "rl_drift_{W}", grid(W=(24, 96, 288)), formula="mean_W(r) / std_W(r) · √W", kind="непрерывный", unit="t",
       deps=("close",), cost="O(n)", mechanism="Устойчивый дрейф против шума", measures="t-статистика среднего дохода за {W} баров",
       why="Сила тренда с поправкой на шум", limits="≈ нормированная доходность окна",
       fn=lambda x, W: x.r.rolling(W, min_periods=W).mean() / x.rv(W) * np.sqrt(W))
family(*RS, "rl_median_abs_{W}", grid(W=(96, 288)), formula="median_W |r| / std_W r", kind="непрерывный", unit="отношение",
       deps=("close",), cost="O(n log W)", mechanism="Мала при толстых хвостах (много тихих баров и редкие скачки)",
       measures="типичный бар относительно стандартного отклонения за {W}", why="Форма распределения просто", limits="—",
       fn=lambda x, W: x.r.abs().rolling(W, min_periods=W).median() / x.rv(W))
for ext, f in (("max", "max"), ("min", "min")):
    family(*RS, f"rl_{ext}_r_{{W}}", grid(W=(24, 288)), formula=f"{ext}_W(r) / std_W(r)", kind="непрерывный", unit="σ",
           deps=("close",), cost="O(n)", mechanism="Самый сильный бар окна", measures=f"самый {'сильный рост' if ext == 'max' else 'сильное падение'} за {{W}} баров в σ",
           why="Был ли в окне выброс", limits="—",
           fn=lambda x, W, f=f: getattr(x.r.rolling(W, min_periods=W), f)() / x.rv(W))

PC = ("Percentiles", "PC")
for name, label, series, wins in (("pc_ret", "доходности бара", lambda x: x.r, (96, 288, 8640)),
                                  ("pc_range", "размаха бара", lambda x: x.rng / x.c, (288, 8640)),
                                  ("pc_volume", "объёма", lambda x: x.v, (288, 8640)),
                                  ("pc_close", "закрытия", lambda x: x.c, (96, 288))):
    family(*PC, f"{name}_{{W}}", grid(W=wins), formula=f"процентиль ({label})_t среди последних W баров", kind="непрерывный",
           unit="доля [0,1]", deps=("close", "high", "low", "volume"), cost="O(n log W)",
           mechanism="Ранг устойчив к выбросам и сравним между монетами", measures=f"ранг {label} за {{W}} баров",
           why="Экстремальность без допущений о распределении", limits="Окно 8640 = 30 суток на 5m",
           fn=lambda x, W, series=series: rolling_pctl(series(x), W))

Z = ("Z-Score", "ZS")
for name, label, series in (("zs_ret", "доходности", lambda x: x.r), ("zs_range", "размаха / цены", lambda x: x.rng / x.c)):
    family(*Z, f"{name}_{{W}}", grid(W=(96, 288)), formula=f"z_W({label})", kind="непрерывный", unit="σ", deps=("close", "high", "low"),
           cost="O(n)", mechanism="Отклонение от обычного в единицах разброса", measures=f"z-оценка {label} за {{W}} баров",
           why="Сила текущего бара относительно окна", limits="Чувствительна к распределению",
           fn=lambda x, W, series=series: rolling_z(series(x), W))
family(*Z, "zs_trades_288", None, formula="z_288( ln trades )", kind="непрерывный", unit="σ", deps=("trades",), cost="O(n)",
       mechanism="Аномальная активность сделок", measures="число сделок в σ", why="Активность в единицах разброса", limits="—",
       fn=lambda x: rolling_z(np.log(x.col("trades").where(x.col("trades") > 0)), 288))
family(*Z, "zs_intrabar_rv_288", None, formula="z_288( ib_rv )", kind="непрерывный", unit="σ", deps=("ib_rv",), cost="O(n)",
       mechanism="Аномальная волатильность внутри бара", measures="внутрибаровая волатильность в σ",
       why="Видит бурю внутри бара с невинной свечой", limits="—", fn=lambda x: rolling_z(x.col("ib_rv"), 288))
