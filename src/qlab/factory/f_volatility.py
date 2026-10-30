"""Volatility: уровень, структура и динамика волатильности."""
import numpy as np

from qlab.factory import WIN, family, grid, rolling_pctl, safe_div

CAT, CODE = "Volatility", "VL"

family(CAT, CODE, "vl_rv_{W}", grid(W=WIN), formula="std_W(r)", kind="непрерывный", unit="доля (лог-доходность)",
       deps=("close",), cost="O(n)", mechanism="Кластеризация волатильности: она предсказуема лучше цены",
       measures="реализованная волатильность закрытий за {W} баров", why="Базовый масштаб риска",
       limits="Не видит движения внутри баров", fn=lambda x, W: x.rv(W))
family(CAT, CODE, "vl_parkinson_{W}", grid(W=WIN), formula="sqrt(mean_W (ln(h/l))² / (4 ln 2))",
       kind="непрерывный", unit="доля", deps=("high", "low"), cost="O(n)",
       mechanism="Размах бара эффективнее закрытий оценивает дисперсию (Parkinson, 1980)",
       measures="волатильность по размаху за {W} баров", why="Точнее std закрытий на тех же данных",
       limits="Смещена при гэпах и трендах", fn=lambda x, W: np.sqrt((np.log(x.h / x.l) ** 2).rolling(W, min_periods=W).mean() / (4 * np.log(2))))
family(CAT, CODE, "vl_garman_klass_{W}", grid(W=(24, 288)),
       formula="sqrt(mean_W [0.5 (ln h/l)² − (2 ln 2 − 1)(ln c/o)²])", kind="непрерывный", unit="доля",
       deps=("open", "high", "low", "close"), cost="O(n)", mechanism="Использует все четыре цены бара (Garman–Klass, 1980)",
       measures="волатильность по OHLC за {W} баров", why="Ещё эффективнее Parkinson", limits="Предполагает отсутствие дрейфа",
       fn=lambda x, W: np.sqrt((0.5 * np.log(x.h / x.l) ** 2 - (2 * np.log(2) - 1) * np.log(x.c / x.o) ** 2)
                               .rolling(W, min_periods=W).mean().clip(lower=0)))
family(CAT, CODE, "vl_rogers_satchell_{W}", grid(W=(24, 288)),
       formula="sqrt(mean_W [ln(h/c)ln(h/o) + ln(l/c)ln(l/o)])", kind="непрерывный", unit="доля",
       deps=("open", "high", "low", "close"), cost="O(n)", mechanism="Оценка, устойчивая к дрейфу (Rogers–Satchell, 1991)",
       measures="волатильность по OHLC с поправкой на тренд за {W} баров", why="Корректна в трендовых периодах",
       limits="Не учитывает гэпы между барами",
       fn=lambda x, W: np.sqrt((np.log(x.h / x.c) * np.log(x.h / x.o) + np.log(x.l / x.c) * np.log(x.l / x.o))
                               .rolling(W, min_periods=W).mean().clip(lower=0)))
family(CAT, CODE, "vl_ratio_{S}_{L}", [{"S": 6, "L": 96}, {"S": 24, "L": 288}], formula="std_{S}(r) / std_{L}(r)",
       kind="непрерывный", unit="отношение", deps=("close",), cost="O(n)",
       mechanism="Короткая волатильность выше длинной — режим ускорения", measures="отношение волатильности {S} и {L} баров",
       why="Ловит смену режима раньше уровня", limits="Шумно при малом S", fn=lambda x, S, L: x.rv(S) / x.rv(L))
family(CAT, CODE, "vl_atr_pctl_{D}d", grid(D=(7, 30, 90)), formula="процентиль ATR среди последних D суток",
       kind="непрерывный", unit="доля [0,1]", deps=("atr",), cost="O(n log W)",
       mechanism="Относительная волатильность важнее абсолютной", measures="ранг текущего ATR за {D} суток",
       why="Сравнимо между монетами и периодами", limits="Медленно реагирует при длинном окне",
       fn=lambda x, D: rolling_pctl(x.col("atr"), D * x.bpd))
family(CAT, CODE, "vl_semivar_ratio_{W}", grid(W=(24, 96, 288)), formula="0.5·ln( Σ_W r²[r>0] / Σ_W r²[r<0] )",
       kind="непрерывный", unit="ln-отношение", deps=("close",), cost="O(n)",
       mechanism="Асимметрия: паника быстрее жадности", measures="перевес волатильности вверх над волатильностью вниз за {W} баров",
       why="Направленная информация внутри волатильности", limits="Неустойчива на коротких окнах",
       fn=lambda x, W: 0.5 * np.log(safe_div((x.r.clip(lower=0) ** 2).rolling(W, min_periods=W).sum(),
                                             (x.r.clip(upper=0) ** 2).rolling(W, min_periods=W).sum())))
family(CAT, CODE, "vl_vol_of_vol_{W}", grid(W=(96, 288)), formula="std_W(std_6(r)) / mean_W(std_6(r))",
       kind="непрерывный", unit="коэффициент вариации", deps=("close",), cost="O(n)",
       mechanism="Нестабильная волатильность — частые выбивания стопов", measures="изменчивость короткой волатильности за {W} баров",
       why="Риск стопов независимо от уровня волатильности", limits="Вторичная оценка, шумная",
       fn=lambda x, W: x.rv(6).rolling(W, min_periods=W).std() / x.rv(6).rolling(W, min_periods=W).mean())
family(CAT, CODE, "vl_jump_share_{W}", grid(W=(24, 288)), formula="max_W r² / Σ_W r²", kind="непрерывный",
       unit="доля [0,1]", deps=("close",), cost="O(n)", mechanism="Скачок (новость, каскад) против диффузии",
       measures="доля дисперсии окна {W} баров, пришедшаяся на один бар", why="Отличает скачки от плавного движения",
       limits="Не говорит, когда был скачок",
       fn=lambda x, W: safe_div((x.r ** 2).rolling(W, min_periods=W).max(), (x.r ** 2).rolling(W, min_periods=W).sum()))
