"""Order Flow Proxy и позиционирование: агрессоры (taker buy), открытый интерес, funding, long/short."""
import numpy as np
import pandas as pd

from qlab.factory import family, grid, rolling_pctl, rolling_z, safe_div

CAT, CODE = "Order Flow Proxy", "OF"


def _share(x):
    return x.memo("taker_share", lambda: safe_div(x.col("taker_buy_volume"), x.v) - 0.5)


family(CAT, CODE, "of_taker_share", None, formula="taker_buy_volume / volume − 0.5", kind="непрерывный", unit="[−0.5,0.5]",
       deps=("taker_buy_volume", "volume"), cost="O(n)", mechanism="Кто был агрессором: покупатели или продавцы",
       measures="перевес агрессивных покупок в баре", why="Прямой след направления рыночных ордеров",
       limits="Агрессор не всегда информирован; ликвидации — тоже рыночные ордера", fn=_share)
family(CAT, CODE, "of_taker_share_{W}", grid(W=(6, 24, 96, 288)), formula="Σ_W taker_buy / Σ_W v − 0.5", kind="непрерывный",
       unit="[−0.5,0.5]", deps=("taker_buy_volume", "volume"), cost="O(n)", mechanism="Накопленное давление агрессоров (CVD-прокси)",
       measures="перевес агрессивных покупок за {W} баров", why="Устойчивое давление против разового",
       limits="Не отличает поглощённую агрессию от продвинувшей цену",
       fn=lambda x, W: safe_div(x.col("taker_buy_volume").rolling(W, min_periods=W).sum(), x.v.rolling(W, min_periods=W).sum()) - 0.5)
family(CAT, CODE, "of_taker_z_288", None, formula="z_288(taker_share)", kind="непрерывный", unit="σ",
       deps=("taker_buy_volume", "volume"), cost="O(n)", mechanism="Аномальность перекоса агрессоров",
       measures="перекос агрессоров в единицах его разброса", why="Сравнимо между монетами", limits="—",
       fn=lambda x: rolling_z(_share(x), 288))
family(CAT, CODE, "of_delta_ret_corr_{W}", grid(W=(96, 288)), formula="corr_W(taker_share, r)", kind="непрерывный",
       unit="[−1,1]", deps=("taker_buy_volume", "volume", "close"), cost="O(n)",
       mechanism="Насколько агрессия двигает цену: низкая связь — поглощение лимитками",
       measures="связь перекоса агрессоров с доходностью за {W} баров", why="Прокси эффективности потока",
       limits="Связь одновременная и почти механическая", fn=lambda x, W: _share(x).rolling(W, min_periods=W).corr(x.r))
family(CAT, CODE, "of_effort_result_{W}", grid(W=(6, 24)),
       formula="ln( (Σ_W v / (W·mean_288 v)) / (|c − c_(t−W)|/a + 0.25) )", kind="непрерывный", unit="ln-отношение",
       deps=("volume", "close", "atr"), cost="O(n)", mechanism="Много объёма при малом сдвиге — поглощение (усилие против результата)",
       measures="объём на единицу пройденной цены за {W} баров", why="Объективная мера абсорбции",
       limits="Константа 0.25 сглаживает деление на малый сдвиг",
       fn=lambda x, W: np.log(safe_div(x.v.rolling(W).sum(), W * x.v.rolling(288, min_periods=288).mean())
                              / ((x.c - x.c.shift(W)).abs() / x.a + 0.25)))
family(CAT, CODE, "of_delta_price_gap_24", None, formula="z_288(Σ_24 taker_share) − z_288(ln(c/c_(t−24)))",
       kind="непрерывный", unit="σ", deps=("taker_buy_volume", "volume", "close"), cost="O(n)",
       mechanism="Агрессия есть, а цена не идёт (или наоборот) — расхождение потока и цены",
       measures="расхождение накопленной агрессии и движения цены за 24 бара", why="Объективная «дивергенция потока»",
       limits="Разность z-оценок шумная",
       fn=lambda x: rolling_z(_share(x).rolling(24).sum(), 288) - rolling_z(x.ret(24), 288))

P, PC = "Positioning", "PO"


def _oi(x):
    return x.memo("oi", lambda: np.log(x.col("oi").where(x.col("oi") > 0)))


family(P, PC, "po_oi_chg_{W}", grid(W=(1, 6, 24, 288)), formula="ln(OI_t / OI_(t−W)), OI известен с задержкой публикации",
       kind="непрерывный", unit="доля", deps=("oi",), cost="O(n)", mechanism="Рост OI — открытие позиций, падение — закрытие",
       measures="изменение открытого интереса за {W} баров", why="Отличает новые позиции от закрытия старых",
       limits="Только Binance; снимки раз в 5 минут", fn=lambda x, W: _oi(x) - _oi(x).shift(W))
family(P, PC, "po_oi_z_288", None, formula="z_288( ln(OI_t/OI_(t−1)) )", kind="непрерывный", unit="σ", deps=("oi",),
       cost="O(n)", mechanism="Аномальное изменение позиций", measures="изменение OI в единицах его разброса",
       why="Сравнимо между монетами", limits="—", fn=lambda x: rolling_z(_oi(x).diff(), 288))
family(P, PC, "po_oi_price_corr_{W}", grid(W=(96, 288)), formula="corr_W( ΔlnOI, r )", kind="непрерывный", unit="[−1,1]",
       deps=("oi", "close"), cost="O(n)", mechanism="Рост цены с ростом OI — новые лонги; с падением OI — закрытие шортов",
       measures="связь изменения OI и цены за {W} баров", why="Тип движения: открытие или закрытие позиций", limits="—",
       fn=lambda x, W: _oi(x).diff().rolling(W, min_periods=W).corr(x.r))
family(P, PC, "po_oi_pctl_30d", None, formula="процентиль OI за 30 суток", kind="непрерывный", unit="доля [0,1]",
       deps=("oi",), cost="O(n log W)", mechanism="Высокий OI — много плеча в системе", measures="уровень OI относительно месяца",
       why="Потенциал каскадов", limits="Растущий рынок смещает уровень", fn=lambda x: rolling_pctl(_oi(x), 30 * x.bpd))
family(P, PC, "po_funding", None, formula="последняя опубликованная ставка funding", kind="непрерывный", unit="доля за период",
       deps=("funding_rate",), cost="O(1)", mechanism="Перекос толпы: платят те, кого больше",
       measures="текущая ставка funding", why="Позиционирование толпы", limits="Меняется раз в 8 часов",
       fn=lambda x: x.col("funding_rate"))
family(P, PC, "po_funding_z_30d", None, formula="z( funding ) за 30 суток баров", kind="непрерывный", unit="σ",
       deps=("funding_rate",), cost="O(n)", mechanism="Экстремальный перекос толпы", measures="funding относительно месяца",
       why="Сравнимо между режимами", limits="Ступенчатый ряд", fn=lambda x: rolling_z(x.col("funding_rate"), 30 * x.bpd))
family(P, PC, "po_funding_chg", None, formula="funding_t − funding 8 часов назад", kind="непрерывный", unit="доля",
       deps=("funding_rate",), cost="O(n)", mechanism="Быстрое набирание плеча одной стороной",
       measures="изменение ставки funding за 8 часов", why="Динамика перекоса", limits="Ступенчато",
       fn=lambda x: x.col("funding_rate") - x.col("funding_rate").shift(8 * 60 // x.tf_minutes))
family(P, PC, "po_ls_accounts", None, formula="long/short ratio аккаунтов (Binance, с задержкой публикации)",
       kind="непрерывный", unit="отношение", deps=("ls_ratio_accounts",), cost="O(1)", mechanism="Розница как контрарный индикатор",
       measures="отношение аккаунтов в лонге к аккаунтам в шорте", why="Позиционирование розницы",
       limits="Одна биржа; число аккаунтов, а не объём", fn=lambda x: x.col("ls_ratio_accounts"))
family(P, PC, "po_ls_accounts_chg_24", None, formula="ln(LS_t / LS_(t−24))", kind="непрерывный", unit="доля",
       deps=("ls_ratio_accounts",), cost="O(n)", mechanism="Розница набирает одну сторону",
       measures="изменение long/short аккаунтов за 24 бара", why="Динамика толпы", limits="—",
       fn=lambda x: np.log(safe_div(x.col("ls_ratio_accounts"), x.col("ls_ratio_accounts").shift(24))))
family(P, PC, "po_taker_ls_ratio", None, formula="ln(taker long/short volume ratio) из метрик Binance", kind="непрерывный",
       unit="ln-отношение", deps=("taker_ls_vol_ratio",), cost="O(1)", mechanism="Агрессия по всему рынку фьючерсов Binance",
       measures="перевес агрессивных лонгов по метрикам Binance", why="Независимый от свечей источник агрессии",
       limits="Снимки раз в 5 минут с задержкой", fn=lambda x: np.log(x.col("taker_ls_vol_ratio").where(x.col("taker_ls_vol_ratio") > 0)))
