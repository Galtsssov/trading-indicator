"""Time, Calendar, Session: когда происходит бар. Часы сессий — в местном времени (zoneinfo, летнее время)."""
import numpy as np
import pandas as pd

from qlab.factory import family

DEPS = ("ts",)


def _cyc(values: pd.Series, period: float, fn) -> pd.Series:
    return fn(2 * np.pi * values / period)


def _local_minutes(x, tz: str) -> pd.Series:
    local = x.ts.dt.tz_convert(tz)
    return (local.dt.hour * 60 + local.dt.minute).astype(float)


def _min_since_local(x, tz: str, hour: int, minute: int, weekdays: bool) -> pd.Series:
    """Минут от последнего открытия (местное время) до закрытия бара; для бирж выходные пропускаются."""
    local = (x.ts + pd.Timedelta(minutes=x.tf_minutes)).dt.tz_convert(tz)
    now = (local.dt.hour * 60 + local.dt.minute).to_numpy()
    t = hour * 60 + minute
    back = (now < t).astype(int)                      # открытие сегодня ещё не наступило → вчерашнее
    if weekdays:
        dow = (local.dt.day_of_week.to_numpy() - back) % 7
        back = back + np.select([dow == 6, dow == 5], [2, 1], 0)   # воскресенье/суббота → пятница
    return pd.Series(now - t + back * 1440, index=x.b.index, dtype=float)


T = dict(category="Time", code="TM")
for name, period, fn, formula in (("tm_hour_sin", 24, np.sin, "sin(2π·час/24)"), ("tm_hour_cos", 24, np.cos, "cos(2π·час/24)"),
                                  ("tm_minute_sin", 60, np.sin, "sin(2π·минута/60)"), ("tm_minute_cos", 60, np.cos, "cos(2π·минута/60)")):
    unit = "час UTC" if "hour" in name else "минута часа"
    family(T["category"], T["code"], name, None, formula=formula + " (UTC)", kind="циклический", unit="[−1,1]",
           deps=DEPS, cost="O(n)", mechanism="Внутридневная сезонность объёма и волатильности",
           measures=f"{unit} в циклическом виде", why="Позволяет модели видеть время суток без разрыва 23→0",
           limits="Сезонность меняется с переходом на летнее время",
           fn=lambda x, p=period, f=fn, part=("hour" if "hour" in name else "minute"):
           _cyc(getattr(x.ts.dt, part).astype(float), p, f))
for name, fn in (("tm_dow_sin", np.sin), ("tm_dow_cos", np.cos)):
    family("Time", "TM", name, None, formula=f"{name[-3:]}(2π·день_недели/7)", kind="циклический", unit="[−1,1]",
           deps=DEPS, cost="O(n)", mechanism="Разный состав участников по дням", measures="день недели в циклическом виде",
           why="Выходные и понедельники ведут себя иначе", limits="Мало независимых недель",
           fn=lambda x, f=fn: _cyc(x.ts.dt.day_of_week.astype(float), 7, f))
family("Time", "TM", "tm_weekend", None, formula="1, если суббота или воскресенье (UTC)", kind="бинарный",
       unit="0/1", deps=DEPS, cost="O(n)", mechanism="Нет институционалов, тонкий рынок", measures="выходной ли день",
       why="Отдельный режим ликвидности", limits="Граница по UTC условна",
       fn=lambda x: (x.ts.dt.day_of_week >= 5).astype(float))
family("Time", "TM", "tm_min_to_funding", None, formula="минут до ближайшего 00:00 / 08:00 / 16:00 UTC",
       kind="непрерывный", unit="минуты", deps=DEPS, cost="O(n)",
       mechanism="Позиции закрывают/открывают к расчёту funding", measures="время до расчёта funding",
       why="Механический поток с известным временем", limits="Интервал funding у отдельных монет бывает 4ч",
       fn=lambda x: (480 - (_local_minutes(x, "UTC") + x.tf_minutes) % 480) % 480)
family("Time", "TM", "tm_min_since_utc_midnight", None, formula="минут с 00:00 UTC до закрытия бара", kind="непрерывный",
       unit="минуты", deps=DEPS, cost="O(n)", mechanism="Дневная свеча биржи, сброс дневных уровней",
       measures="время с начала суток UTC", why="Линейная шкала дня (в дополнение к циклической)",
       limits="Дублирует часы", fn=lambda x: _local_minutes(x, "UTC") + x.tf_minutes)
family("Time", "TM", "tm_min_since_ny_open", None, formula="минут с последнего открытия NYSE (09:30 Нью-Йорк, будни)",
       kind="непрерывный", unit="минуты", deps=DEPS, cost="O(n)",
       mechanism="Приход американских институционалов и ETF-потоков", measures="время с открытия американской сессии",
       why="Корректно с летним временем, в отличие от фиксированных часов UTC", limits="Праздники NYSE не учтены",
       fn=lambda x: _min_since_local(x, "America/New_York", 9, 30, True))
family("Time", "TM", "tm_min_since_london_open", None, formula="минут с последнего открытия Лондона (08:00 местного, будни)",
       kind="непрерывный", unit="минуты", deps=DEPS, cost="O(n)", mechanism="Приход европейского потока",
       measures="время с открытия европейской сессии", why="С учётом летнего времени", limits="Праздники не учтены",
       fn=lambda x: _min_since_local(x, "Europe/London", 8, 0, True))
family("Time", "TM", "tm_us_dst", None, formula="1, если в Нью-Йорке летнее время", kind="бинарный", unit="0/1",
       deps=DEPS, cost="O(n)", mechanism="Сдвиг американской сессии на час по UTC", measures="действует ли летнее время в США",
       why="Объясняет «сезонные» сдвиги эффектов по часам UTC", limits="Почти календарный признак",
       fn=lambda x: pd.Series((x.ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
                               - x.ts.dt.tz_convert("UTC").dt.tz_localize(None)).dt.total_seconds().to_numpy()
                              == -4 * 3600, index=x.b.index, dtype=float))

C = ("Calendar", "CL")
family(*C, "cl_day_of_month", None, formula="число месяца (UTC)", kind="счётчик", unit="день", deps=DEPS, cost="O(n)",
       mechanism="Ребалансировки и выплаты в начале/конце месяца", measures="день месяца", why="Календарные потоки",
       limits="12 месяцев в году — мало наблюдений", fn=lambda x: x.ts.dt.day.astype(float))
family(*C, "cl_days_to_month_end", None, formula="дней до конца месяца", kind="счётчик", unit="дни", deps=DEPS,
       cost="O(n)", mechanism="Ребаланс фондов и закрытие месячных опционов", measures="близость конца месяца",
       why="Отдельно от дня месяца: месяцы разной длины", limits="Мало наблюдений",
       fn=lambda x: (x.ts.dt.days_in_month - x.ts.dt.day).astype(float))
for name, fn in (("cl_month_sin", np.sin), ("cl_month_cos", np.cos)):
    family(*C, name, None, formula=f"{name[-3:]}(2π·месяц/12)", kind="циклический", unit="[−1,1]", deps=DEPS, cost="O(n)",
           mechanism="Сезонность года", measures="месяц в циклическом виде", why="Годовая сезонность",
           limits="6 лет — 6 наблюдений на месяц", fn=lambda x, f=fn: _cyc(x.ts.dt.month.astype(float) - 1, 12, f))
family(*C, "cl_quarter_end_week", None, formula="1, если до конца квартала ≤ 7 дней", kind="бинарный", unit="0/1",
       deps=DEPS, cost="O(n)", mechanism="Квартальные экспирации фьючерсов и опционов, ребаланс",
       measures="последняя неделя квартала", why="Механические потоки экспирации", limits="4 события в год",
       fn=lambda x: ((x.ts.dt.month % 3 == 0) & (x.ts.dt.days_in_month - x.ts.dt.day < 7)).astype(float))
family(*C, "cl_hours_to_weekly_expiry", None, formula="часов до ближайшей пятницы 08:00 UTC (экспирация Deribit)",
       kind="непрерывный", unit="часы", deps=DEPS, cost="O(n)", mechanism="Хеджирование дилеров перед экспирацией (pinning)",
       measures="время до еженедельной экспирации опционов", why="Опционный поток с известным временем",
       limits="Опционы значимы в основном для BTC и ETH",
       fn=lambda x: ((4 - x.ts.dt.day_of_week) % 7 * 24 + 8 - x.ts.dt.hour - x.ts.dt.minute / 60) % 168)
family(*C, "cl_last_friday", None, formula="1, если последняя пятница месяца", kind="бинарный", unit="0/1", deps=DEPS,
       cost="O(n)", mechanism="Месячная экспирация опционов", measures="день месячной экспирации",
       why="Самая крупная регулярная экспирация", limits="12 событий в год",
       fn=lambda x: ((x.ts.dt.day_of_week == 4) & (x.ts.dt.day + 7 > x.ts.dt.days_in_month)).astype(float))
family(*C, "cl_week_of_month", None, formula="ceil(число / 7)", kind="счётчик", unit="1–5", deps=DEPS, cost="O(n)",
       mechanism="Внутримесячная сезонность (зарплаты, выплаты)", measures="неделя месяца", why="Грубая месячная фаза",
       limits="Слабая гипотеза для крипты", fn=lambda x: np.ceil(x.ts.dt.day / 7).astype(float))

S = ("Session", "SS")
SESSION_OPENS = (("asia", "UTC", 0, 0, False), ("london", "Europe/London", 8, 0, True), ("ny", "America/New_York", 9, 30, True))
family(*S, "ss_code", None, formula="0 asia (00–07 UTC), 1 europe (07–13), 2 us (13–20), 3 late (20–24)",
       kind="категориальный", unit="0–3", deps=DEPS, cost="O(n)", mechanism="Разный состав участников по сессиям",
       measures="торговая сессия по часу UTC", why="Грубая, но устойчивая разметка дня", limits="Фиксированные часы UTC",
       fn=lambda x: pd.Series(np.searchsorted([0, 7, 13, 20], x.ts.dt.hour.to_numpy(), side="right") - 1.0, index=x.b.index))


def _day_stats(x):
    def make():
        g = x.day
        return pd.DataFrame({"hi": x.h.groupby(g).cummax(), "lo": x.l.groupby(g).cummin(),
                             "op": x.o.groupby(g).transform("first")})
    return x.memo("day_stats", make)


family(*S, "ss_day_range", None, formula="(max h − min l с 00:00 UTC) / a", kind="непрерывный", unit="ATR",
       deps=("high", "low", "atr", "ts"), cost="O(n)", mechanism="Использованная часть дневного хода",
       measures="размах текущих суток в ATR", why="Сколько дневной волатильности уже реализовано",
       limits="Сутки по UTC условны", fn=lambda x: (_day_stats(x)["hi"] - _day_stats(x)["lo"]) / x.a)
family(*S, "ss_day_pos", None, formula="(c − min l дня) / (max h дня − min l дня)", kind="непрерывный", unit="доля [0,1]",
       deps=("high", "low", "close", "ts"), cost="O(n)", mechanism="Положение в дневном диапазоне",
       measures="где цена внутри текущих суток", why="Объективная «премия/дисконт» дня", limits="В начале суток шумно",
       fn=lambda x: safe(x.c - _day_stats(x)["lo"], _day_stats(x)["hi"] - _day_stats(x)["lo"]))
family(*S, "ss_day_ret", None, formula="(c − open дня) / a", kind="непрерывный", unit="ATR", deps=("open", "close", "atr", "ts"),
       cost="O(n)", mechanism="Якорение к открытию дня", measures="изменение цены с начала суток UTC в ATR",
       why="Направление дня", limits="Сутки по UTC условны", fn=lambda x: (x.c - _day_stats(x)["op"]) / x.a)


def safe(a, b):
    return a / b.where(b != 0)


for name, tz, hh, mm, wd in SESSION_OPENS:
    family(*S, f"ss_min_since_{name}_open", None, formula=f"минут с последнего открытия {name} ({hh:02d}:{mm:02d} {tz})",
           kind="непрерывный", unit="минуты", deps=DEPS, cost="O(n)", mechanism="Фаза сессии: открытие, середина, закрытие",
           measures=f"время с открытия сессии {name}", why="Эффекты открытия затухают со временем",
           limits="Праздники не учтены", fn=lambda x, tz=tz, hh=hh, mm=mm, wd=wd: _min_since_local(x, tz, hh, mm, wd))
