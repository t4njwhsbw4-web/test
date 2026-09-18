"""Kalender-/Saisonalitäts-Strategie für Krypto-Tagesbars.

These
-----
Die Teilnehmerstruktur im Kryptomarkt schwankt mit der Kalenderzeit:

* **Wochenende**: institutionelle Desks und Market-Maker sind reduziert
  besetzt, die Liquidität ist dünner. Dünnere Bücher bedeuten höhere
  Preis-Impact-Kosten und potenziell andere Renditeverteilungen als an
  Werktagen.
* **Montag**: über das Wochenende aufgestaute Information (Makro,
  Nachrichten, Positionierungsentscheidungen) wird zum Wochenstart
  verarbeitet.
* **Monatswechsel (Turn-of-the-Month)**: Sparpläne, Gehaltseingänge und
  Allokationsentscheidungen erzeugen wiederkehrende Zuflüsse um den
  Monatswechsel.

Umsetzung
---------
Das Signal ist das logische UND aus drei Bedingungen:

1. **Wochentagsmaske** (`long_days`): an welchen Wochentagen überhaupt
   Exposure gehalten wird (0 = Montag ... 6 = Sonntag).
2. **Monatsphase** (`month_phase`): "all" | "tom" | "non_tom" |
   "first_half" | "second_half". "tom" = Turn-of-the-Month, definiert als
   die letzten `tom_last` Kalendertage eines Monats plus die ersten
   `tom_first` Tage des Folgemonats.
3. **Trendfilter** (`trend_sma`): 0 = aus, sonst nur long, wenn
   `close[t] > SMA(close, trend_sma)[t]`.

Alignment / Look-Ahead
----------------------
Der Backtest führt `signals.shift(1)` aus: die Position aus Bar t verdient
die Rendite von Bar t -> t+1. Ein Kalendereffekt, der die *am Wochentag X
realisierte* Rendite beschreibt, muss deshalb am Bar VOR diesem Wochentag
eingegangen werden.

Die Kalendermaske wird daher nicht auf dem Zeitstempel von Bar t gebildet,
sondern auf `t + 1 Tag` - dem Zeitstempel des Bars, dessen Rendite die
Position verdient. Das ist **kein** Look-Ahead: der Kalender von morgen ist
heute bekannt, es fliesst keine Preisinformation aus t+1 ein. Der
Trendfilter nutzt ausschliesslich `close` bis einschliesslich Bar t.
Nirgends wird ein `shift(-n)` auf Preisdaten angewendet.

Warnung
-------
Kalendereffekte sind der Lehrbuchfall von Data-Dredging. Bei 7 Wochentagen
x Monatsphasen x Trendfiltern findet man mit Sicherheit *irgendetwas*, das
im Backtest gut aussieht. Jeder Fund aus diesem Modul ist nur zusammen mit
(a) der Anzahl getesteter Kombinationen, (b) einer Nullverteilung des
*besten* Funds und (c) `yearly_sharpe` interpretierbar.
"""
from __future__ import annotations

import pandas as pd

ALL_DAYS = (0, 1, 2, 3, 4, 5, 6)
WEEKDAYS = (0, 1, 2, 3, 4)
WEEKEND = (5, 6)

MONTH_PHASES = ("all", "tom", "non_tom", "first_half", "second_half")


def _effective_calendar(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Zeitstempel des Bars, dessen Rendite die Position aus Bar t verdient.

    Wegen `signals.shift(1)` im Backtest ist das t + 1 Tag. Krypto handelt
    7 Tage/Woche, der nächste Bar ist also der nächste Kalendertag. Es wird
    nur der Kalender verschoben, keine Preisdaten.
    """
    return index + pd.Timedelta(days=1)


def _month_phase_mask(
    cal: pd.DatetimeIndex,
    month_phase: str,
    tom_last: int,
    tom_first: int,
) -> pd.Series:
    dom = pd.Series(cal.day, index=cal)
    days_in_month = pd.Series(cal.days_in_month, index=cal)

    if month_phase == "all":
        mask = pd.Series(True, index=cal)
    elif month_phase in ("tom", "non_tom"):
        is_tom = (dom <= tom_first) | (dom > days_in_month - tom_last)
        mask = is_tom if month_phase == "tom" else ~is_tom
    elif month_phase == "first_half":
        mask = dom <= (days_in_month / 2)
    elif month_phase == "second_half":
        mask = dom > (days_in_month / 2)
    else:
        raise ValueError(f"Unbekannte month_phase: {month_phase!r}")
    return mask


def generate_signals(
    df: pd.DataFrame,
    long_days: tuple[int, ...] = ALL_DAYS,
    month_phase: str = "all",
    tom_last: int = 2,
    tom_first: int = 3,
    trend_sma: int = 0,
) -> pd.Series:
    """Erzeugt 1 = long / 0 = flat für Kalender-Kombinationen.

    df: OHLCV mit DatetimeIndex (Spalte `close` wird für den Trendfilter
    benötigt). Rückgabe: Series auf demselben Index wie df.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df braucht einen DatetimeIndex.")
    if tom_last < 0 or tom_first < 0:
        raise ValueError("tom_last/tom_first müssen >= 0 sein.")

    cal = _effective_calendar(df.index)

    day_mask = pd.Series(pd.Index(cal.dayofweek).isin(long_days), index=df.index)
    phase_mask = _month_phase_mask(cal, month_phase, tom_last, tom_first)
    phase_mask.index = df.index

    signal = day_mask & phase_mask

    if trend_sma and trend_sma > 0:
        # close[t] ist zu Handelsschluss t bekannt -> kein Look-Ahead.
        sma = df["close"].rolling(int(trend_sma), min_periods=int(trend_sma)).mean()
        trend_ok = (df["close"] > sma).fillna(False)
        signal = signal & trend_ok

    return signal.astype(int).rename("signal")


# Die Grid-Grösse ist Teil des Ergebnisses: jede zusätzliche Achse erhöht die
# Wahrscheinlichkeit eines Zufallsfunds. Bewusst klein und ökonomisch
# motiviert gehalten statt "alle 127 Wochentags-Teilmengen".
PARAM_GRID: dict[str, list] = {
    "long_days": [
        ALL_DAYS,
        WEEKDAYS,
        WEEKEND,
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
        (0, 1),          # Montag/Dienstag: Wochenend-Information
        (4, 5, 6),       # verlängertes Wochenende
        (1, 2, 3, 4),    # Werktage ohne Montag
        (0, 1, 2, 3),    # Werktage ohne Freitag
        (0, 5, 6),       # Wochenende + Montag
        (2, 3),          # Wochenmitte
    ],
    "month_phase": list(MONTH_PHASES),
    "trend_sma": [0, 50, 100, 200],
}
