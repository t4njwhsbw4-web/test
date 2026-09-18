# Self-Learning Trading Bot – Grundgerüst

Ein Trading-Bot, der sich periodisch selbst neu trainiert – **kontrolliert**,
nicht als unbeaufsichtigtes Online-Learning mit echtem Geld. Diese
Unterscheidung ist die zentrale Design-Entscheidung dieses Projekts, siehe
[Warum kein Live-Online-Learning](#warum-kein-live-online-learning).

## Status

Infrastruktur steht (Datenpipeline, Feature-Engineering, Walk-Forward-Training,
Backtest-Engine, Risk-Manager, Paper-Trading-Loop). **Es wurde kein Live-Handel
mit echtem Geld durchgeführt, und es gibt derzeit keine Strategie, die das
verdienen würde** – siehe Ergebnisse unten.

## Ergebnisse der Strategie-Suche

Getestet wurden ~1700 Parameter-Kombinationen über fünf ökonomisch
unterschiedlich begründete Thesen, auf Krypto-Tagesbars (BTC/ETH/SOL/DOGE/LTC/ADA),
Entwicklungszeitraum 2020-09 bis 2025-09, mit einem versiegelten Hold-out-Jahr.

| These | Kombinationen | Ergebnis |
|---|---|---|
| Mean-Reversion (z-Score/Bollinger) | 354 | Kein Edge. Keine schlägt Buy-and-Hold. SMA-200-Trendfilter schadet durchgängig |
| Volatility-Breakout (Donchian/Squeeze) | 612 | Kern der These widerlegt: der Kompressionsfilter verschlechtert monoton |
| Kalender-/Saisonalitätseffekte | 320 | Rauschen. Familywise-Nulltest: p = 0.68 – bei 320 Tests ist ein besserer Zufallsfund zu erwarten |
| Cross-Sectional-Momentum (Rotation) | 360 | Dev-Sharpe 1.10 → **Hold-out-Sharpe -1.10**. Overfitting |
| Trendfolge + Vol-Targeting | ~60 | Kein Return-Edge, aber Drawdown-Schutz überträgt sich out-of-sample |

**Zentrale Erkenntnis:** Alle Ansätze, die *Rendite vorhersagen* wollten, sind
gescheitert. Das Einzige, was den Hold-out überlebt hat, ist *Risikomanagement*:
Trendfolge mit Volatilitäts-Targeting senkte den mittleren Drawdown von ~-70 %
(Buy-and-Hold) auf ~-20 %, bei nur 1–3 Trades pro Symbol und Jahr – ohne dabei
Gewinne zu erzeugen. Im Hold-out-Jahr (Bärenmarkt) verlor jede Variante Geld;
die beste Position wäre Cash gewesen, und keine Strategie hat das vorhergesagt.

### Warum der Hold-out der wichtigste Teil des Projekts ist

Cross-Sectional-Momentum bestand in der Entwicklungsphase jede Prüfung, die
man üblicherweise anlegt: keinen Look-Ahead-Bias (verifiziert durch zusätzliche
Ausführungsverzögerung), Grid-Median über dem Benchmark, 94 % des Parameterraums
besser als BTC, und gegen 300 Zufallsrotationen im 99.7. Perzentil. Trotzdem
brach die Strategie out-of-sample vollständig zusammen.

Hinzu kommt ein **Survivorship Bias**, der bei Rotationsstrategien besonders
stark wirkt: Die sechs getesteten Coins wurden im Rückblick ausgewählt, im
Wissen, welche überlebt haben. Ohne SOL – das 2020 niemand in ein Sechs-Coin-
Universum gelegt hätte – fällt der Dev-Sharpe von 1.10 auf 0.71. Ein
punkt-in-der-Zeit-korrektes Universum (Top-N nach Marktkapitalisierung zum
jeweiligen Datum) würde das beheben, ist über yfinance aber nicht verfügbar.

Ohne den versiegelten Hold-out hätte dieses Projekt echtes Geld auf eine
Strategie gesetzt, die im Folgejahr 46 % verloren hätte.

## Architektur

```
Daten (yfinance, gecacht)
        │
        ▼
Feature-Engineering (Momentum, RSI, SMA-Abstand, Volatilität, MACD)
        │
        ▼
Walk-Forward-Training (GradientBoosting, chronologisch, kein Look-Ahead)
        │
        ▼
Retrain-Orchestrierung (src/pipeline/retrain.py)
  → Kandidatenmodell muss aktuelles Produktionsmodell auf
    Out-of-Sample-Sharpe nachweisbar schlagen, sonst keine Beförderung
        │
        ▼
Paper-/Live-Trading-Loop (src/pipeline/run_paper.py)
  → Risk-Manager (unabhängig vom Modell) prüft VOR jedem Trade:
    Tagesverlust-Limit, Max-Drawdown-Kill-Switch, Positionsgröße,
    Stop-Loss/Take-Profit
        │
        ▼
Broker-Abstraktion (PaperBroker Default / AlpacaBroker mit explizitem
Live-Schalter)
```

## Warum kein Live-Online-Learning

Ein Modell, das während des Handelns mit echtem Geld laufend seine Gewichte
anpasst, lernt auf nicht-stationären, verrauschten Märkten genauso leicht
Rauschen wie echte Signale. Ohne strikte Out-of-Sample-Validierung führt das
zu Overfitting auf die jüngste Kursbewegung – und weil niemand mehr live
nachvollziehen kann, warum eine Entscheidung getroffen wurde, ist ein
Fehlverhalten oft erst nach Verlusten sichtbar.

Stattdessen: **geschlossene Retrain-Pipeline**. Modelle werden offline
trainiert, per Walk-Forward auf echten Out-of-Sample-Daten validiert und nur
bei nachweisbarer Verbesserung ins Trading übernommen. Der Risk-Manager
kennt das Modell nicht und lässt sich von ihm nicht umgehen.

## Marktwahl

Start mit **Krypto** (24/7-Daten, keine PDT-Regel, mehr Trainingsmaterial pro
Woche). **Nasdaq-Aktien** sind das Zielbild, aber als *Swing-Strategie auf
Tagesbasis* (nicht Intraday) – das umgeht die PDT-Regel (US-Broker verbieten
Konten unter 25.000 $ häufiges Day-Trading) und passt besser zur Frequenz
eines periodisch nachtrainierten Modells. In hochliquiden Nasdaq-Large-Caps
konkurriert ein Retail-Bot ohnehin mit institutionellen HFT-Systemen – dort
ist auf öffentlichen Tagesdaten kein struktureller Edge zu erwarten.

Konfigurierbar in `config/config.yaml` (`universe.mode: crypto | equities`).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # nur nötig für Alpaca-Anbindung, nicht für den Default (Paper)
```

## Nutzung

```bash
# 1. Modelle trainieren (Walk-Forward-Validierung + Promotion)
python3 -m src.pipeline.retrain

# 2. Paper-Trading-Durchlauf (Default-Broker: lokaler PaperBroker, kein Netzwerk nötig)
python3 -m src.pipeline.run_paper

# Tests (laufen komplett auf synthetischen Daten, kein Netzwerkzugriff nötig)
python3 -m pytest tests/ -v
```

Forschung / Strategie-Suche:

```bash
# Parameter-Grid über mehrere Assets (ML-Ansatz, dokumentiert als Negativergebnis)
python3 -m src.research.grid_search

# Einmaliger Hold-out-Test der vorregistrierten Kandidaten
python3 -m src.research.holdout_test
```

Die Strategie-Familien liegen in `src/strategies/`, jede mit
`generate_signals()` (bzw. `generate_weights()` bei Portfolio-Rotation) und
einem `PARAM_GRID`. `src/research/harness.py` kapselt den versiegelten
Hold-out: `load_data(symbol, period="dev")` für Entwicklung, `"holdout"` nur
für die finale, einmalige Bewertung.

Für periodisches Retraining/Trading außerhalb dieses Repos per Cron/Scheduler
einplanen (kein Dauerprozess im Code selbst) – siehe `config.retrain.schedule`
als Dokumentation der beabsichtigten Frequenz.

## Weg zu echtem Geld

**Voraussetzung, die derzeit nicht erfüllt ist:** Es muss eine Strategie
geben, die out-of-sample über mehrere Marktphasen hält. Nach ~1700 getesteten
Kombinationen gibt es die nicht. Solange das so bleibt, ist der einzige
sachlich richtige Schritt, *kein* echtes Geld anzuschließen – die Schritte
unten beschreiben nur, wie es technisch ginge, nicht dass es angebracht wäre.

1. `execution.broker: alpaca_paper` in `config/config.yaml`, `.env` mit
   Alpaca-Paper-Keys befüllen, mehrere Wochen/Monate laufen lassen.
2. Erst wenn Paper-Performance über einen längeren Zeitraum die Erwartungen
   trifft (nicht nur ein einzelner guter Backtest!): `AlpacaBroker` muss
   explizit mit `confirm_live=True` im Code instanziiert werden – ein
   Config-Flag allein reicht bewusst nicht aus, um mit echtem Geld zu
   handeln.
3. Klein anfangen, Kill-Switch-Limits (`risk.*` in der Config) konservativ
   halten und regelmäßig manuell prüfen.

## Bekannte Grenzen dieser Ausbaustufe

- Backtest ist long/flat only (keine Shorts), kein Slippage-Modell.
- Feature-Set ist bewusst klein gehalten (Overfitting-Risiko bei begrenzter
  Historie) – Erweiterung erst nach belastbarer Baseline sinnvoll.
- Noch keine automatisierte Scheduler-Infrastruktur (Cron liegt außerhalb
  des Repos).
- Keine Garantie für positive Rendite – die Walk-Forward-Validierung zeigt
  nur, ob ein Ansatz historisch überhaupt ein Signal hatte, keine
  Zukunftsgarantie.
- **Der Krypto-Hold-out ist verbraucht.** Er wurde einmal bewertet und deckt
  nur eine Marktphase ab (Bärenmarkt 2025/26). Weitere Kandidaten auf
  denselben Zeitraum zu testen, macht ihn zu einem zweiten Trainingsdatensatz.
  Neue Strategien brauchen einen neuen, vorher unberührten Zeitraum – oder
  längere Historie mit mehreren Regimen (Aktienindizes statt Krypto).
- **Survivorship Bias im Krypto-Universum** (siehe oben) ist nicht behoben und
  betrifft jede Strategie, die zwischen Assets auswählt.
