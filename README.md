# Self-Learning Trading Bot – Grundgerüst

Ein Trading-Bot, der sich periodisch selbst neu trainiert – **kontrolliert**,
nicht als unbeaufsichtigtes Online-Learning mit echtem Geld. Diese
Unterscheidung ist die zentrale Design-Entscheidung dieses Projekts, siehe
[Warum kein Live-Online-Learning](#warum-kein-live-online-learning).

## Status

Frühe Ausbaustufe: Datenpipeline, Feature-Engineering, Walk-Forward-Training,
Backtest-Engine, Risk-Manager und Paper-Trading-Loop stehen. **Es wurde noch
kein Live-Handel mit echtem Geld durchgeführt und die Strategie wurde noch
nicht über einen längeren Zeitraum validiert.**

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

Für periodisches Retraining/Trading außerhalb dieses Repos per Cron/Scheduler
einplanen (kein Dauerprozess im Code selbst) – siehe `config.retrain.schedule`
als Dokumentation der beabsichtigten Frequenz.

## Weg zu echtem Geld

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
