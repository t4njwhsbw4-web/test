"""Haupt-Loop: lädt Produktionsmodelle, generiert Signale, handelt über den
konfigurierten Broker (Default: PaperBroker) unter Aufsicht des RiskManagers.

Gedacht zum periodischen Ausführen (z.B. 1x täglich nach Handelsschluss via
Cron/Scheduler außerhalb dieses Skripts) - kein Dauerprozess, kein Live-
Weight-Update während des Laufs.
"""
from __future__ import annotations

import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.data.fetch import fetch_ohlcv, to_yahoo_symbol
from src.execution.broker import AlpacaBroker, Broker, PaperBroker
from src.features.engineer import build_features
from src.models.train import load_latest_model
from src.risk.manager import AccountState, KillSwitchTriggered, RiskLimits, RiskManager
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_paper")


def build_broker(config: dict) -> Broker:
    exec_cfg = config["execution"]
    if exec_cfg["broker"] == "paper":
        return PaperBroker(starting_cash=config["paper_trading"]["starting_cash"])
    if exec_cfg["broker"] == "alpaca_paper":
        return AlpacaBroker(confirm_live=False)
    if exec_cfg["broker"] == "alpaca_live":
        raise RuntimeError(
            "alpaca_live erfordert einen expliziten Aufruf mit confirm_live=True im Code - "
            "kein Umschalten allein über config.yaml. Siehe src/execution/broker.py."
        )
    raise ValueError(f"Unbekannter Broker: {exec_cfg['broker']!r}")


def build_risk_manager(config: dict) -> RiskManager:
    risk_cfg = config["risk"]
    limits = RiskLimits(
        max_position_pct=risk_cfg["max_position_pct"],
        max_daily_loss_pct=risk_cfg["max_daily_loss_pct"],
        max_drawdown_pct=risk_cfg["max_drawdown_pct"],
        stop_loss_pct=risk_cfg["stop_loss_pct"],
        take_profit_pct=risk_cfg["take_profit_pct"],
        max_open_positions=risk_cfg["max_open_positions"],
    )
    return RiskManager(limits)


def run_once(config: dict, broker: Broker, risk_manager: RiskManager, peak_equity: float) -> float:
    universe_cfg = config["universe"]
    data_cfg = config["data"]
    feat_cfg = config["features"]
    model_cfg = config["model"]

    symbols = universe_cfg["symbols"][universe_cfg["mode"]]
    starting_equity_today = broker.get_equity()
    peak_equity = max(peak_equity, starting_equity_today)

    for symbol in symbols:
        yahoo_symbol = to_yahoo_symbol(symbol)
        raw = fetch_ohlcv(yahoo_symbol, history_days=data_cfg["history_days"], cache_dir=data_cfg["cache_dir"])
        features = build_features(
            raw,
            return_horizons=feat_cfg["return_horizons"],
            rsi_period=feat_cfg["rsi_period"],
            sma_windows=feat_cfg["sma_windows"],
            volatility_window=feat_cfg["volatility_window"],
        ).dropna()

        if features.empty:
            logger.warning("%s: Keine verwertbaren Features, überspringe.", symbol)
            continue

        try:
            model = load_latest_model(model_cfg["artifact_dir"], symbol)
        except FileNotFoundError:
            logger.warning("%s: Kein trainiertes Modell vorhanden, überspringe (erst retrain.py ausführen).", symbol)
            continue

        latest_features = features.iloc[[-1]]
        proba_up = model.predict_proba(latest_features)[0, 1]
        current_price = float(raw["close"].iloc[-1])
        position = broker.get_position(symbol)

        account = AccountState(
            equity=broker.get_equity(),
            starting_equity_today=starting_equity_today,
            peak_equity=peak_equity,
            open_positions=1 if position else 0,
        )

        try:
            risk_manager.check_kill_switch(account)
        except KillSwitchTriggered as e:
            logger.error("KILL-SWITCH AUSGELÖST: %s. Kein weiterer Handel in diesem Lauf.", e)
            break

        if position:
            exit_reason = risk_manager.should_exit(position.entry_price, current_price)
            if exit_reason:
                broker.submit_order(symbol, position.qty, "sell", current_price)
                logger.info("%s: Position geschlossen (%s) bei %.2f", symbol, exit_reason, current_price)
                continue

        if proba_up >= 0.55 and not position:
            qty = risk_manager.position_size(account, current_price)
            if qty > 0:
                broker.submit_order(symbol, qty, "buy", current_price)
                logger.info("%s: Long-Signal (p=%.3f), Order %.4f @ %.2f", symbol, proba_up, qty, current_price)
        elif proba_up < 0.45 and position:
            broker.submit_order(symbol, position.qty, "sell", current_price)
            logger.info("%s: Signal gedreht (p=%.3f), Position geschlossen bei %.2f", symbol, proba_up, current_price)
        else:
            logger.info("%s: Kein Handel (p=%.3f)", symbol, proba_up)

    return peak_equity


if __name__ == "__main__":
    cfg = load_config()
    br = build_broker(cfg)
    rm = build_risk_manager(cfg)
    run_once(cfg, br, rm, peak_equity=br.get_equity())
