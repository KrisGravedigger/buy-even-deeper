#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wąskie przeszukanie parametrów BtD na BTC.

Woła ten sam rdzeń co strategy_runner (run_strategy_core), bez Latin Hypercube.
Prowizja jest liczona po fakcie: nie ma jej w symulatorze.

Założenie rozmiaru pozycji (jak w UI BtD z kontekstu zadania):
100 USDC notional, saldo 1000 USDC, 1x. 1 punkt procentowy zysku = 1 USDC.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from runner_parameters.generation import generate_parameter_combinations
from runner_parameters.models import TradingParameters
from strategy_runner import load_market_data, run_strategy_core

NOTIONAL_USDC = 100.0
BALANCE_USDC = 1000.0
# Binance USD-M, domyślny taker ~0.05% na stronę → 0.10 pp za zamknięcie regułą.
FEE_ROUND_TRIP_PP = 0.10
FEE_ENTRY_ONLY_PP = 0.05


def base_params(**overrides) -> TradingParameters:
    params = TradingParameters(
        check_timeframe=30,
        percentage_buy_threshold=-2.0,
        max_allowed_usd=1000.0,
        add_to_limit_order=2.0,
        sell_profit_target=1.0,
        trailing_enabled=0.0,
        trailing_stop_price=1.0,
        trailing_stop_margin=0.3,
        trailing_stop_time=1,
        stop_loss_enabled=True,
        stop_loss_threshold=-3.0,
        stop_loss_delay_time=0,
        max_open_orders_per_coin=1,
        next_buy_delay=1,
        next_buy_price_lower=0.0,
        pump_detection_enabled=True,
        pump_detection_threshold=5.0,
        pump_detection_disabled_time=30,
        follow_btc_price=False,
        follow_btc_threshold=1.0,
        follow_btc_block_time=30,
        max_open_orders=1,
        stop_loss_disable_buy=False,
        stop_loss_disable_buy_all=False,
        stop_loss_next_buy_lower=0.0,
        stop_loss_no_buy_delay=0,
        trailing_buy_enabled=False,
        trailing_buy_threshold=0.3,
        trailing_buy_time_in_min=15,
    )
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


EXITS = {
    "sell1_sl3": dict(sell_profit_target=1.0, trailing_enabled=0.0, stop_loss_threshold=-3.0),
    "sell1_sl2": dict(sell_profit_target=1.0, trailing_enabled=0.0, stop_loss_threshold=-2.0),
    "sell1_sl5": dict(sell_profit_target=1.0, trailing_enabled=0.0, stop_loss_threshold=-5.0),
    "sell1.5_sl3": dict(sell_profit_target=1.5, trailing_enabled=0.0, stop_loss_threshold=-3.0),
    "tls1_m0.3_sl3": dict(
        sell_profit_target=0.0,
        trailing_enabled=1.0,
        trailing_stop_price=1.0,
        trailing_stop_margin=0.3,
        trailing_stop_time=1,
        stop_loss_threshold=-3.0,
    ),
}


def build_grid() -> list[tuple[str, TradingParameters]]:
    cases: list[tuple[str, TradingParameters]] = []
    for tf in (15, 30, 60):
        for buy in (-1.0, -1.5, -2.0, -3.0):
            for exit_name, exit_kw in EXITS.items():
                label = f"tf{tf}_buy{buy}_{exit_name}"
                cases.append((label, base_params(check_timeframe=tf, percentage_buy_threshold=buy, **exit_kw)))

    cases.append((
        "tf30_buy-5_sell1_sl3",
        base_params(check_timeframe=30, percentage_buy_threshold=-5.0, **EXITS["sell1_sl3"]),
    ))

    anchors = {
        "ui_winner": dict(check_timeframe=30, percentage_buy_threshold=-2.0, **EXITS["sell1_sl3"]),
        "ui_aggressive": dict(check_timeframe=30, percentage_buy_threshold=-1.5, **EXITS["sell1_sl3"]),
    }
    sensitivities = {
        "pump_off": dict(pump_detection_enabled=False),
        "pump2": dict(pump_detection_enabled=True, pump_detection_threshold=2.0),
        "delay60": dict(next_buy_delay=60),
        "sl_delay5": dict(stop_loss_delay_time=5),
        "max_orders3": dict(max_open_orders=3, max_open_orders_per_coin=3),
    }
    for anchor_name, anchor_kw in anchors.items():
        for sens_name, sens_kw in sensitivities.items():
            merged = dict(anchor_kw)
            merged.update(sens_kw)
            cases.append((f"{anchor_name}_{sens_name}", base_params(**merged)))
    return cases


def _split_trades(trades: list[float], open_mtm: int) -> tuple[list[float], list[float]]:
    mtm_n = max(0, min(int(open_mtm), len(trades)))
    if mtm_n == 0:
        return trades, []
    return trades[:-mtm_n], trades[-mtm_n:]


def summarize_run(label: str, params: TradingParameters, trades, open_mtm: int, executed: int, slice_name: str) -> dict:
    trades = [float(t) for t in trades]
    rule, mtm = _split_trades(trades, open_mtm)
    net_rule = [t - FEE_ROUND_TRIP_PP for t in rule]
    net_mtm = [t - FEE_ENTRY_ONLY_PP for t in mtm]
    net = net_rule + net_mtm
    wins = sum(1 for t in net_rule if t > 0)
    losses = sum(1 for t in net_rule if t < 0)
    gross_pp = float(sum(trades))
    net_pp = float(sum(net)) if net else 0.0
    rule_n = len(rule)
    if net:
        equity = np.cumsum(np.array(net, dtype=np.float64))
        peak = np.maximum.accumulate(equity)
        max_dd_pp = float((peak - equity).max()) if len(equity) else 0.0
    else:
        max_dd_pp = 0.0
    gross_wins = sum(t for t in net_rule if t > 0)
    gross_losses = abs(sum(t for t in net_rule if t < 0))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 1e-12 else (float("inf") if gross_wins > 0 else 0.0)
    scale = NOTIONAL_USDC / 100.0
    return {
        "label": label,
        "slice": slice_name,
        "check_timeframe": int(params.check_timeframe),
        "percentage_buy_threshold": float(params.percentage_buy_threshold),
        "sell_profit_target": float(params.sell_profit_target),
        "trailing_enabled": bool(params.trailing_enabled),
        "trailing_stop_price": float(params.trailing_stop_price),
        "trailing_stop_margin": float(params.trailing_stop_margin),
        "stop_loss_threshold": float(params.stop_loss_threshold),
        "stop_loss_delay_time": int(params.stop_loss_delay_time),
        "pump_detection_enabled": bool(params.pump_detection_enabled),
        "pump_detection_threshold": float(params.pump_detection_threshold),
        "pump_detection_disabled_time": int(params.pump_detection_disabled_time),
        "next_buy_delay": int(params.next_buy_delay),
        "max_open_orders": int(params.max_open_orders),
        "entries": int(executed),
        "rule_closes": rule_n,
        "mtm_closes": len(mtm),
        "win_rate_net": (100.0 * wins / rule_n) if rule_n else 0.0,
        "wins": wins,
        "losses": losses,
        "gross_pp": gross_pp,
        "net_pp": net_pp,
        "gross_usdc": gross_pp * scale,
        "net_usdc": net_pp * scale,
        "max_dd_pp": max_dd_pp,
        "max_dd_usdc": max_dd_pp * scale,
        "max_dd_account_pct": (max_dd_pp * scale / BALANCE_USDC) * 100.0,
        "worst_net_pp": float(min(net_rule)) if net_rule else 0.0,
        "profit_factor_net": profit_factor,
        "avg_net_pp": float(np.mean(net_rule)) if net_rule else 0.0,
    }


def run_case(label: str, params: TradingParameters, prices, times, slice_name: str):
    trades, executed, _closed, _checked, _blocks, open_mtm = run_strategy_core(
        prices,
        prices,
        times,
        params.to_array(),
        0,
    )
    trade_list = [float(t) for t in trades]
    row = summarize_run(label, params, trade_list, int(open_mtm), int(executed), slice_name)
    return row, trade_list


def self_test() -> None:
    with open(Path("parametry/btc_ui_winner.json"), "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    combos = generate_parameter_combinations(
        cfg,
        market_data={"symbol": "BTC/USDT"},
        param_file_name="btc_ui_winner.json",
    )
    if len(combos) != 1:
        raise AssertionError(f"oczekiwano 1 kombinacji z btc_ui_winner.json, jest {len(combos)}")
    winner = combos[0]
    if abs(float(winner.stop_loss_threshold) - (-3.0)) > 1e-9:
        raise AssertionError(f"SL nie został zachowany: {winner.stop_loss_threshold}")
    if not bool(winner.stop_loss_enabled):
        raise AssertionError("stop_loss_enabled został wyłączony")
    if int(winner.check_timeframe) != 30 or abs(float(winner.percentage_buy_threshold) + 2.0) > 1e-9:
        raise AssertionError("tf/próg zakupu rozjechały się z JSON")
    if len(winner.to_array()) != 27:
        raise AssertionError("to_array nie zawiera pump_detection_disabled_time")

    n = 80
    times = np.arange(n, dtype=np.int64)
    prices = np.full(n, 100.0, dtype=np.float64)
    prices[40:50] = 97.0
    prices[50:] = 98.0
    dip = base_params(
        check_timeframe=30,
        percentage_buy_threshold=-2.0,
        sell_profit_target=1.0,
        stop_loss_threshold=-3.0,
        pump_detection_enabled=False,
    )
    dip_row, _ = run_case("self_dip", dip, prices, times, "self")
    if dip_row["entries"] < 1 or dip_row["wins"] < 1:
        raise AssertionError(f"syntetyczny dip nie domknął zysku: {dip_row}")

    prices_sl = np.full(n, 100.0, dtype=np.float64)
    prices_sl[40:42] = 97.0
    prices_sl[42:] = 93.0
    sl = base_params(
        check_timeframe=30,
        percentage_buy_threshold=-2.0,
        sell_profit_target=10.0,
        stop_loss_threshold=-3.0,
        stop_loss_delay_time=0,
        pump_detection_enabled=False,
    )
    sl_row, _ = run_case("self_sl", sl, prices_sl, times, "self")
    if sl_row["losses"] < 1 or sl_row["worst_net_pp"] > -3.0:
        raise AssertionError(f"syntetyczny SL nie zadziałał: {sl_row}")

    prices_pump = np.full(90, 106.0, dtype=np.float64)
    prices_pump[:10] = 100.0
    prices_pump[10:20] = 112.0
    prices_pump[20:40] = 100.0
    times_pump = np.arange(90, dtype=np.int64)
    no_pump = base_params(
        check_timeframe=10,
        percentage_buy_threshold=-2.0,
        sell_profit_target=1.0,
        pump_detection_enabled=False,
        stop_loss_threshold=-20.0,
    )
    with_pump = base_params(
        check_timeframe=10,
        percentage_buy_threshold=-2.0,
        sell_profit_target=1.0,
        pump_detection_enabled=True,
        pump_detection_threshold=5.0,
        pump_detection_disabled_time=30,
        stop_loss_threshold=-20.0,
    )
    off, _ = run_case("self_pump_off", no_pump, prices_pump, times_pump, "self")
    on, _ = run_case("self_pump_on", with_pump, prices_pump, times_pump, "self")
    if off["entries"] < 1 or on["entries"] != 0:
        raise AssertionError(f"blokada pump nie działa: off={off['entries']} on={on['entries']}")
    print("self-test OK: generator SL=-3, dip, stop-loss, blokada pump")


def _month_slices(times: np.ndarray) -> list[tuple[str, np.ndarray]]:
    stamps = pd.to_datetime(times.astype(np.int64) * 60, unit="s", utc=True)
    periods = stamps.to_period("M")
    slices = []
    for period in list(dict.fromkeys(periods)):
        mask = np.asarray(periods == period)
        slices.append((str(period), mask))
    return slices


def _score(row: dict) -> float:
    """Heurystyka tylko do sortowania raportu, nie do live-bot score BtD."""
    if row["rule_closes"] < 3:
        return -1e9 + row["net_usdc"]
    trade_penalty = 0.02 * max(0, row["rule_closes"] - 8)
    return row["net_usdc"] - 0.25 * row["max_dd_usdc"] - trade_penalty


def _fmt(row: dict) -> str:
    pf = row["profit_factor_net"]
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    return (
        f"{row['label']}: net {row['net_usdc']:+.2f} USDC "
        f"(gross {row['gross_usdc']:+.2f}), reguły {row['rule_closes']} "
        f"(wejścia {row['entries']}, MTM {row['mtm_closes']}), "
        f"win {row['win_rate_net']:.0f}%, DD {row['max_dd_usdc']:.2f} USDC "
        f"({row['max_dd_account_pct']:.2f}% salda), PF {pf_s}, "
        f"najgorsza {row['worst_net_pp']:.2f} pp"
    )


def write_markdown(path: Path, rows: list[dict], meta: dict, monthly: dict[str, list[dict]]) -> None:
    full = [r for r in rows if r["slice"] == "full"]
    ranked = sorted(full, key=_score, reverse=True)
    lines = [
        "# BTC BtD — skupiona siatka offline",
        "",
        f"- Wygenerowano: {meta['generated_at']}",
        f"- Plik: `{meta['csv']}`",
        f"- Świece: {meta['candles']} ({meta['period_start']} → {meta['period_end']})",
        f"- Cena w symulatorze: `average_price` = (high+low)/2, świece 1m",
        f"- Prowizja doliczona po symulacji: {FEE_ROUND_TRIP_PP:.2f} pp na zamknięcie regułą "
        f"(2 × 0.05% taker USD-M), {FEE_ENTRY_ONLY_PP:.2f} pp na pozycję domkniętą tylko mark-to-market.",
        f"- Przeliczenie na USDC: {NOTIONAL_USDC:.0f} USDC notional, saldo {BALANCE_USDC:.0f}.",
        "- Miesiące są osobnymi startami (pozycja nie przechodzi między miesiącami).",
        "",
        "## Najwyżej w heurystyce (net USDC − 0.25×DD − kara za >8 transakcji)",
        "",
    ]
    for row in ranked[:12]:
        lines.append(f"- {_fmt(row)}")
    lines.extend(["", "## Konfiguracje z UI BtD i warianty", ""])
    interesting = [
        "tf30_buy-2.0_sell1_sl3",
        "tf30_buy-1.5_sell1_sl3",
        "tf30_buy-1.0_sell1_sl3",
        "tf30_buy-2.0_tls1_m0.3_sl3",
        "tf30_buy-1.5_tls1_m0.3_sl3",
        "tf30_buy-3.0_sell1_sl3",
        "tf30_buy-5_sell1_sl3",
        "tf15_buy-2.0_sell1_sl3",
        "tf60_buy-2.0_sell1_sl3",
        "ui_winner_pump_off",
        "ui_winner_pump2",
        "ui_aggressive_pump_off",
        "ui_winner_max_orders3",
        "ui_winner_delay60",
        "ui_winner_sl_delay5",
    ]
    by_label = {r["label"]: r for r in full}
    for label in interesting:
        row = by_label.get(label)
        if row:
            lines.append(f"- {_fmt(row)}")
    lines.extend(["", "## Stabilność miesięczna (osobny start)", ""])
    for label, month_rows in monthly.items():
        bits = [f"{r['slice']} {r['net_usdc']:+.2f} USDC / {r['rule_closes']} reguł" for r in month_rows]
        positive = sum(1 for r in month_rows if r["net_usdc"] > 0 and r["rule_closes"] > 0)
        lines.append(f"- {label}: {positive}/{len(month_rows)} miesięcy na plusie; " + "; ".join(bits))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Skupione przeszukanie parametrów BtD na BTC.")
    parser.add_argument("--csv", required=False, default=None, help="Ścieżka do CSV z fetcherem")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()

    self_test()
    if args.self_test_only:
        return
    if not args.csv:
        raise SystemExit("Podaj --csv (albo uruchom samo --self-test-only).")

    market = load_market_data(Path(args.csv))
    if market is None:
        raise SystemExit(f"Nie udało się wczytać {args.csv}")

    prices = market["prices"].astype(np.float64)
    times = market["times"].astype(np.int64)
    cases = build_grid()
    print(f"Siatka: {len(cases)} konfiguracji, świece: {len(prices)}")

    rows = []
    pkl_results = []
    for idx, (label, params) in enumerate(cases, start=1):
        row, trade_list = run_case(label, params, prices, times, "full")
        rows.append(row)
        pkl_results.append({
            "parameters": {field.name: getattr(params, field.name) for field in fields(params)},
            "trades": trade_list,
            "strategy_id": label,
            "completed": True,
            "total_trades": row["rule_closes"] + row["mtm_closes"],
            "trades_executed": row["entries"],
            "trades_closed": row["rule_closes"] + row["mtm_closes"],
            "trades_checked": int(len(prices)),
            "btc_blocks": 0,
            "open_mtm": row["mtm_closes"],
            "avg_profit": row["avg_net_pp"],
            "symbol": market["symbol"],
            "param_file_name": "btc_btd_focused_grid",
            "label": label,
        })
        print(f"[{idx}/{len(cases)}] {_fmt(row)}")

    focus_labels = [
        "tf30_buy-2.0_sell1_sl3",
        "tf30_buy-1.5_sell1_sl3",
        "tf30_buy-1.0_sell1_sl3",
        "tf30_buy-2.0_tls1_m0.3_sl3",
        "tf60_buy-2.0_sell1_sl3",
        "tf15_buy-2.0_sell1_sl3",
        "tf30_buy-2.0_sell1_sl2",
        "tf30_buy-2.0_sell1_sl5",
        "ui_winner_pump_off",
        "ui_winner_pump2",
    ]
    by_label = {label: params for label, params in cases}
    monthly: dict[str, list[dict]] = {}
    for label in focus_labels:
        monthly[label] = []
        params = by_label[label]
        for slice_name, mask in _month_slices(times):
            if int(mask.sum()) < 1000:
                continue
            month_row, _month_trades = run_case(label, params, prices[mask], times[mask], slice_name)
            monthly[label].append(month_row)
            rows.append(month_row)

    artifacts = Path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_out = artifacts / "btc_btd_grid_summary.csv"
    md_out = artifacts / "btc_btd_grid_summary.md"
    df = pd.DataFrame(rows)
    df.to_csv(csv_out, index=False)
    meta = {
        "generated_at": stamp,
        "csv": str(Path(args.csv)),
        "candles": int(len(prices)),
        "period_start": str(market["period"][0]),
        "period_end": str(market["period"][1]),
    }
    (artifacts / "btc_btd_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_markdown(md_out, rows, meta, monthly)

    out_dir = Path("wyniki/backtesty")
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / f"btc_btd_focused_{stamp}.pkl"
    with pkl_path.open("wb") as handle:
        pickle.dump({
            "results": pkl_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parameters_file": "run_btc_parameter_search.py",
            "market_data_info": {
                "file": str(args.csv),
                "symbol": market["symbol"],
                "period": market["period"],
                "candles": int(len(prices)),
            },
            "mode": "backtest",
        }, handle)
    print(f"Zapisano {csv_out}")
    print(f"Zapisano {md_out}")
    print(f"Zapisano {pkl_path}")


if __name__ == "__main__":
    main()
