#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ponowna ocena dwóch live strategii BtD na BTC/USDT.

Rdzeń jest ten sam co w run_btc_parameter_search.py (run_strategy_core).
Dźwigni w rdzeniu nie ma: wynik jest w punktach procentowych ceny.
Przeliczenie na USDT przyjmuje depozyt pozycji 100 USDT i nominał 100 × dźwignia.
Przy 1× odtwarza to REPORT.md (1 pp = 1 USDT). Ścieżka transakcji nie zależy
od dźwigni, bo SL i sell są progami ceny, a likwidacja isolated jest daleko
za SL −2…−5%.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_btc_parameter_search import (
    BALANCE_USDC,
    FEE_ENTRY_ONLY_PP,
    FEE_ROUND_TRIP_PP,
    base_params,
    self_test,
)
from strategy_runner import load_market_data, run_strategy_core

MARGIN_USDT = 100.0
TFS = (15, 30, 45, 60)
BUYS = (-1.0, -1.25, -1.5, -2.0)
SELLS = (0.8, 1.0, 1.2, 1.5)
SLS = (-2.0, -3.0, -4.0, -5.0)
LEVS = (1, 2)
MIN_TRADES = 10

BOT_A = (60, -2.0, 1.0, -3.0, 1)  # 17469
BOT_B = (15, -1.0, 1.0, -3.0, 2)  # 17473

# Binance USD-M, najniższy tier, rząd wielkości. Nie wchodzi do symulacji.
MAINT_MARGIN_RATE = 0.004


def _minutes(ts: str) -> int:
    stamp = pd.Timestamp(ts, tz="UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return int((stamp - epoch) // pd.Timedelta(minutes=1))


def _label(tf: int, buy: float, sell: float, sl: float, lev: int) -> str:
    return f"tf{tf}_buy{buy:g}_sell{sell:g}_sl{sl:g}_x{lev}"


def build_grid() -> list[tuple]:
    cases = []
    for tf in TFS:
        for buy in BUYS:
            for sell in SELLS:
                for sl in SLS:
                    for lev in LEVS:
                        key = (tf, buy, sell, sl, lev)
                        params = base_params(
                            check_timeframe=tf,
                            percentage_buy_threshold=buy,
                            sell_profit_target=sell,
                            trailing_enabled=0.0,
                            stop_loss_threshold=sl,
                        )
                        cases.append((key, params))
    return cases


def _split_trades(trades: list[float], open_mtm: int) -> tuple[list[float], list[float]]:
    mtm_n = max(0, min(int(open_mtm), len(trades)))
    if mtm_n == 0:
        return trades, []
    return trades[:-mtm_n], trades[-mtm_n:]


def summarize(key, params, trades, open_mtm: int, executed: int, window: str, slice_name: str, span_days: float) -> dict:
    tf, buy, sell, sl, lev = key
    trades = [float(t) for t in trades]
    rule, mtm = _split_trades(trades, open_mtm)
    net_rule = [t - FEE_ROUND_TRIP_PP for t in rule]
    net_mtm = [t - FEE_ENTRY_ONLY_PP for t in mtm]
    net = net_rule + net_mtm
    wins = sum(1 for t in net_rule if t > 0)
    rule_n = len(rule)
    gross_pp = float(sum(trades)) if trades else 0.0
    net_pp = float(sum(net)) if net else 0.0
    realized_net_pp = float(sum(net_rule)) if net_rule else 0.0
    if net:
        equity = np.cumsum(np.array(net, dtype=np.float64))
        peak = np.maximum.accumulate(equity)
        max_dd_pp = float((peak - equity).max())
    else:
        max_dd_pp = 0.0
    scale = float(lev)  # nominał = 100 × dźwignia, 1 pp = 1 USDT × dźwignia
    sl_hits = sum(1 for t in rule if t <= sl + 1e-4)
    non_tp = sum(1 for t in rule if t < sell - 1e-4)
    months = span_days / (365.25 / 12.0) if span_days > 0 else 0.0
    net_usdt = net_pp * scale
    return {
        "window": window,
        "slice": slice_name,
        "label": _label(*key),
        "check_timeframe": tf,
        "percentage_buy_threshold": buy,
        "sell_profit_target": sell,
        "stop_loss_threshold": sl,
        "leverage": lev,
        "notional_usdt": MARGIN_USDT * lev,
        "margin_usdt": MARGIN_USDT,
        "entries": int(executed),
        "rule_closes": rule_n,
        "mtm_closes": len(mtm),
        "open_at_end": int(len(mtm) > 0),
        "sl_hits": int(sl_hits),
        "non_tp_closes": int(non_tp),
        "trades_per_month": (rule_n / months) if months else 0.0,
        "win_rate_net": (100.0 * wins / rule_n) if rule_n else 0.0,
        "wins": wins,
        "gross_pp": gross_pp,
        "net_pp": net_pp,
        "realized_net_pp": realized_net_pp,
        "gross_usdt": gross_pp * scale,
        "net_usdt": net_usdt,
        "realized_net_usdt": realized_net_pp * scale,
        "unrealized_net_usdt": float(sum(net_mtm) * scale) if net_mtm else 0.0,
        "fee_usdt": (rule_n * FEE_ROUND_TRIP_PP + len(mtm) * FEE_ENTRY_ONLY_PP) * scale,
        "roi_account_pct": (net_usdt / BALANCE_USDC) * 100.0,
        "max_dd_pp": max_dd_pp,
        "max_dd_usdt": max_dd_pp * scale,
        "max_dd_account_pct": (max_dd_pp * scale / BALANCE_USDC) * 100.0,
        "worst_net_pp": float(min(net_rule)) if net_rule else 0.0,
        "span_days": span_days,
    }


def run_key(key, params, prices, times, window: str, slice_name: str) -> dict:
    trades, executed, _closed, _checked, _blocks, open_mtm = run_strategy_core(
        prices, prices, times, params.to_array(), 0
    )
    span_days = float(times[-1] - times[0]) / 1440.0 if len(times) > 1 else 0.0
    return summarize(key, params, list(trades), int(open_mtm), int(executed), window, slice_name, span_days)


def _mask(times: np.ndarray, start: str, end: str) -> np.ndarray:
    return (times >= _minutes(start)) & (times < _minutes(end))


def _price_stats(prices: np.ndarray, times: np.ndarray) -> dict:
    stamps = pd.to_datetime(times.astype(np.int64) * 60, unit="s", utc=True)
    rets = np.diff(prices) / prices[:-1] * 100.0
    two = prices[2:] / prices[:-2] - 1.0
    return {
        "candles": int(len(prices)),
        "start": str(stamps[0]),
        "end": str(stamps[-1]),
        "price_start": float(prices[0]),
        "price_end": float(prices[-1]),
        "price_min": float(prices.min()),
        "price_max": float(prices.max()),
        "price_return_pct": float((prices[-1] / prices[0] - 1.0) * 100.0),
        "worst_1m_pct": float(rets.min()) if len(rets) else 0.0,
        "worst_2m_pct": float(two.min() * 100.0) if len(two) else 0.0,
    }


def _neighbors(key) -> list[tuple]:
    tf, buy, sell, sl, lev = key
    axes = (
        (0, TFS, tf),
        (1, BUYS, buy),
        (2, SELLS, sell),
        (3, SLS, sl),
    )
    out = []
    for idx, values, current in axes:
        pos = values.index(current)
        for step in (-1, 1):
            nxt = pos + step
            if 0 <= nxt < len(values):
                mutated = list(key)
                mutated[idx] = values[nxt]
                out.append(tuple(mutated))
    return out


def _component_box(keys: list[tuple]) -> dict:
    return {
        "n": len(keys),
        "leverage": keys[0][4],
        "timeframe": sorted({k[0] for k in keys}),
        "buy": sorted({k[1] for k in keys}),
        "sell": sorted({k[2] for k in keys}),
        "sl": sorted({k[3] for k in keys}),
        "labels": [_label(*k) for k in keys],
    }


def robustness(full_rows: dict[tuple, dict]) -> dict:
    good = {k for k, row in full_rows.items() if row["rule_closes"] >= MIN_TRADES and row["net_usdt"] > 0}
    ranked = sorted(
        (k for k, row in full_rows.items() if row["rule_closes"] >= MIN_TRADES),
        key=lambda k: full_rows[k]["net_usdt"],
        reverse=True,
    )
    top = set(ranked[:15])
    flags = {}
    for key, row in full_rows.items():
        neigh = _neighbors(key)
        good_n = [n for n in neigh if n in good]
        nets = [full_rows[n]["net_usdt"] for n in neigh]
        med = float(np.median(nets)) if nets else 0.0
        frac = (len(good_n) / len(neigh)) if neigh else 0.0
        eligible = row["rule_closes"] >= MIN_TRADES and row["net_usdt"] > 0
        robust = bool(eligible and frac >= 0.5 and med >= 0.5 * row["net_usdt"])
        spike = bool(
            eligible
            and key in top
            and (frac < 0.5 or med < 0.25 * row["net_usdt"])
        )
        flags[key] = {
            "good_neighbor_frac": frac,
            "good_neighbors": len(good_n),
            "neighbors": len(neigh),
            "neighbor_median_net_usdt": med,
            "robust": robust,
            "spike": spike,
        }

    seen = set()
    regions = []
    robust_keys = [k for k, f in flags.items() if f["robust"]]
    for start in robust_keys:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in _neighbors(cur):
                if nxt in flags and flags[nxt]["robust"] and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if len(comp) >= 4:
            nets = [full_rows[k]["net_usdt"] for k in comp]
            box = _component_box(comp)
            box["net_min"] = float(min(nets))
            box["net_median"] = float(np.median(nets))
            box["net_max"] = float(max(nets))
            regions.append(box)
    regions.sort(key=lambda r: r["net_median"], reverse=True)

    spikes = []
    for key, flag in flags.items():
        if flag["spike"]:
            item = dict(full_rows[key])
            item.update(flag)
            spikes.append(item)
    spikes.sort(key=lambda r: r["net_usdt"], reverse=True)
    return {"flags": flags, "regions": regions, "spikes": spikes}


def _public_row(row: dict, extra: dict | None = None) -> dict:
    keep = [
        "label", "window", "slice", "check_timeframe", "percentage_buy_threshold",
        "sell_profit_target", "stop_loss_threshold", "leverage", "notional_usdt",
        "entries", "rule_closes", "trades_per_month", "net_pp", "net_usdt",
        "realized_net_usdt", "unrealized_net_usdt", "roi_account_pct",
        "max_dd_usdt", "max_dd_account_pct", "win_rate_net", "open_at_end",
        "sl_hits", "span_days",
    ]
    out = {k: row[k] for k in keep}
    if extra:
        out.update(extra)
    return out


def main() -> None:
    self_test()
    csv_path = Path("csv/binance_BTC_USDT_1m_2026-07-01_2026-11-26.csv")
    market = load_market_data(csv_path)
    if market is None:
        raise SystemExit(f"Nie wczytano {csv_path}")
    prices = market["prices"].astype(np.float64)
    times = market["times"].astype(np.int64)
    if int(times[-1] - times[0]) < int(len(times) * 0.9):
        raise SystemExit("Zegar świec jest zepsuty względem liczby świec 1m.")

    slices = {
        "recent": _mask(times, "2026-08-25", "2026-11-26"),
        "jul_sep": _mask(times, "2026-07-01", "2026-10-01"),
        "wf_train": _mask(times, "2026-08-25", "2026-10-06"),
        "wf_test": _mask(times, "2026-10-14", "2026-11-26"),
    }
    for name, mask in slices.items():
        if int(mask.sum()) < 1000:
            raise SystemExit(f"Za mało świec w oknie {name}: {int(mask.sum())}")

    cases = build_grid()
    rows: list[dict] = []
    by_window: dict[str, dict[tuple, dict]] = {name: {} for name in slices}
    for idx, (key, params) in enumerate(cases, start=1):
        for name, mask in slices.items():
            row = run_key(key, params, prices[mask], times[mask], name, "full")
            rows.append(row)
            by_window[name][key] = row
        if idx % 64 == 0 or idx == len(cases):
            print(f"[{idx}/{len(cases)}] {_label(*key)} recent net {by_window['recent'][key]['net_usdt']:+.2f}")

    # Miesiące okna recent jako osobne starty (pozycja nie przechodzi).
    recent_times = times[slices["recent"]]
    recent_prices = prices[slices["recent"]]
    stamps = pd.to_datetime(recent_times.astype(np.int64) * 60, unit="s", utc=True)
    periods = stamps.to_period("M")
    for period in list(dict.fromkeys(periods)):
        mask = np.asarray(periods == period)
        if int(mask.sum()) < 1000:
            continue
        slice_name = str(period)
        for key, params in cases:
            rows.append(run_key(key, params, recent_prices[mask], recent_times[mask], "recent_month", slice_name))

    # Spójność 1× / 2×: ta sama ścieżka, podwójny USDT.
    mismatches = 0
    for tf in TFS:
        for buy in BUYS:
            for sell in SELLS:
                for sl in SLS:
                    a = by_window["recent"][(tf, buy, sell, sl, 1)]
                    b = by_window["recent"][(tf, buy, sell, sl, 2)]
                    same_path = (
                        a["entries"] == b["entries"]
                        and a["rule_closes"] == b["rule_closes"]
                        and a["sl_hits"] == b["sl_hits"]
                        and a["open_at_end"] == b["open_at_end"]
                        and abs(a["net_pp"] - b["net_pp"]) < 1e-9
                        and abs(b["net_usdt"] - 2 * a["net_usdt"]) < 1e-6
                    )
                    if not same_path:
                        mismatches += 1
    if mismatches:
        raise SystemExit(f"Dźwignia zmieniła ścieżkę albo skalę: {mismatches} par")

    sl_mismatch = [
        r["label"] for r in by_window["recent"].values()
        if r["sl_hits"] != r["non_tp_closes"]
    ]
    if sl_mismatch:
        raise SystemExit(f"SL nie pokrywa się z zamknięciami poza celem: {sl_mismatch[:5]}")

    recent = by_window["recent"]
    rob = robustness(recent)

    def rank(pool: dict[tuple, dict], lev: int | None = None) -> list[dict]:
        items = []
        for key, row in pool.items():
            if lev is not None and key[4] != lev:
                continue
            if row["rule_closes"] < MIN_TRADES:
                continue
            item = _public_row(row, rob["flags"].get(key) if pool is recent else None)
            items.append(item)
        items.sort(key=lambda r: (-r["net_usdt"], r["max_dd_usdt"], -r["rule_closes"]))
        return items

    def pick(window_name: str, lev: int | None, min_trades: int) -> dict | None:
        pool = [
            row for key, row in by_window[window_name].items()
            if row["rule_closes"] >= min_trades and (lev is None or key[4] == lev)
        ]
        if not pool:
            return None
        best = max(pool, key=lambda r: (r["net_usdt"], -r["max_dd_usdt"], r["rule_closes"]))
        return _public_row(best)

    wf = {}
    for name, lev in (("any", None), ("lev1", 1), ("lev2", 2)):
        chosen = pick("wf_train", lev, MIN_TRADES)
        relaxed = False
        if chosen is None:
            chosen = pick("wf_train", lev, 5)
            relaxed = True
        if chosen is None:
            wf[name] = None
            continue
        key = (
            chosen["check_timeframe"],
            chosen["percentage_buy_threshold"],
            chosen["sell_profit_target"],
            chosen["stop_loss_threshold"],
            chosen["leverage"],
        )
        wf[name] = {
            "relaxed_min_trades": relaxed,
            "train": _public_row(by_window["wf_train"][key]),
            "test": _public_row(by_window["wf_test"][key]),
            "recent": _public_row(recent[key], rob["flags"][key]),
            "jul_sep": _public_row(by_window["jul_sep"][key]),
        }

    def bot_block(key) -> dict:
        return {
            "recent": _public_row(recent[key], rob["flags"][key]),
            "jul_sep": _public_row(by_window["jul_sep"][key]),
            "wf_train": _public_row(by_window["wf_train"][key]),
            "wf_test": _public_row(by_window["wf_test"][key]),
            "months": [
                _public_row(r) for r in rows
                if r["window"] == "recent_month" and r["label"] == _label(*key)
            ],
        }

    # Sąsiedztwo botów (ta sama dźwignia, jeden krok).
    def neighborhood(key) -> list[dict]:
        out = [_public_row(recent[key], rob["flags"][key])]
        for nb in _neighbors(key):
            out.append(_public_row(recent[nb], rob["flags"][nb]))
        out.sort(key=lambda r: r["net_usdt"], reverse=True)
        return out

    highlights = {
        "model": {
            "margin_usdt": MARGIN_USDT,
            "balance_usdt": BALANCE_USDC,
            "notional": "100 × leverage",
            "fee_round_trip_pp": FEE_ROUND_TRIP_PP,
            "fee_mtm_pp": FEE_ENTRY_ONLY_PP,
            "maint_margin_rate_assumed": MAINT_MARGIN_RATE,
            "liq_price_move_1x_pct": (MAINT_MARGIN_RATE - 1.0) * 100.0,
            "liq_price_move_2x_pct": (MAINT_MARGIN_RATE - 0.5) * 100.0,
            "min_trades": MIN_TRADES,
            "note": "net_pp jest wynikiem przy nominale 100 USDT (niezależnym od dźwigni). net_usdt = net_pp × dźwignia.",
        },
        "prices": {
            "file": str(csv_path),
            "full": _price_stats(prices, times),
            "recent": _price_stats(prices[slices["recent"]], times[slices["recent"]]),
            "jul_sep": _price_stats(prices[slices["jul_sep"]], times[slices["jul_sep"]]),
            "wf_train": _price_stats(prices[slices["wf_train"]], times[slices["wf_train"]]),
            "wf_test": _price_stats(prices[slices["wf_test"]], times[slices["wf_test"]]),
        },
        "bots": {"A_17469": bot_block(BOT_A), "B_17473": bot_block(BOT_B)},
        "neighborhood": {"A_17469": neighborhood(BOT_A), "B_17473": neighborhood(BOT_B)},
        "top10_net_usdt": rank(recent)[:10],
        "top10_lev1": rank(recent, 1)[:10],
        "top10_lev2": rank(recent, 2)[:10],
        "top10_net_pp": sorted(
            [_public_row(r, rob["flags"][k]) for k, r in recent.items() if r["rule_closes"] >= MIN_TRADES and k[4] == 1],
            key=lambda r: r["net_pp"],
            reverse=True,
        )[:10],
        "regions": rob["regions"],
        "spikes": [_public_row(s, {k: s[k] for k in ("good_neighbor_frac", "neighbor_median_net_usdt")}) for s in rob["spikes"]],
        "walk_forward": wf,
        "eligible_recent": sum(1 for r in recent.values() if r["rule_closes"] >= MIN_TRADES),
        "positive_eligible_recent": sum(
            1 for r in recent.values() if r["rule_closes"] >= MIN_TRADES and r["net_usdt"] > 0
        ),
        "grid": len(cases),
    }

    artifacts = Path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_out = artifacts / "btc_btd_reeval_2026-11.csv"
    df.to_csv(csv_out, index=False)
    json_out = artifacts / "btc_btd_reeval_2026-11_highlights.json"
    json_out.write_text(json.dumps(highlights, indent=2), encoding="utf-8")
    print(f"Zapisano {csv_out} ({len(df)} wierszy)")
    print(f"Zapisano {json_out}")
    print(
        f"eligible {highlights['eligible_recent']} positive {highlights['positive_eligible_recent']} "
        f"regions {len(rob['regions'])} spikes {len(rob['spikes'])}"
    )


if __name__ == "__main__":
    main()
