# BTC BtD — skupiona siatka offline

- Wygenerowano: 20260930T051140Z
- Plik: `csv/binance_BTC_USDT_1m_2026-07-01_2026-10-01.csv`
- Świece: 131350 (2026-07-01 00:00:00 → 2026-09-30 05:09:00)
- Cena w symulatorze: `average_price` = (high+low)/2, świece 1m
- Prowizja doliczona po symulacji: 0.10 pp na zamknięcie regułą (2 × 0.05% taker USD-M), 0.05 pp na pozycję domkniętą tylko mark-to-market.
- Przeliczenie na USDC: 100 USDC notional, saldo 1000.
- Miesiące są osobnymi startami (pozycja nie przechodzi między miesiącami).

## Najwyżej w heurystyce (net USDC − 0.25×DD − kara za >8 transakcji)

- tf60_buy-1.0_sell1_sl5: net +24.08 USDC (gross +27.43), reguły 33 (wejścia 34, MTM 1), win 97%, DD 5.17 USDC (0.52% salda), PF 5.95, najgorsza -5.17 pp
- tf60_buy-1.0_sell1.5_sl3: net +21.31 USDC (gross +24.46), reguły 31 (wejścia 32, MTM 1), win 84%, DD 4.78 USDC (0.48% salda), PF 2.45, najgorsza -3.19 pp
- tf60_buy-1.0_tls1_m0.3_sl3: net +19.51 USDC (gross +22.76), reguły 32 (wejścia 33, MTM 1), win 84%, DD 4.80 USDC (0.48% salda), PF 2.33, najgorsza -3.19 pp
- tf60_buy-1.0_sell1_sl3: net +18.53 USDC (gross +22.38), reguły 38 (wejścia 39, MTM 1), win 89%, DD 3.55 USDC (0.35% salda), PF 2.59, najgorsza -3.21 pp
- tf15_buy-1.0_sell1.5_sl3: net +15.76 USDC (gross +17.31), reguły 15 (wejścia 16, MTM 1), win 93%, DD 3.23 USDC (0.32% salda), PF 6.40, najgorsza -3.23 pp
- tf30_buy-1.0_tls1_m0.3_sl3: net +15.61 USDC (gross +17.86), reguły 22 (wejścia 23, MTM 1), win 86%, DD 4.22 USDC (0.42% salda), PF 2.85, najgorsza -3.23 pp
- tf30_buy-1.0_sell1.5_sl3: net +15.26 USDC (gross +17.41), reguły 21 (wejścia 22, MTM 1), win 86%, DD 3.37 USDC (0.34% salda), PF 2.81, najgorsza -3.23 pp
- tf15_buy-1.0_tls1_m0.3_sl3: net +13.32 USDC (gross +14.97), reguły 16 (wejścia 17, MTM 1), win 94%, DD 3.23 USDC (0.32% salda), PF 5.64, najgorsza -3.23 pp
- tf30_buy-1.0_sell1_sl3: net +12.78 USDC (gross +15.23), reguły 24 (wejścia 25, MTM 1), win 92%, DD 4.43 USDC (0.44% salda), PF 3.33, najgorsza -3.23 pp
- ui_aggressive_max_orders3: net +12.21 USDC (gross +14.81), reguły 26 (wejścia 26, MTM 0), win 88%, DD 9.47 USDC (0.95% salda), PF 2.29, najgorsza -3.19 pp
- ui_winner_max_orders3: net +9.46 USDC (gross +10.46), reguły 10 (wejścia 10, MTM 0), win 100%, DD 0.00 USDC (0.00% salda), PF inf, najgorsza 0.90 pp
- tf30_buy-1.0_sell1_sl5: net +10.82 USDC (gross +12.87), reguły 20 (wejścia 21, MTM 1), win 95%, DD 5.35 USDC (0.54% salda), PF 3.39, najgorsza -5.35 pp

## Konfiguracje z UI BtD i warianty

- tf30_buy-2.0_sell1_sl3: net +3.88 USDC (gross +4.28), reguły 4 (wejścia 4, MTM 0), win 100%, DD 0.00 USDC (0.00% salda), PF inf, najgorsza 0.90 pp
- tf30_buy-1.5_sell1_sl3: net +4.46 USDC (gross +5.36), reguły 9 (wejścia 9, MTM 0), win 89%, DD 3.16 USDC (0.32% salda), PF 2.41, najgorsza -3.16 pp
- tf30_buy-1.0_sell1_sl3: net +12.78 USDC (gross +15.23), reguły 24 (wejścia 25, MTM 1), win 92%, DD 4.43 USDC (0.44% salda), PF 3.33, najgorsza -3.23 pp
- tf30_buy-2.0_tls1_m0.3_sl3: net +0.06 USDC (gross +0.46), reguły 4 (wejścia 4, MTM 0), win 75%, DD 3.12 USDC (0.31% salda), PF 1.02, najgorsza -3.12 pp
- tf30_buy-1.5_tls1_m0.3_sl3: net +4.52 USDC (gross +5.37), reguły 8 (wejścia 9, MTM 1), win 88%, DD 3.16 USDC (0.32% salda), PF 2.85, najgorsza -3.16 pp
- tf30_buy-3.0_sell1_sl3: net +0.00 USDC (gross +0.00), reguły 0 (wejścia 0, MTM 0), win 0%, DD 0.00 USDC (0.00% salda), PF 0.00, najgorsza 0.00 pp
- tf30_buy-5_sell1_sl3: net +0.00 USDC (gross +0.00), reguły 0 (wejścia 0, MTM 0), win 0%, DD 0.00 USDC (0.00% salda), PF 0.00, najgorsza 0.00 pp
- tf15_buy-2.0_sell1_sl3: net +2.01 USDC (gross +2.21), reguły 2 (wejścia 2, MTM 0), win 100%, DD 0.00 USDC (0.00% salda), PF inf, najgorsza 0.90 pp
- tf60_buy-2.0_sell1_sl3: net +4.60 USDC (gross +5.10), reguły 5 (wejścia 5, MTM 0), win 100%, DD 0.00 USDC (0.00% salda), PF inf, najgorsza 0.90 pp
- ui_winner_pump_off: net +3.88 USDC (gross +4.28), reguły 4 (wejścia 4, MTM 0), win 100%, DD 0.00 USDC (0.00% salda), PF inf, najgorsza 0.90 pp
- ui_winner_pump2: net +3.88 USDC (gross +4.28), reguły 4 (wejścia 4, MTM 0), win 100%, DD 0.00 USDC (0.00% salda), PF inf, najgorsza 0.90 pp
- ui_aggressive_pump_off: net +4.46 USDC (gross +5.36), reguły 9 (wejścia 9, MTM 0), win 89%, DD 3.16 USDC (0.32% salda), PF 2.41, najgorsza -3.16 pp
- ui_winner_max_orders3: net +9.46 USDC (gross +10.46), reguły 10 (wejścia 10, MTM 0), win 100%, DD 0.00 USDC (0.00% salda), PF inf, najgorsza 0.90 pp
- ui_winner_delay60: net +3.88 USDC (gross +4.28), reguły 4 (wejścia 4, MTM 0), win 100%, DD 0.00 USDC (0.00% salda), PF inf, najgorsza 0.90 pp
- ui_winner_sl_delay5: net +3.88 USDC (gross +4.28), reguły 4 (wejścia 4, MTM 0), win 100%, DD 0.00 USDC (0.00% salda), PF inf, najgorsza 0.90 pp

## Stabilność miesięczna (osobny start)

- tf30_buy-2.0_sell1_sl3: 2/3 miesięcy na plusie; 2026-07 +0.00 USDC / 0 reguł; 2026-08 +0.90 USDC / 1 reguł; 2026-09 +2.97 USDC / 3 reguł
- tf30_buy-1.5_sell1_sl3: 2/3 miesięcy na plusie; 2026-07 +1.26 USDC / 1 reguł; 2026-08 +2.79 USDC / 3 reguł; 2026-09 -0.27 USDC / 4 reguł
- tf30_buy-1.0_sell1_sl3: 2/3 miesięcy na plusie; 2026-07 +5.56 USDC / 6 reguł; 2026-08 +6.32 USDC / 11 reguł; 2026-09 -0.38 USDC / 6 reguł
- tf30_buy-2.0_tls1_m0.3_sl3: 1/3 miesięcy na plusie; 2026-07 +0.00 USDC / 0 reguł; 2026-08 +1.14 USDC / 1 reguł; 2026-09 -1.08 USDC / 3 reguł
- tf60_buy-2.0_sell1_sl3: 3/3 miesięcy na plusie; 2026-07 +0.93 USDC / 1 reguł; 2026-08 +1.83 USDC / 2 reguł; 2026-09 +1.84 USDC / 2 reguł
- tf15_buy-2.0_sell1_sl3: 2/3 miesięcy na plusie; 2026-07 +0.00 USDC / 0 reguł; 2026-08 +0.90 USDC / 1 reguł; 2026-09 +1.11 USDC / 1 reguł
- tf30_buy-2.0_sell1_sl2: 2/3 miesięcy na plusie; 2026-07 +0.00 USDC / 0 reguł; 2026-08 +0.90 USDC / 1 reguł; 2026-09 +2.97 USDC / 3 reguł
- tf30_buy-2.0_sell1_sl5: 2/3 miesięcy na plusie; 2026-07 +0.00 USDC / 0 reguł; 2026-08 +0.90 USDC / 1 reguł; 2026-09 +2.97 USDC / 3 reguł
- ui_winner_pump_off: 2/3 miesięcy na plusie; 2026-07 +0.00 USDC / 0 reguł; 2026-08 +0.90 USDC / 1 reguł; 2026-09 +2.97 USDC / 3 reguł
- ui_winner_pump2: 2/3 miesięcy na plusie; 2026-07 +0.00 USDC / 0 reguł; 2026-08 +0.90 USDC / 1 reguł; 2026-09 +2.97 USDC / 3 reguł
