# buy-even-deeper

Offline’owy backtester w stylu Buy-the-Dip, sprzed wbudowanych backtestów MaxData BtD.
Symulacja jest hipotezą roboczą do strojenia parametrów. Nie składa zleceń i nie używa kluczy API.

Nazwy z `_INSTRUKCJA_.txt` (`binance_data_fetcher_c.py`, `strategy_runner_c.py`, …) odpowiadają plikom bez sufiksu `_c`.

## Instalacja

```bash
python3 -m pip install -r requirements.txt
```

Ścieżka backtestu BTC potrzebuje: `pandas`, `numpy`, `numba`, `ccxt`, `tqdm`.
`scikit-learn` jest używany przez `strategy_analyzer.py` przy grupowaniu strategii.

## Headless: dane BTC i jeden backtest

Fetcher bierze wyłącznie publiczne OHLCV (ccxt / Binance spot), bez kluczy. Gdy `api.binance.com` zwraca HTTP 451, zapytania spot idą na publiczne lustro `data-api.binance.vision`. Futures (`fapi`/`dapi`) nie są odpytywane. Koniec zakresu jest północą podanej daty, więc dzień końcowy trzeba podać jako dzień następny.

```bash
python3 binance_data_fetcher.py --non-interactive \
  --symbol BTC/USDT --timeframe 1m \
  --start 2026-07-01 --end 2026-10-01
```

Bez TTY skrypty `binance_data_fetcher.py`, `strategy_runner.py` i `strategy_analyzer.py` same przechodzą w tryb nieinteraktywny.

Konfiguracja zbliżona do zwycięzcy z UI BtD (tf 30 min, próg −2%, sell 1%, SL −3%, pump włączony) jest w `parametry/btc_ui_winner.json`.

```bash
python3 strategy_runner.py --non-interactive \
  --mode backtest \
  --csv-file csv/binance_BTC_USDT_1m_2026-07-01_2026-10-01.csv \
  --param-file parametry/btc_ui_winner.json \
  --output-prefix btc_ui_winner
```

Siatka `min/max/step` bez losowego Latin Hypercube: dodaj `--exact-grid`.

```bash
python3 strategy_analyzer.py --no-recommendations --skip-prefiltering --min-trades 1
```

`parameter_configurator.py` dalej pyta w terminalu. W trybie headless parametry edytuje się jako JSON (stała: `{"type":"numeric","value":-3.0}` albo zakres: `{"type":"numeric","range":[-3,-1,1]}`).

## Siatka wokół wyników z UI BtD

```bash
python3 run_btc_parameter_search.py --self-test-only
python3 run_btc_parameter_search.py \
  --csv csv/binance_BTC_USDT_1m_2026-07-01_2026-10-01.csv
```

Wynik ląduje w `artifacts/btc_btd_grid_summary.csv` i `.md` oraz w `wyniki/backtesty/*.pkl`.
Raport interpretacji: `REPORT.md`.

## Mapowanie na pola BtD

| BtD (UI) | Pole lokalne | Uwagi |
| --- | --- | --- |
| Check Timeframe | `check_timeframe` | Liczba świec 1m, czyli minuty przy danych 1m. Zegar świecy to minuty od epoch; przy pandas 3 (datetime w mikrosekundach) nie wolno dzielić `int64 // 60e9` |
| Percentage Buy Threshold | `percentage_buy_threshold` | Ujemny próg zmiany `(high+low)/2` |
| Sell Enabled | `sell_profit_target` | Działa, gdy `trailing_enabled` jest wyłączone |
| Trailing Stop price / margin / time | `trailing_stop_price`, `trailing_stop_margin`, `trailing_stop_time` | Wyklucza się ze zwykłym sell |
| Stop Loss | `stop_loss_threshold` | Musi być ujemny, np. −3 |
| Stop Loss delay | `stop_loss_delay_time` | Minuty; pierwsze naruszenie tylko uzbraja timer |
| Pump Detection | `pump_detection_enabled`, `pump_detection_threshold`, `pump_detection_disabled_time` | Po wzroście ≥ próg zakupy są wstrzymane na `disabled_time` minut |
| Next Buy Delay | `next_buy_delay` | Minuty od ostatniego wejścia |
| Next buy price lower | `next_buy_price_lower` | Kolejna pozycja tylko głębiej |
| Max open orders | `max_open_orders` | `max_open_orders_per_coin` nie wchodzi do rdzenia numba |
| Trailing buy | `trailing_buy_*` | Osobna ścieżka wejścia |
| Follow BTC | `follow_btc_*` | Przy BTC/USDT i BTC/USDC warunek jest martwy (ta sama seria) |

Czego lokalny model nie liczy tak jak live BtD: prowizji w rdzeniu, dźwigni, isolated margin, limitu obrotu `max_allowed_usd`, poślizgu `add_to_limit_order`, filli tickowych (jest jedna cena na świecę: środek high/low) oraz domknięcia pozycji, które na końcu serii są mark-to-market, a nie sprzedażą z reguły.
