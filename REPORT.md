# Raport: parametry BtD dla BTC

Offline’owy backtester **potwierdza** zestaw z UI (check timeframe 30 min, próg zakupu −2%, sell 1% bez trailingu, SL −3%) i **doprecyzowuje** go: na tym oknie lepiej trzyma się ten sam próg i te same wyjścia przy timeframe **60 min**. Nie zaprzecza zwycięzcy 18120. Trailing 1% / 0,3% przy progu −2% w tym modelu wypada słabiej niż stały sell 1%.

Dane: publiczne spot **BTC/USDT** 1m z `data-api.binance.vision` (api.binance.com zwraca HTTP 451). 131 350 świec, **2026-07-01 00:00 UTC → 2026-09-30 05:09 UTC**. Cena od 57 800 do 87 396 USDT. To okno wzrostowe, więc wysoki win rate nie jest własnością strategii na każdym reżimie.

Przeliczenie jak w UI: 100 USDC notional, saldo 1000, 1x. 1 punkt procentowy na zamknięciu = 1 USDC. Prowizja **nie siedzi w rdzeniu**; doliczona po fakcie jako 0,10 pp na zamknięcie regułą (2 × 0,05% taker USD-M).

Pełna siatka: `artifacts/btc_btd_grid_summary.csv` i `artifacts/btc_btd_grid_summary.md` (71 konfiguracji plus osobny start każdego miesiąca).

## Rekomendacja

### Konserwatywny zestaw (do live BtD na BTC)

| Pole | Wartość |
| --- | --- |
| Check Timeframe | **60 min** |
| Percentage Buy Threshold | **−2%** |
| Sell Enabled | **1%**, trailing wyłączony |
| Stop Loss | **−3%**, delay 0 |
| Max open orders | **1** |
| Next Buy Delay | 1 min (na tej ścieżce i 60 min nic nie zmienia) |
| Pump Detection | włączone, próg 5%, blokada 30 min (na BTC w tym oknie filtr nie odciął żadnego wejścia) |

Dowód, pełne okno: **5 zamknięć regułą, 0 mark-to-market, win rate 100%, gross +5,10 USDC, net +4,60 USDC, max DD zamknięć 0**. Osobny start miesiąca: lipiec +0,93 / 1 transakcja, sierpień +1,83 / 2, wrzesień +1,84 / 2. Jedyny wariant progu −2% ze stałym sellem 1%, który jest na plusie w każdym z trzech miesięcy.

`strategy_analyzer.py` (score, bez pre-filtra) ustawił ten sam zestaw jako reprezentanta klastra nr 1 (score 1162, profit factor nieskończony, bo nie było straty). Score analizatora zeruje strategie poniżej 5 transakcji, więc bliźniak 30 min (4 transakcje) w tym rankingu w ogóle nie startuje.

### Zestaw zgodny z UI 18120 (zostawić, jeśli timeframe ma zostać 30)

Te same wyjścia, **Check Timeframe 30 min**, próg **−2%**, max 1 pozycja.

Pełne okno: **4 zamknięcia, win 100%, gross +4,28 USDC, net +3,88 USDC, DD 0**, najsłabsze zamknięcie +0,90 pp. `strategy_runner.py` na `parametry/btc_ui_winner.json` daje ten sam wynik (suma tradów 4,275 pp, SL −3,0). UI miało około **+5 USDC na 5 transakcjach**. Rząd wielkości i liczba tradów się zgadzają; brakuje jednej transakcji i około 0,7 USDC gross.

Miesiące: lipiec 0 transakcji, sierpień +0,90, wrzesień +2,97. Lipiec jest martwy, więc 30 min jest rzadsze niż 60 min, a nie „gorsze jakościowo” na zamknięciach, które w ogóle powstały.

SL −2% i SL −5% dają **identyczny** wynik przy progu −2% (żadne zamknięcie nie doszło do stopu). −3% zostaje jako ubezpieczenie z UI, nie jako coś, co na tej próbce odróżnia P&L.

### Agresywny, opcjonalny

Check Timeframe **30 min**, próg **−1,5%**, sell **1%**, trailing off, SL **−3%**, max **1** pozycja. To odpowiednik 18124.

Pełne okno: **9 zamknięć, win 89%, gross +5,36 USDC, net +4,46 USDC, DD 3,16 USDC (0,32% salda 1000), najgorsza transakcja −3,16 pp**. Po prowizji przewaga nad konserwatywnym 30 min to około +0,6 USDC, przy jednym pełnym stopie. Wrzesień osobno: **−0,27 USDC / 4 transakcje**. Wyższy obrót zjada większość dodatkowego gross.

Próg **−1%** daje wyższy net (tf 30: +12,78 USDC / 24 zamknięcia; tf 60 i SL −5%: +24 USDC / 33 zamknięcia), ale to zbieranie płytkich dołków w trendzie wzrostowym, ze stopami i czerwonym wrześniem przy tf 30 (−0,38 USDC). Na live BTC tego nie ustawiałbym jako domyślnego.

Trzy równoległe pozycje przy progu −2% / tf 30 podnoszą wynik do net +9,46 USDC / 10 zielonych zamknięć. DD serii zamknięć zostaje 0, a jednocześnie w rynku może leżeć do 300 USDC. Model nie liczy drawdownu niezrealizowanego nakładających się pozycji. Zostawiam to jako obserwację, nie jako ustawienie.

## Czego siatka nie wspiera

- Próg zakupu **−3% i −5%** przy timeframe 15 i 30 min: **zero transakcji**. Przy 60 min próg −3% odpala raz (+0,91 USDC). Zgadza się ze skanem UI, w którym na krótkim oknie BTC ruszały tylko −1 / −1,5 / −2%.
- Sell **1,5%** przy tf 30 / −2%: net **−1,21 USDC**, bo jedna pozycja nie dobiła do 1,5% (została dociągnięta mark-to-market albo stopem). Przy progu −2% cel 1% jest częścią wyniku, nie kosmetyką.
- Trailing TLS **1% / margin 0,3% / czas 1 min**, SL −3%, próg −2%, tf 30: net **+0,06 USDC**, win 75%, wrzesień **−1,08 USDC**. W tym symulatorze to nie jest bliski drugi wynik. Przy progu −1,5% trailing jest prawie równy stałemu sellowi 1% (net +4,52 vs +4,46). Przy progu −1% trailing jest nieco lepszy od sella 1%, za cenę większego DD.

## Luki modelu względem live BtD

Rdzeń (`run_strategy_core`) patrzy raz na świecę na cenę `(high+low)/2`. Wejście jest wtedy, gdy zmiana tej ceny na `check_timeframe` świecach zejdzie pod próg. Nie ma fillu tickowego na poziomie progu, nie ma poślizgu (`add_to_limit_order` jest w JSON i nie wchodzi do symulacji), nie ma isolated margin ani dźwigni.

Prowizja, saldo 1000 i notional 100 są tylko w raporcie. `max_allowed_usd` i `max_open_orders_per_coin` nie sterują pętlą; limit pozycji to `max_open_orders`.

Stop loss: pierwsze naruszenie progu tylko uzbraja timer, zamknięcie jest na kolejnych świecach, gdy strata nadal trwa. Przy delay 0 to co najmniej jedna minuta później. Trailing uzbraja się dopiero po `trailing_stop_time` minutach powyżej `trailing_stop_price`, potem schodzi o `trailing_stop_margin` od lokalnego szczytu. Live „TLS 1% / 0,3%” może uzbrajać się od razu. Stały sell i trailing wykluczają się.

Pump: sam wzrost na bieżącej świecy nie koliduje z warunkiem dipu (jeden jest dodatni, drugi ujemny). Blokada zakupów na `pump_detection_disabled_time` minut była w konfiguracji i w walidacji, a rdzeń jej nie czytał. Jest podłączona. Przy progu 5% i 2% na tf 30 / kupno −2% wynik się nie zmienia: w tym oknie filtr nie stanął przed żadnym wejściem BTC. Na altach albo przy niższym progu może.

Follow BTC na parze BTC jest martwy (ta sama seria). Koniec serii domyka otwarte pozycje po ostatniej cenie (`open_mtm`). Zwycięskie ścieżki −2% nie miały takiej pozycji; przy płytszych progach jedna sztuka MTM siedzi w liczbie zamknięć.

DD w tabeli to obsunięcie **sumy zamkniętych** tradów, nie equity co świecę. Miesiące są osobnymi startami od zera.

Para w UI to BTCUSDC (futures, isolated). Tutaj jest spot BTCUSDT. Dla procentów to bliski substytut, nie ten sam order book.

Generator kombinacji wcześniej **wyrzucał** stały `stop_loss_threshold` i wstawiał −20%, oraz wymuszał włączony stop. Bez tej poprawki porównanie z SL −3% byłoby fałszywe. Zegar świec (`int64 // 60e9`) przy pandas 3 (mikrosekundy) kompresował czas około tysiąckrotnie; opóźnienia i trailing liczyłyby się w tysiącach minut. Liczby w tym raporcie są po obu poprawkach.

## Jak to powtórzyć

```bash
python3 -m pip install -r requirements.txt
python3 binance_data_fetcher.py --non-interactive \
  --symbol BTC/USDT --timeframe 1m --start 2026-07-01 --end 2026-10-01
python3 run_btc_parameter_search.py \
  --csv csv/binance_BTC_USDT_1m_2026-07-01_2026-10-01.csv
python3 strategy_runner.py --non-interactive --mode backtest \
  --csv-file csv/binance_BTC_USDT_1m_2026-07-01_2026-10-01.csv \
  --param-file parametry/btc_ui_winner.json --output-prefix btc_ui_winner
python3 strategy_analyzer.py --no-recommendations --skip-prefiltering \
  --pkl <plik z wyniki/backtesty>
```

Koniec zakresu fetchera jest północą podanej daty, stąd `--end 2026-10-01`. Surowe CSV zostaje lokalnie (gitignore).
