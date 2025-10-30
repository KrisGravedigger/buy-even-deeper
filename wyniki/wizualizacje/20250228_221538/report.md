# Raport strategii tradingowej dla TON

## Informacje o danych

* **Symbol:** TON
* **Okres:** 2025-02-12 11:00 - 2025-03-14 10:59
* **Długość analizy:** 29 days 23:59:00
* **Liczba świeczek:** 43200

## Statystyki rynku

* **Cena początkowa:** 3.701557
* **Cena końcowa:** 2.759638
* **Zmiana rynku:** -25.45%

## Statystyki strategii

* **Liczba transakcji:** 84
* **Całkowity zysk:** 6.19%
* **Średni zysk na transakcję:** 0.07%
* **Win rate:** 69.05%

## Parametry strategii

| Parametr | Wartość | Opis |
|---------|---------|------|
| check_timeframe | 15 | Określa zakres czasowy/świeczkę do analizy. Przykład: wartość 1 oznacza analiza 1-minutowych świeczek, wartość 30 oznacza analiza 30-minutowych świeczek. Im większy przedział czasowy, tym większa szansa na złapanie spadku, ale głównym celem jest łapanie krótkich spadków z szybkim odbiciem. |
| percentage_buy_threshold | -1.0 | Określa procentowy spadek względem wybranego timeframe'u. Należy dostosować wielkość spadku do wybranego czasu - większy timeframe pozwala na większy spadek. |
| max_allowed_usd | 100000000.0 | Maksymalny obrót jaki może wykonać aplikacja. Po przekroczeniu tej wartości aplikacja przestaje dokonywać zakupów. |
| add_to_limit_order | 2.0 | Zabezpieczenie przy zakupie - określa maksymalną różnicę procentową powyżej ceny zakupu. Przy sprzedaży określa maksymalną różnicę poniżej ceny sprzedaży. Sugerowana wartość: 2%. |
| sell_profit_target | 0.0 | Określa próg procentowy zysku, przy którym pozycja zostanie sprzedana w trybie zwykłej sprzedaży. |
| trailing_enabled | 1.0 | Włącza/wyłącza funkcję trailing stop (podążający stop loss). Trailing śledzi wzrost ceny i automatycznie podnosi poziom stop loss. |
| trailing_stop_price | 0.6 | Określa próg procentowy zysku, przy którym zostanie aktywowany trailing stop. |
| trailing_stop_margin | 0.4 | Określa o ile procent poniżej aktualnej ceny ma być ustawiony trailing stop loss. Sugerowana minimalna wartość: 0.5%. |
| trailing_stop_time | 0 | Czas w minutach, przez który cena musi utrzymać się powyżej progu trailing stop, zanim zostanie on aktywowany. |
| stop_loss_enabled | 1.0 | Włącza/wyłącza funkcję stop loss (automatyczna sprzedaż ze stratą w celu jej ograniczenia). |
| stop_loss_threshold | 2.5 | Określa próg procentowy poniżej ceny zakupu, przy którym zostanie aktywowany stop loss. |
| stop_loss_delay_time | 0 | Czas w minutach przed aktywacją stop loss. Przydatne przy większych spadkach/flash crashach - chroni przed zbyt szybkim "wycięciem" przez stop loss. Używane głównie przy wysokich progach spadkowych. |
| max_open_orders_per_coin | 4 | Maksymalna liczba jednocześnie otwartych pozycji dla jednej kryptowaluty. |
| next_buy_delay | 60 | Minimalny czas w minutach, jaki musi upłynąć przed kolejnym zakupem tej samej kryptowaluty. |
| next_buy_price_lower | 2.0 | O ile procent niżej od poprzedniego zakupu musi być cena, aby system mógł dokonać kolejnego zakupu tej samej kryptowaluty (po upływie czasu next_buy_delay). |
| pump_detection_enabled | 1.0 | Włącza/wyłącza funkcję wykrywania pump'ów (gwałtownych wzrostów ceny) - chroni przed zakupami na szczytach. |
| pump_detection_threshold | 3.0 | Określa próg procentowy wzrostu ceny (w czasie określonym przez check_timeframe), powyżej którego system uzna ruch za pump i wstrzyma zakupy. |
| pump_detection_disabled_time | 120 | Minimalny czas w minutach, na jaki zostanie wyłączone kupowanie danej kryptowaluty po wykryciu pump'a. Jeśli wzrost będzie kontynuowany, czas może się wydłużyć. |
| follow_btc_price | 1.0 | Włącza/wyłącza funkcję śledzenia ceny BTC. Gdy włączona, system wstrzyma zakupy altcoinów jeśli BTC spadnie mocniej i szybciej niż dany altcoin - chroni przed silnymi spadkami rynku. |
| max_open_orders | 4 | Maksymalna liczba wszystkich jednocześnie otwartych pozycji dla jednej strategii. |
| stop_loss_disable_buy | 1.0 | Włącza/wyłącza funkcję modyfikacji kolejnych zakupów po aktywacji stop loss dla danej kryptowaluty. |
| stop_loss_disable_buy_all | 0.0 | Włącza/wyłącza funkcję czasowego wstrzymania zakupów wszystkich kryptowalut po aktywacji stop loss - funkcja chroniąca przed silnymi spadkami rynku. |
| stop_loss_next_buy_lower | -16.0 | Określa o ile procent niżej od ceny sprzedaży po stop loss system będzie próbował kupić ponownie tę samą kryptowalutę. |
| stop_loss_no_buy_delay | 120.0 | Czas w minutach, który musi upłynąć od aktywacji stop loss zanim system sprawdzi cenę i spróbuje dokonać ponownego zakupu. W przypadku włączonej opcji stop_loss_disable_buy_all określa czas wstrzymania wszystkich zakupów. |
| trailing_buy_enabled | 0.0 | Włącza/wyłącza funkcję trailing buy (oczekiwanie na dalszy spadek ceny przed zakupem). Wartość 1.0 oznacza włączone, 0.0 oznacza wyłączone. |
| follow_btc_threshold | -2.0 | Określa próg spadku ceny BTC w procentach. Jeśli BTC spadnie o więcej niż zadany próg, system wstrzyma zakupy altcoinów - chroni przed silnymi spadkami rynku. |
| follow_btc_block_time | 20.0 | Czas w minutach, przez który system będzie wstrzymywał zakupy po wykryciu silnego spadku BTC. |

## Wykres

![Wykres strategii](chart_TON.png)

## Szczegóły transakcji

Szczegółowa tabela transakcji dostępna w pliku: [transactions.md](transactions.md)
