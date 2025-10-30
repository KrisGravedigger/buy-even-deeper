# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math # Do sprawdzania nan
import time
import os    # Do operacji na ścieżkach i plikach
import glob  # Do wyszukiwania plików

# --- Konfiguracja ---
CSV_SUBDIRECTORY = 'csv'      # Nazwa podkatalogu z plikami CSV
# INPUT_FILEPATH zostanie ustawiony dynamicznie poniżej

# Parametry strategii
DIP_PERCENTAGE = -1.0         # Procent spadku wyzwalający zakup (wymagany ujemny)
DIP_WINDOW_MINUTES = 30        # Okno czasowe (w minutach/świecach) do szukania spadku (liczba świec poprzedzających)
TAKE_PROFIT_PERCENTAGE = 1.0   # Procent zysku od ceny wejścia (wymagany dodatni)
STOP_LOSS_PERCENTAGE = -2    # Procent straty od ceny wejścia (wymagany ujemny)

# Sprawdzenie poprawności parametrów procentowych
if DIP_PERCENTAGE >= 0:
    raise ValueError("DIP_PERCENTAGE musi być wartością ujemną (np. -1.25).")
if TAKE_PROFIT_PERCENTAGE <= 0:
    raise ValueError("TAKE_PROFIT_PERCENTAGE musi być wartością dodatnią (np. 1.0).")
if STOP_LOSS_PERCENTAGE >= 0:
    raise ValueError("STOP_LOSS_PERCENTAGE musi być wartością ujemną (np. -0.5).")
if not isinstance(DIP_WINDOW_MINUTES, int) or DIP_WINDOW_MINUTES <= 0:
     raise ValueError("DIP_WINDOW_MINUTES musi być dodatnią liczbą całkowitą.")

def find_latest_csv_in_subdir(subdir_name='csv'):
    """
    Znajduje najnowszy plik .csv w podkatalogu względem lokalizacji skryptu.
    Nie przeszukuje rekursywnie.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_dir_path = os.path.join(script_dir, subdir_name)

        if not os.path.isdir(csv_dir_path):
            print(f"Błąd: Podkatalog '{subdir_name}' nie istnieje w lokalizacji skryptu: {script_dir}")
            return None

        # Wyszukaj pliki .csv tylko w tym katalogu (bez podkatalogów)
        list_of_files = glob.glob(os.path.join(csv_dir_path, '*.csv'))

        if not list_of_files:
            print(f"Błąd: Brak plików .csv w podkatalogu '{subdir_name}'.")
            return None

        # Znajdź najnowszy plik na podstawie czasu modyfikacji
        latest_file = max(list_of_files, key=os.path.getmtime)
        print(f"Znaleziono najnowszy plik CSV w '{subdir_name}': {os.path.basename(latest_file)}")
        return latest_file

    except Exception as e:
        print(f"Wystąpił błąd podczas wyszukiwania najnowszego pliku CSV: {e}")
        return None

# --- Koniec Konfiguracji (dynamiczne ustawienie ścieżki pliku) ---

def load_data(filepath):
    """
    Wczytuje dane z pliku CSV, przetwarza timestamp i identyfikuje kolumny OHLC
    dla głównej pary (pierwsze wystąpienia nie zaczynające się od 'btc_').
    """
    if filepath is None:
        print("Błąd: Ścieżka do pliku nie została określona.")
        return None

    print(f"Wczytywanie danych z pliku: {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print(f"Wczytano {len(df):,} wierszy.")

        # Sprawdzenie i konwersja timestamp
        if 'timestamp' not in df.columns:
            raise ValueError("Brak kolumny 'timestamp' w pliku CSV.")
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            raise ValueError(f"Nie można przekonwertować kolumny 'timestamp' na datę/czas. Błąd: {e}. Sprawdź format.")

        df = df.set_index('timestamp')
        df = df.sort_index() # Upewnij się, że dane są posortowane chronologicznie

        # Automatyczne wykrywanie kolumn dla głównej pary (pierwsze wystąpienia)
        required_cols = ['open', 'high', 'low', 'close']
        main_symbol_cols = {}
        found_cols_count = 0
        potential_cols = {}

        # Najpierw zbierz wszystkie pasujące kolumny
        for col in df.columns:
             # Użyjemy tylko części przed pierwszym '_' jeśli istnieje, chyba że to 'btc_'
             is_btc_col = col.lower().startswith('btc_')
             base_col_name = col
             # Sprawdźmy, czy kolumna zawiera '_' i nie jest kolumną BTC
             # Jeśli tak, weź część przed pierwszym '_', w przeciwnym razie całą nazwę
             # To powinno poprawnie obsłużyć nazwy typu 'TON_open' oraz 'open'
             if not is_btc_col:
                if '_' in col:
                    potential_base = col.split('_')[0]
                    # Dodatkowe sprawdzenie, czy po podzieleniu otrzymujemy jedną z wymaganych nazw
                    if potential_base in required_cols:
                        base_col_name = potential_base
                    else: # Jeśli nie, użyj pełnej nazwy (np. dla 'open')
                        base_col_name = col
                else: # Jeśli nie ma '_', to jest to potencjalnie 'open', 'high', 'low', 'close'
                    base_col_name = col

             # Zapisz potencjalne kolumny nie-BTC tylko jeśli pasują do wymaganych nazw
             if base_col_name in required_cols and not is_btc_col:
                 # Weź pierwsze znalezione wystąpienie dla danego typu (open, high, etc.)
                 if base_col_name not in potential_cols:
                     potential_cols[base_col_name] = col


        # Przypisz znalezione kolumny
        main_symbol_cols = potential_cols
        found_cols_count = len(main_symbol_cols)

        # Sprawdzenie, czy znaleziono wszystkie wymagane kolumny
        missing = [rc for rc in required_cols if rc not in main_symbol_cols]
        if missing:
            print(f"Ostrzeżenie: Nie znaleziono automatycznie wszystkich wymaganych kolumn: {missing}.")
            # Sprawdź, czy brakujące kolumny istnieją pod standardowymi nazwami
            can_proceed = True
            for miss in missing:
                if miss in df.columns and miss not in main_symbol_cols.values():
                    print(f"  Używam standardowej nazwy '{miss}' dla brakującej kolumny.")
                    main_symbol_cols[miss] = miss
                else:
                    # Jeśli kolumny nie ma nawet pod standardową nazwą, to jest problem
                    if miss not in main_symbol_cols: # Upewnij się, że naprawdę brakuje
                        print(f"  Krytyczny błąd: Brak wymaganej kolumny '{miss}' (ani jej wariantu) w pliku CSV.")
                        can_proceed = False
            if not can_proceed:
                 raise ValueError(f"Nie można kontynuować z powodu brakujących kolumn: {missing}")

        print(f"Używane kolumny dla głównej pary: {main_symbol_cols}")

        # Wybierz i zmień nazwy kolumn na standardowe
        # Upewnij się, że używasz poprawnych nazw z main_symbol_cols
        selected_cols_original_names = [main_symbol_cols['open'], main_symbol_cols['high'], main_symbol_cols['low'], main_symbol_cols['close']]
        df_main = df[selected_cols_original_names].copy()
        df_main.columns = ['open', 'high', 'low', 'close'] # Zmień nazwy na standardowe

        # Konwersja na typ numeryczny, błędy zamień na NaN
        for col in df_main.columns:
             df_main[col] = pd.to_numeric(df_main[col], errors='coerce')

        # Usunięcie wierszy z brakującymi danymi w kluczowych kolumnach
        initial_rows = len(df_main)
        df_main = df_main.dropna(subset=['open', 'high', 'low', 'close'])
        removed_rows = initial_rows - len(df_main)
        if removed_rows > 0:
            print(f"Usunięto {removed_rows:,} wierszy z brakującymi danymi (NaN) w kolumnach open, high, low lub close.")

        if len(df_main) == 0:
            print("Błąd: Po przetworzeniu DataFrame jest pusty.")
            return None

        print(f"Dane gotowe do analizy: {len(df_main):,} wierszy.")
        return df_main

    except FileNotFoundError:
        print(f"Błąd: Plik {filepath} nie został znaleziony.")
        return None
    except Exception as e:
        print(f"Wystąpił krytyczny błąd podczas wczytywania danych: {e}")
        import traceback
        traceback.print_exc() # Pokaż pełny ślad błędu
        return None

def run_backtest(df, dip_percentage, dip_window_minutes, take_profit_percentage, stop_loss_percentage):
    """
    Przeprowadza symulację strategii kupowania dołków na danych minutowych.
    (Reszta funkcji run_backtest pozostaje bez zmian - wklejam ją ponownie dla kompletności)
    """
    if df is None or df.empty:
        print("DataFrame jest pusty, przerywam backtest.")
        return None

    start_time = time.time()
    print("\nRozpoczynam backtesting...")
    print(f"Parametry: Spadek={dip_percentage:.2f}%, Okno={dip_window_minutes} min, TP={take_profit_percentage:.2f}%, SL={stop_loss_percentage:.2f}%")

    open_positions = []
    closed_positions = []
    actual_entry_dips = [] # Lista rzeczywistych spadków przy wejściu

    # Iteracja przez każdą świecę (minutę)
    total_candles = len(df)
    for i in range(total_candles):
        # Wyświetlanie postępu co 10000 świec
        if (i + 1) % 10000 == 0:
             elapsed = time.time() - start_time
             print(f"Przetworzono {i+1:,}/{total_candles:,} świec ({elapsed:.1f}s)...")

        current_time = df.index[i]
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        current_open = df['open'].iloc[i]
        current_close = df['close'].iloc[i]

        # --- Sprawdzenie warunku wejścia ---
        if i >= dip_window_minutes:
            start_index = i - dip_window_minutes
            end_index = i
            window_highs = df['high'].iloc[start_index:end_index]

            if not window_highs.empty:
                window_max_high = window_highs.max()

                if not math.isnan(window_max_high) and window_max_high > 0:
                     entry_threshold = window_max_high * (1 + dip_percentage / 100.0)

                     if current_low <= entry_threshold:
                         entry_price = entry_threshold
                         tp_level = entry_price * (1 + take_profit_percentage / 100.0)
                         sl_level = entry_price * (1 + stop_loss_percentage / 100.0)
                         actual_dip = (current_low / window_max_high - 1) * 100.0
                         actual_entry_dips.append(actual_dip)

                         position = {
                             'entry_price': entry_price, 'entry_time': current_time, 'entry_index': i,
                             'tp_level': tp_level, 'sl_level': sl_level, 'status': 'open',
                             'exit_price': None, 'exit_time': None, 'exit_index': None,
                             'pnl_perc': None, 'max_potential_profit_perc': 0.0,
                             'actual_dip_at_entry': actual_dip
                         }
                         open_positions.append(position)

        # --- Zarządzanie otwartymi pozycjami ---
        positions_to_close_indices = []
        for j, pos in enumerate(open_positions):
            candle_high = current_high
            candle_low = current_low

            potential_profit = (candle_high / pos['entry_price'] - 1) * 100.0
            pos['max_potential_profit_perc'] = max(pos['max_potential_profit_perc'], potential_profit if potential_profit > -100 else -100)

            hit_tp = candle_high >= pos['tp_level']
            hit_sl = candle_low <= pos['sl_level']

            if hit_tp and hit_sl:
                pos['status'] = 'undecided'; pos['exit_price'] = None; pos['pnl_perc'] = None
                pos['exit_time'] = current_time; pos['exit_index'] = i
                closed_positions.append(pos); positions_to_close_indices.append(j)
            elif hit_tp:
                pos['status'] = 'profit'; pos['exit_price'] = pos['tp_level']; pos['pnl_perc'] = take_profit_percentage
                pos['exit_time'] = current_time; pos['exit_index'] = i
                closed_positions.append(pos); positions_to_close_indices.append(j)
            elif hit_sl:
                final_potential_profit = (candle_high / pos['entry_price'] - 1) * 100.0
                pos['max_potential_profit_perc'] = max(pos['max_potential_profit_perc'], final_potential_profit if final_potential_profit > -100 else -100)
                pos['status'] = 'loss'; pos['exit_price'] = pos['sl_level']; pos['pnl_perc'] = stop_loss_percentage
                pos['exit_time'] = current_time; pos['exit_index'] = i
                closed_positions.append(pos); positions_to_close_indices.append(j)

        if positions_to_close_indices:
            for index in sorted(positions_to_close_indices, reverse=True):
                if index < len(open_positions):
                    del open_positions[index]
                else:
                     print(f"Ostrzeżenie: Próba usunięcia nieprawidłowego indeksu {index} z open_positions.")

    end_time = time.time()
    print(f"Zakończono pętlę symulacji. Czas trwania: {end_time - start_time:.2f}s")

    # --- Zamknięcie pozycji, które pozostały otwarte ---
    if open_positions:
        print(f"Zamykanie {len(open_positions)} pozycji pozostałych otwartych na koniec symulacji...")
        final_time = df.index[-1]
        final_price = df['close'].iloc[-1]
        last_high = df['high'].iloc[-1] # Potrzebne do aktualizacji max profit dla ostatniej świecy
        for pos in open_positions:
            potential_profit = (last_high / pos['entry_price'] - 1) * 100.0
            pos['max_potential_profit_perc'] = max(pos['max_potential_profit_perc'], potential_profit if potential_profit > -100 else -100)
            pos['status'] = 'closed_at_end'; pos['exit_price'] = final_price
            pos['exit_time'] = final_time; pos['exit_index'] = total_candles - 1
            pos['pnl_perc'] = (final_price / pos['entry_price'] - 1) * 100.0
            closed_positions.append(pos)

    print(f"Zakończono backtesting. Przetworzono {total_candles:,} świec.")

    # --- Obliczanie statystyk ---
    total_entries = len(closed_positions)
    if total_entries == 0:
        print("\nNie znaleziono żadnych okazji do wejścia na podstawie zadanych kryteriów.")
        return {}

    total_profits = sum(1 for pos in closed_positions if pos['status'] == 'profit')
    total_losses = sum(1 for pos in closed_positions if pos['status'] == 'loss')
    total_undecided = sum(1 for pos in closed_positions if pos['status'] == 'undecided')
    total_closed_at_end = sum(1 for pos in closed_positions if pos['status'] == 'closed_at_end')

    avg_actual_dip = np.mean(actual_entry_dips) if actual_entry_dips else 0
    median_actual_dip = np.median(actual_entry_dips) if actual_entry_dips else 0
    min_actual_dip = min(actual_entry_dips) if actual_entry_dips else 0
    max_actual_dip = max(actual_entry_dips) if actual_entry_dips else 0

    max_profit_no_sl_list = [pos['max_potential_profit_perc'] for pos in closed_positions if pos['status'] != 'loss']
    avg_max_profit_no_sl = np.mean(max_profit_no_sl_list) if max_profit_no_sl_list else 0
    median_max_profit_no_sl = np.median(max_profit_no_sl_list) if max_profit_no_sl_list else 0
    min_max_profit_no_sl = min(max_profit_no_sl_list) if max_profit_no_sl_list else 0
    max_max_profit_no_sl = max(max_profit_no_sl_list) if max_profit_no_sl_list else 0

    # --- Prezentacja wyników ---
    print("\n--- Wyniki Symulacji ---")
    print(f"Parametry:")
    print(f"  - Spadek wyzwalający [%]: {dip_percentage:.2f}")
    print(f"  - Okno spadku [min]:     {dip_window_minutes}")
    print(f"  - Take Profit [%]:       {take_profit_percentage:.2f}")
    print(f"  - Stop Loss [%]:         {stop_loss_percentage:.2f}")
    print("-" * 30)
    print(f"Liczba wejść ogółem:              {total_entries:,}")
    print(f"Liczba zrealizowanych zysków (TP): {total_profits:,}")
    print(f"Liczba zrealizowanych strat (SL):  {total_losses:,}")
    print(f"Liczba nierozstrzygniętych (TP i SL w tej samej świecy): {total_undecided:,}")
    print(f"Liczba zamkniętych na koniec symulacji: {total_closed_at_end:,}")
    print("-" * 30)
    resolved_trades = total_profits + total_losses
    win_rate = 0 # Domyślnie 0
    reward_risk_ratio = 0 # Domyślnie 0
    if resolved_trades > 0:
        win_rate = total_profits / resolved_trades
        print(f"Win Rate (TP / (TP + SL)):      {win_rate:.2%}")
        if stop_loss_percentage != 0:
             reward_risk_ratio = abs(take_profit_percentage / stop_loss_percentage)
             print(f"Ustawiony stosunek Zysk/Ryzyko: {reward_risk_ratio:.2f}")
        else:
             print(f"Ustawiony stosunek Zysk/Ryzyko: Nieskończony (SL=0)")

    else:
        print("Win Rate: Brak pozycji zakończonych przez TP lub SL.")

    print("\n--- Dodatkowe Statystyki ---")
    print("Statystyki RZECZYWISTYCH spadków [%] przy wejściu (dla wszystkich wejść):")
    if actual_entry_dips:
        print(f"  - Średni spadek:   {avg_actual_dip:.2f}%")
        print(f"  - Mediana spadku:  {median_actual_dip:.2f}%")
        print(f"  - Minimalny spadek:{min_actual_dip:.2f}%")
        print(f"  - Maksymalny spadek:{max_actual_dip:.2f}%")
    else:
        print("  - Brak danych (nie dokonano żadnych wejść).")

    print("\nStatystyki MAKSYMALNYCH WZROSTÓW [%] od ceny wejścia")
    print("(tylko dla pozycji, które NIE zostały zamknięte przez Stop Loss):")
    if max_profit_no_sl_list:
        print(f"  - Średni maks. wzrost:    {avg_max_profit_no_sl:.2f}%")
        print(f"  - Mediana maks. wzrostu:  {median_max_profit_no_sl:.2f}%")
        print(f"  - Minimalny maks. wzrost: {min_max_profit_no_sl:.2f}%")
        print(f"  - Maksymalny maks. wzrost:{max_max_profit_no_sl:.2f}%")
        print(f"  (Liczba pozycji w statystyce: {len(max_profit_no_sl_list):,})")
    else:
        print("  - Brak pozycji niezakończonych stratą lub brak wejść.")

    return {
        'params': {
            'dip_percentage': dip_percentage, 'dip_window_minutes': dip_window_minutes,
            'take_profit_percentage': take_profit_percentage, 'stop_loss_percentage': stop_loss_percentage,
        },
        'results': {
            'total_entries': total_entries, 'total_profits': total_profits, 'total_losses': total_losses,
            'total_undecided': total_undecided, 'total_closed_at_end': total_closed_at_end,
            'win_rate': win_rate, 'reward_risk_ratio': reward_risk_ratio,
        },
        'stats': {
            'avg_actual_dip': avg_actual_dip, 'median_actual_dip': median_actual_dip,
            'min_actual_dip': min_actual_dip, 'max_actual_dip': max_actual_dip,
            'avg_max_profit_no_sl': avg_max_profit_no_sl, 'median_max_profit_no_sl': median_max_profit_no_sl,
            'min_max_profit_no_sl': min_max_profit_no_sl, 'max_max_profit_no_sl': max_max_profit_no_sl,
            'count_actual_dips': len(actual_entry_dips), 'count_max_profit_no_sl': len(max_profit_no_sl_list),
        }
    }

# --- Główny blok wykonawczy ---
if __name__ == "__main__":
    print("="*50)
    print("Rozpoczynanie Skryptu Backtestingu Strategii Kupowania Dołków")
    print("="*50)

    # Znajdź najnowszy plik CSV w podkatalogu
    INPUT_FILEPATH = find_latest_csv_in_subdir(CSV_SUBDIRECTORY)

    if INPUT_FILEPATH:
        # Wczytaj dane
        data_df = load_data(INPUT_FILEPATH)

        # Uruchom backtest, jeśli dane zostały wczytane poprawnie
        if data_df is not None and not data_df.empty:
            results = run_backtest(data_df,
                                   DIP_PERCENTAGE,
                                   DIP_WINDOW_MINUTES,
                                   TAKE_PROFIT_PERCENTAGE,
                                   STOP_LOSS_PERCENTAGE)

            if results:
                print("\n" + "="*50)
                print("Zakończono analizę.")
                print("="*50)
            else:
                print("\nBacktest nie wygenerował wyników (prawdopodobnie nie było żadnych wejść lub wystąpił błąd).")
        else:
            print("\nNie udało się wczytać lub przetworzyć danych z pliku. Przerywam działanie.")
    else:
        # Komunikat o błędzie został już wyświetlony w find_latest_csv_in_subdir
        print("\nNie można kontynuować bez pliku wejściowego. Przerywam działanie.")