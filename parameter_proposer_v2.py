# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import time
import os
import glob
import statistics
from collections import defaultdict

# --- Konfiguracja ---
CSV_SUBDIRECTORY = 'csv'
OPTIMIZATION_SUBDIR = 'parameter_optimization' # Na razie nieużywany

# Parametry strategii (początkowe)
INITIAL_PARAMS = {
    'dip_percentage': -1.60,  # Zmieniono na -1.00% dla przykładu z logu
    'dip_window_minutes': 30,
    'take_profit_percentage': 1.00, # Zmieniono na 1.00%
    'stop_loss_percentage': -2.00 # Zmieniono na -0.80%
}

# Walidacja parametrów początkowych (jak poprzednio)
# ... (kod walidacji bez zmian) ...
if INITIAL_PARAMS['dip_percentage'] >= 0: raise ValueError("Początkowy DIP_PERCENTAGE musi być ujemny.")
if INITIAL_PARAMS['take_profit_percentage'] <= 0: raise ValueError("Początkowy TAKE_PROFIT_PERCENTAGE musi być dodatni.")
if INITIAL_PARAMS['stop_loss_percentage'] >= 0: raise ValueError("Początkowy STOP_LOSS_PERCENTAGE musi być ujemny.")
if not isinstance(INITIAL_PARAMS['dip_window_minutes'], int) or INITIAL_PARAMS['dip_window_minutes'] <= 0: raise ValueError("Początkowy DIP_WINDOW_MINUTES musi być dodatnią liczbą całkowitą.")


# --- Funkcje pomocnicze (find_latest_csv_in_subdir, load_data - bez zmian) ---
# ... (kod find_latest_csv_in_subdir i load_data bez zmian) ...
def find_latest_csv_in_subdir(subdir_name='csv'):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_dir_path = os.path.join(script_dir, subdir_name)
        if not os.path.isdir(csv_dir_path):
            print(f"Błąd: Podkatalog '{subdir_name}' nie istnieje w lokalizacji skryptu: {script_dir}")
            return None
        list_of_files = glob.glob(os.path.join(csv_dir_path, '*.csv'))
        if not list_of_files:
            print(f"Błąd: Brak plików .csv w podkatalogu '{subdir_name}'.")
            return None
        latest_file = max(list_of_files, key=os.path.getmtime)
        print(f"Znaleziono najnowszy plik CSV w '{subdir_name}': {os.path.basename(latest_file)}")
        return latest_file
    except Exception as e:
        print(f"Wystąpił błąd podczas wyszukiwania najnowszego pliku CSV: {e}")
        return None

def load_data(filepath):
    if filepath is None: return None
    print(f"Wczytywanie danych z pliku: {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print(f"Wczytano {len(df):,} wierszy.")
        if 'timestamp' not in df.columns: raise ValueError("Brak kolumny 'timestamp'")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        required_cols = ['open', 'high', 'low', 'close']
        main_symbol_cols = {}
        potential_cols = {}
        for col in df.columns:
             is_btc_col = col.lower().startswith('btc_')
             base_col_name = col
             if not is_btc_col:
                potential_base = col.split('_')[0] if '_' in col else col
                if potential_base in required_cols:
                    base_col_name = potential_base
                else:
                    base_col_name = col # Keep original if split part isn't required

             if base_col_name in required_cols and not is_btc_col:
                 if base_col_name not in potential_cols:
                     potential_cols[base_col_name] = col

        main_symbol_cols = potential_cols
        missing = [rc for rc in required_cols if rc not in main_symbol_cols]
        if missing:
            print(f"Ostrzeżenie: Nie znaleziono automatycznie: {missing}.")
            can_proceed = True
            for miss in missing:
                if miss in df.columns and miss not in main_symbol_cols.values():
                    print(f"  Używam standardowej nazwy '{miss}'.")
                    main_symbol_cols[miss] = miss
                else:
                    if miss not in main_symbol_cols:
                         print(f"  Krytyczny błąd: Brak kolumny '{miss}'.")
                         can_proceed = False
            if not can_proceed: raise ValueError(f"Brakujące kolumny: {missing}")

        print(f"Używane kolumny: {main_symbol_cols}")
        selected_cols_original_names = [main_symbol_cols[rc] for rc in required_cols]
        df_main = df[selected_cols_original_names].copy()
        df_main.columns = ['open', 'high', 'low', 'close']
        for col in df_main.columns: df_main[col] = pd.to_numeric(df_main[col], errors='coerce')
        initial_rows = len(df_main)
        df_main = df_main.dropna(subset=['open', 'high', 'low', 'close'])
        removed_rows = initial_rows - len(df_main)
        if removed_rows > 0: print(f"Usunięto {removed_rows:,} wierszy z NaN.")
        if len(df_main) == 0: return None
        print(f"Dane gotowe: {len(df_main):,} wierszy.")
        return df_main
    except Exception as e:
        print(f"Krytyczny błąd podczas wczytywania danych: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Funkcja Backtestu (bez zmian w logice wewnętrznej) ---
# ... (kod run_backtest bez zmian - ważne: nadal zwraca summary_results, closed_positions_details) ...
def run_backtest(df, dip_percentage, dip_window_minutes, take_profit_percentage, stop_loss_percentage):
    if df is None or df.empty: return None, []

    start_time = time.time()
    print("\nRozpoczynam backtesting...")
    print(f"Parametry: Spadek={dip_percentage:.2f}%, Okno={dip_window_minutes} min, TP={take_profit_percentage:.2f}%, SL={stop_loss_percentage:.2f}%")

    open_positions = []
    closed_positions_details = []
    actual_entry_dips = []

    total_candles = len(df)
    for i in range(total_candles):
        if (i + 1) % 20000 == 0: print(f"Przetworzono {i+1:,}/{total_candles:,} świec ({(time.time() - start_time):.1f}s)...")

        current_time = df.index[i]
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]

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
                             'pnl_perc': None,
                             'max_potential_profit_perc': 0.0,
                             'max_drawdown_perc': 0.0,
                             'duration_candles': 0,
                             'actual_dip_at_entry': actual_dip
                         }
                         open_positions.append(position)

        positions_to_close_indices = []
        for j, pos in enumerate(open_positions):
            potential_profit = (current_high / pos['entry_price'] - 1) * 100.0
            pos['max_potential_profit_perc'] = max(pos['max_potential_profit_perc'], potential_profit if potential_profit > -100 else -100)
            current_drawdown = (current_low / pos['entry_price'] - 1) * 100.0
            pos['max_drawdown_perc'] = min(pos['max_drawdown_perc'], current_drawdown if current_drawdown < 100 else 100)

            hit_tp = current_high >= pos['tp_level']
            hit_sl = current_low <= pos['sl_level']
            closed = False
            if hit_tp and hit_sl:
                pos['status'] = 'undecided'; pos['exit_price'] = None; pos['pnl_perc'] = None; closed = True
            elif hit_tp:
                pos['status'] = 'profit'; pos['exit_price'] = pos['tp_level']; pos['pnl_perc'] = take_profit_percentage; closed = True
            elif hit_sl:
                final_potential_profit = (current_high / pos['entry_price'] - 1) * 100.0
                pos['max_potential_profit_perc'] = max(pos['max_potential_profit_perc'], final_potential_profit if final_potential_profit > -100 else -100)
                final_drawdown = (current_low / pos['entry_price'] - 1) * 100.0
                pos['max_drawdown_perc'] = min(pos['max_drawdown_perc'], final_drawdown if final_drawdown < 100 else 100)
                pos['status'] = 'loss'; pos['exit_price'] = pos['sl_level']; pos['pnl_perc'] = stop_loss_percentage; closed = True

            if closed:
                pos['exit_time'] = current_time; pos['exit_index'] = i
                pos['duration_candles'] = pos['exit_index'] - pos['entry_index']
                closed_positions_details.append(pos.copy())
                positions_to_close_indices.append(j)

        if positions_to_close_indices:
            for index in sorted(positions_to_close_indices, reverse=True):
                if index < len(open_positions): del open_positions[index]

    end_time = time.time()
    print(f"Zakończono pętlę symulacji. Czas trwania: {end_time - start_time:.2f}s")

    if open_positions:
        print(f"Zamykanie {len(open_positions)} pozycji pozostałych otwartych...")
        final_time = df.index[-1]; final_price = df['close'].iloc[-1]
        last_high = df['high'].iloc[-1]; last_low = df['low'].iloc[-1]
        for pos in open_positions:
            potential_profit = (last_high / pos['entry_price'] - 1) * 100.0
            pos['max_potential_profit_perc'] = max(pos['max_potential_profit_perc'], potential_profit if potential_profit > -100 else -100)
            current_drawdown = (last_low / pos['entry_price'] - 1) * 100.0
            pos['max_drawdown_perc'] = min(pos['max_drawdown_perc'], current_drawdown if current_drawdown < 100 else 100)
            pos['status'] = 'closed_at_end'; pos['exit_price'] = final_price
            pos['exit_time'] = final_time; pos['exit_index'] = total_candles - 1
            pos['duration_candles'] = pos['exit_index'] - pos['entry_index']
            pos['pnl_perc'] = (final_price / pos['entry_price'] - 1) * 100.0
            closed_positions_details.append(pos.copy())

    print(f"Zakończono backtesting. Przetworzono {total_candles:,} świec.")

    summary_results = {}
    total_entries = len(closed_positions_details)
    if total_entries == 0:
        print("\nNie znaleziono żadnych okazji do wejścia.")
        return None, []

    status_counts = defaultdict(int)
    for pos in closed_positions_details: status_counts[pos['status']] += 1
    total_profits = status_counts['profit']; total_losses = status_counts['loss']
    total_undecided = status_counts['undecided']; total_closed_at_end = status_counts['closed_at_end']
    resolved_trades = total_profits + total_losses
    win_rate = (total_profits / resolved_trades * 100.0) if resolved_trades > 0 else 0
    total_profit_perc = total_profits * take_profit_percentage
    total_loss_perc = abs(total_losses * stop_loss_percentage)
    profit_factor = total_profit_perc / total_loss_perc if total_loss_perc > 0 else float('inf')
    estimated_pnl = (total_profits * take_profit_percentage) + (total_losses * stop_loss_percentage)
    avg_actual_dip = np.mean(actual_entry_dips) if actual_entry_dips else 0
    median_actual_dip = np.median(actual_entry_dips) if actual_entry_dips else 0

    summary_results = {
        'params': {
            'dip_percentage': dip_percentage, 'dip_window_minutes': dip_window_minutes,
            'take_profit_percentage': take_profit_percentage, 'stop_loss_percentage': stop_loss_percentage,
        },
        'results': {
            'total_entries': total_entries, 'total_profits': total_profits, 'total_losses': total_losses,
            'total_undecided': total_undecided, 'total_closed_at_end': total_closed_at_end,
            'win_rate_perc': win_rate, 'profit_factor': profit_factor, 'estimated_pnl_points': estimated_pnl,
        },
        'stats': {'avg_actual_dip': avg_actual_dip, 'median_actual_dip': median_actual_dip,}
    }

    print("\n--- Wyniki Symulacji (Podsumowanie) ---")
    print(f"Liczba wejść: {total_entries:,}, Zyski (TP): {total_profits:,}, Straty (SL): {total_losses:,}, Nierozstrz.: {total_undecided:,}, Zamknięte na koniec: {total_closed_at_end:,}")
    if resolved_trades > 0: print(f"Win Rate: {win_rate:.2f}%, Profit Factor: {profit_factor:.2f}, Szacowany PnL (% pkt): {estimated_pnl:.2f}")
    print(f"Średni rzeczywisty spadek przy wejściu: {avg_actual_dip:.2f}% (Mediana: {median_actual_dip:.2f}%)")
    print("-" * 30)

    return summary_results, closed_positions_details


# --- ZMODYFIKOWANA funkcja Analizy Detali ---
def analyze_run_details(positions_details, summary_results):
    """
    Analizuje szczegółowe dane pozycji i sugeruje nowe TP/SL ORAZ Dip%.
    """
    print("\n--- Analiza Szczegółowa Przebiegu ---")

    # Sprawdź czy mamy wyniki do analizy
    if not positions_details or summary_results is None:
        print("Brak wystarczających danych do analizy.")
        return None, None, None # Brak sugestii

    current_params = summary_results['params']
    current_sl = current_params['stop_loss_percentage']
    current_tp = current_params['take_profit_percentage']
    current_dip = current_params['dip_percentage']

    # Inicjalizacja sugestii (domyślnie brak zmian)
    suggested_sl = current_sl
    suggested_tp = current_tp
    suggested_dip = current_dip

    # --- Logika sugestii SL/TP (jak poprzednio) ---
    profitable_trades = [p for p in positions_details if p['status'] == 'profit']
    non_losing_trades = [p for p in positions_details if p['status'] != 'loss']

    print("\nAnaliza Stop Loss (max drawdown [%] dla ZYSKOWNYCH):")
    if profitable_trades:
        drawdowns = [p['max_drawdown_perc'] for p in profitable_trades if p['max_drawdown_perc'] < 0]
        if drawdowns:
            median_dd = np.median(drawdowns)
            p75_dd = np.percentile(drawdowns, 75)
            print(f"  Liczba poz.: {len(profitable_trades)}, Mediana DD: {median_dd:.2f}%, 75perc DD: {p75_dd:.2f}%")
            # Sugestia SL: nieco poniżej 75 percentyla
            suggested_sl_raw = p75_dd * 1.05
            temp_sl = math.floor(suggested_sl_raw * 20) / 20.0 # Zaokr. w dół do 0.05
            suggested_sl = min(temp_sl, -0.1) # Musi być przynajmniej -0.1%
            print(f"  -> Sugerowany Stop Loss: {suggested_sl:.2f}% (obecny: {current_sl:.2f}%)")
        else: print("  Brak zyskownych pozycji z ujemnym drawdownem.")
    else: print("  Brak zyskownych pozycji.")

    print("\nAnaliza Take Profit (max potencjalny zysk [%] dla NIE-STRATNYCH):")
    if non_losing_trades:
        max_profits = [p['max_potential_profit_perc'] for p in non_losing_trades if p['max_potential_profit_perc'] > 0]
        if max_profits:
            median_mp = np.median(max_profits)
            p25_mp = np.percentile(max_profits, 25) # Użyjmy 25 percentyla dla ostrożniejszej sugestii
            print(f"  Liczba poz.: {len(non_losing_trades)}, Mediana MP: {median_mp:.2f}%, 25perc MP: {p25_mp:.2f}%")
            # Sugestia TP: blisko 25 percentyla
            suggested_tp_raw = p25_mp * 0.98 # Nieco poniżej 25 perc.
            temp_tp = math.ceil(suggested_tp_raw * 20) / 20.0 # Zaokr. w górę do 0.05
            suggested_tp = max(temp_tp, 0.1) # Musi być przynajmniej 0.1%
            print(f"  -> Sugerowany Take Profit: {suggested_tp:.2f}% (obecny: {current_tp:.2f}%)")
        else: print("  Brak nie-stratnych pozycji z dodatnim potencjałem.")
    else: print("  Brak nie-stratnych pozycji.")

    # --- NOWA Logika sugestii Dip% ---
    print("\nAnaliza Progu Wejścia (Dip%):")
    run_pnl = summary_results['results']['estimated_pnl_points']
    median_actual_dip = summary_results['stats']['median_actual_dip']
    total_entries = summary_results['results']['total_entries']

    dip_suggestion_reason = "Rentowny przebieg." # Domyślny powód braku zmiany

    # Główny warunek: czy przebieg był nierentowny?
    if run_pnl < 0:
        dip_suggestion_reason = f"Nierentowny przebieg (PnL {run_pnl:.2f} pkt)."
        # Proponuj głębszy spadek o stały krok
        suggested_dip_raw = current_dip - 0.25
        # Zaokrąglij do najbliższych 0.05% w dół (dalej od zera)
        suggested_dip = math.floor(suggested_dip_raw * 20) / 20.0

        # Dodatkowy komentarz jeśli mediana spadków była głębsza
        if median_actual_dip < current_dip - 0.1: # Jeśli mediana była znacząco głębsza
             dip_suggestion_reason += f" Mediana akt. spadków ({median_actual_dip:.2f}%) była głębsza niż próg."

        # Dodatkowy komentarz jeśli dużo wejść
        if total_entries > 2000: # Próg do ewentualnej kalibracji
             dip_suggestion_reason += f" Wysoka liczba wejść ({total_entries:,})."

        # Ograniczenie, żeby nie sugerować absurdalnie głębokich spadków
        suggested_dip = max(suggested_dip, -5.0) # Nie sugeruj głębiej niż -5%

    else:
        # Jeśli rentowny, generalnie nie zmieniamy
        suggested_dip = current_dip
        dip_suggestion_reason = f"Rentowny przebieg (PnL {run_pnl:.2f} pkt), nie zmieniam progu wejścia."

    print(f"  Obecny Dip: {current_dip:.2f}%, Mediana akt. spadków: {median_actual_dip:.2f}%")
    print(f"  Powód sugestii: {dip_suggestion_reason}")
    print(f"  -> Sugerowany Dip %: {suggested_dip:.2f}% (obecny: {current_dip:.2f}%)")


    return suggested_dip, suggested_sl, suggested_tp

# --- Funkcja Podsumowania Sesji (bez zmian) ---
# ... (kod summarize_session_runs bez zmian) ...
def summarize_session_runs(session_runs_data):
    print("\n" + "="*50)
    print("Podsumowanie Sesji Optymalizacji")
    print("="*50)
    if not session_runs_data:
        print("Brak danych z przebiegów do podsumowania.")
        return None

    optimization_metric = 'estimated_pnl_points'
    best_run_index = -1
    best_metric_value = -float('inf')

    print("\nWyniki poszczególnych przebiegów:")
    print("-" * 80)
    print(f"{'Nr':>3} | {'Dip%':>6} | {'Win(m)':>6} | {'TP%':>6} | {'SL%':>6} | {'Trades':>6} | {'Win%':>7} | {'PF':>6} | {'PnL(p)':>8}")
    print("-" * 80)

    for i, run_data in enumerate(session_runs_data):
        p = run_data['params']
        r = run_data['results']
        metric_value = r.get(optimization_metric, -float('inf'))
        print(f"{i+1:>3} | {p['dip_percentage']:>6.2f} | {p['dip_window_minutes']:>6} | {p['take_profit_percentage']:>6.2f} | {p['stop_loss_percentage']:>6.2f} | {r['total_entries']:>6,} | {r['win_rate_perc']:>6.2f}% | {r['profit_factor']:>6.2f} | {r['estimated_pnl_points']:>8.2f}")
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_run_index = i

    print("-" * 80)
    if best_run_index != -1:
        best_run_data = session_runs_data[best_run_index]
        p = best_run_data['params']
        r = best_run_data['results']
        print("\nNajlepszy znaleziony przebieg (wg PnL pkt):")
        print(f"  Przebieg numer: {best_run_index + 1}")
        print(f"  Parametry: Dip={p['dip_percentage']:.2f}%, Okno={p['dip_window_minutes']}min, TP={p['take_profit_percentage']:.2f}%, SL={p['stop_loss_percentage']:.2f}%")
        print(f"  Wyniki: Wejścia={r['total_entries']:,}, WinRate={r['win_rate_perc']:.2f}%, PF={r['profit_factor']:.2f}, PnL Pkt={r['estimated_pnl_points']:.2f}")
        return best_run_data
    else:
        print("\nNie udało się określić najlepszego przebiegu.")
        return None


# --- Główny blok wykonawczy z ZMODYFIKOWANĄ pętlą ---
if __name__ == "__main__":
    print("="*50)
    print("Interaktywna Optymalizacja Parametrów Wejścia/TP/SL")
    print("="*50)

    INPUT_FILEPATH = find_latest_csv_in_subdir(CSV_SUBDIRECTORY)
    if not INPUT_FILEPATH: exit()
    data_df = load_data(INPUT_FILEPATH)
    if data_df is None or data_df.empty: exit()

    current_params = INITIAL_PARAMS.copy()
    session_runs = []
    run_counter = 0

    while True:
        run_counter += 1
        print("\n" + "="*20 + f" Przebieg nr {run_counter} " + "="*20)

        summary_results, positions_details = run_backtest(
            data_df,
            current_params['dip_percentage'],
            current_params['dip_window_minutes'],
            current_params['take_profit_percentage'],
            current_params['stop_loss_percentage']
        )

        if summary_results is None:
            print("Backtest nie wygenerował wyników. Zmień parametry wejściowe (Dip%, Okno).")
            if session_runs: # Jeśli były jakieś wcześniejsze przebiegi
                summarize_session_runs(session_runs)
            break

        session_runs.append(summary_results)

        # Analizuj i uzyskaj sugestie dla wszystkich 3 parametrów
        suggested_dip, suggested_sl, suggested_tp = analyze_run_details(
            positions_details,
            summary_results # Przekaż podsumowanie do analizy rentowności
        )

        # Pytanie do użytkownika
        while True:
            print("\nCo chcesz zrobić dalej?")
            # Zmieniono T na A (Apply)
            print("  A: Uruchom kolejny przebieg z ZASUGEROWANYMI parametrami (Dip%, TP%, SL%).")
            print("  M: Uruchom kolejny przebieg z RĘCZNIE wprowadzonymi parametrami (Dip%, Okno, TP%, SL%).")
            print("  N: Zakończ sesję i wyświetl podsumowanie wszystkich przebiegów.")
            choice = input("Wybór (A/M/N): ").upper()

            if choice == 'A':
                if suggested_dip is not None and suggested_sl is not None and suggested_tp is not None:
                    print(f"-> Stosuję sugestie: Dip={suggested_dip:.2f}%, TP={suggested_tp:.2f}%, SL={suggested_sl:.2f}%")
                    current_params['dip_percentage'] = suggested_dip
                    current_params['take_profit_percentage'] = suggested_tp
                    current_params['stop_loss_percentage'] = suggested_sl
                    # Okno pozostaje bez zmian w tym trybie
                    break
                else:
                    print("Nie wygenerowano pełnych sugestii. Wybierz inną opcję.")
            elif choice == 'M':
                try:
                    print("\n--- Wprowadź nowe parametry ---")
                    new_dip = float(input(f"Nowy Dip [%] (ujemny, obecnie {current_params['dip_percentage']:.2f}): "))
                    new_win = int(input(f"Nowe Okno [min] (dodatnie, obecnie {current_params['dip_window_minutes']}): "))
                    new_tp = float(input(f"Nowy Take Profit [%] (dodatni, obecnie {current_params['take_profit_percentage']:.2f}): "))
                    new_sl = float(input(f"Nowy Stop Loss [%] (ujemny, obecnie {current_params['stop_loss_percentage']:.2f}): "))

                    if new_dip < 0 and new_win > 0 and new_tp > 0 and new_sl < 0:
                        current_params['dip_percentage'] = new_dip
                        current_params['dip_window_minutes'] = new_win
                        current_params['take_profit_percentage'] = new_tp
                        current_params['stop_loss_percentage'] = new_sl
                        print(f"-> Ustawiono ręcznie: Dip={new_dip:.2f}%, Okno={new_win}min, TP={new_tp:.2f}%, SL={new_sl:.2f}%")
                        break
                    else: print("Błędne wartości. Sprawdź warunki.")
                except ValueError: print("Nieprawidłowy format liczby.")
            elif choice == 'N':
                print("Kończenie sesji...")
                summarize_session_runs(session_runs)
                exit()
            else: print("Nieprawidłowy wybór.")

    # Ten kod wykona się tylko jeśli pętla zostanie przerwana przez 'break' bez 'exit()'
    print("\nZakończono pętlę optymalizacji.")
    if session_runs: # Pokaż podsumowanie jeśli coś się wykonało
        summarize_session_runs(session_runs)