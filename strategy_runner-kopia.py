#!/opt/homebrew/bin/python3.11
# -*- coding: utf-8 -*-

"""
Zoptymalizowany skrypt strategii tradingowej z obsługą Follow BTC Price.
Rozszerzony o tryb fronttest dla testowania na wielu scenariuszach.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict, fields
import uuid
import pickle
import warnings
from numba import njit, float32, int64, boolean
import numba as nb
import multiprocessing as mp
from tqdm import tqdm

# Import z nowego modułu
from runner_parameters.models import TradingParameters
from runner_parameters.generation import generate_parameter_combinations
from runner_parameters.validation import validate_csv_directory, get_parameter_files

# Importy dla wizualizacji (przeniesione wyżej dla porządku)
try:
    from runner_parameters.trading_visualisation import (
        visualize_strategy, validate_visualization_parameters, create_trading_parameters
    )
    from utils.config import (
        get_newest_visualization_csv, get_newest_visualization_params,
        get_visualization_output_dir, get_fronttest_output_dir # Dodano get_fronttest_output_dir
    )
    VISUALIZATION_ENABLED = True
except ImportError as e:
    VISUALIZATION_ENABLED = False
    visualization_import_error = e

warnings.filterwarnings('ignore')

# Inicjalizacja ścieżek
PARAMETRY_DIR = Path('parametry')
CSV_DIR = Path('csv')
WYNIKI_DIR = Path('wyniki')
LOGI_DIR = Path('logi')
WYNIKI_BACKTEST_DIR = WYNIKI_DIR / 'backtesty'
WYNIKI_FRONTTEST_DIR = WYNIKI_DIR / 'fronttesty'
WYNIKI_ANALIZA_DIR = WYNIKI_DIR / 'analizy'

for dir_path in [PARAMETRY_DIR, CSV_DIR, WYNIKI_DIR, LOGI_DIR,
                 WYNIKI_BACKTEST_DIR, WYNIKI_FRONTTEST_DIR, WYNIKI_ANALIZA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

log_file = LOGI_DIR / f'strategy_runner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_latest_scenario_directory():
    """Znajduje najnowszy podkatalog ze scenariuszami w folderze symulacji."""
    base_dir = Path('csv/symulacje')
    if not base_dir.exists() or not base_dir.is_dir():
        logger.warning(f"Katalog {base_dir} nie istnieje")
        return None

    # Znajdź wszystkie podkatalogi, wykluczając folder logi
    subdirs = [d for d in base_dir.iterdir()
               if d.is_dir() and d.name != 'logi']

    if not subdirs:
        logger.warning(f"Brak podkatalogów ze scenariuszami w {base_dir}")
        return None

    # Posortuj według daty modyfikacji (od najnowszego)
    latest_dir = max(subdirs, key=lambda d: d.stat().st_mtime)

    logger.info(f"Wybrano najnowszy katalog scenariuszy: {latest_dir}")
    return latest_dir

@njit(cache=True, fastmath=True)
def precompute_price_changes(prices: np.ndarray, timeframe: int) -> np.ndarray:
    """Prekompiluje zmiany cen"""
    if timeframe <= 0 or timeframe >= len(prices):
        return np.zeros_like(prices) # Zwraca zera, jeśli timeframe jest nieprawidłowy
    result = np.empty_like(prices)
    result[:timeframe] = 0
    result[timeframe:] = ((prices[timeframe:] / prices[:-timeframe]) - 1.0) * 100.0
    return result

@njit(cache=True, fastmath=True)
def run_strategy_core(
    main_prices: np.ndarray,
    btc_prices: np.ndarray,
    times: np.ndarray,
    params: np.ndarray,
    stop_flag: int
) -> Tuple[List[float], int, int, int, int]: # Zmieniamy typ zwracany List[float] na bardziej ogólny, bo Numba zwraca swój typ listy
#) -> Tuple[nb.typed.List, int, int, int, int]: # Można też tak, ale Tuple[List[float], ...] jest czytelniejsze dla Pythona
    """Główna logika strategii zoptymalizowana pod numba"""

    # Sprawdzenie czy dane wejściowe nie są puste
    if len(main_prices) == 0 or len(btc_prices) == 0 or len(times) == 0:
        # Zwracamy PUSTĄ LISTĘ Z TYPEM float64
        return nb.typed.List.empty_list(nb.float64), 0, 0, 0, 0 # <--- ZMIANA

    # Sprawdzenie, czy długości tablic są zgodne
    if not (len(main_prices) == len(btc_prices) == len(times)):
        # Zwracamy PUSTĄ LISTĘ Z TYPEM float64
        return nb.typed.List.empty_list(nb.float64), 0, 0, 0, 0 # <--- ZMIANA

    # Prekompilacja zmian cen
    timeframe = int(params[0])
    # Dodatkowe zabezpieczenie przed nieprawidłowym timeframe
    if timeframe <= 0 or timeframe >= len(main_prices):
         # Zwracamy PUSTĄ LISTĘ Z TYPEM float64
        return nb.typed.List.empty_list(nb.float64), 0, 0, 0, 0 # <--- ZMIANA

    main_changes = precompute_price_changes(main_prices, timeframe)
    btc_changes = precompute_price_changes(btc_prices, timeframe)

    # Inicjalizacja struktur danych
    max_positions = int(params[11])
    # Używamy float64 dla pozycji, bo ceny i czasy mogą wymagać większej precyzji
    positions = np.zeros((max_positions, 5), dtype=np.float64)
    position_count = 0
    # INICJALIZACJA LISTY Z TYPEM
    trades_profit = nb.typed.List.empty_list(nb.float64) # <--- ZMIANA

    # Blokady czasowe [global_block, btc_block, coin_block, last_buy]
    blocking_times = np.zeros(4, dtype=np.int64)
    stop_loss_price = 0.0

    # Statystyki
    trades_checked = 0
    trades_executed = 0
    positions_closed = 0
    btc_blocks = 0

    # Parametry trailing buy
    trailing_buy_enabled = params[23] > 0
    trailing_buy_threshold = params[24]
    trailing_buy_window = int(params[25]) # Upewnijmy się, że jest int

    # Stan trailing buy
    trailing_buy_active = False
    trailing_buy_start_price = 0.0
    trailing_buy_start_time = 0
    trailing_buy_lowest_price = 0.0

    # Główna pętla
    for i in range(timeframe, len(main_prices)):
        if stop_flag == 1:
            break

        trades_checked += 1
        current_time = times[i]
        current_price = main_prices[i]

        # Analiza pozycji
        j = 0
        while j < position_count:
            # Dodatkowe sprawdzenie, czy cena wejścia nie jest zerowa (bezpiecznik)
            if positions[j, 0] <= 0:
                # Usunięcie nieprawidłowej pozycji (choć nie powinna się zdarzyć)
                if j < position_count - 1:
                    positions[j] = positions[position_count - 1]
                positions[position_count - 1] = np.zeros(5)
                position_count -= 1
                continue # Przejdź do następnej iteracji pętli while

            profit_pct = ((current_price - positions[j, 0]) / positions[j, 0]) * 100.0
            should_close = False

            # Stop Loss
            if params[8] > 0 and profit_pct <= params[9]: # stop_loss_enabled and stop_loss_pct
                if positions[j, 4] == 0: # stop_loss_time (timer start)
                    positions[j, 4] = current_time
                # Sprawdź, czy czas stop loss minął (stop_loss_cooldown)
                elif (current_time - positions[j, 4]) >= params[10]:
                    should_close = True

            # Trailing stop lub zwykły sell (wzajemnie się wykluczają)
            if not should_close:
                if params[4] > 0:  # jeśli trailing_enabled
                    # Warunek aktywacji trailing stop (trailing_start_profit)
                    if profit_pct >= params[5]:
                        if positions[j, 2] == 0: # trailing_start_time (inicjalizacja)
                            positions[j, 2] = current_time
                            positions[j, 3] = current_price # trailing_high_price
                        else:
                            # Sprawdzenie, czy upłynął czas holdingu dla trailing (trailing_hold_time)
                            if (current_time - positions[j, 2]) >= params[7]:
                                # Aktualizacja najwyższej ceny od aktywacji trailing stop
                                if current_price > positions[j, 3]:
                                    positions[j, 3] = current_price
                                # Obliczenie ceny aktywacji trailing stop loss (trailing_stop_loss_pct)
                                trailing_stop_price = positions[j, 3] * (1 - params[6]/100)
                                if current_price <= trailing_stop_price:
                                    should_close = True
                elif params[3] > 0:  # jeśli nie trailing to sprawdź zwykły sell (sell_profit_target)
                    if profit_pct >= params[3]:
                        should_close = True

            if should_close:
                trades_profit.append(profit_pct)
                positions_closed += 1

                # Logika blokad po zamknięciu pozycji (szczególnie po stop loss)
                # block_trade_on_stoploss
                if params[19] > 0 and profit_pct <= params[9]: # Jeśli SL był aktywny i zamknęliśmy na nim
                    stop_loss_price = current_price # Zapisz cenę zamknięcia SL
                     # block_time_after_stoploss (blokada na ten coin)
                    blocking_times[2] = current_time + int(params[22])

                    # block_all_trades_on_stoploss (blokada globalna)
                    if params[20] > 0:
                        blocking_times[0] = current_time + int(params[22])

                # Usunięcie pozycji z tablicy
                if j < position_count - 1:
                    positions[j] = positions[position_count - 1]
                positions[position_count - 1] = np.zeros(5)
                position_count -= 1
                # Nie inkrementujemy 'j', bo na jego miejsce wskoczyła inna pozycja (lub pusta)
            else:
                j += 1 # Przechodzimy do następnej pozycji tylko jeśli nie zamknęliśmy

        # Sprawdzenie warunków nowego zakupu
        if position_count >= max_positions:
            continue

        # Sprawdź blokady czasowe (globalną i dla coina)
        if current_time <= blocking_times[0] or current_time <= blocking_times[2]:
            trailing_buy_active = False # Reset trailing buy podczas blokady
            continue

        # Sprawdź czas od ostatniego zakupu (cooldown_between_trades)
        if blocking_times[3] > 0 and (current_time - blocking_times[3]) < params[12]:
            continue

        # Sprawdź cenę po stop loss (wait_price_recover)
        # Jeśli wait_price_recover_pct > 0 i mieliśmy SL
        if stop_loss_price > 0 and params[21] > 0:
             # Oblicz wymaganą niższą cenę
            required_price = stop_loss_price * (1 - params[21]/100)
            if current_price > required_price:
                continue # Cena nie spadła wystarczająco po SL
        # Jeśli cena spadła wystarczająco lub nie było SL, resetujemy stop_loss_price
        # aby ten warunek nie blokował kolejnych wejść, jeśli nie jest już potrzebny
        elif stop_loss_price > 0 and params[21] <=0 : #Jeśli nie ma wait_price_recover
            stop_loss_price = 0.0
        elif stop_loss_price > 0 and current_price <= stop_loss_price * (1 - params[21]/100):
             stop_loss_price = 0.0 # Warunek spełniony, można kupować, reset

        # Follow BTC Price (btc_follow_enabled)
        if params[16] > 0:
            btc_change = btc_changes[i]
            main_change = main_changes[i]
            # Sprawdź warunek blokady: BTC spada, a jego spadek jest większy niż spadek (lub wzrost) coina
            if btc_change < 0 and abs(btc_change) > params[17] and abs(btc_change) > abs(main_change) + params[18]:
                btc_blocks += 1
                trailing_buy_active = False # Reset trailing buy przy bloku BTC
                continue

        # Sprawdź warunek minimalnej ceny dla kolejnego zakupu (next_buy_price_lower)
        if position_count > 0 and params[13] > 0:
            min_entry_price = np.inf
            for j in range(position_count):
                if positions[j, 0] > 0: # Upewnij się, że pozycja jest aktywna
                    min_entry_price = min(min_entry_price, positions[j, 0])

            # Jeśli mamy aktywne pozycje, oblicz wymaganą niższą cenę
            if min_entry_price != np.inf:
                 required_price = min_entry_price * (1 - params[13]/100)
                 if current_price > required_price:
                     continue # Cena nie jest wystarczająco niska dla kolejnego zakupu

        # Pump Detection (pump_detection_enabled)
        if params[14] > 0 and main_changes[i] >= params[15]: # pump_change_threshold
            trailing_buy_active = False  # Reset trailing buy przy pump detection
            continue

        # Główny warunek wejścia (spadek ceny) - buy_change_threshold
        # Ten warunek jest teraz sprawdzany wewnątrz logiki trailing buy lub standardowego zakupu

        # Logika zakupu z trailing buy
        if trailing_buy_enabled:
            if not trailing_buy_active:
                # Warunek aktywacji trailing buy: spadek ceny poniżej progu
                if main_changes[i] <= params[1]: # buy_change_threshold
                    trailing_buy_active = True
                    trailing_buy_start_price = current_price
                    trailing_buy_start_time = current_time
                    trailing_buy_lowest_price = current_price
            else: # Trailing buy jest aktywny
                # Aktualizuj najniższą cenę od aktywacji
                trailing_buy_lowest_price = min(trailing_buy_lowest_price, current_price)

                # Opcjonalny reset, jeśli cena wzrosła za bardzo od startu (np. o 0.3%)
                # Można to dostosować lub usunąć, jeśli nie jest potrzebne
                if current_price > trailing_buy_start_price * 1.003:
                    trailing_buy_active = False
                    continue # Anuluj ten cykl trailing buy

                # Sprawdzenie okna czasowego trailing buy
                if current_time - trailing_buy_start_time > trailing_buy_window:
                    # Okno czasowe minęło, resetujemy i czekamy na nowy sygnał spadku
                    # Można też rozważyć reset tylko jeśli cena nie odbiła
                    trailing_buy_active = False
                    # trailing_buy_start_price = current_price # Reset startu - alternatywne podejście
                    # trailing_buy_start_time = current_time
                    # trailing_buy_lowest_price = current_price
                    continue

                # Obliczenie odbicia od najniższej ceny w oknie
                price_bounce = 0.0
                if trailing_buy_lowest_price > 0: # Unikaj dzielenia przez zero
                    price_bounce = ((current_price - trailing_buy_lowest_price) / trailing_buy_lowest_price) * 100.0

                # Warunek zakupu: odbicie osiągnęło próg (trailing_buy_threshold)
                if price_bounce >= trailing_buy_threshold:
                    # Dokonaj zakupu
                    positions[position_count] = np.array([current_price, current_time, 0, 0, 0])
                    position_count += 1
                    blocking_times[3] = current_time # Zapisz czas ostatniego zakupu
                    trades_executed += 1
                    trailing_buy_active = False # Zakończ trailing buy po zakupie
        
        # Standardowa logika zakupu (jeśli trailing buy jest wyłączony)
        elif not trailing_buy_enabled and main_changes[i] <= params[1]: # buy_change_threshold
            # Dokonaj zakupu
            positions[position_count] = np.array([current_price, current_time, 0, 0, 0])
            position_count += 1
            blocking_times[3] = current_time # Zapisz czas ostatniego zakupu
            trades_executed += 1

    # Zamknij pozostałe pozycje na końcu symulacji (tutaj już trades_profit ma poprawny typ)
    if position_count > 0 and stop_flag == 0:
        last_price = main_prices[-1]
        for j in range(position_count):
            if positions[j, 0] > 0: # Upewnij się, że pozycja jest aktywna
                # Użyj float64 dla spójności z listą
                profit_pct = np.float64(((last_price - positions[j, 0]) / positions[j, 0]) * 100.0)
                trades_profit.append(profit_pct)
                positions_closed += 1

    return trades_profit, trades_executed, positions_closed, trades_checked, btc_blocks


def run_strategy(args):
    """Wykonuje pojedynczą strategię, opakowuje run_strategy_core."""
    market_data, params, strategy_index, total_combinations, stop_flag = args
    worker_id = mp.current_process().pid
    strategy_id = str(uuid.uuid4())[:8]

    try:
        # Wywołanie rdzenia strategii
        trades_profit, trades_executed, positions_closed, trades_checked, btc_blocks = run_strategy_core(
            market_data['prices'],
            market_data['btc_prices'],
            market_data['times'],
            params.to_array(),
            stop_flag.value
        )

        # Obliczenie średniego profitu
        avg_profit = np.mean(trades_profit) if trades_profit else 0.0

        # Konwersja trades na floaty
        try:
            trades_profit_python_floats = [float(p) for p in trades_profit]
        except (TypeError, ValueError) as e:
            logger.error(f"Strategia {strategy_id}: Błąd konwersji trades_profit na float: {e}. Zwracam oryginalną listę.", exc_info=False)
            trades_profit_python_floats = trades_profit # Fallback

        # Pobranie nazwy pliku parametrów
        param_file_name = getattr(params, '__param_file_name', getattr(params, '_TradingParameters__param_file_name', "unknown"))

        # Używamy asdict do konwersji obiektu parametrów na słownik
        try:
            # Tworzymy słownik tylko z pól zdefiniowanych w TradingParameters
            parameters_dict = {field.name: getattr(params, field.name) for field in fields(params)}
        except Exception as e_asdict:
            logger.error(f"Strategia {strategy_id}: Błąd podczas tworzenia słownika parametrów: {e_asdict}. Zwracam słownik z błędem.", exc_info=False)
            parameters_dict = {"__error_during_param_dict_creation__": str(e_asdict)}


        # Zwracamy wyniki w czystej formie
        return {
            'parameters': parameters_dict,
            'trades': trades_profit_python_floats,
            'strategy_id': strategy_id,
            'worker_id': worker_id,
            'completed': stop_flag.value == 0,
            'total_trades': len(trades_profit_python_floats),
            'trades_executed': trades_executed,
            'trades_closed': positions_closed,
            'trades_checked': trades_checked,
            'btc_blocks': btc_blocks,
            'avg_profit': float(avg_profit), # Upewnijmy się, że to float
            'symbol': market_data['symbol'],
            'param_file_name': param_file_name
        }

    except Exception as e:
        logger.error(f"Błąd podczas wykonywania strategii (worker {worker_id}, strat_id {strategy_id}): {str(e)}", exc_info=True)

        error_params_dict = {"__error_in_strategy_execution__": str(e)}
        error_param_file = "unknown_error"
        try:
            if params is not None:
                 # Próba utworzenia słownika również tutaj
                 error_params_dict = {field.name: getattr(params, field.name) for field in fields(params)}
                 error_param_file = getattr(params, '__param_file_name', getattr(params, '_TradingParameters__param_file_name', "unknown"))
        except Exception as e_dict_err:
             logger.error(f"Błąd podczas tworzenia słownika parametrów w bloku except dla strategii {strategy_id}: {e_dict_err}", exc_info=False)
             error_params_dict["__error_during_param_dict_creation_in_except__"] = str(e_dict_err)

        return {
            'parameters': error_params_dict,
            'error': str(e),
            'strategy_id': strategy_id,
            'worker_id': worker_id,
            'completed': False,
            'total_trades': 0,
            'trades_executed': 0,
            'trades_closed': 0,
            'trades_checked': 0,
            'btc_blocks': 0,
            'avg_profit': 0.0,
            'symbol': market_data.get('symbol', 'ERROR_SYMBOL'),
            'param_file_name': error_param_file
        }

# --- NOWA FUNKCJA ---
def load_market_data(csv_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Wczytuje dane rynkowe z pliku CSV, obsługując różne formaty kolumn.

    Args:
        csv_path: Ścieżka do pliku CSV.

    Returns:
        Słownik zawierający 'prices', 'btc_prices', 'times', 'symbol' jako numpy arrays,
        lub None jeśli wystąpił błąd wczytywania lub brak kluczowych danych.
    """
    try:
        logger.debug(f"Wczytywanie danych z: {csv_path}")
        df = pd.read_csv(csv_path)

        # --- Podstawowe sprawdzenie kolumn ---
        if 'timestamp' not in df.columns:
            logger.error(f"Brak kolumny 'timestamp' w pliku {csv_path.name}")
            return None
        if 'average_price' not in df.columns:
            logger.error(f"Brak kolumny 'average_price' w pliku {csv_path.name}")
            return None

        # --- Konwersja czasu ---
        try:
            # Próba konwersji ze świadomością strefy czasowej, jeśli istnieje
            df['minutes'] = pd.to_datetime(df['timestamp'], utc=True).astype(np.int64) // 60e9
        except TypeError:
             # Fallback dla timestampów bez informacji o strefie czasowej
             df['minutes'] = pd.to_datetime(df['timestamp']).astype(np.int64) // 60e9
        except Exception as e_time:
            logger.error(f"Błąd konwersji kolumny 'timestamp' na minuty w pliku {csv_path.name}: {e_time}")
            return None
        times_np = df['minutes'].to_numpy(dtype=np.int64)

        # --- Określenie głównego symbolu ---
        main_symbol = "UNKNOWN"
        if 'main_symbol' in df.columns and df['main_symbol'].notna().any():
            # Bierzemy pierwszy nie-nullowy symbol
            main_symbol = df['main_symbol'].dropna().iloc[0]
        else:
            # Awaryjne wykrywanie z nazwy pliku (prosta heurystyka)
            filename_parts = csv_path.stem.split('_')
            if len(filename_parts) > 2 and "USDT" in filename_parts[1].upper():
                 main_symbol = f"{filename_parts[0].upper()}/{filename_parts[1].upper()}" # Np. BTC/USDT
                 logger.warning(f"Brak kolumny 'main_symbol' w pliku {csv_path.name}. Wykryto {main_symbol} z nazwy pliku.")
            elif "BTC" in csv_path.stem.upper() and "USDT" in csv_path.stem.upper():
                 main_symbol = "BTC/USDT" # Ostateczny fallback dla BTC
                 # Zmieniono poziom logowania
                 logger.info(f"Brak kolumny 'main_symbol'. Wydedukowano {main_symbol} z nazwy pliku {csv_path.name}.") # Zmieniono też treść na bardziej pozytywną

        # --- Przygotowanie cen ---
        prices_np = df['average_price'].to_numpy(dtype=np.float32)
        btc_prices_np = np.array([], dtype=np.float32) # Domyślnie pusta

        # Normalizacja symbolu do porównania (np. BTC/USDT -> BTCUSDT)
        normalized_symbol = main_symbol.replace('/', '').upper()

        if normalized_symbol == 'BTCUSDT':
            # Dla BTC/USDT, ceny BTC są takie same jak ceny główne
            btc_prices_np = prices_np
            logger.debug(f"Plik {csv_path.name}: Wykryto parę BTC/USDT. Używam 'average_price' jako ceny BTC.")
        else:
            # Dla innych par, szukamy dedykowanej kolumny btc_average_price
            if 'btc_average_price' in df.columns:
                btc_prices_np = df['btc_average_price'].to_numpy(dtype=np.float32)
                logger.debug(f"Plik {csv_path.name}: Znaleziono 'btc_average_price' dla symbolu {main_symbol}.")
            else:
                # To jest problem dla par innych niż BTC/USDT
                logger.error(f"Brak kolumny 'btc_average_price' w pliku {csv_path.name} dla symbolu {main_symbol}. Nie można kontynuować z tym plikiem.")
                return None # Zwracamy None, bo nie mamy kluczowych danych

        # --- Ostateczne sprawdzenie ---
        if prices_np.size == 0 or btc_prices_np.size == 0 or times_np.size == 0:
            logger.error(f"Nie udało się poprawnie załadować wszystkich danych (puste tablice numpy) z pliku {csv_path.name}.")
            return None
        if not (prices_np.size == btc_prices_np.size == times_np.size):
             logger.error(f"Niezgodność rozmiarów danych w pliku {csv_path.name}: prices={prices_np.size}, btc_prices={btc_prices_np.size}, times={times_np.size}")
             return None

        logger.info(f"Pomyślnie wczytano dane dla {main_symbol} z {csv_path.name} ({len(prices_np)} rekordów).")
        return {
            'prices': prices_np,
            'btc_prices': btc_prices_np,
            'times': times_np,
            'symbol': main_symbol,
             # Dodajemy info o DataFrame na wszelki wypadek, gdyby było potrzebne gdzieś indziej
             # 'dataframe': df
             # Dodajmy też informację o zakresie dat
             'period': (df['timestamp'].iloc[0], df['timestamp'].iloc[-1]) if not df.empty else ('N/A', 'N/A'),
             'candles': len(df)
        }

    except FileNotFoundError:
        logger.error(f"Nie znaleziono pliku: {csv_path}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"Plik CSV jest pusty: {csv_path.name}")
        return None
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas wczytywania pliku {csv_path.name}: {str(e)}", exc_info=True)
        return None

# --- KONIEC NOWEJ FUNKCJI ---


def process_fronttest_scenarios(scenario_files, parameter_files, args):
    """
    Przetwarza wiele plików CSV ze scenariuszami w trybie fronttest.
    Używa funkcji load_market_data.
    """
    # Użycie funkcji z utils.config, jeśli dostępna
    if 'get_fronttest_output_dir' in globals():
         output_dir = get_fronttest_output_dir()
    else:
         # Fallback, jeśli utils.config nie jest dostępne
         output_dir = WYNIKI_FRONTTEST_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
         output_dir.mkdir(parents=True, exist_ok=True)
         logger.warning("Nie znaleziono funkcji get_fronttest_output_dir z utils.config. Używam domyślnej ścieżki.")

    logger.info(f"Wyniki fronttestu zostaną zapisane w katalogu: {output_dir}")

    num_processes = args.processes or mp.cpu_count()
    manager = mp.Manager()
    stop_flag = manager.Value('i', 0)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = args.output_prefix or f'fronttest_{session_id}'
    results_by_scenario = {}
    processed_scenarios_count = 0 # Licznik pomyślnie przetworzonych scenariuszy

    for scenario_file in scenario_files:
        scenario_name = scenario_file.stem
        logger.info(f"\n--- Przetwarzanie scenariusza: {scenario_name} ({scenario_file.name}) ---")

        # --- ZMODYFIKOWANE WCZYTYWANIE DANYCH ---
        market_data = load_market_data(scenario_file)

        if market_data is None:
            logger.error(f"Pominięto scenariusz {scenario_name} z powodu błędu wczytywania danych.")
            continue # Przejdź do następnego pliku scenariusza
        # --- KONIEC ZMIAN WE WCZYTYWANIU ---

        # Przetwarzanie każdego pliku parametrów dla tego scenariusza
        scenario_results = []
        scenario_processed_successfully = False # Flaga czy ten scenariusz dał jakiekolwiek wyniki

        for param_file in parameter_files:
            try:
                logger.info(f"Przetwarzanie parametrów: {param_file.name} dla scenariusza {scenario_name}")

                # Wczytanie parametrów (bez zmian)
                with open(param_file, 'r') as f:
                    params_config = json.load(f)

                # Generowanie kombinacji parametrów z określonym trybem fronttest
                parameter_combinations = generate_parameter_combinations(
                    params_config,
                    market_data=market_data, # Przekazujemy wczytane dane
                    max_combinations=args.limit,
                    mode="fronttest",
                    param_file_name=param_file.name
                )

                if not parameter_combinations:
                    logger.warning(f"Brak poprawnych kombinacji parametrów w pliku {param_file.name} dla scenariusza {scenario_name}")
                    continue

                logger.info(f"Rozpoczynam testy: {len(parameter_combinations)} kombinacji, {num_processes} procesów dla {scenario_name}")

                # Uruchomienie testów (bez zmian w logice pool)
                results = []
                with mp.Pool(num_processes) as pool:
                    combinations_list = [(market_data, p, i, len(parameter_combinations), stop_flag)
                                                    for i, p in enumerate(parameter_combinations)]

                    with tqdm(total=len(parameter_combinations), desc=f"Postęp {scenario_name[:30]}...", leave=False) as pbar:
                        for result in pool.imap_unordered(run_strategy, combinations_list):
                            if result is not None and 'error' not in result: # Dodajemy tylko udane wyniki
                                result['scenario'] = scenario_name
                                result['param_file'] = param_file.name
                                results.append(result)
                            elif result is not None and 'error' in result:
                                logger.warning(f"Strategia {result.get('strategy_id','N/A')} dla scenariusza {scenario_name} zwróciła błąd: {result['error']}")
                            pbar.update(1)

                # Dodanie wyników do listy tego scenariusza
                if results:
                     scenario_results.extend(results)
                     scenario_processed_successfully = True # Mamy wyniki dla tego scenariusza

            except Exception as e:
                logger.error(f"Krytyczny błąd podczas przetwarzania pliku parametrów {param_file.name} dla scenariusza {scenario_name}: {str(e)}", exc_info=True)
                continue # Przejdź do następnego pliku parametrów

        # Zapisujemy wyniki dla tego scenariusza tylko jeśli jakieś uzyskano
        if scenario_processed_successfully:
            results_by_scenario[scenario_name] = scenario_results
            processed_scenarios_count += 1 # Zwiększ licznik udanych scenariuszy

            # Zapisujemy również wyniki dla każdego scenariusza osobno
            scenario_output_file = output_dir / f'{output_prefix}_scenario_{scenario_name}.pkl'

            # Przygotowujemy dane do zapisu
            scenario_data = {
                'results': scenario_results,
                'timestamp': datetime.now().isoformat(),
                'scenario_file': str(scenario_file),
                'scenario_name': scenario_name,
                'parameters_files': [str(p) for p in parameter_files],
                'market_data_info': {
                    'symbol': market_data['symbol'],
                    'period': market_data['period'],
                    'candles': market_data['candles']
                }
            }

            # Zapisujemy plik PKL
            try:
                with open(scenario_output_file, 'wb') as f:
                    pickle.dump(scenario_data, f)
                logger.info(f"Zapisano wyniki dla scenariusza {scenario_name} do {scenario_output_file}")
            except Exception as e_pickle:
                logger.error(f"Błąd podczas zapisywania pliku pickle {scenario_output_file}: {e_pickle}")

        else:
            logger.warning(f"Scenariusz {scenario_name} nie wygenerował żadnych pomyślnych wyników.")


    # Zapis zbiorczych wyników (tylko jeśli cokolwiek przetworzono)
    if results_by_scenario:
        output_file = output_dir / f'{output_prefix}_all_scenarios.pkl'

        # Przygotowujemy dane do zapisu
        all_scenarios_data = {
            'results_by_scenario': results_by_scenario,
            'timestamp': datetime.now().isoformat(),
            'scenarios_processed_paths': [str(f) for f in scenario_files if f.stem in results_by_scenario], # Zapisz tylko te przetworzone
            'parameters_files': [str(f) for f in parameter_files],
            'session_id': session_id
        }

        # Zapisujemy plik PKL
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(all_scenarios_data, f)
            logger.info(f"Zapisano zbiorcze wyniki fronttestu ({len(results_by_scenario)} scenariuszy) do {output_file}")
            return results_by_scenario, str(output_file), processed_scenarios_count # Zwracamy liczbę przetworzonych
        except Exception as e_pickle:
            logger.error(f"Błąd podczas zapisywania zbiorczego pliku pickle {output_file}: {e_pickle}")
            return results_by_scenario, None, processed_scenarios_count # Zwracamy None jako ścieżkę pliku przy błędzie
    else:
        logger.warning("Nie przetworzono pomyślnie żadnych scenariuszy. Nie zapisano zbiorczego pliku wyników.")
        return {}, None, 0 # Zwracamy puste wyniki i brak pliku


def main():
    """Główna funkcja programu"""
    try:
        manager = mp.Manager()
        stop_flag = manager.Value('i', 0)

        import argparse
        parser = argparse.ArgumentParser(description='Strategy Runner')
        parser.add_argument('--limit', type=int, default=None, help='Limit kombinacji parametrów')
        parser.add_argument('--processes', type=int, default=None, help='Liczba procesów (domyślnie: CPU count)')
        # Usunięto --continue, bo nie było używane
        parser.add_argument('--non-interactive', action='store_true', help='Tryb nieinteraktywny (używa argumentów wiersza poleceń)')
        parser.add_argument('--mode', choices=['backtest', 'fronttest', 'visualization'], default=None, help='Tryb działania (dla trybu nieinteraktywnego)')
        parser.add_argument('--scenarios-dir', type=str, default=None, help='Katalog ze scenariuszami (dla trybu fronttest)')
        parser.add_argument('--output-prefix', type=str, default=None, help='Prefiks dla plików wynikowych')
        parser.add_argument('--csv-file', type=str, default=None, help='Plik CSV z danymi (dla trybu backtest/visualization)')
        parser.add_argument('--param-file', type=str, default=None, help='Pojedynczy plik JSON z parametrami (głównie dla visualization, opcjonalnie dla backtest)')
        parser.add_argument('--param-dir', type=str, default=None, help='Katalog z plikami parametrów JSON (dla backtest/fronttest)')
        parser.add_argument('--output-dir', type=str, default=None, help='Katalog wyjściowy dla wyników')

        args = parser.parse_args()

        # Tryb działania
        mode = args.mode
        if not args.non_interactive and not mode:
            print("\n=== BTC Follow Strategy Runner ===")
            print("Wybierz tryb działania:")
            print("1. Backtest (domyślny) - testowanie wielu kombinacji parametrów na jednym pliku CSV")
            print("2. Fronttest - testowanie na wielu scenariuszach CSV")
            print("3. Wizualizacja - wykres i tabela dla pojedynczego zestawu parametrów i pliku CSV")
            choice = input("\nTwój wybór (1-3) [1]: ").strip() or "1"
            if choice == "1": mode = "backtest"
            elif choice == "2": mode = "fronttest"
            elif choice == "3": mode = "visualization"
            else: mode = "backtest"
            print(f"Wybrano tryb: {mode.capitalize()}")
        elif not mode:
            mode = "backtest" # Domyślny tryb, jeśli nie podano

        # Wybór trybu działania
        if mode == 'backtest':
            logger.info("Uruchamiam tryb Backtest")

            # --- Wybór pliku CSV ---
            csv_file_path_str = args.csv_file
            if not args.non_interactive and not csv_file_path_str:
                 use_default_csv = input(f"\nCzy użyć domyślnego pliku CSV z katalogu '{CSV_DIR}'? (t/n) [t]: ").strip().lower() or 't'
                 if use_default_csv == 'n':
                     custom_csv = input("Podaj ścieżkę do pliku CSV: ").strip()
                     csv_file_path_str = custom_csv if custom_csv and Path(custom_csv).exists() else None
            if not csv_file_path_str:
                 is_valid, csv_file_path = validate_csv_directory(CSV_DIR) # Używamy domyślnego katalogu CSV
                 if not is_valid:
                     logger.error("Nie znaleziono lub nie można użyć domyślnego pliku CSV.")
                     return
                 csv_file_path_str = str(csv_file_path)
            else:
                 csv_file_path = Path(csv_file_path_str)
                 if not csv_file_path.exists() or not csv_file_path.is_file():
                      logger.error(f"Podany plik CSV '{csv_file_path_str}' nie istnieje lub nie jest plikiem.")
                      return

            # --- Wczytanie danych rynkowych ---
            market_data = load_market_data(Path(csv_file_path_str))
            if market_data is None:
                 logger.error(f"Nie udało się wczytać danych rynkowych z {csv_file_path_str}. Przerwanie trybu backtest.")
                 return

            # --- Wybór plików parametrów ---
            parameter_files = []
            param_source_info = ""
            if args.param_file: # Jeśli podano konkretny plik parametrów
                 param_file_path = Path(args.param_file)
                 if param_file_path.exists() and param_file_path.is_file():
                      parameter_files = [param_file_path]
                      param_source_info = f"z pliku: {param_file_path.name}"
                 else:
                      logger.error(f"Podany plik parametrów '{args.param_file}' nie istnieje.")
                      return
            elif args.param_dir: # Jeśli podano katalog parametrów
                 param_dir_path = Path(args.param_dir)
                 if param_dir_path.is_dir():
                      parameter_files = get_parameter_files(param_dir_path)
                      param_source_info = f"z katalogu: {param_dir_path}"
                 else:
                      logger.error(f"Podany katalog parametrów '{args.param_dir}' nie istnieje.")
                      return
            else: # Domyślne lub interaktywne
                 if not args.non_interactive:
                      use_default_params = input(f"\nCzy użyć plików parametrów z domyślnego katalogu '{PARAMETRY_DIR}'? (t/n) [t]: ").strip().lower() or 't'
                      if use_default_params == 'n':
                           custom_params_dir = input("Podaj ścieżkę do katalogu z parametrami: ").strip()
                           if custom_params_dir and Path(custom_params_dir).is_dir():
                                parameter_files = get_parameter_files(Path(custom_params_dir))
                                param_source_info = f"z katalogu: {custom_params_dir}"
                           else:
                                logger.warning("Podana ścieżka do parametrów jest nieprawidłowa. Używam domyślnego katalogu.")
                                parameter_files = get_parameter_files(PARAMETRY_DIR)
                                param_source_info = f"z domyślnego katalogu: {PARAMETRY_DIR}"
                      else:
                           parameter_files = get_parameter_files(PARAMETRY_DIR)
                           param_source_info = f"z domyślnego katalogu: {PARAMETRY_DIR}"
                 else:
                      parameter_files = get_parameter_files(PARAMETRY_DIR)
                      param_source_info = f"z domyślnego katalogu: {PARAMETRY_DIR}"

            if not parameter_files:
                logger.error(f"Nie znaleziono żadnych plików parametrów {param_source_info}.")
                return
            logger.info(f"Znaleziono {len(parameter_files)} plików parametrów {param_source_info}.")

            num_processes = args.processes or mp.cpu_count()
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(args.output_dir) if args.output_dir else WYNIKI_BACKTEST_DIR
            output_dir.mkdir(parents=True, exist_ok=True)

            # Przetwarzanie każdego pliku parametrów
            for param_file in parameter_files:
                try:
                    logger.info(f"\n--- Przetwarzanie pliku parametrów: {param_file.name} ---")
                    with open(param_file, 'r') as f:
                        params_config = json.load(f)

                    # Generowanie kombinacji parametrów
                    parameter_combinations = generate_parameter_combinations(
                        params_config,
                        market_data=market_data,
                        max_combinations=args.limit,
                        mode="backtest",
                        param_file_name=param_file.name # Przekaż nazwę pliku
                    )

                    if not parameter_combinations:
                        logger.warning(f"Brak poprawnych kombinacji parametrów w pliku {param_file.name}")
                        continue

                    logger.info(f"Rozpoczynam testy: {len(parameter_combinations)} kombinacji, {num_processes} procesów")

                    # Uruchomienie testów
                    results = []
                    with mp.Pool(num_processes) as pool:
                        combinations_list = [(market_data, p, i, len(parameter_combinations), stop_flag)
                                          for i, p in enumerate(parameter_combinations)]

                        with tqdm(total=len(parameter_combinations), desc=f"Postęp {param_file.stem[:30]}...", leave=False) as pbar:
                            for result in pool.imap_unordered(run_strategy, combinations_list):
                                if result is not None and 'error' not in result:
                                    results.append(result)
                                elif result is not None and 'error' in result:
                                    logger.warning(f"Strategia {result.get('strategy_id','N/A')} dla {param_file.name} zwróciła błąd: {result['error']}")
                                pbar.update(1)

                    if not results:
                        logger.warning(f"Brak pomyślnych wyników do zapisania dla pliku {param_file.name}")
                        continue

                    # Zapis wyników
                    param_file_stem = param_file.stem
                    output_prefix = args.output_prefix or f"backtest_{market_data['symbol'].replace('/','-')}"
                    output_file = output_dir / f'{output_prefix}_{param_file_stem}_{session_id}.pkl'

                    with open(output_file, 'wb') as f:
                        pickle.dump({
                            'results': results,
                            'timestamp': datetime.now().isoformat(),
                            'parameters_file': str(param_file),
                            'parameters_config': params_config, # Oryginalna konfiguracja z pliku
                            'market_data_info': {
                                'file': str(csv_file_path_str),
                                'symbol': market_data['symbol'],
                                'period': market_data['period'],
                                'candles': market_data['candles']
                            },
                            'mode': 'backtest'
                        }, f)
                    logger.info(f"Zapisano wyniki backtestu do {output_file}")

                except json.JSONDecodeError as e_json:
                    logger.error(f"Błąd odczytu pliku JSON {param_file.name}: {e_json}")
                    continue
                except Exception as e:
                    logger.error(f"Nieoczekiwany błąd podczas przetwarzania pliku {param_file.name}: {str(e)}", exc_info=True)
                    continue

        elif mode == 'fronttest':
            logger.info("Uruchamiam tryb Fronttest")

            # --- Wybór katalogu scenariuszy ---
            scenarios_dir_path_str = args.scenarios_dir
            if not args.non_interactive and not scenarios_dir_path_str:
                use_default_scenarios = input("\nCzy użyć domyślnego katalogu scenariuszy (najnowszego w csv/symulacje)? (t/n) [t]: ").strip().lower() or 't'
                if use_default_scenarios == 'n':
                    custom_scenarios_dir = input("Podaj ścieżkę do katalogu ze scenariuszami: ").strip()
                    scenarios_dir_path_str = custom_scenarios_dir if custom_scenarios_dir and Path(custom_scenarios_dir).is_dir() else None
            if not scenarios_dir_path_str:
                scenarios_dir_path = get_latest_scenario_directory()
                if not scenarios_dir_path:
                    logger.error("Nie znaleziono katalogu scenariuszy. Użyj --scenarios-dir lub wygeneruj scenariusze.")
                    return
                scenarios_dir_path_str = str(scenarios_dir_path)
            else:
                 scenarios_dir_path = Path(scenarios_dir_path_str)
                 if not scenarios_dir_path.is_dir():
                      logger.error(f"Podany katalog scenariuszy '{scenarios_dir_path_str}' nie istnieje lub nie jest katalogiem.")
                      return

            # Pobranie listy plików CSV ze scenariuszami
            scenario_files = sorted(list(Path(scenarios_dir_path_str).glob('*.csv'))) # Sortujemy dla spójności
            if not scenario_files:
                logger.error(f"Brak plików CSV w katalogu scenariuszy: {scenarios_dir_path_str}")
                return
            logger.info(f"Znaleziono {len(scenario_files)} plików scenariuszy w {scenarios_dir_path_str}")

            # --- Wybór plików parametrów (logika jak w backtest) ---
            parameter_files = []
            param_source_info = ""
            # ... (skopiowana logika wyboru parametrów z trybu backtest) ...
            if args.param_file:
                 param_file_path = Path(args.param_file)
                 if param_file_path.exists() and param_file_path.is_file(): parameter_files = [param_file_path]; param_source_info = f"z pliku: {param_file_path.name}"
                 else: logger.error(f"Podany plik parametrów '{args.param_file}' nie istnieje."); return
            elif args.param_dir:
                 param_dir_path = Path(args.param_dir)
                 if param_dir_path.is_dir(): parameter_files = get_parameter_files(param_dir_path); param_source_info = f"z katalogu: {param_dir_path}"
                 else: logger.error(f"Podany katalog parametrów '{args.param_dir}' nie istnieje."); return
            else:
                 if not args.non_interactive:
                      use_default_params = input(f"\nCzy użyć plików parametrów z domyślnego katalogu '{PARAMETRY_DIR}'? (t/n) [t]: ").strip().lower() or 't'
                      if use_default_params == 'n':
                           custom_params_dir = input("Podaj ścieżkę do katalogu z parametrami: ").strip()
                           if custom_params_dir and Path(custom_params_dir).is_dir(): parameter_files = get_parameter_files(Path(custom_params_dir)); param_source_info = f"z katalogu: {custom_params_dir}"
                           else: logger.warning("Ścieżka nieprawidłowa. Używam domyślnych."); parameter_files = get_parameter_files(PARAMETRY_DIR); param_source_info = f"z domyślnego katalogu: {PARAMETRY_DIR}"
                      else: parameter_files = get_parameter_files(PARAMETRY_DIR); param_source_info = f"z domyślnego katalogu: {PARAMETRY_DIR}"
                 else: parameter_files = get_parameter_files(PARAMETRY_DIR); param_source_info = f"z domyślnego katalogu: {PARAMETRY_DIR}"

            if not parameter_files:
                logger.error(f"Nie znaleziono żadnych plików parametrów {param_source_info}.")
                return
            logger.info(f"Znaleziono {len(parameter_files)} plików parametrów {param_source_info}.")


            # Uruchomienie przetwarzania scenariuszy
            results_by_scenario, output_file, processed_count = process_fronttest_scenarios(scenario_files, parameter_files, args)

            logger.info(f"Zakończono przetwarzanie. Pomyślnie przetworzono {processed_count} z {len(scenario_files)} scenariuszy.")
            if output_file:
                logger.info(f"Zbiorcze wyniki zapisano w pliku: {output_file}")
            else:
                 logger.warning("Nie zapisano zbiorczego pliku wyników (prawdopodobnie brak udanych scenariuszy lub błąd zapisu).")

        elif mode == 'visualization':
             logger.info("Uruchamiam tryb Wizualizacji")
             if not VISUALIZATION_ENABLED:
                 logger.error(f"Moduły wizualizacji nie mogły zostać zaimportowane. Błąd: {visualization_import_error}")
                 logger.error("Upewnij się, że zainstalowano zależności (plotly, etc.) i moduły są dostępne.")
                 return

             # --- Wybór pliku CSV (logika jak w backtest, ale może użyć get_newest_visualization_csv) ---
             csv_file_path_str = args.csv_file
             if not args.non_interactive and not csv_file_path_str:
                  use_default_csv = input("\nCzy użyć domyślnego pliku CSV z katalogu wizualizacje? (t/n) [t]: ").strip().lower() or 't'
                  if use_default_csv == 'n':
                       custom_csv = input("Podaj ścieżkę do pliku CSV: ").strip()
                       csv_file_path_str = custom_csv if custom_csv and Path(custom_csv).exists() else None
                  else:
                       csv_file_path_str = get_newest_visualization_csv() # Użyj funkcji z utils
             elif not csv_file_path_str: # Tryb nieinteraktywny, brak argumentu
                  csv_file_path_str = get_newest_visualization_csv()

             if not csv_file_path_str or not Path(csv_file_path_str).exists():
                  logger.error(f"Nie znaleziono pliku CSV z danymi dla wizualizacji: '{csv_file_path_str or 'Brak ścieżki'}'. Użyj --csv-file lub umieść plik w odpowiednim katalogu.")
                  return

             # --- Wybór pliku parametrów (musi być *jeden* plik z *konkretnymi* wartościami) ---
             param_file_path_str = args.param_file
             if not args.non_interactive and not param_file_path_str:
                  use_default_params = input("\nCzy użyć domyślnego pliku parametrów z katalogu wizualizacje? (t/n) [t]: ").strip().lower() or 't'
                  if use_default_params == 'n':
                       custom_params = input("Podaj ścieżkę do pliku parametrów JSON: ").strip()
                       param_file_path_str = custom_params if custom_params and Path(custom_params).exists() else None
                  else:
                       param_file_path_str = get_newest_visualization_params() # Użyj funkcji z utils
             elif not param_file_path_str: # Tryb nieinteraktywny, brak argumentu
                  param_file_path_str = get_newest_visualization_params()

             if not param_file_path_str or not Path(param_file_path_str).exists():
                  logger.error(f"Nie znaleziono pliku z parametrami dla wizualizacji: '{param_file_path_str or 'Brak ścieżki'}'. Użyj --param-file lub umieść plik w odpowiednim katalogu.")
                  return

             # --- Wczytanie danych (używamy tej samej funkcji co reszta) ---
             # Najpierw wczytujemy DataFrame, bo visualize_strategy go oczekuje
             try:
                  logger.info(f"Wczytywanie danych z {csv_file_path_str}")
                  df = pd.read_csv(csv_file_path_str)
                  # Upewnij się, że mamy potrzebne kolumny (można by użyć load_market_data i wziąć df)
                  required_vis_columns = ['timestamp', 'average_price'] # Minimum dla wizualizacji ceny
                  # Sprawdź czy jest btc_average_price LUB czy plik dotyczy BTC
                  is_btc_file = False
                  if 'main_symbol' in df.columns and df['main_symbol'].notna().any():
                      is_btc_file = "BTC" in df['main_symbol'].dropna().iloc[0].upper()
                  elif "BTC" in Path(csv_file_path_str).stem.upper():
                      is_btc_file = True

                  if not is_btc_file and 'btc_average_price' not in df.columns:
                      logger.warning(f"Plik {Path(csv_file_path_str).name} nie zawiera 'btc_average_price' i nie jest plikiem BTC. Wizualizacja cen BTC może być niedostępna.")
                  else:
                      required_vis_columns.append('btc_average_price') # Dodaj, jeśli jest lub jeśli to plik BTC

                  missing_columns = [col for col in required_vis_columns if col not in df.columns and not (col == 'btc_average_price' and is_btc_file)]
                  if missing_columns:
                      logger.error(f"Plik {csv_file_path_str} nie zawiera wymaganych kolumn do wizualizacji: {', '.join(missing_columns)}")
                      return

                  # Dodaj kolumnę main_symbol jeśli nie istnieje (heurystyka)
                  if 'main_symbol' not in df.columns:
                      symbol = "UNKNOWN"
                      parts = Path(csv_file_path_str).stem.split('_')
                      if len(parts) > 2 and "USDT" in parts[1].upper():
                          symbol = f"{parts[0].upper()}/{parts[1].upper()}"
                      elif is_btc_file:
                           symbol = "BTC/USDT"
                      df['main_symbol'] = symbol
                      logger.info(f"Dodano brakującą kolumnę 'main_symbol': {symbol}")

             except Exception as e:
                  logger.error(f"Błąd podczas wczytywania pliku CSV {csv_file_path_str} dla wizualizacji: {e}")
                  return

             # Wczytanie i walidacja parametrów
             logger.info(f"Wczytywanie parametrów z {param_file_path_str}")
             try:
                 with open(param_file_path_str, 'r') as f:
                     params_dict = json.load(f)
                 if not validate_visualization_parameters(params_dict):
                     logger.error("Walidacja parametrów dla wizualizacji nie powiodła się. Plik musi zawierać pojedyncze wartości, a nie zakresy.")
                     return
                 params_obj = create_trading_parameters(params_dict)
             except json.JSONDecodeError as e_json:
                 logger.error(f"Błąd odczytu pliku JSON {param_file_path_str}: {e_json}")
                 return
             except Exception as e_param:
                 logger.error(f"Błąd przetwarzania parametrów z pliku {param_file_path_str}: {e_param}")
                 return

             # Ustalenie katalogu wyjściowego
             output_dir_vis = Path(args.output_dir) if args.output_dir else get_visualization_output_dir()
             output_dir_vis.mkdir(parents=True, exist_ok=True)

             # Uruchomienie wizualizacji
             logger.info("Uruchamiam wizualizację strategii...")
             results = visualize_strategy(df, params_obj, params_dict, output_dir_vis)

             if results and results.get("success"):
                 logger.info("Wizualizacja zakończona pomyślnie.")
                 if results.get('chart_path'): logger.info(f"  Wykres: {results['chart_path']}")
                 if results.get('table_path'): logger.info(f"  Tabela transakcji: {results['table_path']}")
                 summary = results.get("summary", {})
                 if summary:
                     logger.info("  Podsumowanie:")
                     logger.info(f"    Liczba transakcji: {summary.get('total_trades', 'N/A')}")
                     logger.info(f"    Średni zysk: {summary.get('avg_profit', 'N/A'):.2f}%" if isinstance(summary.get('avg_profit'), (int, float)) else 'N/A')
                     logger.info(f"    Win rate: {summary.get('win_rate', 'N/A'):.2f}%" if isinstance(summary.get('win_rate'), (int, float)) else 'N/A')
                 logger.info(f"  Wszystkie wyniki zapisano w katalogu: {results.get('output_dir', 'N/A')}")
             else:
                 logger.error(f"Wizualizacja nie powiodła się. Szczegóły: {results.get('error', 'Brak informacji o błędzie')}")

    except KeyboardInterrupt:
        logger.info("\nOtrzymano sygnał przerwania - inicjuję bezpieczne zamknięcie...")
        stop_flag.value = 1
        # Daj chwilę procesom potomnym na reakcję
        import time
        time.sleep(1)

    except Exception as e:
        logger.critical(f"Nieoczekiwany błąd krytyczny na głównym poziomie programu: {str(e)}", exc_info=True)
        # Można dodać sys.exit(1) jeśli błąd ma zatrzymać wykonanie skryptu
        # sys.exit(1)

    finally:
        # Opcjonalnie: zamknij otwarte zasoby, np. pool menedżera, jeśli byłby tworzony globalnie
        logger.info("Zakończono działanie skryptu.")


if __name__ == "__main__":
    # Dodatkowe zabezpieczenie dla multiprocessing na niektórych systemach
    if sys.platform.startswith('win'):
         # On Windows calling this function is necessary.
         mp.freeze_support()
    elif sys.platform.startswith('darwin') and sys.version_info >= (3, 8):
         # Na macOS z Python 3.8+ domyślna metoda startu to 'spawn', co jest bezpieczniejsze
         try:
              if mp.get_start_method() != 'spawn':
                   mp.set_start_method('spawn', force=True)
                   logger.debug("Ustawiono metodę startową multiprocessing na 'spawn' dla macOS.")
         except RuntimeError:
              logger.debug("Nie można zmienić metody startowej multiprocessing (prawdopodobnie już uruchomiono procesy).")

    main()