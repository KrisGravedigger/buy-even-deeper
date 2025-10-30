"""
Funkcje pomocnicze dla symulatora rynku.
Zawiera funkcje do przetwarzania danych, obliczania statystyk i generowania symulacji.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import os
from math import sqrt
from datetime import datetime, timedelta


def setup_logger(output_dir) -> logging.Logger:
    """
    Konfiguracja loggera
    
    Args:
        output_dir: Katalog wyjściowy
        
    Returns:
        logging.Logger: Skonfigurowany logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)  # Zmienione z INFO na WARNING
    
    if not logger.handlers:
        # Ustawienie formatowania
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Handler konsoli
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.WARNING)  # Explicit ustawienie poziomu dla konsoli
        logger.addHandler(console_handler)
        
        # Handler pliku
        log_dir = output_dir / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        file_handler = logging.FileHandler(log_dir / 'market_simulator.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def prepare_source_data(simulator) -> None:
    """
    Przygotowuje dane źródłowe, zapewniając poprawny format timestamp itp.
    
    Args:
        simulator: Instancja MarketSimulator
    """
    # Konwersja timestamp na datetime jeśli to string
    if simulator.data['timestamp'].dtype == 'object':
        simulator.data['timestamp'] = pd.to_datetime(simulator.data['timestamp'])
    
    # Sortowanie danych chronologicznie
    simulator.data.sort_values('timestamp', inplace=True)
    
    # Sprawdzenie odstępów czasowych i interwału danych
    time_diffs = simulator.data['timestamp'].diff()[1:].dt.total_seconds()
    simulator.interval_seconds = time_diffs.median()
    
    # Jeśli mamy dane BTC z osobnego pliku, potrzebujemy je zsynchronizować
    if simulator.btc_data is not None:
        if simulator.btc_data['timestamp'].dtype == 'object':
            simulator.btc_data['timestamp'] = pd.to_datetime(simulator.btc_data['timestamp'])
        
        simulator.btc_data.sort_values('timestamp', inplace=True)
        
        # Połączenie danych głównej kryptowaluty z BTC
        simulator.data = merge_with_btc_data(simulator)
        
    # Ustalenie symbolu głównej kryptowaluty na podstawie nazwy pliku
    try:
        filename = os.path.basename(simulator.source_data_path)
        parts = filename.split('_')
        if len(parts) > 1:
            simulator.main_symbol = parts[1]
            if '_' in simulator.main_symbol:
                simulator.main_symbol = simulator.main_symbol.split('_')[0]
        else:
            simulator.main_symbol = "UNKNOWN"  # Domyślna wartość w przypadku niepowodzenia
            simulator.logger.warning(f"Nie udało się wyekstrahować symbolu tokena z nazwy pliku: {filename}, używam UNKNOWN")
    except Exception as e:
        simulator.main_symbol = "UNKNOWN"
        simulator.logger.warning(f"Błąd podczas ekstrakcji symbolu tokena: {e}, używam UNKNOWN")
        
    # Ustalenie symbolu BTC
    simulator.btc_symbol = "BTC/USDT"


def merge_with_btc_data(simulator) -> pd.DataFrame:
    """
    Łączy dane głównej kryptowaluty z danymi BTC
    
    Args:
        simulator: Instancja MarketSimulator
        
    Returns:
        pd.DataFrame: Połączone dane
    """
    if simulator.btc_data is None:
        simulator.logger.warning("Brak danych BTC do łączenia. Zwracam oryginalne dane.")
        return simulator.data.copy()
    
    try:
        # Sprawdzenie wymaganych kolumn w danych BTC
        required_cols = ['timestamp', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in simulator.btc_data.columns]
        
        if missing_cols:
            simulator.logger.warning(f"Brak wymaganych kolumn w danych BTC: {missing_cols}. Zwracam oryginalne dane.")
            return simulator.data.copy()
        
        # Wybieramy potrzebne kolumny z danych BTC
        btc_cols = ['timestamp', 'open', 'high', 'low', 'close']
        # Dodajemy volume, jeśli jest dostępny
        if 'volume' in simulator.btc_data.columns:
            btc_cols.append('volume')
            
        btc_df = simulator.btc_data[btc_cols].copy()
        
        # Zmiana nazw kolumn
        btc_df.columns = ['timestamp'] + [f'btc_{col}' for col in btc_cols[1:]]
        
        # Łączenie danych
        merged_df = pd.merge_asof(
            simulator.data, 
            btc_df, 
            on='timestamp', 
            direction='nearest'
        )
        
        # Sprawdzenie, czy łączenie się powiodło
        if len(merged_df) < len(simulator.data):
            simulator.logger.warning(f"Po łączeniu z danymi BTC utracono {len(simulator.data) - len(merged_df)} rekordów.")
        
        return merged_df
    except Exception as e:
        simulator.logger.error(f"Błąd podczas łączenia danych BTC: {e}. Zwracam oryginalne dane.")
        return simulator.data.copy()


def calculate_historical_stats(simulator) -> None:
    """
    Oblicza statystyki historyczne: średnie zwroty, odchylenia standardowe, korelację.
    
    Args:
        simulator: Instancja MarketSimulator
    """
    # Obliczanie logarytmicznych zwrotów
    simulator.data['log_return'] = np.log(simulator.data['close'] / simulator.data['close'].shift(1))
    
    # Usunięcie wierszy z NaN z obliczeń
    returns_data = simulator.data.dropna(subset=['log_return'])
    
    # Podstawowe statystyki dla głównej kryptowaluty
    simulator.mean_return = returns_data['log_return'].mean()
    simulator.std_dev = returns_data['log_return'].std()
    
    # Jeśli mamy dane BTC, obliczamy statystyki również dla nich
    if simulator.has_btc_data or simulator.btc_data is not None:
        simulator.data['btc_log_return'] = np.log(simulator.data['btc_close'] / simulator.data['btc_close'].shift(1))
        returns_data = simulator.data.dropna(subset=['log_return', 'btc_log_return'])
        
        simulator.btc_mean_return = returns_data['btc_log_return'].mean()
        simulator.btc_std_dev = returns_data['btc_log_return'].std()
        
        # Obliczanie korelacji
        simulator.correlation = returns_data[['log_return', 'btc_log_return']].corr().iloc[0, 1]
        
        # Obliczanie macierzy kowariancji
        simulator.cov_matrix = returns_data[['log_return', 'btc_log_return']].cov().values
    else:
        simulator.btc_mean_return = 0
        simulator.btc_std_dev = 0
        simulator.correlation = 0
        simulator.cov_matrix = np.array([[simulator.std_dev**2, 0], [0, 0]])
    
    # Inne przydatne statystyki
    simulator.price_range_ratio = (simulator.data['high'] / simulator.data['low']).mean()
    simulator.volume_mean = simulator.data['volume'].mean()
    simulator.volume_std = simulator.data['volume'].std()
    
    # Zapisanie statystyk do logów
    simulator.logger.info(f"Statystyki historyczne dla {simulator.source_data_path}:")
    simulator.logger.info(f"Średni zwrot: {simulator.mean_return:.6f}")
    simulator.logger.info(f"Odchylenie standardowe: {simulator.std_dev:.6f}")
    simulator.logger.info(f"Korelacja z BTC: {simulator.correlation:.6f}")


def simulate_gbm(initial_price: float, mean_return: float, std_dev: float, intervals: int) -> np.ndarray:
    """
    Symuluje geometryczny ruch Browna (GBM)
    
    Args:
        initial_price: Początkowa cena
        mean_return: Średni zwrot
        std_dev: Odchylenie standardowe
        intervals: Liczba interwałów do symulacji
        
    Returns:
        np.ndarray: Tablica z symulowanymi cenami
    """
    # Generowanie losowego ruchu Wienera
    dt = 1  # Interwał czasowy
    dW = np.random.normal(0, sqrt(dt), size=intervals)
    W = np.cumsum(dW)
    
    # Obliczanie ścieżki cen
    t = np.arange(intervals)
    S = initial_price * np.exp((mean_return - 0.5 * std_dev**2) * t + std_dev * W)
    
    return S


def simulate_correlated_gbm(initial_price1: float, initial_price2: float, mean_return1: float, 
                           mean_return2: float, std_dev1: float, std_dev2: float, 
                           correlation: float, intervals: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symuluje dwa skorelowane procesy GBM

    Args:
        initial_price1: Początkowa cena pierwszego aktywa (BTC)
        initial_price2: Początkowa cena drugiego aktywa (token)
        mean_return1: Średni zwrot pierwszego aktywa (BTC)
        mean_return2: Średni zwrot drugiego aktywa (token)
        std_dev1: Odchylenie standardowe pierwszego aktywa (BTC)
        std_dev2: Odchylenie standardowe drugiego aktywa (token)
        correlation: Korelacja między aktywami
        intervals: Liczba interwałów do symulacji

    Returns:
        Tuple[np.ndarray, np.ndarray]: Dwie tablice z symulowanymi cenami (BTC, token)
    """
    # Tworzenie macierzy kowariancji
    cov_matrix = np.array([
        [std_dev1**2, correlation * std_dev1 * std_dev2],
        [correlation * std_dev1 * std_dev2, std_dev2**2]
    ])

    # Dekompozycja Choleskiego
    L = np.linalg.cholesky(cov_matrix)

    # Generowanie nieskorelowanych zmiennych normalnych
    dt = 1
    Z = np.random.normal(0, sqrt(dt), size=(2, intervals))

    # Tworzenie skorelowanych procesów
    W = L @ Z

    # Generowanie ścieżek cenowych
    t = np.arange(intervals)
    S1 = initial_price1 * np.exp((mean_return1 - 0.5 * std_dev1**2) * t + W[0])
    S2 = initial_price2 * np.exp((mean_return2 - 0.5 * std_dev2**2) * t + W[1])

    return S1, S2


def create_simulated_dataframe(simulator, simulated_prices: np.ndarray, 
                              simulated_btc_prices: Optional[np.ndarray], 
                              days: int) -> pd.DataFrame:
    """
    Tworzy DataFrame z symulowanymi danymi w formacie zgodnym z danymi Binance
    
    Args:
        simulator: Instancja MarketSimulator
        simulated_prices: Tablica z symulowanymi cenami
        simulated_btc_prices: Tablica z symulowanymi cenami BTC (lub None)
        days: Liczba dni
        
    Returns:
        pd.DataFrame: DataFrame z symulowanymi danymi
    """
    # Ustalenie liczby interwałów
    intervals_per_day = int(24 * 60 * 60 / simulator.interval_seconds)
    total_intervals = days * intervals_per_day
    
    # Ustalenie czasu początkowego
    start_time = simulator.data['timestamp'].iloc[-1]
    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time)
    
    # Generowanie timestampów
    timestamps = [start_time + timedelta(seconds=i * simulator.interval_seconds) 
                for i in range(1, total_intervals + 1)]
    
    # Inicjalizacja DataFrame
    df = pd.DataFrame({'timestamp': timestamps})
    
    # Generowanie danych OHLCV dla głównej kryptowaluty
    df['close'] = simulated_prices
    
    # Generowanie danych high, low, open
    volatility_factor = 0.5 * simulator.std_dev * sqrt(simulator.interval_seconds / (24 * 60 * 60))
    df['high'] = df['close'] * np.exp(np.random.normal(0, volatility_factor, len(df)))
    df['low'] = df['close'] * np.exp(np.random.normal(0, volatility_factor, len(df)))
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    
    # Upewniamy się, że high jest najwyższe, a low najniższe
    df['high'] = np.maximum(np.maximum(df['high'], df['close']), df['open'])
    df['low'] = np.minimum(np.minimum(df['low'], df['close']), df['open'])
    
    # Generowanie wolumenu - POPRAWKA: ograniczenie wartości dla uniknięcia przepełnienia
    # Zabezpieczenie przed wartościami bliskimi zeru
    log_volume_mean = np.log(max(simulator.volume_mean, 1e-10))
    # Bezpieczniejsze obliczanie odchylenia standardowego dla logarytmicznego wolumenu
    if simulator.volume_mean > 1e-10:
        log_volume_ratio = simulator.volume_std / simulator.volume_mean
        log_volume_std = min(log_volume_ratio, 2.0)  # Ograniczenie odchylenia
    else:
        log_volume_std = 0.5  # Wartość domyślna jeśli średni wolumen jest zbyt mały

    # Używanie bezpieczniejszej metody generowania wolumenu
    try:
        df['volume'] = np.exp(np.random.normal(log_volume_mean, log_volume_std, len(df)))
        # Dodatkowe ograniczenie zakresu dla bezpieczeństwa
        df['volume'] = df['volume'].clip(simulator.volume_mean * 0.1, simulator.volume_mean * 10)
    except Exception as e:
        # Alternatywna metoda w przypadku błędu
        simulator.logger.warning(f"Wystąpił problem przy generowaniu wolumenu: {e}, używam alternatywnej metody")
        mean_volume = simulator.volume_mean
        std_volume = min(simulator.volume_std, simulator.volume_mean * 2)
        df['volume'] = np.abs(np.random.normal(mean_volume, std_volume, len(df)))
        df['volume'] = df['volume'].clip(mean_volume * 0.1, mean_volume * 10)  # Ograniczenie zakresu
    
    # Jeśli mamy dane BTC, dodajemy je również
    if simulated_btc_prices is not None:
        df['btc_close'] = simulated_btc_prices
        
        # Generowanie danych high, low, open dla BTC
        btc_volatility_factor = 0.5 * simulator.btc_std_dev * sqrt(simulator.interval_seconds / (24 * 60 * 60))
        df['btc_high'] = df['btc_close'] * np.exp(np.random.normal(0, btc_volatility_factor, len(df)))
        df['btc_low'] = df['btc_close'] * np.exp(np.random.normal(0, btc_volatility_factor, len(df)))
        df['btc_open'] = df['btc_close'].shift(1).fillna(df['btc_close'].iloc[0])
        
        # Upewniamy się, że high jest najwyższe, a low najniższe dla BTC
        df['btc_high'] = np.maximum(np.maximum(df['btc_high'], df['btc_close']), df['btc_open'])
        df['btc_low'] = np.minimum(np.minimum(df['btc_low'], df['btc_close']), df['btc_open'])
        
        # Generowanie wolumenu dla BTC
        if 'btc_volume' in simulator.data.columns:
            btc_volume_mean = simulator.data['btc_volume'].mean()
            btc_volume_std = simulator.data['btc_volume'].std()
            log_btc_volume_mean = np.log(btc_volume_mean)
            log_btc_volume_std = min(btc_volume_std / btc_volume_mean, 2.0)  # Ograniczenie odchylenia
            
            # Używanie bezpieczniejszej metody generowania wolumenu
            try:
                df['btc_volume'] = np.exp(np.random.normal(log_btc_volume_mean, log_btc_volume_std, len(df)))
            except:
                # Alternatywna metoda w przypadku błędu
                simulator.logger.warning("Wystąpił problem przy generowaniu wolumenu BTC, używam alternatywnej metody")
                df['btc_volume'] = np.abs(np.random.normal(btc_volume_mean, min(btc_volume_std, btc_volume_mean * 2), len(df)))
                df['btc_volume'] = df['btc_volume'].clip(btc_volume_mean * 0.1, btc_volume_mean * 10)  # Ograniczenie zakresu
        else:
            # Jeśli nie mamy historycznych danych o wolumenie BTC, generujemy fikcyjne
            df['btc_volume'] = np.random.lognormal(10, 1, len(df))
    
    # Dodanie dodatkowych metryk
    add_derived_metrics(df, simulator.has_btc_data or simulated_btc_prices is not None, simulator.main_symbol, simulator.btc_symbol)
    
    return df


def add_derived_metrics(df: pd.DataFrame, has_btc: bool = False, main_symbol: str = "UNKNOWN", btc_symbol: str = "BTC/USDT") -> None:
    """
    Dodaje dodatkowe metryki do DataFrame, zgodne z formatem danych Binance
    
    Args:
        df: DataFrame do uzupełnienia
        has_btc: Czy dodać metryki dla BTC
        main_symbol: Symbol głównej kryptowaluty
        btc_symbol: Symbol BTC
    """
    # Metryki dla głównej kryptowaluty
    df['average_price'] = (df['open'] + df['close']) / 2
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['price_range'] = df['high'] - df['low']
    df['returns'] = df['close'].pct_change()
    # Zastąpienie pierwszego NaN w returns
    df.loc[0, 'returns'] = 0
    
    # Obliczenie zmienności jako odchylenie standardowe zwrotów z okna 20 okresów
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Metryki wolumenu
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['relative_volume'] = df['volume'] / df['volume_ma']
    
    # Momentum cenowy (zmiana ceny w ciągu ostatnich 10 okresów)
    df['price_momentum'] = df['close'].diff(10)
    
    # True Range
    df['true_range'] = np.maximum(df['high'], df['close'].shift(1)) - np.minimum(df['low'], df['close'].shift(1))
    
    # Jeśli mamy dane BTC, dodajemy podobne metryki dla BTC
    if has_btc:
        df['btc_average_price'] = (df['btc_open'] + df['btc_close']) / 2
        df['btc_typical_price'] = (df['btc_high'] + df['btc_low'] + df['btc_close']) / 3
        df['btc_price_range'] = df['btc_high'] - df['btc_low']
        df['btc_returns'] = df['btc_close'].pct_change()
        df.loc[0, 'btc_returns'] = 0
        
        df['btc_volatility'] = df['btc_returns'].rolling(window=20).std()
        
        if 'btc_volume' in df.columns:
            df['btc_volume_ma'] = df['btc_volume'].rolling(window=20).mean()
            df['btc_relative_volume'] = df['btc_volume'] / df['btc_volume_ma']
        
        df['btc_price_momentum'] = df['btc_close'].diff(10)
        df['btc_true_range'] = np.maximum(df['btc_high'], df['btc_close'].shift(1)) - np.minimum(df['btc_low'], df['btc_close'].shift(1))
        
        # Dodatkowe metryki porównawcze
        df['price_change_ratio'] = abs(df['returns'] / df['btc_returns'])
        df['volatility_ratio'] = df['volatility'] / df['btc_volatility']
        
        # Dodanie informacji o symbolach
        df['main_symbol'] = main_symbol
        df['btc_symbol'] = btc_symbol
    
    # Zastąpienie wartości NaN
    df.fillna(0, inplace=True)


def save_to_csv(simulator, simulated_df: pd.DataFrame, scenario_name: str) -> str:
    """
    Zapisuje wygenerowane dane do pliku CSV.
    
    Args:
        simulator: Instancja MarketSimulator
        simulated_df: DataFrame z symulowanymi danymi
        scenario_name: Nazwa scenariusza (zostanie użyta w nazwie pliku)
        
    Returns:
        str: Ścieżka do zapisanego pliku
    """
    # Formatowanie timestampów
    if pd.api.types.is_datetime64_any_dtype(simulated_df['timestamp']):
        simulated_df['timestamp'] = simulated_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Generowanie nazwy pliku
    main_symbol = os.path.basename(simulator.source_data_path).split('_')[1]
    if '_' in main_symbol:
        main_symbol = main_symbol.split('_')[0]
        
    # Usunięcie niedozwolonych znaków z nazwy scenariusza
    safe_scenario_name = "".join(c for c in scenario_name if c.isalnum() or c in "_-")
    
    # Generowanie daty dla nazwy pliku
    start_date = pd.to_datetime(simulated_df['timestamp'].iloc[0])
    end_date = pd.to_datetime(simulated_df['timestamp'].iloc[-1])
    date_str = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
    
    # Tworzenie nazwy pliku
    if simulator.has_btc_data or simulator.btc_data is not None:
        filename = f"simulated_{main_symbol}_USDT_1m_{date_str}_{safe_scenario_name}_with_btc.csv"
    else:
        filename = f"simulated_{main_symbol}_USDT_1m_{date_str}_{safe_scenario_name}.csv"
    
    # Ścieżka do zapisu
    csv_path = simulator.output_dir / filename
    
    # Zapisanie pliku
    simulated_df.to_csv(csv_path, index=False)
    
    return str(csv_path)