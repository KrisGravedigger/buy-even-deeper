#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moduł do szczegółowego porównywania wyników backtestingu z rzeczywistymi transakcjami.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
import multiprocessing as mp
from numba import njit, float32, int64
import pickle
import warnings
import re
import sys
import traceback
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Struktura katalogów
BASE_DIR = Path('testowanie')
ARCHIVE_DIR = BASE_DIR / 'archiwum'
TRADES_DIR = BASE_DIR / 'zakupy'
TRADES_ARCHIVE = TRADES_DIR / 'archiwum'
RESULTS_DIR = BASE_DIR / 'wyniki'
PARAMETRY_DIR = Path('parametry')
LOGS_DIR = Path('logi')

DATETIME_FORMAT = "%Y%m%d_%H%M%S"

# Tworzenie struktury katalogów
for dir_path in [BASE_DIR, ARCHIVE_DIR, TRADES_DIR, TRADES_ARCHIVE, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def setup_logging() -> logging.Logger:
    """Konfiguracja loggera z zapisem do głównego folderu logów"""
    log_file = LOGS_DIR / f'comparator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    logger = logging.getLogger('strategy_comparator')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

class Trade(NamedTuple):
    """Struktura pojedynczej transakcji"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    profit_pct: float
    volume: float
    symbol: str

@dataclass
class ComparisonParameters:
    window_size: int = 5
    price_tolerance: float = 0.1
    volume_tolerance: float = 0.1
    time_unit: str = 'T'
    min_profit_threshold: float = 0.5
    check_timeframe: int = 1

@dataclass
class DetailedParameters:
    check_timeframe: int
    percentage_buy_threshold: float
    max_open_orders_per_coin: int = 1
    next_buy_delay: int = 60
    next_buy_price_lower: float = 1.0
    trailing_enabled: bool = False
    trailing_stop_price: float = 1.0
    trailing_stop_margin: float = 0.5
    trailing_stop_time: int = 1
    stop_loss_enabled: bool = False
    stop_loss_threshold: float = 2.0
    stop_loss_delay_time: int = 1
    follow_btc_price: bool = False
    follow_btc_threshold: float = 1.0
    follow_btc_block_time: int = 30
    pump_detection_enabled: bool = False
    pump_detection_threshold: float = 5.0
    stop_loss_disable_buy: bool = False
    stop_loss_disable_buy_all: bool = False
    stop_loss_next_buy_lower: float = 1.0
    stop_loss_no_buy_delay: int = 15

    def to_array(self) -> np.ndarray:
        """Konwersja parametrów do tablicy numpy dla funkcji njit"""
        return np.array([
            self.check_timeframe,
            self.percentage_buy_threshold,
            float(self.trailing_enabled),
            self.trailing_stop_price,
            self.trailing_stop_margin,
            self.trailing_stop_time,
            float(self.stop_loss_enabled),
            self.stop_loss_threshold,
            self.stop_loss_delay_time,
            float(self.follow_btc_price),
            self.follow_btc_threshold,
            self.follow_btc_block_time,
            float(self.pump_detection_enabled),
            self.pump_detection_threshold
        ], dtype=np.float64)

class TradesParser:
    @staticmethod
    def _parse_trade_line(line: str) -> Optional[Trade]:
        """Parsuje pojedynczą linię danych transakcji"""
        try:
            if not line.strip():
                return None
                
            parts = line.strip().split('\t')
            if len(parts) != 9:
                return None
                
            symbol = parts[0]
            profit_pct = float(parts[2].replace("%", ""))
            volume = float(parts[4])
            entry_price = float(parts[5])
            
            # Korekta czasów o -1 godzinę dla synchronizacji
            entry_time = datetime.strptime(parts[6], '%d.%m.%Y %H:%M:%S') - timedelta(hours=1)
            exit_time = datetime.strptime(parts[7], '%d.%m.%Y %H:%M:%S') - timedelta(hours=1)
            
            exit_price = float(parts[8])
            
            return Trade(
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                profit_pct=profit_pct,
                volume=volume,
                symbol=symbol
            )
            
        except Exception as e:
            logger.error(f"Błąd parsowania linii: {e}")
            return None

    def parse_trades_file(self, file_path: Path) -> pd.DataFrame:
        """Parsuje plik z transakcjami do DataFrame"""
        trades = []
        try:
            logger.info(f"Przetwarzanie pliku: {file_path}")
            
            encodings = ['utf-8', 'utf-8-sig', 'cp1250', 'iso-8859-1']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
                    
            if not content:
                raise ValueError(f"Nie udało się odczytać pliku z żadnym z kodowań: {encodings}")
            
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            
            for line in lines:
                if trade := self._parse_trade_line(line):
                    trades.append(trade)
                
            if not trades:
                raise ValueError("Nie znaleziono poprawnych transakcji w pliku")
            
            df = pd.DataFrame(trades)
            df.sort_values('entry_time', inplace=True)
            logger.info(f"Znaleziono {len(trades)} transakcji")
            return df
            
        except Exception as e:
            logger.error(f"Błąd przetwarzania pliku {file_path}: {str(e)}")
            raise

    def archive_trades_file(self, file_path: Path):
        """Przenosi przetworzony plik do archiwum"""
        try:
            archive_path = TRADES_ARCHIVE / file_path.name
            file_path.rename(archive_path)
            logger.info(f"Zarchiwizowano plik: {file_path} -> {archive_path}")
        except Exception as e:
            logger.error(f"Błąd archiwizacji pliku {file_path}: {str(e)}")

@njit(cache=True)
def calculate_profit_pct(entry_price: float, current_price: float) -> float:
    """Oblicza procent zysku z transakcji"""
    return ((current_price - entry_price) / entry_price) * 100.0

@njit(cache=True)
def precompute_changes(prices: np.ndarray, timeframe: int) -> np.ndarray:
    """Oblicza procentowe zmiany cen dla danego timeframe"""
    result = np.empty_like(prices)
    result[:timeframe] = 0
    result[timeframe:] = ((prices[timeframe:] / prices[:-timeframe]) - 1.0) * 100.0
    return result

@njit(cache=True)
def detailed_strategy_core(
    main_prices: np.ndarray,
    btc_prices: np.ndarray,
    timestamps: np.ndarray,
    params: np.ndarray
) -> List[Tuple[int, float, int, float, int, str]]:
    """
    Logika strategii z zapisem szczegółów decyzji
    
    Returns:
        Lista krotek (timestamp, cena, typ, wartość_parametru, indeks_parametru, powód)
        typ: 0=wejście, 1=wyjście
    """
    timeframe = int(params[0])
    main_changes = precompute_changes(main_prices, timeframe)
    btc_changes = precompute_changes(btc_prices, timeframe) if params[9] > 0 else np.zeros_like(main_prices)
    
    signals = []
    position_open = False
    entry_price = 0.0
    entry_time = 0
    trailing_high = 0.0
    trailing_start_time = 0
    stop_loss_trigger_time = 0
    last_buy_time = 0
    stop_loss_block_time = 0
    
    # Główna pętla analizy
    for i in range(timeframe, len(main_prices)):
        current_price = main_prices[i]
        current_time = timestamps[i]
        
        if stop_loss_block_time > 0 and current_time < stop_loss_block_time:
            continue
            
        if position_open:
            # Analiza otwartej pozycji
            profit_pct = calculate_profit_pct(entry_price, current_price)
            should_close = False
            close_reason = ""
            
            # Sprawdzanie stop loss
            if params[6] > 0 and profit_pct <= params[7]:
                if stop_loss_trigger_time == 0:
                    stop_loss_trigger_time = current_time
                elif (current_time - stop_loss_trigger_time) >= params[8]:
                    should_close = True
                    close_reason = "STOP_LOSS"
                    if params[17] > 0:
                        stop_loss_block_time = current_time + params[20] * 60
            
            # Sprawdzanie trailing stop
            if not should_close and params[2] > 0:
                if profit_pct >= params[3]:
                    if trailing_start_time == 0:
                        trailing_start_time = current_time
                        trailing_high = current_price
                    elif current_price > trailing_high:
                        trailing_high = current_price
                    elif calculate_profit_pct(trailing_high, current_price) <= -params[4]:
                        if (current_time - trailing_start_time) >= params[5]:
                            should_close = True
                            close_reason = "TRAILING_STOP"
            
            if should_close:
                signals.append((
                    current_time, current_price, 1, profit_pct,
                    1, close_reason
                ))
                position_open = False
                trailing_start_time = 0
                trailing_high = 0.0
                stop_loss_trigger_time = 0
                continue
                
        else:
            # Szukanie okazji do wejścia
            if current_time - last_buy_time < params[11]:
                continue
                
            # Sprawdzanie BTC
            if params[9] > 0:
                btc_change = btc_changes[i]
                if btc_change < 0 and abs(btc_change) > abs(main_changes[i]):
                    continue
            
            # Sprawdzanie pump detection
            if params[12] > 0 and main_changes[i] >= params[13]:
                continue
            
            # Sprawdzanie ceny po stop loss
            if stop_loss_block_time > 0:
                required_price = entry_price * (1 - abs(params[19])/100)
                if current_price > required_price:
                    continue
            
            # Warunek wejścia
            if main_changes[i] <= params[1]:
                entry_reason = "BASE_CONDITION" if params[9] == 0 else "BASE_WITH_BTC_FOLLOW"
                
                signals.append((
                    current_time, current_price, 0, params[1],
                    1, entry_reason
                ))
                position_open = True
                entry_price = current_price
                entry_time = current_time
                last_buy_time = current_time
                trailing_start_time = 0
                trailing_high = 0
                stop_loss_trigger_time = 0
    
    # Zamknięcie ostatniej pozycji
    if position_open:
        signals.append((
            timestamps[-1], main_prices[-1], 1, 
            calculate_profit_pct(entry_price, main_prices[-1]),
            0, "END_OF_DATA"
        ))
    
    return signals

def analyze_market_conditions(market_data: pd.DataFrame, timestamp: pd.Timestamp, window_size: int = 5) -> Dict:
    """Analiza warunków rynkowych w zadanym oknie czasowym"""
    try:
        start_time = timestamp - pd.Timedelta(minutes=window_size)
        end_time = timestamp + pd.Timedelta(minutes=window_size)
        
        window_data = market_data[
            (market_data['timestamp'] >= start_time) & 
            (market_data['timestamp'] <= end_time)
        ]
        
        if window_data.empty:
            return {}
            
        price_change = ((window_data['close'].iloc[-1] / window_data['close'].iloc[0]) - 1) * 100
        volatility = window_data['price_range'].mean()
        volume = window_data['volume'].mean()
        
        return {
            'price_change': price_change,
            'volatility': volatility,
            'volume': volume,
            'avg_price': window_data['average_price'].mean(),
            'data_points': len(window_data)
        }
        
    except Exception as e:
        logger.error(f"Błąd analizy warunków rynkowych: {str(e)}")
        return {}

def validate_market_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """Sprawdza poprawność danych rynkowych"""
    required_base_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    btc_columns = ['btc_open', 'btc_high', 'btc_low', 'btc_close', 'btc_volume']
    
    missing_base = [col for col in required_base_columns if col not in df.columns]
    if missing_base:
        return False, f"Brakujące podstawowe kolumny: {', '.join(missing_base)}"
    
    has_btc_data = all(col in df.columns for col in btc_columns)
    if not has_btc_data:
        logger.warning("Brak kolumn z danymi BTC - strategia follow BTC będzie niedostępna")
    
    return True, "OK"

def main():
    """Główna funkcja programu"""
    try:
        logger.info("Rozpoczynam porównywanie strategii")
        
        # Sprawdzanie plików z transakcjami
        trades_files = [f for f in TRADES_DIR.glob('*') if f.is_file() and not f.name.startswith('.')]
        logger.info(f"Znaleziono {len(trades_files)} plików z transakcjami")
        
        if not trades_files:
            logger.error("Brak plików z transakcjami w katalogu zakupy/")
            return
            
        # Wczytanie danych rynkowych
        csv_files = list(Path('csv').glob('*.csv'))
        if not csv_files:
            logger.error("Brak plików CSV z danymi")
            return
        if len(csv_files) > 1:
            logger.error("Znaleziono więcej niż jeden plik CSV")
            return
            
        # Wczytanie parametrów
        try:
            with open(PARAMETRY_DIR / 'trading_parameters.json', 'r') as f:
                params_config = json.load(f)

            def get_param_value(param_config):
                if 'value' in param_config:
                    return param_config['value']
                elif 'range' in param_config:
                    min_val, max_val, _ = param_config['range']
                    return (min_val + max_val) / 2
                else:
                    raise ValueError(f"Nieprawidłowa konfiguracja parametru")

            default_params = {
                'check_timeframe': {'value': 1},
                'percentage_buy_threshold': {'value': -0.5},
                'trailing_enabled': {'value': True},
                'trailing_stop_price': {'value': 1.0},
                'trailing_stop_margin': {'value': 0.5},
                'trailing_stop_time': {'value': 1},
                'stop_loss_enabled': {'value': True},
                'stop_loss_threshold': {'value': -2.0},
                'stop_loss_delay_time': {'value': 1},
                'follow_btc_price': {'value': True},
                'follow_btc_threshold': {'value': 1.0},
                'follow_btc_block_time': {'value': 30},
                'pump_detection_enabled': {'value': True},
                'pump_detection_threshold': {'value': 5.0}
            }

            # Uzupełnienie brakujących parametrów wartościami domyślnymi
            for param_name, default_config in default_params.items():
                if param_name not in params_config:
                    params_config[param_name] = default_config

            params = DetailedParameters(
                check_timeframe=int(get_param_value(params_config['check_timeframe'])),
                percentage_buy_threshold=float(get_param_value(params_config['percentage_buy_threshold'])),
                trailing_enabled=bool(get_param_value(params_config['trailing_enabled'])),
                trailing_stop_price=float(get_param_value(params_config['trailing_stop_price'])),
                trailing_stop_margin=float(get_param_value(params_config['trailing_stop_margin'])),
                trailing_stop_time=int(get_param_value(params_config['trailing_stop_time'])),
                stop_loss_enabled=bool(get_param_value(params_config.get('stop_loss_enabled', {'value': True}))),
                stop_loss_threshold=float(get_param_value(params_config['stop_loss_threshold'])),
                stop_loss_delay_time=int(get_param_value(params_config['stop_loss_delay_time'])),
                follow_btc_price=bool(get_param_value(params_config.get('follow_btc_price', {'value': True}))),
                follow_btc_threshold=float(get_param_value(params_config.get('follow_btc_threshold', {'value': 1.0}))),
                follow_btc_block_time=int(get_param_value(params_config.get('follow_btc_block_time', {'value': 30}))),
                pump_detection_enabled=bool(get_param_value(params_config.get('pump_detection_enabled', {'value': True}))),
                pump_detection_threshold=float(get_param_value(params_config.get('pump_detection_threshold', {'value': 5.0})))
            )

        except Exception as e:
            logger.error(f"Błąd wczytywania parametrów: {e}")
            return
            
        trades_parser = TradesParser()
        
        for trades_file in trades_files:
            try:
                real_trades = trades_parser.parse_trades_file(trades_file)
                if real_trades.empty:
                    logger.warning(f"Brak transakcji w pliku {trades_file}")
                    continue
                
                # Wczytanie danych rynkowych
                logger.info(f"Wczytywanie danych rynkowych")
                market_data = pd.read_csv(csv_files[0])
                market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
                market_data['minutes'] = market_data['timestamp'].astype(np.int64) // 60e9
                
                valid, message = validate_market_data(market_data)
                if not valid:
                    logger.error(f"Nieprawidłowe dane rynkowe: {message}")
                    continue

                # Przygotowanie danych do backtestingu
                main_prices = market_data['average_price'].to_numpy(dtype=np.float32)
                timestamps = market_data['minutes'].to_numpy(dtype=np.int64)
                
                btc_prices = (market_data['btc_average_price'].to_numpy(dtype=np.float32) 
                            if 'btc_average_price' in market_data.columns else main_prices)
                
                # Wykonanie symulacji strategii
                signals = detailed_strategy_core(
                    main_prices,
                    btc_prices,
                    timestamps,
                    params.to_array()
                )
                
                # Generowanie raportu
                report_file = RESULTS_DIR / f'comparison_report_{datetime.now().strftime(DATETIME_FORMAT)}.txt'
                
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write("=== Raport z analizy porównawczej strategii ===\n")
                    f.write(f"Data wygenerowania: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    real_trades = real_trades.sort_values('entry_time')
                    
                    f.write("PORÓWNANIE TRANSAKCJI:\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for idx, real_trade in real_trades.iterrows():
                        f.write(f"Transakcja {idx + 1}:\n")
                        f.write("-" * 30 + "\n")
                        
                        # Dane z bota
                        f.write("BOT:\n")
                        f.write(f"Wejście: {real_trade.entry_time} @ {real_trade.entry_price}\n")
                        f.write(f"Wyjście: {real_trade.exit_time} @ {real_trade.exit_price}\n")
                        f.write(f"Zysk: {real_trade.profit_pct}%\n")
                        f.write(f"Wolumen: {real_trade.volume}\n")
                        
                        # Warunki rynkowe
                        entry_conditions = analyze_market_conditions(
                            market_data, 
                            real_trade.entry_time,
                            window_size=5
                        )
                        if entry_conditions:
                            f.write("\nWarunki rynkowe przy wejściu:\n")
                            f.write(f"Zmiana ceny: {entry_conditions['price_change']:.2f}%\n")
                            f.write(f"Zmienność: {entry_conditions['volatility']:.4f}\n")
                            f.write(f"Średni wolumen: {entry_conditions['volume']:.6f}\n")
                            f.write(f"Średnia cena: {entry_conditions['avg_price']:.2f}\n")
                        
                        # Porównanie ze strategią
                        entry_time_minutes = int(real_trade.entry_time.timestamp() // 60)
                        matching_signals = [
                            s for s in signals 
                            if abs(s[0] - entry_time_minutes) <= 5 and s[2] == 0
                        ]
                        
                        f.write("\nSTRATEGIA:\n")
                        if matching_signals:
                            for signal in matching_signals:
                                entry_time = pd.Timestamp(signal[0] * 60, unit='s')
                                exit_signals = [
                                    s for s in signals 
                                    if s[0] > signal[0] and s[2] == 1 
                                    and not any(x[0] < s[0] and x[0] > signal[0] and x[2] == 0 for x in signals)
                                ]
                                if exit_signals:
                                    exit_signal = exit_signals[0]
                                    exit_time = pd.Timestamp(exit_signal[0] * 60, unit='s')
                                    f.write(f"Wejście: {entry_time} @ {signal[1]:.2f} ({signal[5]})\n")
                                    f.write(f"Wyjście: {exit_time} @ {exit_signal[1]:.2f} ({exit_signal[5]})\n")
                                    f.write(f"Zysk: {exit_signal[3]:.2f}%\n")
                        else:
                            f.write("Brak sygnału wejścia\n")
                            f.write("Możliwe przyczyny:\n")
                            f.write(f"- Percentage Buy Threshold (próg: {params.percentage_buy_threshold}%)\n")
                        
                        f.write("\n" + "=" * 50 + "\n\n")

                    # Statystyki zbiorcze
                    f.write("\nSTATYSTYKI ZBIORCZE:\n")
                    f.write("=" * 50 + "\n")
                    
                    real_trades_count = len(real_trades)
                    strategy_trades_count = len([s for s in signals if s[2] == 0])
                    real_profit = real_trades['profit_pct'].sum()
                    
                    strategy_trades = []
                    for signal in [s for s in signals if s[2] == 0]:
                        exit_signals = [
                            s for s in signals 
                            if s[0] > signal[0] and s[2] == 1 
                            and not any(x[0] < s[0] and x[0] > signal[0] and x[2] == 0 for x in signals)
                        ]
                        if exit_signals:
                            strategy_trades.append(exit_signals[0][3])
                    
                    strategy_profit = sum(strategy_trades)
                    
                    f.write(f"Liczba transakcji bot: {real_trades_count}\n")
                    f.write(f"Liczba transakcji strategia: {strategy_trades_count}\n")
                    f.write(f"Całkowity zysk bot: {real_profit:.2f}%\n")
                    f.write(f"Całkowity zysk strategia: {strategy_profit:.2f}%\n")
                    f.write(f"Średni zysk bot: {real_profit/real_trades_count:.2f}%\n")
                    if strategy_trades:
                        f.write(f"Średni zysk strategia: {strategy_profit/len(strategy_trades):.2f}%\n")
                
                trades_parser.archive_trades_file(trades_file)
                logger.info(f"Zakończono analizę pliku {trades_file}")
                
            except Exception as e:
                logger.error(f"Błąd przetwarzania pliku {trades_file}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Błąd krytyczny: {str(e)}")
        raise

if __name__ == "__main__":
    main()