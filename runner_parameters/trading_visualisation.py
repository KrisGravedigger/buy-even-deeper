#!/opt/homebrew/bin/python3.11
# -*- coding: utf-8 -*-

"""
Moduł do wizualizacji wyników strategii tradingowej.
Generuje wykresy i raporty na podstawie uruchomienia strategii na danych historycznych.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict

# Import funkcji pomocniczych z innych modułów
from utils.config import get_visualization_output_dir

logger = logging.getLogger(__name__)

def validate_visualization_parameters(params_dict: Dict) -> bool:
    """
    Sprawdza czy parametry są poprawne dla wizualizacji - muszą mieć pojedyncze wartości, nie zakresy.
    
    Args:
        params_dict: Słownik z parametrami strategii
        
    Returns:
        bool: True jeśli parametry są poprawne dla wizualizacji
    """
    invalid_params = []
    
    for param_name, param_info in params_dict.items():
        if isinstance(param_info, dict) and 'range' in param_info:
            invalid_params.append(param_name)
            
    if invalid_params:
        logger.error(f"Parametry zawierają zakresy wartości zamiast pojedynczych wartości: {', '.join(invalid_params)}")
        logger.error("Dla wizualizacji wymagane są pojedyncze wartości parametrów (pole 'value' zamiast 'range')")
        return False
        
    return True

def create_trading_parameters(params_dict: Dict) -> Any:
    """
    Tworzy obiekt TradingParameters z słownika parametrów.
    
    Args:
        params_dict: Słownik z parametrami strategii
        
    Returns:
        TradingParameters: Obiekt z parametrami strategii
    """
    from runner_parameters.models import TradingParameters
    
    # Konwersja słownika parametrów do formatu odpowiedniego dla TradingParameters
    param_values = {}
    
    for param_name, param_info in params_dict.items():
        if param_name.startswith('__'):  # Pomijamy specjalne parametry
            continue
            
        if isinstance(param_info, dict):
            if 'value' in param_info:
                param_values[param_name] = param_info['value']
        else:
            # Jeśli bezpośrednio podana wartość
            param_values[param_name] = param_info
    
    return TradingParameters(**param_values)

def run_strategy_with_details(
    main_prices: np.ndarray, 
    btc_prices: np.ndarray, 
    times: np.ndarray,
    df: pd.DataFrame,
    params: np.ndarray
) -> List[Dict]:
    """
    Uruchamia strategię i zwraca szczegółowe informacje o każdej transakcji.
    
    Args:
        main_prices: Tablica cen głównego aktywa
        btc_prices: Tablica cen BTC
        times: Tablica czasów (w minutach)
        df: Oryginalny DataFrame z danymi
        params: Tablica z parametrami strategii
        
    Returns:
        List[Dict]: Lista słowników z detalami transakcji
    """
    from strategy_runner import precompute_price_changes
    
    # Prekompilacja zmian cen
    timeframe = int(params[0])
    main_changes = precompute_price_changes(main_prices, timeframe)
    btc_changes = precompute_price_changes(btc_prices, timeframe)
    
    # Inicjalizacja struktur danych
    max_positions = int(params[11])
    positions = np.zeros((max_positions, 5), dtype=np.float64)  # [entry_price, entry_time, trailing_start_time, trailing_high_price, stop_loss_time]
    position_count = 0
    trades = []
    
    # Blokady czasowe [global_block, btc_block, coin_block, last_buy]
    blocking_times = np.zeros(4, dtype=np.int64)
    stop_loss_price = 0.0
    
    # Parametry trailing buy
    trailing_buy_enabled = params[23] > 0
    trailing_buy_threshold = params[24]
    trailing_buy_window = params[25]
    
    # Stan trailing buy
    trailing_buy_active = False
    trailing_buy_start_price = 0.0
    trailing_buy_start_time = 0
    trailing_buy_lowest_price = 0.0
    
    # Główna pętla
    for i in range(timeframe, len(main_prices)):
        current_time = times[i]
        current_price = main_prices[i]
        
        # Analiza pozycji
        j = 0
        while j < position_count:
            if positions[j, 0] == 0:
                j += 1
                continue
                
            entry_price = positions[j, 0]
            entry_time = positions[j, 1]
            profit_pct = ((current_price - entry_price) / entry_price) * 100.0
            should_close = False
            close_reason = ""
            
            # Stop Loss
            if params[8] > 0 and profit_pct <= -abs(params[9]):
                if positions[j, 4] == 0:
                    positions[j, 4] = current_time
                elif (current_time - positions[j, 4]) >= params[10]:
                    should_close = True
                    close_reason = "Stop Loss"
            
            # Trailing stop lub zwykły sell (wzajemnie się wykluczają)
            if not should_close:
                if params[4] > 0:  # jeśli trailing_enabled
                    if profit_pct >= params[5]:  # trailing_stop_price
                        if positions[j, 2] == 0:
                            positions[j, 2] = current_time
                            positions[j, 3] = current_price
                        else:
                            if (current_time - positions[j, 2]) >= params[7]:
                                if current_price > positions[j, 3]:
                                    positions[j, 3] = current_price
                                trailing_stop_price = positions[j, 3] * (1 - params[6]/100)
                                if current_price <= trailing_stop_price:
                                    should_close = True
                                    close_reason = "Trailing Stop"
                elif params[3] > 0:  # jeśli nie trailing to sprawdź zwykły sell
                    if profit_pct >= params[3]:  # sell_profit_target
                        should_close = True
                        close_reason = "Take Profit"
            
            if should_close:
                # Pobierz czas i timestamp z DataFrame dla tej pozycji
                entry_index = np.where(times == entry_time)[0][0]
                exit_index = i
                
                entry_timestamp = df.iloc[entry_index]['timestamp']
                exit_timestamp = df.iloc[exit_index]['timestamp']
                
                trade = {
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'entry_index': entry_index,
                    'entry_timestamp': entry_timestamp,
                    'exit_time': current_time,
                    'exit_price': current_price,
                    'exit_index': exit_index,
                    'exit_timestamp': exit_timestamp,
                    'profit_pct': profit_pct,
                    'profit_amount': current_price - entry_price,
                    'close_reason': close_reason
                }
                trades.append(trade)
                
                if params[19] > 0 and profit_pct <= -abs(params[9]):
                    stop_loss_price = current_price
                    blocking_times[2] = current_time + int(params[22])
                    
                    if params[20] > 0:
                        blocking_times[0] = current_time + int(params[22])
                
                if j < position_count - 1:
                    positions[j] = positions[position_count - 1]
                positions[position_count - 1] = np.zeros(5)
                position_count -= 1
            else:
                j += 1
        
        # Sprawdzenie warunków nowego zakupu
        if position_count >= max_positions:
            continue
            
        # Sprawdź blokady czasowe
        if current_time <= blocking_times[0] or current_time <= blocking_times[2]:
            trailing_buy_active = False
            continue
        
        # Sprawdź czas od ostatniego zakupu
        if blocking_times[3] > 0 and (current_time - blocking_times[3]) < params[12]:
            continue
        
        # Sprawdź cenę po stop loss
        if stop_loss_price > 0:
            required_price = stop_loss_price * (1 - params[21]/100)
            if current_price > required_price:
                continue
        
        # Follow BTC Price
        if params[16] > 0:
            btc_change = btc_changes[i]
            main_change = main_changes[i]
            if btc_change < 0 and abs(btc_change) > abs(main_change):
                continue
        
        # Sprawdź aktywne pozycje dla next_buy_price_lower
        if position_count > 0:
            min_entry_price = np.inf
            for j in range(position_count):
                if positions[j, 0] > 0:
                    min_entry_price = min(min_entry_price, positions[j, 0])
            
            required_price = min_entry_price * (1 - params[13]/100)
            if current_price > required_price:
                continue
        
        # Pump Detection
        if params[14] > 0 and main_changes[i] >= params[15]:
            trailing_buy_active = False  # Reset trailing buy przy pump detection
            continue

        # Logika zakupu z trailing buy
        buy_executed = False
        
        if trailing_buy_enabled:
            if not trailing_buy_active:
                if main_changes[i] <= params[1]:
                    trailing_buy_active = True
                    trailing_buy_start_price = current_price
                    trailing_buy_start_time = current_time
                    trailing_buy_lowest_price = current_price
            else:
                trailing_buy_lowest_price = min(trailing_buy_lowest_price, current_price)
                
                if current_price > trailing_buy_start_price:
                    trailing_buy_active = False
                    continue
                    
                if current_time - trailing_buy_start_time > trailing_buy_window:
                    trailing_buy_start_price = current_price
                    trailing_buy_start_time = current_time
                    trailing_buy_lowest_price = current_price
                    continue
                
                price_bounce = ((current_price - trailing_buy_lowest_price) / trailing_buy_lowest_price) * 100.0
                if price_bounce >= trailing_buy_threshold:
                    positions[position_count] = np.array([current_price, current_time, 0, 0, 0])
                    position_count += 1
                    blocking_times[3] = current_time
                    buy_executed = True
                    trailing_buy_active = False
        
        # Standardowa logika zakupu
        elif not trailing_buy_enabled and main_changes[i] <= params[1]:
            positions[position_count] = np.array([current_price, current_time, 0, 0, 0])
            position_count += 1
            blocking_times[3] = current_time
            buy_executed = True
    
    # Zamknij pozostałe pozycje
    if position_count > 0:
        for j in range(position_count):
            if positions[j, 0] > 0:
                entry_price = positions[j, 0]
                entry_time = positions[j, 1]
                current_price = main_prices[-1]
                current_time = times[-1]
                profit_pct = ((current_price - entry_price) / entry_price) * 100.0
                
                # Pobierz czas i timestamp z DataFrame dla tej pozycji
                entry_index = np.where(times == entry_time)[0][0]
                exit_index = len(main_prices) - 1
                
                entry_timestamp = df.iloc[entry_index]['timestamp']
                exit_timestamp = df.iloc[exit_index]['timestamp']
                
                trade = {
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'entry_index': entry_index,
                    'entry_timestamp': entry_timestamp,
                    'exit_time': current_time,
                    'exit_price': current_price,
                    'exit_index': exit_index,
                    'exit_timestamp': exit_timestamp,
                    'profit_pct': profit_pct,
                    'profit_amount': current_price - entry_price,
                    'close_reason': "End of Data"
                }
                trades.append(trade)
    
    return trades

def generate_chart(df: pd.DataFrame, trades: List[Dict], output_dir: Path) -> str:
    """
    Generuje wykres z oznaczonymi punktami wejścia i wyjścia.
    
    Args:
        df: DataFrame z danymi rynkowymi
        trades: Lista transakcji
        output_dir: Katalog wyjściowy
        
    Returns:
        str: Ścieżka do wygenerowanego wykresu
    """
    plt.figure(figsize=(16, 8))
    
    # Konwersja timestampów na daty
    dates = pd.to_datetime(df['timestamp'])
    
    # Wykres ceny
    plt.plot(dates, df['average_price'], label='Cena', color='blue')
    
    # Oznaczenie punktów wejścia
    for trade in trades:
        entry_idx = trade['entry_index']
        exit_idx = trade['exit_index']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        
        # Punkt wejścia (zielony trójkąt)
        plt.scatter(dates[entry_idx], entry_price, color='green', marker='^', s=100)
        
        # Punkt wyjścia (czerwony trójkąt w dół dla straty, zielony dla zysku)
        marker_color = 'red' if trade['profit_pct'] < 0 else 'green'
        plt.scatter(dates[exit_idx], exit_price, color=marker_color, marker='v', s=100)
        
        # Linia łącząca wejście z wyjściem
        plt.plot([dates[entry_idx], dates[exit_idx]], [entry_price, exit_price], 
                color='gray', linestyle='--', alpha=0.6)
    
    # Formatowanie wykresu
    plt.title(f"Wykres ceny {df['main_symbol'].iloc[0]} z oznaczonymi transakcjami")
    plt.xlabel('Data')
    plt.ylabel('Cena')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Formatowanie osi X (daty)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Zapisanie wykresu
    chart_path = output_dir / f"chart_{df['main_symbol'].iloc[0]}.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(chart_path)

def generate_transaction_table(trades: List[Dict], output_dir: Path) -> str:
    """
    Generuje tabelę z szczegółami transakcji w formacie Markdown.
    
    Args:
        trades: Lista transakcji
        output_dir: Katalog wyjściowy
        
    Returns:
        str: Ścieżka do wygenerowanej tabeli
    """
    # Utworzenie nagłówka tabeli Markdown
    table_content = "# Szczegóły transakcji\n\n"
    table_content += "| Lp. | Data zakupu | Cena zakupu | Data sprzedaży | Cena sprzedaży | Zysk % | Zysk kwotowy | Powód zamknięcia |\n"
    table_content += "|-----|-------------|-------------|---------------|----------------|--------|--------------|------------------|\n"
    
    # Dodanie wierszy dla każdej transakcji
    for i, trade in enumerate(trades, 1):
        entry_date = pd.to_datetime(trade['entry_timestamp']).strftime('%Y-%m-%d %H:%M')
        exit_date = pd.to_datetime(trade['exit_timestamp']).strftime('%Y-%m-%d %H:%M')
        
        table_content += f"| {i} | {entry_date} | {trade['entry_price']:.6f} | {exit_date} | {trade['exit_price']:.6f} | "
        table_content += f"{trade['profit_pct']:.2f}% | {trade['profit_amount']:.6f} | {trade['close_reason']} |\n"
    
    # Dodanie podsumowania
    if trades:
        profits = [t['profit_pct'] for t in trades]
        win_trades = sum(1 for p in profits if p > 0)
        lose_trades = sum(1 for p in profits if p <= 0)
        
        table_content += "\n## Podsumowanie\n\n"
        table_content += f"* **Liczba transakcji:** {len(trades)}\n"
        table_content += f"* **Zyskowne transakcje:** {win_trades}\n"
        table_content += f"* **Stratne transakcje:** {lose_trades}\n"
        table_content += f"* **Win rate:** {(win_trades/len(trades)*100):.2f}%\n"
        table_content += f"* **Średni zysk:** {np.mean(profits):.2f}%\n"
        table_content += f"* **Mediana zysku:** {np.median(profits):.2f}%\n"
        table_content += f"* **Maksymalny zysk:** {max(profits):.2f}%\n"
        table_content += f"* **Maksymalna strata:** {min(profits):.2f}%\n"
    
    # Zapisanie tabeli do pliku
    table_path = output_dir / "transactions.md"
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(table_content)
    
    return str(table_path)

def visualize_strategy(df: pd.DataFrame, params_obj: Any, params_dict: Dict, output_dir: Optional[Path] = None) -> Dict:
    """
    Główna funkcja do wizualizacji strategii.
    
    Args:
        df: DataFrame z danymi rynkowymi
        params_obj: Obiekt parametrów
        params_dict: Słownik z parametrami
        output_dir: Katalog wyjściowy (opcjonalnie)
        
    Returns:
        Dict: Wyniki wizualizacji
    """
    # Ustalenie katalogu wyjściowego
    if output_dir is None:
        output_dir = get_visualization_output_dir()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Rozpoczynam wizualizację strategii. Wyniki będą zapisane w: {output_dir}")
    
    # Przygotowanie danych rynkowych
    df['minutes'] = pd.to_datetime(df['timestamp']).astype(np.int64) // 60e9
    
    # Uruchomienie strategii z pełnymi danymi o transakcjach
    params_array = params_obj.to_array()
    trades = run_strategy_with_details(
        df['average_price'].to_numpy(dtype=np.float32), 
        df['btc_average_price'].to_numpy(dtype=np.float32), 
        df['minutes'].to_numpy(dtype=np.int64),
        df,
        params_array
    )
    
    logger.info(f"Strategia zakończona. Znaleziono {len(trades)} transakcji.")
    
    # Generowanie wykresu
    chart_path = generate_chart(df, trades, output_dir)
    logger.info(f"Wygenerowano wykres: {chart_path}")
    
    # Generowanie tabeli transakcji
    table_path = generate_transaction_table(trades, output_dir)
    logger.info(f"Wygenerowano tabelę transakcji: {table_path}")
    
    # Podsumowanie
    summary = {}
    if trades:
        profits = [t['profit_pct'] for t in trades]
        win_trades = sum(1 for p in profits if p > 0)
        lose_trades = sum(1 for p in profits if p <= 0)
        
        summary = {
            'total_trades': len(trades),
            'win_trades': win_trades,
            'lose_trades': lose_trades,
            'win_rate': (win_trades/len(trades)*100) if trades else 0,
            'avg_profit': np.mean(profits) if trades else 0,
            'max_profit': max(profits) if trades else 0,
            'max_loss': min(profits) if trades else 0
        }
    
    return {
        "trades": trades,
        "chart_path": chart_path,
        "table_path": table_path,
        "summary": summary,
        "output_dir": str(output_dir),
        "success": True
    }