# analiza_techniczna/trendy.py
import numpy as np
import pandas as pd
import talib

def detect_trend(data, column='close', period=20, method='ema'):
    """
    Wykrywa kierunek trendu (rosnący/malejący/boczny).
    
    Args:
        data: DataFrame lub Series z danymi cenowymi
        column: Nazwa kolumny z ceną zamknięcia (tylko gdy data to DataFrame)
        period: Okres analizy
        method: Metoda wykrywania ('sma', 'ema', 'linear')
        
    Returns:
        pd.Series: Trend jako wartości (-1 = spadkowy, 0 = boczny, 1 = wzrostowy)
    """
    series = data[column] if isinstance(data, pd.DataFrame) and column else data
    
    # Inicjalizacja serii wynikowej
    trend = pd.Series(0, index=series.index)
    
    if method == 'sma':
        # Metoda oparta na SMA
        ma = pd.Series(talib.SMA(series.values, timeperiod=period), index=series.index)
        slope = ma.diff()
        threshold = 0.01 * ma.shift(1)
        
        # Określenie trendu
        trend[slope > threshold] = 1  # Wzrostowy
        trend[slope < -threshold] = -1  # Spadkowy
        
    elif method == 'ema':
        # Metoda oparta na EMA
        ma = pd.Series(talib.EMA(series.values, timeperiod=period), index=series.index)
        slope = ma.diff()
        threshold = 0.01 * ma.shift(1)
        
        # Określenie trendu
        trend[slope > threshold] = 1  # Wzrostowy
        trend[slope < -threshold] = -1  # Spadkowy
    
    return trend

def measure_trend_strength(data, column='close', period=14, method='linear'):
    """
    Mierzy siłę trendu.
    
    Args:
        data: DataFrame lub Series z danymi cenowymi
        column: Nazwa kolumny z ceną zamknięcia (tylko gdy data to DataFrame)
        period: Okres analizy
        method: Metoda pomiaru ('linear', 'volatility')
        
    Returns:
        pd.Series: Siła trendu jako wartość od 0 do 100
    """
    series = data[column] if isinstance(data, pd.DataFrame) and column else data
    
    if method == 'linear':
        # Metoda oparta na R² regresji liniowej
        window_count = min(len(series), period)
        r_squared = pd.Series(np.zeros(len(series)), index=series.index)
        
        # Tylko jeśli mamy wystarczająco danych
        if len(series) >= period:
            try:
                # Przygotowanie rolling window
                rolling_data = np.array([
                    series.values[i:i+window_count] 
                    for i in range(len(series) - window_count + 1)
                ])
                x = np.arange(window_count)
                
                # Obliczenie R² dla każdego okna
                from scipy.stats import linregress
                
                r_squared_values = []
                for window_data in rolling_data:
                    try:
                        # Sprawdzenie czy dane w oknie są stałe
                        if np.std(window_data) == 0:
                            r_squared_values.append(0)
                        else:
                            r_value = linregress(x, window_data)[2]  # indeks 2 to r_value
                            r_squared_values.append(r_value ** 2 * 100)
                    except Exception:
                        # W przypadku błędu, użyj wartości 0
                        r_squared_values.append(0)
                
                # Konwersja do tablicy numpy
                r_squared_values = np.array(r_squared_values)
                
                # Przypisanie wartości do serii wynikowej
                if len(r_squared_values) > 0:
                    if period + len(r_squared_values) <= len(r_squared):
                        r_squared.iloc[period:period+len(r_squared_values)] = r_squared_values
                    else:
                        # Przycinamy r_squared_values, aby pasowało do dostępnej długości
                        max_length = len(r_squared) - period
                        if max_length > 0:
                            r_squared.iloc[period:len(r_squared)] = r_squared_values[:max_length]
            except Exception as e:
                print(f"Błąd podczas obliczania siły trendu: {e}")
                # Nie modyfikujemy r_squared, zostawiając zera
        
        return r_squared
    
    elif method == 'volatility':
        # Metoda oparta na stosunku kierunkowej zmienności do całkowitej zmienności
        window_count = min(len(series), period)
        directional_vol = pd.Series(np.zeros(len(series)), index=series.index)
        
        if len(series) >= period:
            # Obliczenie zmiany netto dla każdego okna
            net_changes = np.abs(series.values[period:] - series.values[:-period])
            
            # Obliczenie całkowitej zmienności dla każdego okna
            diffs = np.abs(series.diff().values)
            # Suma absolutnych zmian w każdym oknie
            total_volatility = np.array([
                np.sum(diffs[i+1:i+period+1])  # +1 bo diff wprowadza NaN na początku
                for i in range(len(series) - period)
            ])
            
            # Unikanie dzielenia przez zero
            total_volatility = np.maximum(total_volatility, np.finfo(float).eps)
            
            # Stosunek zmiany netto do całkowitej zmienności
            directional_strength = (net_changes / total_volatility) * 100
            directional_vol.iloc[period:] = directional_strength
        
        return directional_vol
    
    return pd.Series(np.zeros(len(series)), index=series.index)