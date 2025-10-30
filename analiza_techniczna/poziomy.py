# analiza_techniczna/poziomy.py
import numpy as np
import pandas as pd
from scipy import signal

def find_support_resistance(data, column='close', window=20, sensitivity=0.05, method='peaks'):
    """
    Wykrywa poziomy wsparcia i oporu.
    
    Args:
        data: DataFrame lub Series z danymi cenowymi
        column: Nazwa kolumny z ceną zamknięcia (tylko gdy data to DataFrame)
        window: Okres analizy
        sensitivity: Czułość wykrywania (wyższa = więcej poziomów)
        method: Metoda wykrywania ('peaks', 'fractals')
        
    Returns:
        tuple: (poziomy wsparcia, poziomy oporu)
    """
    series = data[column] if isinstance(data, pd.DataFrame) and column else data
    
    if method == 'peaks':
        # Metoda oparta na wykrywaniu ekstremów lokalnych
        # Normalizacja danych dla sensitivity
        min_val, max_val = series.min(), series.max()
        price_range = max_val - min_val
        peak_height = sensitivity * price_range
        
        # Znalezienie szczytów (opory)
        peaks, _ = signal.find_peaks(series, height=None, distance=window//2, prominence=peak_height)
        resistance_levels = series.iloc[peaks].tolist()
        
        # Znalezienie dołków (wsparcia) - odwrócenie serii
        inverted_series = -series
        troughs, _ = signal.find_peaks(inverted_series, height=None, distance=window//2, prominence=peak_height)
        support_levels = series.iloc[troughs].tolist()
        
    elif method == 'fractals':
        # Metoda oparta na fraktalach Williama
        support_levels = []
        resistance_levels = []
        
        if len(series) > 4:  # Potrzebujemy co najmniej 5 punktów dla fraktala
            # Użycie wektoryzacji dla przyspieszenia
            highs = np.zeros(len(series), dtype=bool)
            lows = np.zeros(len(series), dtype=bool)
            
            # Sprawdzenie fraktali wzrostowych i spadkowych
            for i in range(2, len(series) - 2):
                # Fraktal wzrostowy (opór)
                highs[i] = (series.iloc[i] > series.iloc[i-1] and 
                          series.iloc[i] > series.iloc[i-2] and 
                          series.iloc[i] > series.iloc[i+1] and 
                          series.iloc[i] > series.iloc[i+2])
                
                # Fraktal spadkowy (wsparcie)
                lows[i] = (series.iloc[i] < series.iloc[i-1] and 
                         series.iloc[i] < series.iloc[i-2] and 
                         series.iloc[i] < series.iloc[i+1] and 
                         series.iloc[i] < series.iloc[i+2])
            
            # Pobranie wartości na podstawie masek
            resistance_levels = series.iloc[highs].tolist()
            support_levels = series.iloc[lows].tolist()
    
    return support_levels, resistance_levels