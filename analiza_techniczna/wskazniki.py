# analiza_techniczna/wskazniki.py
import numpy as np
import pandas as pd
import talib

def calculate_rsi(data, column='close', period=14):
    """
    Oblicza wskaźnik względnej siły (Relative Strength Index).
    
    Args:
        data: DataFrame lub Series z danymi cenowymi
        column: Nazwa kolumny z ceną zamknięcia (tylko gdy data to DataFrame)
        period: Okres RSI
        
    Returns:
        pd.Series: Wartości RSI
    """
    series = data[column] if isinstance(data, pd.DataFrame) and column else data
    return pd.Series(talib.RSI(series.values, timeperiod=period), index=series.index)

def calculate_ema(data, column='close', period=20):
    """
    Oblicza wykładniczą średnią kroczącą (Exponential Moving Average).
    
    Args:
        data: DataFrame lub Series z danymi cenowymi
        column: Nazwa kolumny z ceną zamknięcia (tylko gdy data to DataFrame)
        period: Okres EMA
        
    Returns:
        pd.Series: Wartości EMA
    """
    series = data[column] if isinstance(data, pd.DataFrame) and column else data
    return pd.Series(talib.EMA(series.values, timeperiod=period), index=series.index)

def calculate_obv(data, close_col='close', volume_col='volume'):
    """
    Oblicza On-Balance Volume.
    
    Args:
        data: DataFrame z danymi cenowymi i wolumenem
        close_col: Nazwa kolumny z ceną zamknięcia
        volume_col: Nazwa kolumny z wolumenem
        
    Returns:
        pd.Series: Wartości OBV
    """
    # Sprawdzenie istnienia wymaganych kolumn
    required_cols = [close_col, volume_col]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Brak wymaganej kolumny: {col}")
    
    # Obliczenie OBV
    obv = talib.OBV(data[close_col].values, data[volume_col].values)
    return pd.Series(obv, index=data.index)