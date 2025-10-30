"""
Moduł przetwarzania danych analizy technicznej.
Zawiera funkcje do przetwarzania, analizy i modyfikacji danych rynkowych.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from analiza_techniczna.wskazniki import calculate_rsi, calculate_ema, calculate_obv
from analiza_techniczna.trendy import detect_trend, measure_trend_strength
from analiza_techniczna.poziomy import find_support_resistance


def prepare_hourly_data(orchestrator) -> Optional[pd.DataFrame]:
    """
    Przygotowuje dane godzinowe na podstawie danych historycznych.
    Jeśli dane są już w interwale godzinowym lub dłuższym, używa ich bezpośrednio.
    
    Args:
        orchestrator: Instancja ATOrchestrator
        
    Returns:
        Optional[pd.DataFrame]: Dane godzinowe lub None jeśli nie można ich przygotować
    """
    if orchestrator.historical_data is None or len(orchestrator.historical_data) == 0:
        orchestrator.logger.warning("Brak danych historycznych do przygotowania danych godzinowych")
        return None
    
    # Sprawdzenie czy kolumna timestamp jest w odpowiednim formacie
    if 'timestamp' not in orchestrator.historical_data.columns:
        orchestrator.logger.warning("Brak kolumny timestamp w danych historycznych")
        orchestrator.use_hourly_data = False
        return None
    
    # Konwersja timestamp na format datetime jeśli nie jest
    if not pd.api.types.is_datetime64_any_dtype(orchestrator.historical_data['timestamp']):
        try:
            orchestrator.historical_data['timestamp'] = pd.to_datetime(orchestrator.historical_data['timestamp'])
        except Exception as e:
            orchestrator.logger.error(f"Błąd przy konwersji timestamp na datetime: {e}")
            orchestrator.use_hourly_data = False
            return None
    
    # Sprawdzenie interwału danych
    try:
        # Sortowanie danych chronologicznie
        sorted_data = orchestrator.historical_data.sort_values('timestamp')
        
        # Obliczenie interwału czasowego w sekundach
        time_diffs = sorted_data['timestamp'].diff()[1:].dt.total_seconds()
        median_interval = time_diffs.median()
        
        # Jeśli interwał jest już godzinowy lub większy, użyj oryginalnych danych
        if median_interval >= 3600:  # 3600 sekund = 1 godzina
            orchestrator.logger.info("Dane są już w interwale godzinowym lub większym, używanie oryginalnych danych")
            return orchestrator.historical_data.copy()
        
        orchestrator.logger.info(f"Wykryto interwał danych: {median_interval} sekund, grupowanie do interwału godzinowego")
    except Exception as e:
        orchestrator.logger.error(f"Błąd przy określaniu interwału danych: {e}")
        orchestrator.use_hourly_data = False
        return None
    
    # Grupowanie danych do interwału godzinowego
    try:
        # Ustawienie timestamp jako indeksu
        df_with_index = orchestrator.historical_data.set_index('timestamp')
        
        # Grupowanie danych OHLCV
        hourly_data = df_with_index.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Dodanie kolumn BTC jeśli istnieją
        btc_columns = [col for col in orchestrator.historical_data.columns if col.startswith('btc_')]
        if btc_columns:
            btc_ohlcv = {
                'btc_open': 'first',
                'btc_high': 'max',
                'btc_low': 'min',
                'btc_close': 'last'
            }
            
            # Dodanie btc_volume jeśli istnieje
            if 'btc_volume' in orchestrator.historical_data.columns:
                btc_ohlcv['btc_volume'] = 'sum'
            
            # Grupowanie danych BTC
            btc_hourly = df_with_index.resample('1h').agg({
                col: agg_func for col, agg_func in btc_ohlcv.items() 
                if col in orchestrator.historical_data.columns
            })
            
            # Dodanie kolumn BTC do danych godzinowych
            for col in btc_hourly.columns:
                hourly_data[col] = btc_hourly[col]
        
        # Resetowanie indeksu, aby timestamp był zwykłą kolumną
        hourly_data = hourly_data.reset_index()
        
        # Usunięcie wierszy z brakującymi danymi
        hourly_data = hourly_data.dropna(subset=['open', 'high', 'low', 'close'])
        
        orchestrator.logger.info(f"Przygotowano dane godzinowe: {len(hourly_data)} rekordów")
        return hourly_data
    except Exception as e:
        orchestrator.logger.error(f"Błąd przy tworzeniu danych godzinowych: {e}")
        orchestrator.use_hourly_data = False
        return None


def project_hourly_results_to_original(orchestrator) -> Dict[str, Any]:
    """
    Rzutuje wyniki analizy z danych godzinowych na oryginalne dane.
    
    Args:
        orchestrator: Instancja ATOrchestrator
        
    Returns:
        Dict[str, Any]: Wyniki analizy dla oryginalnych danych
    """
    if orchestrator.historical_data is None or orchestrator.hourly_data is None or not orchestrator.hourly_analysis_results:
        return {}
    
    projected_results = {}
    
    # Przekształcenie timestamp do formatu datetime
    if not pd.api.types.is_datetime64_any_dtype(orchestrator.historical_data['timestamp']):
        historical_data_time = pd.to_datetime(orchestrator.historical_data['timestamp'])
    else:
        historical_data_time = orchestrator.historical_data['timestamp']
    
    if not pd.api.types.is_datetime64_any_dtype(orchestrator.hourly_data['timestamp']):
        hourly_data_time = pd.to_datetime(orchestrator.hourly_data['timestamp'])
    else:
        hourly_data_time = orchestrator.hourly_data['timestamp']
    
    # Przetwarzanie każdego wskaźnika
    for indicator_name, hourly_values in orchestrator.hourly_analysis_results.items():
        if indicator_name in ['support_levels', 'resistance_levels']:
            # Dla poziomów wsparcia/oporu po prostu kopiujemy wartości
            projected_results[indicator_name] = hourly_values
            continue
        
        # Dla wskaźników szeregów czasowych
        if isinstance(hourly_values, pd.Series):
            # Inicjalizacja pustej serii dla oryginalnych danych
            projected_values = pd.Series(index=orchestrator.historical_data.index)
            
            # Mapowanie każdego punktu oryginalnych danych do najbliższej godziny
            for i, timestamp in enumerate(historical_data_time):
                # Zaokrąglenie do pełnej godziny w dół
                hour_timestamp = timestamp.floor('h')
                
                # Znalezienie indeksu tej godziny w danych godzinowych
                hour_indices = np.where(hourly_data_time == hour_timestamp)[0]
                
                if len(hour_indices) > 0:
                    hour_idx = hour_indices[0]
                    if hour_idx < len(hourly_values):
                        projected_values.iloc[i] = hourly_values.iloc[hour_idx]
            
            projected_results[indicator_name] = projected_values.copy()
    
    return projected_results


def analyze_current_conditions(orchestrator, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analizuje bieżące warunki rynkowe dla jednego punktu danych.
    
    Args:
        orchestrator: Instancja ATOrchestrator
        data: DataFrame z jednym punktem danych (aktualnym) i historią
    
    Returns:
        Dict[str, Any]: Warunki rynkowe
    """
    conditions = {}
    
    # Aktualna cena
    current_price = data['close'].iloc[-1]
    conditions['current_price'] = current_price
    
    # Sprawdzenie RSI
    if 'rsi' in orchestrator.analysis_results:
        current_rsi = orchestrator.analysis_results['rsi'].iloc[-1]
        conditions['current_rsi'] = current_rsi
        
        # Sprawdzenie czy RSI jest wykupiony/wyprzedany
        rsi_params = orchestrator.parameters['technical_indicators']['rsi']
        conditions['rsi_overbought'] = current_rsi > rsi_params.get('overbought', 70)
        conditions['rsi_oversold'] = current_rsi < rsi_params.get('oversold', 30)
    
    # Sprawdzenie trendu
    if 'trend' in orchestrator.analysis_results:
        current_trend = orchestrator.analysis_results['trend'].iloc[-1]
        conditions['current_trend'] = current_trend
        
        if 'trend_strength' in orchestrator.analysis_results:
            current_trend_strength = orchestrator.analysis_results['trend_strength'].iloc[-1]
            conditions['current_trend_strength'] = current_trend_strength
    
    # Sprawdzenie poziomów wsparcia/oporu
    if 'support_levels' in orchestrator.analysis_results and 'resistance_levels' in orchestrator.analysis_results:
        support_levels = orchestrator.analysis_results['support_levels']
        resistance_levels = orchestrator.analysis_results['resistance_levels']
        
        # Najbliższy poziom wsparcia
        if support_levels:
            levels_below = [level for level in support_levels if level < current_price]
            if levels_below:
                nearest_support = max(levels_below)
                conditions['nearest_support'] = nearest_support
                conditions['distance_to_support'] = (current_price - nearest_support) / current_price
        
        # Najbliższy poziom oporu
        if resistance_levels:
            levels_above = [level for level in resistance_levels if level > current_price]
            if levels_above:
                nearest_resistance = min(levels_above)
                conditions['nearest_resistance'] = nearest_resistance
                conditions['distance_to_resistance'] = (nearest_resistance - current_price) / current_price
        
        # Czy jesteśmy blisko wsparcia/oporu
        threshold = 0.01  # 1% odległości
        conditions['near_support'] = conditions.get('distance_to_support', 1.0) < threshold
        conditions['near_resistance'] = conditions.get('distance_to_resistance', 1.0) < threshold
    
    return conditions


def analyze_btc_conditions(orchestrator, data: pd.DataFrame, btc_analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analizuje bieżące warunki rynkowe BTC.
    
    Args:
        orchestrator: Instancja ATOrchestrator
        data: DataFrame z danymi historycznymi
        btc_analysis_results: Wyniki analizy BTC
        
    Returns:
        Dict[str, Any]: Warunki rynkowe BTC
    """
    conditions = {}
    
    # Sprawdzenie, czy mamy dane BTC
    if 'btc_close' not in data.columns:
        return conditions
    
    # Aktualna cena BTC
    current_price = data['btc_close'].iloc[-1]
    conditions['current_price'] = current_price
    
    # Sprawdzenie RSI dla BTC
    if 'rsi' in btc_analysis_results:
        current_rsi = btc_analysis_results['rsi'].iloc[-1]
        conditions['current_rsi'] = current_rsi
        
        # Sprawdzenie czy RSI jest wykupiony/wyprzedany
        rsi_params = orchestrator.parameters['technical_indicators']['rsi']
        conditions['rsi_overbought'] = current_rsi > rsi_params.get('overbought', 70)
        conditions['rsi_oversold'] = current_rsi < rsi_params.get('oversold', 30)
    
    # Sprawdzenie trendu BTC
    if 'trend' in btc_analysis_results:
        current_trend = btc_analysis_results['trend'].iloc[-1]
        conditions['current_trend'] = current_trend
        
        if 'trend_strength' in btc_analysis_results:
            current_trend_strength = btc_analysis_results['trend_strength'].iloc[-1]
            conditions['current_trend_strength'] = current_trend_strength
    
    # Sprawdzenie poziomów wsparcia/oporu BTC
    if 'support_levels' in btc_analysis_results and 'resistance_levels' in btc_analysis_results:
        support_levels = btc_analysis_results['support_levels']
        resistance_levels = btc_analysis_results['resistance_levels']
        
        # Najbliższy poziom wsparcia
        if support_levels:
            levels_below = [level for level in support_levels if level < current_price]
            if levels_below:
                nearest_support = max(levels_below)
                conditions['nearest_support'] = nearest_support
                conditions['distance_to_support'] = (current_price - nearest_support) / current_price
        
        # Najbliższy poziom oporu
        if resistance_levels:
            levels_above = [level for level in resistance_levels if level > current_price]
            if levels_above:
                nearest_resistance = min(levels_above)
                conditions['nearest_resistance'] = nearest_resistance
                conditions['distance_to_resistance'] = (nearest_resistance - current_price) / current_price
        
        # Czy jesteśmy blisko wsparcia/oporu
        threshold = 0.01  # 1% odległości
        conditions['near_support'] = conditions.get('distance_to_support', 1.0) < threshold
        conditions['near_resistance'] = conditions.get('distance_to_resistance', 1.0) < threshold
    
    # Obliczenie ogólnego biasu BTC na podstawie wskaźników
    btc_bias = 0.0
    bias_count = 0
    
    # RSI
    if 'current_rsi' in conditions:
        if conditions.get('rsi_overbought', False):
            btc_bias -= 0.5  # Efekt negatywny dla wykupionego RSI
        elif conditions.get('rsi_oversold', False):
            btc_bias += 0.5  # Efekt pozytywny dla wyprzedanego RSI
        bias_count += 1
    
    # Trend
    if 'current_trend' in conditions:
        trend = conditions['current_trend']
        if trend != 0:  # Jeśli nie jest trendem bocznym
            btc_bias += 0.3 * trend  # Wpływ trendu
        bias_count += 1
    
    # Wsparcie/opór
    if conditions.get('near_support', False):
        btc_bias += 0.4  # Efekt pozytywny dla bliskości wsparcia
        bias_count += 1
    
    if conditions.get('near_resistance', False):
        btc_bias -= 0.4  # Efekt negatywny dla bliskości oporu
        bias_count += 1
    
    # Uśrednienie biasu
    if bias_count > 0:
        btc_bias /= bias_count
    
    conditions['btc_bias'] = btc_bias
    
    return conditions


def calculate_mean_adjustment(conditions: Dict[str, Any], price_reactions: Dict[str, float]) -> float:
    """
    Oblicza modyfikację średniej zwrotu na podstawie warunków rynkowych.
    
    Args:
        conditions: Słownik z warunkami rynkowymi
        price_reactions: Parametry reakcji ceny na wskaźniki
        
    Returns:
        float: Modyfikacja średniej zwrotu
    """
    mean_adjustment = 0.0
    
    # RSI
    if conditions.get('rsi_overbought', False):
        effect = price_reactions.get('rsi_overbought_effect', -0.5)
        mean_adjustment += effect
    
    if conditions.get('rsi_oversold', False):
        effect = price_reactions.get('rsi_oversold_effect', 0.5)
        mean_adjustment += effect
    
    # Trend
    if 'current_trend' in conditions:
        trend = conditions['current_trend']
        if trend != 0:  # Jeśli nie jest trendem bocznym
            effect = price_reactions.get('trend_following_effect', 0.3) * trend
            mean_adjustment += effect
    
    # Wsparcie/opór
    if conditions.get('near_support', False):
        effect = price_reactions.get('support_bounce_effect', 0.4)
        mean_adjustment += effect
    
    if conditions.get('near_resistance', False):
        effect = price_reactions.get('resistance_bounce_effect', -0.4)
        mean_adjustment += effect
    
    return mean_adjustment


def calculate_volatility_adjustment(conditions: Dict[str, Any], volatility_params: Dict[str, float]) -> float:
    """
    Oblicza modyfikację zmienności na podstawie warunków rynkowych.
    
    Args:
        conditions: Słownik z warunkami rynkowymi
        volatility_params: Parametry modyfikacji zmienności
        
    Returns:
        float: Modyfikacja zmienności
    """
    volatility_adjustment = 0.0
    
    # Siła trendu
    if 'current_trend_strength' in conditions:
        strength = conditions['current_trend_strength']
        effect = volatility_params.get('trend_strength_effect', -0.2) * strength / 100
        volatility_adjustment += effect
    
    # Bliskość wsparcia/oporu
    if conditions.get('near_support', False) or conditions.get('near_resistance', False):
        effect = volatility_params.get('near_support_resistance_effect', 0.3)
        volatility_adjustment += effect
    
    return volatility_adjustment


# Znajdź funkcję modify_simulated_data i zastąp ją tą zmodyfikowaną wersją:

def modify_simulated_data(orchestrator, simulated_data: pd.DataFrame) -> pd.DataFrame:
    """
    Modyfikuje symulowane dane na podstawie analizy technicznej z wykorzystaniem
    okien czasowych i osłabiającego się biasu.
    
    Args:
        orchestrator: Instancja ATOrchestrator
        simulated_data: DataFrame z symulowanymi danymi
        
    Returns:
        pd.DataFrame: Zmodyfikowane dane
    """
    # Bezpośredni wydruk na konsolę - nie przez system logowania
    print("\n" + "!" * 80)
    print("!!!!!!!!!! ROZPOCZYNANIE MODYFIKACJI PRZEZ ANALIZĘ TECHNICZNĄ !!!!!!!!!!")
    print("!" * 80)
    
    # Kopia danych do modyfikacji
    modified_data = simulated_data.copy()
    
    # Połączenie danych historycznych i symulowanych dla ciągłości analizy
    # Używamy tylko ostatnich N punktów danych historycznych
    history_points = 100  # Liczba punktów historycznych do użycia
    combined_data = pd.concat([
        orchestrator.historical_data.iloc[-history_points:].reset_index(drop=True),
        modified_data.reset_index(drop=True)
    ]).reset_index(drop=True)
    
    # Inicjalizacja modyfikacji dla każdego punktu danych
    mean_adjustments = np.zeros(len(modified_data))
    volatility_adjustments = np.zeros(len(modified_data))

    # Śledzenie skumulowanego odchylenia
    cumulative_deviation = np.zeros(len(modified_data))
    # Maksymalne dozwolone skumulowane odchylenie (%) - ZWIĘKSZONE z 0.2 do 0.6
    # max_cumulative_deviation = 0.6  # Poziom 3
    # max_cumulative_deviation = 1.2  # Poziom 6-7
    max_cumulative_deviation = 2.0  # Poziom 10 (oryginalny)
    # Próg, przy którym zaczynamy tłumić wpływ AT - ZWIĘKSZONE z 0.1 do 0.3
    # damping_threshold = 0.3  # Poziom 3
    # damping_threshold = 0.6  # Poziom 6-7
    damping_threshold = 1.0  # Poziom 10 (oryginalny)

    # Sprawdzenie czy mamy dane BTC i analizę dla BTC
    has_btc = 'btc_close' in modified_data.columns and 'btc_close' in orchestrator.historical_data.columns
    btc_analysis_results = {}
    
    # Jeśli mamy dane BTC, wykonaj analizę techniczną dla BTC
    if has_btc:
        orchestrator.logger.info("Wykryto dane BTC - wykonywanie analizy technicznej dla BTC")
        
        # Przygotowanie danych BTC (konwersja ze struktury btc_X na strukturę główną)
        btc_data = orchestrator.historical_data.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if f'btc_{col}' in btc_data.columns:
                btc_data[col] = btc_data[f'btc_{col}']
        
        # Analiza techniczna dla BTC
        btc_indicators = orchestrator.parameters['technical_indicators']
        
        # RSI dla BTC
        if btc_indicators['rsi']['enabled']:
            period = btc_indicators['rsi']['period']
            btc_analysis_results['rsi'] = calculate_rsi(btc_data, 'close', period)
            orchestrator.logger.info(f"Obliczono RSI dla BTC z okresem {period}")
        
        # EMA dla BTC
        if btc_indicators['ema']['enabled']:
            period = btc_indicators['ema']['period']
            btc_analysis_results['ema'] = calculate_ema(btc_data, 'close', period)
            orchestrator.logger.info(f"Obliczono EMA dla BTC z okresem {period}")
        
        # Wykrywanie trendu dla BTC
        if orchestrator.parameters['trend_detection']['enabled']:
            method = orchestrator.parameters['trend_detection']['method']
            period = orchestrator.parameters['trend_detection']['period']
            btc_analysis_results['trend'] = detect_trend(btc_data, 'close', period, method)
            btc_analysis_results['trend_strength'] = measure_trend_strength(btc_data, 'close', period)
            orchestrator.logger.info(f"Wykryto trend BTC metodą {method} z okresem {period}")
        
        # Poziomy wsparcia/oporu dla BTC
        if orchestrator.parameters['support_resistance']['enabled']:
            method = orchestrator.parameters['support_resistance']['method']
            sensitivity = orchestrator.parameters['support_resistance']['sensitivity']
            support_levels, resistance_levels = find_support_resistance(
                btc_data, 'close', 20, sensitivity, method
            )
            btc_analysis_results['support_levels'] = support_levels
            btc_analysis_results['resistance_levels'] = resistance_levels
            orchestrator.logger.info(f"Wykryto poziomy wsparcia/oporu BTC metodą {method}")
    
    # Parametry okna czasowego dla analizy technicznej
    window_hours = 12  # Długość okna w godzinach
    minutes_per_hour = 60  # Liczba minut w godzinie (dla danych minutowych)
    window_size = window_hours * minutes_per_hour  # Rozmiar okna w interwałach danych
    
    # Zaawansowane opcje wpływu analizy technicznej
    adv_options = orchestrator.parameters.get('advanced_options', {})
    price_reactions = adv_options.get('price_reaction_to_indicators', {})
    volatility_adjustments_params = adv_options.get('volatility_adjustment', {})
    
    # Wzmocnienie wagi RSI dla lepszej cykliczności - ZWIĘKSZONE z 1.5 na 2.0
    # rsi_weight_multiplier = 2.0  # Poziom 3
    # rsi_weight_multiplier = 2.5  # Poziom 6-7
    rsi_weight_multiplier = 3.0  # Poziom 10 (oryginalny)
    if 'rsi' in price_reactions:
        if 'rsi_overbought_effect' in price_reactions:
            price_reactions['rsi_overbought_effect'] *= rsi_weight_multiplier
        if 'rsi_oversold_effect' in price_reactions:
            price_reactions['rsi_oversold_effect'] *= rsi_weight_multiplier
    
    # Przetwarzanie danych w oknach czasowych
    for window_start_idx in range(0, len(modified_data), window_size):
        window_end_idx = min(window_start_idx + window_size, len(modified_data))
        orchestrator.logger.info(f"Przetwarzanie okna czasowego {window_start_idx//window_size + 1}: punkty {window_start_idx} do {window_end_idx}")
        
        # Dane historyczne do początku okna
        history_to_window = combined_data.iloc[:history_points + window_start_idx].copy()
        
        # Analiza warunków rynkowych na początku okna
        conditions = analyze_current_conditions(orchestrator, history_to_window)
        
        # Analiza BTC jeśli dostępna
        btc_conditions = {}
        if has_btc:
            btc_conditions = analyze_btc_conditions(orchestrator, history_to_window, btc_analysis_results)
            
            # Konfiguracja wpływu BTC
            btc_influence = orchestrator.parameters.get('btc_influence', {})
            btc_weight = btc_influence.get('weight', 30) / 100.0
            correlation_threshold = btc_influence.get('correlation_threshold', 0.5)
            use_dynamic_correlation = btc_influence.get('use_dynamic_correlation', False)

            # Obliczenie korelacji między tokenem głównym a BTC
            correlation = history_to_window['close'].corr(history_to_window['btc_close'])
            orchestrator.logger.info(f"Korelacja między głównym tokenem a BTC: {correlation:.4f}")

            # Określenie wpływu BTC - dynamiczna korelacja lub próg
            if use_dynamic_correlation:
                # Dynamiczna korelacja - wpływ proporcjonalny do siły korelacji
                correlation_factor = abs(correlation)
                orchestrator.logger.info(f"Używam dynamicznej korelacji: {correlation_factor:.4f}")
                # Dostosowanie wagi BTC na podstawie siły korelacji
                effective_btc_weight = btc_weight * correlation_factor
            else:
                # Tradycyjny próg korelacji
                correlation_factor = 1.0 if abs(correlation) >= correlation_threshold else 0.0
                orchestrator.logger.info(f"Korelacja {'przekracza' if correlation_factor > 0 else 'nie przekracza'} próg {correlation_threshold}")
                effective_btc_weight = btc_weight * correlation_factor
        
        # Początkowe modyfikacje dla okna
        initial_mean_adjustment = calculate_mean_adjustment(conditions, price_reactions)
        initial_volatility_adjustment = calculate_volatility_adjustment(conditions, volatility_adjustments_params)
        
        # Dodanie wpływu BTC jeśli dostępny
        if has_btc and 'btc_bias' in btc_conditions and effective_btc_weight > 0:
            btc_bias = btc_conditions['btc_bias']
            orchestrator.logger.info(f"Wpływ BTC na początkowe modyfikacje: bias={btc_bias:.4f}, waga={effective_btc_weight:.2f}")
            
            if correlation > 0:  # Dodatnia korelacja
                initial_mean_adjustment += btc_bias * effective_btc_weight
            else:  # Ujemna korelacja
                initial_mean_adjustment -= btc_bias * effective_btc_weight
        
        # Ograniczenie początkowej modyfikacji - ZWIĘKSZONE limity
        # initial_mean_adjustment = np.clip(initial_mean_adjustment, -1.0, 1.0)  # Poziom 3
        # initial_mean_adjustment = np.clip(initial_mean_adjustment, -1.5, 1.5)  # Poziom 6-7
        initial_mean_adjustment = np.clip(initial_mean_adjustment, -2.0, 2.0)  # Poziom 10 (oryginalny)
        # initial_volatility_adjustment = np.clip(initial_volatility_adjustment, -0.5, 0.8)  # Poziom 3
        # initial_volatility_adjustment = np.clip(initial_volatility_adjustment, -0.8, 1.2)  # Poziom 6-7
        initial_volatility_adjustment = np.clip(initial_volatility_adjustment, -1.0, 1.5)  # Poziom 10 (oryginalny)
        
        orchestrator.logger.info(f"Początkowe modyfikacje dla okna: mean={initial_mean_adjustment:.4f}, volatility={initial_volatility_adjustment:.4f}")
        
        # Dla każdego punktu w oknie
        for i in range(window_start_idx, window_end_idx):
            # Obliczenie współczynnika osłabienia (od 1.0 do 0.2) - wykładnicze osłabienie
            relative_pos_in_window = (i - window_start_idx) / window_size
            decay_factor = max(0.2, np.exp(-3 * relative_pos_in_window))  # Wykładnicze osłabienie
            
            # Dodatkowe tłumienie bazujące na skumulowanym odchyleniu
            if i > 0:
                cumulative_deviation[i] = cumulative_deviation[i-1]
                # Jeśli przekroczyliśmy próg, zaczynamy tłumić
                if abs(cumulative_deviation[i]) > damping_threshold:
                    # Współczynnik tłumienia rośnie nieliniowo z odchyleniem
                    damping_factor = min(0.9, (abs(cumulative_deviation[i]) - damping_threshold) / 
                                          (max_cumulative_deviation - damping_threshold))
                    # Zastosowanie tłumienia
                    decay_factor *= (1.0 - damping_factor)
            
            # Zastosowanie osłabionego biasu
            mean_adjustments[i] = initial_mean_adjustment * decay_factor
            volatility_adjustments[i] = initial_volatility_adjustment * decay_factor
            
            if i % 60 == 0:  # Log co godzinę
                orchestrator.logger.debug(f"Punkt {i}: decay={decay_factor:.2f}, mean={mean_adjustments[i]:.4f}, vol={volatility_adjustments[i]:.4f}")
    
    # Wagi wskaźników
    total_weight = 0
    if orchestrator.parameters['technical_indicators']['rsi']['enabled']:
        total_weight += orchestrator.parameters['technical_indicators']['rsi']['weight']
    if orchestrator.parameters['technical_indicators']['ema']['enabled']:
        total_weight += orchestrator.parameters['technical_indicators']['ema']['weight']
    if orchestrator.parameters['technical_indicators']['obv']['enabled']:
        total_weight += orchestrator.parameters['technical_indicators']['obv']['weight']
    if orchestrator.parameters['trend_detection']['enabled']:
        total_weight += orchestrator.parameters['trend_detection']['weight']
    if orchestrator.parameters['support_resistance']['enabled']:
        total_weight += orchestrator.parameters['support_resistance']['weight']
    
    # Normalizacja modyfikacji tylko jeśli suma wag > 0
    if total_weight > 0:
        # ZWIĘKSZONE skalowanie
        # mean_adjustments = mean_adjustments / total_weight * 70  # Poziom 3
        # mean_adjustments = mean_adjustments / total_weight * 150  # Poziom 6-7
        mean_adjustments = mean_adjustments / total_weight * 300  # Poziom 10 (oryginalny)
        # volatility_adjustments = volatility_adjustments / total_weight * 70  # Poziom 3
        # volatility_adjustments = volatility_adjustments / total_weight * 150  # Poziom 6-7
        volatility_adjustments = volatility_adjustments / total_weight * 300  # Poziom 10 (oryginalny)
    
    # Ograniczenie modyfikacji do sensownych wartości - ZWIĘKSZONE limity
    # mean_adjustments = np.clip(mean_adjustments, -0.8, 0.8)  # Poziom 3
    # mean_adjustments = np.clip(mean_adjustments, -1.5, 1.5)  # Poziom 6-7
    mean_adjustments = np.clip(mean_adjustments, -3.0, 3.0)  # Poziom 10 (oryginalny)

    # volatility_adjustments = np.clip(volatility_adjustments, -0.5, 0.7)  # Poziom 3
    # volatility_adjustments = np.clip(volatility_adjustments, -0.8, 1.2)  # Poziom 6-7
    volatility_adjustments = np.clip(volatility_adjustments, -1.5, 2.0)  # Poziom 10 (oryginalny)
    
    # Śledzenie referencyjnej ścieżki cen
    reference_prices = simulated_data['close'].copy()
    
    # Zastosowanie modyfikacji do cen
    for i in range(1, len(modified_data)):
        # Aktualna cena
        current_price = modified_data['close'].iloc[i-1]  # Używamy poprzedniej ceny close jako punkt odniesienia
        
        # Oryginalny zwrot procentowy
        if simulated_data['close'].iloc[i-1] > 0:  # Używamy oryginalnych danych
            original_return = (simulated_data['close'].iloc[i] / simulated_data['close'].iloc[i-1]) - 1
        else:
            original_return = 0
        
        # Modyfikacja zwrotu z ZWIĘKSZONYM wpływem
        # modified_return = original_return * (1 + 0.5 * mean_adjustments[i])  # Poziom 3
        # modified_return = original_return * (1 + 0.8 * mean_adjustments[i])  # Poziom 6-7
        modified_return = original_return * (1 + 1.5 * mean_adjustments[i])  # Poziom 10 (oryginalny)
        
        # Ograniczenie modyfikacji do rozsądnych wartości - ZWIĘKSZONE limity
        # max_return_limit = 0.08  # Poziom 3
        # max_return_limit = 0.15  # Poziom 6-7
        max_return_limit = 0.25  # Poziom 10 (oryginalny)
        # min_return_limit = -0.08  # Poziom 3
        # min_return_limit = -0.15  # Poziom 6-7
        min_return_limit = -0.25  # Poziom 10 (oryginalny)
        
        modified_return = np.clip(modified_return, min_return_limit, max_return_limit)
        
        # Obliczenie nowej ceny
        new_close = current_price * (1 + modified_return)
        
        # Ograniczenie odchylenia od referencyjnej ścieżki
        ref_price = reference_prices.iloc[i]
        price_deviation = (new_close / ref_price) - 1
        
        # Aktualizacja skumulowanego odchylenia
        cumulative_deviation[i] = cumulative_deviation[i-1] + price_deviation
        
        # Ograniczenie skumulowanego odchylenia
        if abs(cumulative_deviation[i]) > max_cumulative_deviation:
            # Jeśli przekroczyliśmy maksymalne odchylenie, przyciągamy cenę z powrotem do referencyjnej ścieżki
            correction_strength = 0.3  # Siła korekcji
            correction = -np.sign(cumulative_deviation[i]) * correction_strength * abs(price_deviation)
            new_close = new_close * (1 + correction)
            
            # Aktualizacja skumulowanego odchylenia po korekcji
            price_deviation = (new_close / ref_price) - 1
            cumulative_deviation[i] = cumulative_deviation[i-1] + price_deviation
        
        # Dodatkowe zabezpieczenie przed zbyt dużym jednorazowym odchyleniem - ZWIĘKSZONE
        # max_single_deviation = 0.15  # Poziom 3
        # max_single_deviation = 0.3  # Poziom 6-7
        max_single_deviation = 0.5  # Poziom 10 (oryginalny)
        if abs(price_deviation) > max_single_deviation:
            # Ograniczenie odchylenia
            new_close = ref_price * (1 + np.sign(price_deviation) * max_single_deviation)
        
        # Zapobieganie zbyt małym wartościom
        min_close_value = 1e-3
        if new_close < min_close_value:
            new_close = min_close_value
        
        # Sprawdzenie, czy wynik jest sensowny
        if np.isfinite(new_close) and new_close > 0:
            modified_data.loc[modified_data.index[i], 'close'] = new_close
        else:
            # W przypadku problemu, używamy oryginalnej niemodyfikowanej wartości
            orchestrator.logger.warning(f"Nieprawidłowa wartość new_close={new_close}, używam oryginalnej wartości")
            modified_data.loc[modified_data.index[i], 'close'] = simulated_data.iloc[i]['close']
        
        # Modyfikacja zmienności dla high/low
        if volatility_adjustments[i] != 0:
            # Bardziej naturalne określanie high/low
            original_high = simulated_data['high'].iloc[i]
            original_low = simulated_data['low'].iloc[i]
            original_close = simulated_data['close'].iloc[i]
            
            # Określenie stosunku high i low do close w oryginalnych danych
            if original_close > 0:
                high_ratio = original_high / original_close
                low_ratio = original_low / original_close
            else:
                high_ratio = 1.01  # Domyślne wartości
                low_ratio = 0.99
            
            # Zastosowanie modyfikacji zmienności z ograniczeniem - ZWIĘKSZONE
            # vol_factor = 1 + 0.5 * volatility_adjustments[i]  # Poziom 3
            # vol_factor = 1 + 0.8 * volatility_adjustments[i]  # Poziom 6-7
            vol_factor = 1 + 1.5 * volatility_adjustments[i]  # Poziom 10 (oryginalny)
            # vol_factor = np.clip(vol_factor, 0.4, 2.5)  # Poziom 3
            # vol_factor = np.clip(vol_factor, 0.3, 3.5)  # Poziom 6-7
            vol_factor = np.clip(vol_factor, 0.2, 5.0)  # Poziom 10 (oryginalny)
            
            # Obliczenie nowych high/low z zachowaniem proporcji
            new_high = new_close * max(1.001, (high_ratio - 1) * vol_factor + 1)
            new_low = new_close * min(0.999, (low_ratio - 1) * vol_factor + 1)
            
            # Ograniczenie maksymalnego zakresu - ZWIĘKSZONE
            # max_range_pct = 0.08  # Poziom 3
            # max_range_pct = 0.15  # Poziom 6-7
            max_range_pct = 0.25  # Poziom 10 (oryginalny)
            if (new_high / new_close - 1) > max_range_pct:
                new_high = new_close * (1 + max_range_pct)
            if (1 - new_low / new_close) > max_range_pct:
                new_low = new_close * (1 - max_range_pct)
            
            # Upewnienie się, że low < close < high
            new_high = max(new_high, new_close * 1.0001)
            new_low = min(new_low, new_close * 0.9999)
            
            if np.isfinite(new_high) and new_high > 0:
                modified_data.loc[modified_data.index[i], 'high'] = new_high
            if np.isfinite(new_low) and new_low > 0:
                modified_data.loc[modified_data.index[i], 'low'] = new_low
        
        # Ustawienie open dla następnego punktu tylko jeśli nie jest to ostatni punkt
        if i < len(modified_data) - 1:
            modified_data.loc[modified_data.index[i+1], 'open'] = new_close
    
    # Finalne sprawdzenie i naprawa danych
    orchestrator.logger.info("Finalne sprawdzenie i naprawa danych...")

    # Znajdź wiersze z ekstremalnie małymi wartościami
    extreme_low_rows = (modified_data['close'] < 1e-3) | (modified_data['high'] < 1e-3) | (modified_data['low'] < 1e-3)
    extreme_count = extreme_low_rows.sum()

    if extreme_count > 0:
        orchestrator.logger.warning(f"Znaleziono {extreme_count} wierszy z ekstremalnie małymi wartościami - naprawiam")
        
        # Dla każdego wiersza z problemem, przywróć oryginalne wartości
        for i in range(len(modified_data)):
            if i < len(simulated_data) and (modified_data['close'].iloc[i] < 1e-3 or 
                                            modified_data['high'].iloc[i] < 1e-3 or 
                                            modified_data['low'].iloc[i] < 1e-3):
                modified_data.loc[modified_data.index[i], 'open'] = simulated_data.iloc[i]['open']
                modified_data.loc[modified_data.index[i], 'high'] = simulated_data.iloc[i]['high']
                modified_data.loc[modified_data.index[i], 'low'] = simulated_data.iloc[i]['low']
                modified_data.loc[modified_data.index[i], 'close'] = simulated_data.iloc[i]['close']

    # Sprawdź czy średnia cena nie spadła drastycznie
    orig_mean = simulated_data['close'].mean()
    mod_mean = modified_data['close'].mean()

    if orig_mean > 0 and mod_mean/orig_mean < 0.5:  # Jeśli spadła więcej niż o 50%
        orchestrator.logger.warning(f"Średnia cena po modyfikacji spadła zbyt drastycznie: z {orig_mean:.4f} do {mod_mean:.4f}")
        orchestrator.logger.warning("Stosuję naprawę - ograniczam maksymalne odchylenie każdego punktu danych")
        
        # Przywracamy wartości z ograniczonym odchyleniem
        for i in range(len(modified_data)):
            if i < len(simulated_data):
                orig_close = simulated_data.iloc[i]['close']
                mod_close = modified_data.iloc[i]['close']
                
                if orig_close > 0:
                    deviation = mod_close / orig_close
                    
                    # Jeśli odchylenie jest zbyt duże, naprawiamy
                    if deviation < 0.5:  # Spadek o ponad 50%
                        modified_data.loc[modified_data.index[i], 'close'] = orig_close * 0.5
                        modified_data.loc[modified_data.index[i], 'low'] = min(modified_data.iloc[i]['low'], orig_close * 0.45)
                        modified_data.loc[modified_data.index[i], 'high'] = max(modified_data.iloc[i]['high'], orig_close * 0.55)
                    elif deviation > 2.0:  # Wzrost o ponad 100%
                        modified_data.loc[modified_data.index[i], 'close'] = orig_close * 2.0
                        modified_data.loc[modified_data.index[i], 'low'] = min(modified_data.iloc[i]['low'], orig_close * 1.8)
                        modified_data.loc[modified_data.index[i], 'high'] = max(modified_data.iloc[i]['high'], orig_close * 2.2)
    
    # Finalne wygładzenie serii danych
    if len(modified_data) > 3:
        # Wygładzanie przez średnią ruchomą tylko jeśli mamy drastyczne skoki
        close_diff = np.abs(modified_data['close'].pct_change())
        if close_diff.max() > 0.1:  # Jeśli jest skok większy niż 10%
            orchestrator.logger.warning(f"Wykryto drastyczne skoki cenowe (max: {close_diff.max()*100:.2f}%) - stosuję wygładzanie")
            
            # Używamy maski do identyfikacji punktów ze skokami
            jump_mask = close_diff > 0.05  # Skoki większe niż 5%
            
            for i in range(1, len(modified_data)-1):
                if jump_mask.iloc[i]:
                    # Wygładzanie przez średnią z sąsiednich punktów
                    smooth_close = (modified_data['close'].iloc[i-1] + modified_data['close'].iloc[i+1]) / 2
                    # Ograniczamy wygładzanie do 50% różnicy
                    current = modified_data['close'].iloc[i]
                    modified_data.loc[modified_data.index[i], 'close'] = current * 0.5 + smooth_close * 0.5
                    
                    # Aktualizacja high/low dla zachowania spójności
                    modified_data.loc[modified_data.index[i], 'high'] = max(modified_data.iloc[i]['high'], modified_data.iloc[i]['close'])
                    modified_data.loc[modified_data.index[i], 'low'] = min(modified_data.iloc[i]['low'], modified_data.iloc[i]['close'])
    
    # Tymczasowy log pokazujący wpływ AT na symulowane dane
    orchestrator.logger.warning("\n" + "!" * 80)
    orchestrator.logger.warning("!!!!!!!!!! WPŁYW ANALIZY TECHNICZNEJ NA DANE SYMULOWANE !!!!!!!!!!")
    orchestrator.logger.warning("!" * 80)

    # Dodajmy bardzo wyraźny log porównania pierwszych 5 świeczek
    orchestrator.logger.warning("\nPorównanie pierwszych 5 świeczek:")
    orchestrator.logger.warning("-" * 60)
    for i in range(min(5, len(simulated_data))):
        orig_close = simulated_data.iloc[i]['close']
        mod_close = modified_data.iloc[i]['close']
        if orig_close > 0:
            pct_diff = ((mod_close / orig_close) - 1) * 100
        else:
            pct_diff = 0
        orchestrator.logger.warning(f"Świeczka {i+1}: Oryg={orig_close:.4f}, Po AT={mod_close:.4f}, Zmiana={pct_diff:.2f}%")

    # Podsumowanie z dodatkowym zabezpieczeniem
    avg_close_before = simulated_data['close'].mean()
    avg_close_after = modified_data['close'].mean()
    if avg_close_before > 0:
        avg_pct_diff = ((avg_close_after / avg_close_before) - 1) * 100
    else:
        avg_pct_diff = 0

    # Bezpieczne obliczenie zmienności
    try:
        vol_before = simulated_data['close'].pct_change(fill_method=None).dropna().std() * 100
        vol_after = modified_data['close'].pct_change(fill_method=None).dropna().std() * 100
        if vol_before > 0:
            vol_ratio = vol_after / vol_before
        else:
            vol_ratio = 1.0
    except Exception as e:
        orchestrator.logger.warning(f"Błąd przy obliczaniu zmienności: {e}")
        vol_before = 0
        vol_after = 0
        vol_ratio = 1.0

    # Szczegółowe statystyki wpływu AT
    orchestrator.logger.warning(f"\nŚrednia cena: Przed={avg_close_before:.4f}, Po={avg_close_after:.4f}, Różnica={avg_pct_diff:.2f}%")
    orchestrator.logger.warning(f"Zmienność: Przed={vol_before:.4f}%, Po={vol_after:.4f}%, Stosunek={vol_ratio:.2f}")
    
    # Dodatkowe statystyki modyfikacji
    mean_adj_avg = np.mean(mean_adjustments)
    vol_adj_avg = np.mean(volatility_adjustments)
    mean_adj_max = np.max(np.abs(mean_adjustments))
    vol_adj_max = np.max(np.abs(volatility_adjustments))
    
    orchestrator.logger.warning(f"\nStatystyki modyfikacji AT:")
    orchestrator.logger.warning(f"Średnia modyfikacja zwrotu: {mean_adj_avg:.4f}")
    orchestrator.logger.warning(f"Średnia modyfikacja zmienności: {vol_adj_avg:.4f}")
    orchestrator.logger.warning(f"Maksymalna modyfikacja zwrotu: {mean_adj_max:.4f}")
    orchestrator.logger.warning(f"Maksymalna modyfikacja zmienności: {vol_adj_max:.4f}")
    
    # Statystyki odchyleń
    orchestrator.logger.warning(f"\nStatystyki odchyleń od oryginalnej symulacji:")
    orchestrator.logger.warning(f"Średnie odchylenie: {np.mean(cumulative_deviation):.4f}")
    orchestrator.logger.warning(f"Maksymalne odchylenie: {np.max(np.abs(cumulative_deviation)):.4f}")
    
    if hasattr(orchestrator, 'simulator') and hasattr(orchestrator.simulator, 'at_bias_level'):
        bias_level = orchestrator.simulator.at_bias_level
        bias_description = ""
        if bias_level > 0.7:
            bias_description = "(silny wzrostowy)"
        elif bias_level > 0.3:
            bias_description = "(umiarkowany wzrostowy)"
        elif bias_level > 0.0:
            bias_description = "(lekki wzrostowy)"
        elif bias_level == 0.0:
            bias_description = "(neutralny)"
        elif bias_level > -0.3:
            bias_description = "(lekki spadkowy)"
        elif bias_level > -0.7:
            bias_description = "(umiarkowany spadkowy)"
        else:
            bias_description = "(silny spadkowy)"
        
        orchestrator.logger.warning(f"AT bias level: {bias_level:.4f} {bias_description}")
    
    orchestrator.logger.warning("!" * 80 + "\n")

    # Bezpośredni wydruk podsumowania modyfikacji
    print("\n" + "!" * 80)
    print("!!!!!!!!!! ZAKOŃCZONO MODYFIKACJĘ PRZEZ ANALIZĘ TECHNICZNĄ !!!!!!!!!!")
    
    # Podsumowanie zmian
    print(f"Średnia cena przed: {avg_close_before:.4f}, po: {avg_close_after:.4f}")
    print(f"Średnia zmiana: {avg_pct_diff:.2f}%")
    print("!" * 80 + "\n")
    
    orchestrator.logger.info("Zakończenie modyfikacji symulowanych danych")
    return modified_data