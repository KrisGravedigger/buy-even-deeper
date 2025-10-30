"""
Orkiestrator analizy technicznej.
Koordynuje proces analizy technicznej i modyfikacji danych wygenerowanych przez symulator rynku.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import concurrent.futures
import os

# Import modułów analizy technicznej
from analiza_techniczna.wskazniki import calculate_rsi, calculate_ema, calculate_obv
from analiza_techniczna.trendy import detect_trend, measure_trend_strength
from analiza_techniczna.poziomy import find_support_resistance

# Import konfiguratora parametrów
from analiza_techniczna.at_parameter_configurator import ATParameterConfigurator

# Import procesora danych analizy technicznej
from analiza_techniczna.at_data_processor import (
    prepare_hourly_data, 
    project_hourly_results_to_original, 
    analyze_current_conditions,
    calculate_mean_adjustment, 
    calculate_volatility_adjustment,
    modify_simulated_data
)


class ATOrchestrator:
    """
    Orkiestrator analizy technicznej.
    """
    
    def __init__(self, config_file: Optional[str] = None, logger: Optional[logging.Logger] = None, at_data_file: Optional[str] = None):
        """
        Inicjalizacja orkiestratora analizy technicznej.
        
        Args:
            config_file: Ścieżka do pliku konfiguracyjnego (jeśli None, używa domyślnej lokalizacji)
            logger: Obiekt loggera (opcjonalnie)
            at_data_file: Ścieżka do pliku z danymi do analizy technicznej (jeśli None, zostanie użyty najnowszy plik z folderu csv/AT)
        """
        self.logger = logger or self._setup_logger()
        
        # Inicjalizacja konfiguratora parametrów
        self.config = ATParameterConfigurator(config_file)
        
        # Wczytanie konfiguracji
        self.parameters = self.config.get_config()
        
        # Wyniki analizy technicznej
        self.analysis_results = {}
        self.historical_data = None
        
        # Dodanie nowych pól dla danych i analizy godzinowej
        self.hourly_data = None
        self.hourly_analysis_results = {}
        self.use_hourly_data = self.parameters.get('use_hourly_data', True)
        self.max_workers = self.parameters.get('max_workers', min(4, os.cpu_count() or 4))
        
        # Ścieżka do pliku z danymi AT
        self.at_data_file = at_data_file
        
        self.logger.info("Orkiestrator analizy technicznej został zainicjalizowany")

    def load_at_data(self) -> bool:
        """
        Wczytuje dane AT z pliku określonego przez at_data_file 
        lub z najnowszego pliku CSV w katalogu csv/AT.
        
        Returns:
            bool: True jeśli wczytanie się powiodło, False w przeciwnym wypadku
        """
        from utils.config import get_newest_at_csv_file, AT_CSV_DIR
        
        # Jeśli nie podano pliku, użyj najnowszego z katalogu AT
        if self.at_data_file is None:
            self.at_data_file = get_newest_at_csv_file()
            
        if self.at_data_file is None:
            self.logger.error(f"Nie znaleziono pliku z danymi AT w katalogu {AT_CSV_DIR}")
            return False
        
        try:
            at_data = pd.read_csv(self.at_data_file)
            self.logger.info(f"Wczytano dane AT z pliku: {self.at_data_file}")
            self.set_historical_data(at_data)
            return True
        except Exception as e:
            self.logger.error(f"Błąd podczas wczytywania danych AT: {e}")
            return False
    
    def _setup_logger(self) -> logging.Logger:
        """
        Konfiguracja loggera.
        
        Returns:
            logging.Logger: Skonfigurowany logger
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Ustawienie formatowania
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # Handler konsoli
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def set_historical_data(self, data: pd.DataFrame) -> None:
        """
        Ustawia dane historyczne do analizy.
        
        Args:
            data: DataFrame z danymi historycznymi
        """
        required_columns = ['open', 'high', 'low', 'close']
        
        # Sprawdzenie, czy wszystkie wymagane kolumny są dostępne
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.warning(f"Brak wymaganych kolumn w danych historycznych: {missing_columns}")
        
        self.historical_data = data
        self.logger.info(f"Ustawiono dane historyczne: {len(data)} rekordów")
        
        # Przygotowanie danych godzinowych jeśli flaga jest ustawiona
        if self.use_hourly_data:
            self.hourly_data = prepare_hourly_data(self)
    
    def analyze_historical_data(self) -> Dict[str, Any]:
        """
        Analizuje dane historyczne przy użyciu wskaźników analizy technicznej.
        Jeśli włączona jest obsługa danych godzinowych, analiza jest przeprowadzana na nich,
        a wyniki są rzutowane z powrotem na oryginalne dane.
        
        Returns:
            Dict[str, Any]: Wyniki analizy
        """
        if self.historical_data is None:
            self.logger.error("Brak danych historycznych do analizy")
            return {}
        
        self.logger.info("Rozpoczęcie analizy danych historycznych")
        
        # Wybór danych do analizy
        if self.use_hourly_data and self.hourly_data is not None:
            data_to_analyze = self.hourly_data
            self.logger.info("Używanie danych godzinowych do analizy")
        else:
            data_to_analyze = self.historical_data
            self.logger.info("Używanie oryginalnych danych do analizy")
        
        # Inicjalizacja wyników analizy
        results = {}
        
        # Analiza wskaźników technicznych
        tech_indicators = self.parameters['technical_indicators']
        
        # Użycie wielowątkowości do równoległego obliczania wskaźników
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Przygotowanie zadań do wykonania
            futures = []
            
            # RSI
            if tech_indicators['rsi']['enabled']:
                futures.append(executor.submit(
                    self._calculate_indicator, 
                    'rsi', 
                    calculate_rsi, 
                    data_to_analyze, 
                    'close', 
                    tech_indicators['rsi']['period']
                ))
            
            # EMA
            if tech_indicators['ema']['enabled']:
                futures.append(executor.submit(
                    self._calculate_indicator, 
                    'ema', 
                    calculate_ema, 
                    data_to_analyze, 
                    'close', 
                    tech_indicators['ema']['period']
                ))
            
            # OBV
            if tech_indicators['obv']['enabled'] and 'volume' in data_to_analyze.columns:
                futures.append(executor.submit(
                    self._calculate_indicator, 
                    'obv', 
                    calculate_obv, 
                    data_to_analyze, 
                    'close', 
                    'volume'
                ))
            
            # Wykrywanie trendu
            if self.parameters['trend_detection']['enabled']:
                method = self.parameters['trend_detection']['method']
                period = self.parameters['trend_detection']['period']
                
                futures.append(executor.submit(
                    self._calculate_indicator, 
                    'trend', 
                    detect_trend, 
                    data_to_analyze, 
                    'close', 
                    period, 
                    method
                ))
                
                futures.append(executor.submit(
                    self._calculate_indicator, 
                    'trend_strength', 
                    measure_trend_strength, 
                    data_to_analyze, 
                    'close', 
                    period
                ))
            
            # Poziomy wsparcia/oporu
            if self.parameters['support_resistance']['enabled']:
                method = self.parameters['support_resistance']['method']
                sensitivity = self.parameters['support_resistance']['sensitivity']
                
                futures.append(executor.submit(
                    self._calculate_support_resistance, 
                    data_to_analyze, 
                    'close', 
                    20, 
                    sensitivity, 
                    method
                ))
            
            # Pobieranie wyników
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        # Jeśli wynik to krotka, to mamy poziomy wsparcia/oporu
                        if isinstance(result, tuple) and len(result) == 3:
                            indicator_name1, indicator_name2, indicator_values = result
                            results[indicator_name1] = indicator_values[0]
                            results[indicator_name2] = indicator_values[1]
                        else:
                            indicator_name, indicator_values = result
                            results[indicator_name] = indicator_values
                except Exception as e:
                    self.logger.error(f"Błąd podczas równoległego obliczania wskaźnika: {e}")
        
        # Zapisanie wyników analizy
        if self.use_hourly_data:
            # Zapisanie wyników dla danych godzinowych
            self.hourly_analysis_results = results.copy()
            
            # Rzutowanie wyników na oryginalne dane
            self.analysis_results = project_hourly_results_to_original(self)
        else:
            self.analysis_results = results
        
        self.logger.info("Zakończenie analizy danych historycznych")
        return self.analysis_results
    
    def _calculate_indicator(self, name, func, data, *args):
        """
        Pomocnicza funkcja do obliczania wskaźnika w osobnym wątku.
        
        Args:
            name: Nazwa wskaźnika
            func: Funkcja obliczająca wskaźnik
            data: Dane do analizy
            *args: Dodatkowe argumenty dla funkcji
            
        Returns:
            Tuple[str, pd.Series]: Nazwa wskaźnika i obliczony wskaźnik
        """
        try:
            result = func(data, *args)
            self.logger.info(f"Obliczono wskaźnik {name}")
            return name, result
        except Exception as e:
            self.logger.error(f"Błąd przy obliczaniu wskaźnika {name}: {e}")
            return None
    
    def _calculate_support_resistance(self, data, column, window, sensitivity, method):
        """
        Pomocnicza funkcja do obliczania poziomów wsparcia/oporu w osobnym wątku.
        
        Args:
            data: Dane do analizy
            column: Kolumna z ceną
            window: Rozmiar okna
            sensitivity: Czułość
            method: Metoda
            
        Returns:
            Tuple[str, str, Tuple[List, List]]: Nazwy wskaźników i lista poziomów
        """
        try:
            support_levels, resistance_levels = find_support_resistance(
                data, column, window, sensitivity, method
            )
            self.logger.info(f"Wykryto poziomy wsparcia/oporu metodą {method}")
            return 'support_levels', 'resistance_levels', (support_levels, resistance_levels)
        except Exception as e:
            self.logger.error(f"Błąd przy wykrywaniu poziomów wsparcia/oporu: {e}")
            return None
    
    def modify_simulated_data(self, simulated_data: pd.DataFrame) -> pd.DataFrame:
        """
        Modyfikuje symulowane dane na podstawie analizy technicznej.
        
        Args:
            simulated_data: DataFrame z symulowanymi danymi
            
        Returns:
            pd.DataFrame: Zmodyfikowane dane
        """
        if self.historical_data is None:
            self.logger.error("Brak danych historycznych. Użyj najpierw set_historical_data() i analyze_historical_data()")
            return simulated_data
        
        if not self.analysis_results:
            self.logger.warning("Brak wyników analizy technicznej. Wykonuję analizę teraz.")
            self.analyze_historical_data()
        
        # Wymuśmy wyświetlenie komunikatów bezpośrednio na konsoli
        print("\n" + "!" * 80)
        print("!!! ROZPOCZYNANIE MODYFIKACJI DANYCH PRZEZ ANALIZĘ TECHNICZNĄ !!!")
        print("!" * 80)
        
        self.logger.info("Rozpoczęcie modyfikacji symulowanych danych")
        
        # Wywołanie funkcji modyfikacji danych z modułu procesora danych
        result = modify_simulated_data(self, simulated_data)
        
        # Dodajmy bezpośrednie logowanie również po zakończeniu
        print("\n" + "!" * 80)
        print("!!! ZAKOŃCZONO MODYFIKACJĘ DANYCH PRZEZ ANALIZĘ TECHNICZNĄ !!!")
        print("!" * 80)
        
        return result
    
    def get_modification_parameters(self, base_scenario: str) -> Dict[str, float]:
        """
        Zwraca parametry modyfikacji dla danego bazowego scenariusza.
        
        Args:
            base_scenario: Typ bazowego scenariusza ('normal', 'bootstrap', 'stress')
            
        Returns:
            Dict[str, float]: Parametry modyfikacji
        """
        # Domyślne parametry
        params = {
            'mean_multiplier': 1.0,
            'volatility_multiplier': 1.0,
            'stress_factor': 2.0,
            'crash_probability': 0.5,
            'at_bias_level': 0.0
        }
        
        # Analiza metryk wskaźników aby określić bias
        if self.analysis_results:
            at_bias = 0.0
            bias_count = 0
            
            # RSI
            if 'rsi' in self.analysis_results:
                last_rsi = self.analysis_results['rsi'].iloc[-1]
                if last_rsi > 70:
                    # Wykupiony - oczekiwany spadek
                    at_bias -= 0.5
                elif last_rsi < 30:
                    # Wyprzedany - oczekiwany wzrost
                    at_bias += 0.5
                bias_count += 1
            
            # Trend
            if 'trend' in self.analysis_results:
                last_trend = self.analysis_results['trend'].iloc[-1]
                if last_trend != 0:  # Jeśli nie jest trendem bocznym
                    at_bias += 0.3 * last_trend
                bias_count += 1
            
            # Uśrednienie biasu
            if bias_count > 0:
                at_bias /= bias_count
                params['at_bias_level'] = at_bias
            
            # Dostosowanie parametrów w zależności od biasu
            if at_bias > 0:
                # Pozytywny bias - zwiększamy średni zwrot
                params['mean_multiplier'] = 1.0 + min(at_bias, 0.5)
            elif at_bias < 0:
                # Negatywny bias - zmniejszamy średni zwrot
                params['mean_multiplier'] = 1.0 + max(at_bias, -0.5)
            
            # Określenie zmienności
            if 'trend_strength' in self.analysis_results:
                trend_strength = self.analysis_results['trend_strength'].iloc[-1]
                # Silny trend = mniejsza zmienność
                volatility_adj = -0.2 * (trend_strength / 100)
                params['volatility_multiplier'] = max(0.5, 1.0 + volatility_adj)
        
        return params