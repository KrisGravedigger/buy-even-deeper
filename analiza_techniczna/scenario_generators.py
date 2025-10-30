"""
Generatory scenariuszy dla symulatora rynku.
Zawiera klasy odpowiedzialne za generowanie różnych typów scenariuszy symulacyjnych.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from math import sqrt
import time
import os
import random
from datetime import timedelta
from pathlib import Path

from analiza_techniczna.simulator_utils import (
    simulate_gbm, simulate_correlated_gbm, create_simulated_dataframe, save_to_csv
)


class BaseScenarioGenerator:
    """
    Bazowa klasa dla generatorów scenariuszy.
    """
    
    def __init__(self, simulator):
        """
        Inicjalizacja generatora scenariuszy.
        
        Args:
            simulator: Instancja MarketSimulator
        """
        self.simulator = simulator
        self.logger = simulator.logger
    
    def generate(self, days: int, scenario_name: str, **kwargs) -> str:
        """
        Abstrakcyjna metoda generacji scenariusza. Powinna być nadpisana przez klasy pochodne.
        
        Args:
            days: Liczba dni do symulacji
            scenario_name: Nazwa scenariusza
            **kwargs: Dodatkowe parametry specyficzne dla danego typu scenariusza
            
        Returns:
            str: Ścieżka do wygenerowanego pliku CSV
        """
        raise NotImplementedError("Ta metoda musi być zaimplementowana przez klasę pochodną")


class NormalScenarioGenerator(BaseScenarioGenerator):
    """
    Generator normalnego scenariusza, opartego na geometrycznym ruchu Browna.
    """
    
    def generate(self, days: int, scenario_name: str, mean_multiplier: float = 1.0, 
               volatility_multiplier: float = 1.0) -> str:
        """
        Generuje normalny scenariusz z dostosowanymi parametrami.
        
        Args:
            days: Liczba dni do symulacji
            scenario_name: Nazwa scenariusza
            mean_multiplier: Mnożnik dla średniego zwrotu (>1 oznacza bardziej optymistyczny)
            volatility_multiplier: Mnożnik dla zmienności (>1 oznacza większą zmienność)
            
        Returns:
            str: Ścieżka do wygenerowanego pliku CSV
        """
        self.logger.info(f"Generowanie scenariusza normalnego: {scenario_name}")
        
        # Ustalenie liczby kroków do symulacji na podstawie interwału danych
        intervals_per_day = int(24 * 60 * 60 / self.simulator.interval_seconds)
        total_intervals = days * intervals_per_day
        
        # Dostosowanie parametrów
        adjusted_mean = self.simulator.mean_return * mean_multiplier
        adjusted_std = self.simulator.std_dev * volatility_multiplier
        
        # Ustalenie początkowych cen 
        initial_price = self.simulator.data['close'].iloc[-1]
        
        # Generowanie symulowanych danych dla głównej kryptowaluty
        simulated_prices = simulate_gbm(
            initial_price, 
            adjusted_mean, 
            adjusted_std, 
            total_intervals
        )
        
        # Jeśli mamy dane BTC, generujemy również dla nich
        if self.simulator.has_btc_data or self.simulator.btc_data is not None:
            initial_btc_price = self.simulator.data['btc_close'].iloc[-1]
            adjusted_btc_mean = self.simulator.btc_mean_return * mean_multiplier
            adjusted_btc_std = self.simulator.btc_std_dev * volatility_multiplier
            
            # Generowanie skorelowanych danych dla BTC
            simulated_btc_data = simulate_correlated_gbm(
                initial_btc_price,
                initial_price,
                adjusted_btc_mean,
                adjusted_mean,
                adjusted_btc_std,
                adjusted_std,
                self.simulator.correlation,
                total_intervals
            )[1]  # Bierzemy tylko drugą macierz (BTC)
        else:
            simulated_btc_data = None
            
        # Tworzenie dataframe z symulowanymi danymi
        simulated_df = create_simulated_dataframe(
            self.simulator, 
            simulated_prices, 
            simulated_btc_data, 
            days
        )
        
        # Zapis do CSV
        csv_path = save_to_csv(self.simulator, simulated_df, scenario_name)
        
        self.logger.info(f"Wygenerowano scenariusz: {csv_path}")
        return csv_path


class BootstrapScenarioGenerator(BaseScenarioGenerator):
    """
    Generator scenariusza bootstrap, wykorzystujący historyczne dane.
    """
    
    def generate(self, days: int, scenario_name: str) -> str:
        """
        Generuje scenariusz bootstrap.
        
        Args:
            days: Liczba dni do symulacji
            scenario_name: Nazwa scenariusza
            
        Returns:
            str: Ścieżka do wygenerowanego pliku CSV
        """
        self.logger.info(f"Generowanie scenariusza bootstrap: {scenario_name}")
        
        # Ustalenie liczby kroków do symulacji na podstawie interwału danych
        intervals_per_day = int(24 * 60 * 60 / self.simulator.interval_seconds)
        total_intervals = days * intervals_per_day
        
        # Obliczenie historycznych logarytmicznych zmian cen
        log_returns = np.log(self.simulator.data['close'] / self.simulator.data['close'].shift(1)).dropna().values
        
        # Losowanie zwrotów z historycznego rozkładu
        sampled_returns = np.random.choice(log_returns, size=total_intervals)
        
        # Ustalenie początkowej ceny
        initial_price = self.simulator.data['close'].iloc[-1]
        
        # Generowanie symulowanych cen
        simulated_prices = np.zeros(total_intervals)
        simulated_prices[0] = initial_price
        
        for i in range(1, total_intervals):
            simulated_prices[i] = simulated_prices[i-1] * np.exp(sampled_returns[i])
            
        # Jeśli mamy dane BTC, generujemy również dla nich
        if self.simulator.has_btc_data or self.simulator.btc_data is not None:
            btc_log_returns = np.log(self.simulator.data['btc_close'] / self.simulator.data['btc_close'].shift(1)).dropna().values
            
            # Aby zachować korelację, używamy tej samej sekwencji losowań, ale z danymi BTC
            indices = np.random.choice(len(btc_log_returns), size=total_intervals)
            sampled_btc_returns = btc_log_returns[indices]
            
            # Ustalenie początkowej ceny BTC
            initial_btc_price = self.simulator.data['btc_close'].iloc[-1]
            
            # Generowanie symulowanych cen BTC
            simulated_btc_prices = np.zeros(total_intervals)
            simulated_btc_prices[0] = initial_btc_price
            
            for i in range(1, total_intervals):
                simulated_btc_prices[i] = simulated_btc_prices[i-1] * np.exp(sampled_btc_returns[i])
        else:
            simulated_btc_prices = None
            
        # Tworzenie dataframe z symulowanymi danymi
        simulated_df = create_simulated_dataframe(
            self.simulator, 
            simulated_prices, 
            simulated_btc_prices, 
            days
        )
        
        # Zapis do CSV
        csv_path = save_to_csv(self.simulator, simulated_df, scenario_name)
        
        self.logger.info(f"Wygenerowano scenariusz: {csv_path}")
        return csv_path


class StressScenarioGenerator(BaseScenarioGenerator):
    """
    Generator scenariusza stress testu z podwyższoną zmiennością i nagłymi spadkami.
    """
    
    def generate(self, days: int, scenario_name: str, stress_factor: float = 2.0,
                num_flash_crashes: int = 2) -> str:
        """
        Generuje scenariusz stresowy.
        
        Args:
            days: Liczba dni do symulacji
            scenario_name: Nazwa scenariusza
            stress_factor: Mnożnik zwiększający zmienność
            num_flash_crashes: Liczba nagłych spadków do zasymulowania
            
        Returns:
            str: Ścieżka do wygenerowanego pliku CSV
        """
        self.logger.info(f"Generowanie scenariusza stresowego: {scenario_name}")
        
        # Najpierw generujemy normalne dane z podwyższoną zmiennością
        intervals_per_day = int(24 * 60 * 60 / self.simulator.interval_seconds)
        total_intervals = days * intervals_per_day
        
        # Dostosowanie parametrów dla scenariusza stresowego
        adjusted_mean = self.simulator.mean_return * 0.5  # Zmniejszony średni zwrot
        adjusted_std = self.simulator.std_dev * stress_factor  # Zwiększona zmienność
        
        # Ustalenie początkowych cen 
        initial_price = self.simulator.data['close'].iloc[-1]
        
        # Generowanie symulowanych danych dla głównej kryptowaluty
        simulated_prices = simulate_gbm(
            initial_price, 
            adjusted_mean, 
            adjusted_std, 
            total_intervals
        )
        
        # Jeśli mamy dane BTC, generujemy również dla nich
        if self.simulator.has_btc_data or self.simulator.btc_data is not None:
            initial_btc_price = self.simulator.data['btc_close'].iloc[-1]
            adjusted_btc_mean = self.simulator.btc_mean_return * 0.5
            adjusted_btc_std = self.simulator.btc_std_dev * stress_factor
            
            # Generowanie skorelowanych danych dla BTC
            simulated_btc_prices = simulate_correlated_gbm(
                initial_btc_price,
                initial_price,
                adjusted_btc_mean,
                adjusted_mean,
                adjusted_btc_std,
                adjusted_std,
                self.simulator.correlation,
                total_intervals
            )[1]  # Bierzemy tylko drugą macierz (BTC)
        else:
            simulated_btc_prices = None
            
        # Dodajemy nagłe spadki (flash crash)
        if num_flash_crashes > 0:
            # Równomierne rozłożenie flash crashy w czasie symulacji
            flash_crash_points = np.linspace(
                total_intervals // 10,  # Nie na samym początku
                total_intervals - total_intervals // 10,  # Nie na samym końcu
                num_flash_crashes
            ).astype(int)
            
            # Dodanie nagłych spadków do symulacji
            for point in flash_crash_points:
                # Losowy spadek między 5% a 15%
                crash_pct = np.random.uniform(0.05, 0.15)
                crash_length = np.random.randint(intervals_per_day // 12, intervals_per_day // 4)  # Od 2h do 6h
                
                # Aplikowanie spadku dla głównej kryptowaluty
                for i in range(point, min(point + crash_length, total_intervals)):
                    decay_factor = 1 - (i - point) / crash_length
                    crash_effect = 1 - crash_pct * decay_factor
                    simulated_prices[i] *= crash_effect
                
                # Jeśli mamy BTC, dodajemy mniejszy spadek skorelowany
                if simulated_btc_prices is not None:
                    btc_crash_pct = crash_pct * np.random.uniform(0.5, 0.8)  # BTC spada mniej
                    for i in range(point, min(point + crash_length, total_intervals)):
                        decay_factor = 1 - (i - point) / crash_length
                        btc_crash_effect = 1 - btc_crash_pct * decay_factor
                        simulated_btc_prices[i] *= btc_crash_effect
        
        # Tworzenie dataframe z symulowanymi danymi
        simulated_df = create_simulated_dataframe(
            self.simulator, 
            simulated_prices, 
            simulated_btc_prices, 
            days
        )
        
        # Zapis do CSV
        csv_path = save_to_csv(self.simulator, simulated_df, scenario_name)
        
        self.logger.info(f"Wygenerowano scenariusz stresowy: {csv_path}")
        return csv_path

def prepare_scenario_configs(base_days: int = 30, 
                           normal_scenarios: int = 10,
                           bootstrap_scenarios: int = 2,
                           stress_scenarios: int = 2) -> List[Dict]:
    """
    Przygotowuje konfiguracje dla różnych typów scenariuszy.
    
    Args:
        base_days: Podstawowa liczba dni do symulacji
        normal_scenarios: Liczba normalnych scenariuszy
        bootstrap_scenarios: Liczba scenariuszy bootstrap
        stress_scenarios: Liczba scenariuszy stresowych
        
    Returns:
        List[Dict]: Lista konfiguracji scenariuszy
    """
    configs = []
    
    # Scenariusze normalne - dokładnie tyle ile żądano
    for i in range(normal_scenarios):
        mean_mult = np.random.uniform(0.8, 1.2)
        vol_mult = np.random.uniform(0.8, 1.2)
        configs.append({
            'type': 'normal',
            'days': base_days,
            'name': f"normal_{i+1}_mean{mean_mult:.1f}_vol{vol_mult:.1f}",
            'mean_multiplier': mean_mult,
            'volatility_multiplier': vol_mult
        })
    
    # Scenariusze bootstrap - dokładnie tyle ile żądano
    for i in range(bootstrap_scenarios):
        configs.append({
            'type': 'bootstrap',
            'days': base_days,
            'name': f"bootstrap_{i+1}"
        })
    
    # Scenariusze stresowe - dokładnie tyle ile żądano
    for i in range(stress_scenarios):
        stress_factor = np.random.uniform(1.5, 3.0)
        num_crashes = np.random.randint(1, 4)
        configs.append({
            'type': 'stress',
            'days': base_days,
            'name': f"stress_{i+1}_factor{stress_factor:.1f}_crashes{num_crashes}",
            'stress_factor': stress_factor,
            'num_flash_crashes': num_crashes
        })
    
    # Wyraźne logowanie wygenerowanych konfiguracji
    print(f"\nWygenerowane konfiguracje scenariuszy:")
    print(f"- Normalnych: {normal_scenarios}")
    print(f"- Bootstrap: {bootstrap_scenarios}")
    print(f"- Stresowych: {stress_scenarios}")
    print(f"Łącznie: {len(configs)} scenariuszy\n")
    
    return configs