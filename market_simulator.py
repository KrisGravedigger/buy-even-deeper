"""
Symulator rynku do forward-testingu strategii tradingowych.
Generuje symulowane dane rynkowe w formacie zgodnym z danymi z Binance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import multiprocessing as mp
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time
import json
import random
import os

from analiza_techniczna.simulator_utils import setup_logger, prepare_source_data, merge_with_btc_data, calculate_historical_stats, create_simulated_dataframe, add_derived_metrics, save_to_csv
from analiza_techniczna.scenario_generators import BaseScenarioGenerator, NormalScenarioGenerator, BootstrapScenarioGenerator, StressScenarioGenerator, prepare_scenario_configs
from utils.config import get_simulation_output_dir, get_simulation_parameters_dir, get_newest_csv_file


class MarketSimulator:
    def __init__(self, 
                source_data_path: str,
                output_dir: str = None,
                random_seed: Optional[int] = None,
                num_processes: Optional[int] = None,
                btc_source_data_path: Optional[str] = None):
        """
        Inicjalizacja symulatora rynku.
        
        Args:
            source_data_path: Ścieżka do źródłowego pliku CSV z danymi Binance
            output_dir: Katalog wyjściowy dla symulowanych danych (jeśli None, zostanie wygenerowany automatycznie)
            random_seed: Ziarno losowości (dla powtarzalności)
            num_processes: Liczba procesów do użycia
            btc_source_data_path: Opcjonalna ścieżka do danych BTC (jeśli nie zawarte w source_data_path)
        """
        self.source_data_path = source_data_path
        
        # Użycie funkcji z config.py do uzyskania ścieżki wyjściowej
        if output_dir is None:
            self.output_dir = get_simulation_output_dir(source_data_path)
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Konfiguracja losowości dla powtarzalności wyników
        if random_seed is not None:
            self.random_seed = random_seed
            np.random.seed(random_seed)
            random.seed(random_seed)
        else:
            self.random_seed = int(time.time())
            
        # Konfiguracja wielowątkowości
        if num_processes is None:
            self.num_processes = max(1, mp.cpu_count() - 1)
        else:
            self.num_processes = num_processes
            
        # Wczytanie danych źródłowych
        self.data = pd.read_csv(source_data_path)
        
        # Sprawdzenie czy dane zawierają informacje o BTC
        self.has_btc_data = any(col.startswith('btc_') for col in self.data.columns)
        self.btc_source_data_path = btc_source_data_path
        
        # Jeśli dane nie zawierają BTC, a podano ścieżkę do danych BTC, wczytujemy je
        if not self.has_btc_data and btc_source_data_path:
            self.btc_data = pd.read_csv(btc_source_data_path)
        else:
            self.btc_data = None
        
        # Najpierw inicjalizujemy logger
        self.logger = setup_logger(self.output_dir)
        
        # Potem przygotowujemy dane i obliczamy statystyki
        prepare_source_data(self)
        calculate_historical_stats(self)
        
        # Inicjalizacja generatorów scenariuszy
        self.normal_generator = NormalScenarioGenerator(self)
        self.bootstrap_generator = BootstrapScenarioGenerator(self)
        self.stress_generator = StressScenarioGenerator(self)
    
    def generate_normal_scenario(self, 
                            days: int, 
                            scenario_name: str,
                            mean_multiplier: float = 1.0,
                            volatility_multiplier: float = 1.0) -> str:
        """
        Generuje symulację z normalnym rozkładem cen na podstawie historycznych danych.
        
        Args:
            days: Liczba dni do symulacji
            scenario_name: Nazwa scenariusza (zostanie użyta w nazwie pliku)
            mean_multiplier: Mnożnik dla średniego zwrotu (>1 oznacza bardziej optymistyczny)
            volatility_multiplier: Mnożnik dla zmienności (>1 oznacza większą zmienność)
            
        Returns:
            str: Ścieżka do wygenerowanego pliku CSV
        """
        return self.normal_generator.generate(days, scenario_name, mean_multiplier, volatility_multiplier)
    
    def generate_historical_bootstrap(self, 
                                    days: int, 
                                    scenario_name: str) -> str:
        """
        Generuje symulację metodą bootstrap (losowanie historycznych zwrotów).
        
        Args:
            days: Liczba dni do symulacji
            scenario_name: Nazwa scenariusza (zostanie użyta w nazwie pliku)
            
        Returns:
            str: Ścieżka do wygenerowanego pliku CSV
        """
        return self.bootstrap_generator.generate(days, scenario_name)
    
    def generate_stress_scenario(self, 
                            days: int, 
                            scenario_name: str,
                            stress_factor: float = 2.0,
                            num_flash_crashes: int = 2) -> str:
        """
        Generuje scenariusz stress testu z podwyższoną zmiennością i nagłymi spadkami.
        
        Args:
            days: Liczba dni do symulacji
            scenario_name: Nazwa scenariusza (zostanie użyta w nazwie pliku)
            stress_factor: Mnożnik zwiększający zmienność
            num_flash_crashes: Liczba nagłych spadków do zasymulowania
            
        Returns:
            str: Ścieżka do wygenerowanego pliku CSV
        """
        return self.stress_generator.generate(days, scenario_name, stress_factor, num_flash_crashes)
       
    def generate_custom_scenario(self, 
                            days: int, 
                            scenario_name: str,
                            custom_scenario_func, 
                            **kwargs) -> str:
        """
        Generuje scenariusz przy użyciu niestandardowej funkcji.
        
        Args:
            days: Liczba dni do symulacji
            scenario_name: Nazwa scenariusza (zostanie użyta w nazwie pliku)
            custom_scenario_func: Funkcja generująca niestandardowy scenariusz
            **kwargs: Dodatkowe argumenty dla funkcji
            
        Returns:
            str: Ścieżka do wygenerowanego pliku CSV
        """
        self.logger.info(f"Generowanie niestandardowego scenariusza: {scenario_name}")
        
        # Ustalenie liczby kroków do symulacji na podstawie interwału danych
        intervals_per_day = int(24 * 60 * 60 / self.interval_seconds)
        total_intervals = days * intervals_per_day
        
        # Ustalenie początkowych cen 
        initial_price = self.data['close'].iloc[-1]
        
        # Przygotowanie argumentów dla funkcji niestandardowej
        base_args = {
            'initial_price': initial_price,
            'mean_return': self.mean_return,
            'std_dev': self.std_dev,
            'intervals': total_intervals,
            'interval_seconds': self.interval_seconds,
            'historical_data': self.data
        }
        
        # Jeśli mamy dane BTC, dodajemy je również
        if self.has_btc_data or self.btc_data is not None:
            base_args.update({
                'initial_btc_price': self.data['btc_close'].iloc[-1],
                'btc_mean_return': self.btc_mean_return,
                'btc_std_dev': self.btc_std_dev,
                'correlation': self.correlation,
                'cov_matrix': self.cov_matrix
            })
        
        # Łączenie ze przekazanymi argumentami
        all_args = {**base_args, **kwargs}
        
        # Wywołanie funkcji niestandardowej
        result = custom_scenario_func(**all_args)
        
        # Sprawdzenie formatu wyniku
        if isinstance(result, tuple) and len(result) == 2:
            simulated_prices, simulated_btc_prices = result
        else:
            simulated_prices = result
            simulated_btc_prices = None
        
        # Tworzenie dataframe z symulowanymi danymi
        simulated_df = create_simulated_dataframe(self, simulated_prices, simulated_btc_prices, days)
        
        # Zapis do CSV
        csv_path = save_to_csv(self, simulated_df, scenario_name)
        
        self.logger.info(f"Wygenerowano niestandardowy scenariusz: {csv_path}")
        return csv_path
    
    def generate_multiple_scenarios(self, 
                                  scenario_config: List[Dict],
                                  parallel: bool = True) -> List[str]:
        """
        Generuje wiele scenariuszy na podstawie konfiguracji.
        
        Args:
            scenario_config: Lista słowników z konfiguracją scenariuszy
            parallel: Czy generować scenariusze równolegle
            
        Returns:
            List[str]: Lista ścieżek do wygenerowanych plików CSV
        """
        self.logger.info(f"Generowanie {len(scenario_config)} scenariuszy")
        
        if parallel and len(scenario_config) > 1 and self.num_processes > 1:
            # Generowanie równoległe
            with mp.Pool(self.num_processes) as pool:
                results = pool.map(self._generate_scenario, scenario_config)
            return results
        else:
            # Generowanie sekwencyjne
            return [self._generate_scenario(config) for config in scenario_config]
    
    def _generate_scenario(self, config: Dict) -> str:
        """
        Pomocnicza funkcja do generowania pojedynczego scenariusza na podstawie konfiguracji.
        Używana głównie w generowaniu równoległym.
        
        Args:
            config: Słownik z konfiguracją scenariusza
            
        Returns:
            str: Ścieżka do wygenerowanego pliku CSV
        """
        scenario_type = config.get('type', 'normal')
        days = config.get('days', 30)
        scenario_name = config.get('name', f"scenario_{int(time.time())}")
        use_at = config.get('use_at', False)
        at_config_file = config.get('at_config_file')
        
        # Dodajmy wyraźne logowanie
        if use_at:
            print(f"\nGenerowanie scenariusza {scenario_type} z analizą techniczną: {scenario_name}")
        else:
            print(f"\nGenerowanie standardowego scenariusza {scenario_type}: {scenario_name}")
                        
        # Generowanie bazowego scenariusza
        csv_path = None
        if scenario_type == 'normal':
            mean_multiplier = config.get('mean_multiplier', 1.0)
            volatility_multiplier = config.get('volatility_multiplier', 1.0)
            csv_path = self.generate_normal_scenario(
                days, 
                scenario_name, 
                mean_multiplier, 
                volatility_multiplier
            )
        elif scenario_type == 'bootstrap':
            csv_path = self.generate_historical_bootstrap(days, scenario_name)
        elif scenario_type == 'stress':
            stress_factor = config.get('stress_factor', 2.0)
            num_flash_crashes = config.get('num_flash_crashes', 2)
            csv_path = self.generate_stress_scenario(
                days, 
                scenario_name, 
                stress_factor, 
                num_flash_crashes
            )
        elif scenario_type == 'custom':
            custom_func = config.get('custom_func')
            if custom_func is None:
                raise ValueError("Brak funkcji niestandardowej w konfiguracji")
            kwargs = config.get('kwargs', {})
            csv_path = self.generate_custom_scenario(
                days, 
                scenario_name, 
                custom_func, 
                **kwargs
            )
        else:
            raise ValueError(f"Nieznany typ scenariusza: {scenario_type}")

        # Zastosowanie analizy technicznej, jeśli wymagana
        if use_at and csv_path is not None:
            print(f"\n{'!' * 80}")
            print(f"!!! ZASTOSOWANIE ANALIZY TECHNICZNEJ DO SCENARIUSZA: {scenario_name} !!!")
            print(f"{'!' * 80}")
            
            # Wczytanie wygenerowanych danych
            import pandas as pd
            from analiza_techniczna.at_orchestrator import ATOrchestrator
            
            # Inicjalizacja orkiestratora AT
            at_config_file = config.get('at_config_file')
            orchestrator = ATOrchestrator(config_file=at_config_file, logger=self.logger)
            
            # Wczytanie danych historycznych
            orchestrator.set_historical_data(self.data.copy())
            orchestrator.analyze_historical_data()
            
            try:
                # Wczytanie wygenerowanych danych
                generated_data = pd.read_csv(csv_path)
                
                # Zastosowanie AT do modyfikacji danych
                print("\nModyfikowanie danych przez analizę techniczną...")
                modified_data = orchestrator.modify_simulated_data(generated_data)
                
                # Zapisanie zmodyfikowanych danych
                modified_data.to_csv(csv_path, index=False)
                print(f"Zapisano zmodyfikowane dane do {csv_path}\n")
            except Exception as e:
                self.logger.error(f"Błąd podczas stosowania analizy technicznej: {e}")
                print(f"BŁĄD ANALIZY TECHNICZNEJ: {e}")

        return csv_path

def generate_parameters_files(strategy_params: Dict, 
                            output_dir: str = None,
                            num_variations: int = 5) -> List[str]:
    """
    Generuje pliki parametrów dla symulacji.
    
    Args:
        strategy_params: Podstawowe parametry strategii
        output_dir: Katalog wyjściowy (jeśli None, użyje katalogu z config.py)
        num_variations: Liczba wariantów parametrów do wygenerowania
        
    Returns:
        List[str]: Lista ścieżek do wygenerowanych plików parametrów
    """
    if output_dir is None:
        output_path = get_simulation_parameters_dir()
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    param_files = []
    
    # Kopiowanie podstawowych parametrów
    base_params_path = output_path / "base_params.json"
    with open(base_params_path, 'w') as f:
        json.dump(strategy_params, f, indent=4)
    param_files.append(str(base_params_path))
    
    # Generowanie wariantów
    for i in range(1, num_variations):
        varied_params = strategy_params.copy()
        
        # Modyfikacja parametrów
        for key, value in varied_params.items():
            # Modyfikacja tylko parametrów numerycznych
            if isinstance(value, (int, float)) and key != "add_to_limit_order":
                # Losowa zmiana w zakresie ±20%
                variation_factor = np.random.uniform(0.8, 1.2)
                varied_params[key] = value * variation_factor
            
            # Specjalne traktowanie dla parametrów słownikowych z zakresami
            elif isinstance(value, dict) and 'range' in value:
                if isinstance(value['range'], list) and len(value['range']) >= 2:
                    # Losowa zmiana zakresu
                    min_val, max_val = value['range'][0], value['range'][1]
                    range_width = max_val - min_val
                    
                    new_min = min_val + np.random.uniform(-0.2, 0.2) * range_width
                    new_max = max_val + np.random.uniform(-0.2, 0.2) * range_width
                    
                    # Upewniamy się, że min < max
                    if new_min >= new_max:
                        new_min, new_max = min_val, max_val
                    
                    # Aktualizacja zakresu
                    value['range'][0] = new_min
                    value['range'][1] = new_max
        
        # Zapisanie wariantu
        variant_path = output_path / f"variant_{i}_params.json"
        with open(variant_path, 'w') as f:
            json.dump(varied_params, f, indent=4)
        param_files.append(str(variant_path))
    
    return param_files

def main():
    """Główna funkcja demonstrująca użycie klasy MarketSimulator"""
    import argparse
    import sys
    
    # Sprawdzenie, czy uruchomiono z argumentami
    if len(sys.argv) > 1:
        # Tryb pracy z argumentami linii poleceń
        parser = argparse.ArgumentParser(description='Symulator rynku dla forward-testingu')
        parser.add_argument('--source-data', '-s', help='Ścieżka do pliku z danymi źródłowymi (domyślnie: najnowszy plik CSV w głównym folderze csv/)')
        parser.add_argument('--output-dir', '-o', help='Katalog wyjściowy (domyślnie: automatycznie generowany)')
        parser.add_argument('--seed', type=int, help='Ziarno losowości (domyślnie: brak - używa aktualnego czasu)')
        parser.add_argument('--days', type=int, default=30, help='Liczba dni do symulacji (domyślnie: 30)')
        parser.add_argument('--scenarios', '-n', type=int, default=24, help='Całkowita liczba scenariuszy (domyślnie: 24 - proporcja 10:1:1)')
        parser.add_argument('--at-config', '-c', help='Ścieżka do pliku konfiguracyjnego analizy technicznej')
        
        args = parser.parse_args()
        run_simulator(args)
    else:
        # Tryb interaktywny z menu
        run_interactive_mode()

def run_interactive_mode():
    """Uruchamia symulator w trybie interaktywnym z menu"""
    from utils.config import get_newest_csv_file, get_newest_at_config_file
    
    print("=" * 60)
    print("                   SYMULATOR RYNKU                   ")
    print("=" * 60)
    print("Wybierz tryb pracy:")
    print("1. Tryb prosty (standardowa symulacja)")
    print("2. Tryb zaawansowany (z analizą techniczną)")
    
    while True:
        try:
            choice = int(input("Wybór [1-2]: "))
            if choice in [1, 2]:
                break
            else:
                print("Nieprawidłowy wybór. Wybierz 1 lub 2.")
        except ValueError:
            print("Wprowadź liczbę 1 lub 2.")
    
    # Przygotowanie parametrów symulacji
    params = {}
    
    # Pobieranie ścieżki do pliku źródłowego
    source_data = get_newest_csv_file()
    if source_data is None:
        print("Błąd: Nie znaleziono plików CSV w głównym katalogu csv/")
        return
    print(f"Dane źródłowe: {source_data}")
    params['source_data'] = source_data
    
    # Ustalenie liczby dni symulacji
    days = input("Liczba dni do symulacji [30]: ")
    params['days'] = int(days) if days.strip() else 30
    
    # Drugi krok menu - konfiguracja liczby scenariuszy
    print("\nKonfiguracja liczby scenariuszy:")
    
    if choice == 1:  # Tryb prosty
        normal_input = input("Liczba normalnych scenariuszy [10]: ")
        normal_scenarios = int(normal_input) if normal_input.strip() else 10
        
        bootstrap_input = input("Liczba scenariuszy bootstrap [0]: ")
        bootstrap_scenarios = int(bootstrap_input) if bootstrap_input.strip() else 0
        
        stress_input = input("Liczba scenariuszy stresowych [0]: ")
        stress_scenarios = int(stress_input) if stress_input.strip() else 0
        
    else:  # Tryb zaawansowany
        print("Podaj liczbę scenariuszy każdego typu:")
        
        normal_input = input("Normalnych scenariuszy [10]: ")
        normal_scenarios = int(normal_input) if normal_input.strip() else 10
        
        bootstrap_input = input("Scenariuszy bootstrap [2]: ")
        bootstrap_scenarios = int(bootstrap_input) if bootstrap_input.strip() else 2
        
        stress_input = input("Scenariuszy stresowych [2]: ")
        stress_scenarios = int(stress_input) if stress_input.strip() else 2
                        
        # Sprawdzenie pliku konfiguracyjnego
        at_config = get_newest_at_config_file()
        if at_config is None:
            print("\nUWAGA: Nie znaleziono pliku konfiguracyjnego analizy technicznej!")
            print("Symulacja zostanie uruchomiona z domyślnymi parametrami.")
            print("Aby utworzyć plik konfiguracyjny, uruchom at_parameter_configurator.py z opcją --create-template")
        else:
            print(f"Plik konfiguracyjny AT: {at_config}")
        
        params['at_config'] = at_config
    
    params['normal_scenarios'] = normal_scenarios
    params['bootstrap_scenarios'] = bootstrap_scenarios
    params['stress_scenarios'] = stress_scenarios
    params['advanced_mode'] = (choice == 2)
    
    # Uruchomienie symulacji
    print("\nRozpoczynanie symulacji z parametrami:")
    print(f"- Dni symulacji: {params['days']}")
    print(f"- Scenariusze normalne: {normal_scenarios}")
    print(f"- Scenariusze bootstrap: {bootstrap_scenarios}")
    print(f"- Scenariusze stresowe: {stress_scenarios}")
    if choice == 2:
        print(f"- Zastosowanie analizy technicznej: TAK")
    
    confirmation = input("\nCzy kontynuować? [T/n]: ")
    if confirmation.lower() in ['', 't', 'tak', 'y', 'yes']:
        # Konwersja do formatu args
        class Args:
            pass
        
        args = Args()
        args.source_data = params['source_data']
        args.output_dir = None
        args.seed = None
        args.days = params['days']
        args.normal_scenarios = params['normal_scenarios']
        args.bootstrap_scenarios = params['bootstrap_scenarios']
        args.stress_scenarios = params['stress_scenarios']
        args.scenarios = args.normal_scenarios + args.bootstrap_scenarios + args.stress_scenarios
        args.at_config = params.get('at_config')
        args.advanced_mode = params.get('advanced_mode', False)
        
        # Uruchomienie symulacji
        run_simulator(args)
        
        # Komunikat końcowy dla trybu zaawansowanego bez konfiguracji
        if choice == 2 and params.get('at_config') is None:
            print("\nUWAGA: Symulacja została wykonana z domyślnymi parametrami analizy technicznej.")
            print("Aby skonfigurować analizę techniczną, utwórz plik konfiguracyjny używając at_parameter_configurator.py")
    else:
        print("Anulowano symulację.")

def run_simulator(args):
    """Uruchamia symulator z podanymi argumentami"""
    # Jeśli nie podano pliku źródłowego, używamy funkcji z config.py
    if args.source_data is None:
        args.source_data = get_newest_csv_file()
        if args.source_data is None:
            print("Błąd: Nie znaleziono plików CSV w głównym katalogu csv/")
            return
        print(f"Automatycznie wybrano najnowszy plik: {args.source_data}")
    
    # Inicjalizacja symulatora
    simulator = MarketSimulator(
        source_data_path=args.source_data,
        output_dir=args.output_dir,
        random_seed=args.seed
    )
    
    # Dodajmy informację o użyciu analizy technicznej (jeśli jest używana)
    if hasattr(args, 'advanced_mode') and args.advanced_mode:
        print("\n" + "!" * 80)
        print("!!! UWAGA: UŻYWANIE ANALIZY TECHNICZNEJ DO MODYFIKACJI WSZYSTKICH SCENARIUSZY !!!")
        print("!" * 80 + "\n")

    # Określenie liczby scenariuszy poszczególnych typów
    if hasattr(args, 'normal_scenarios') and hasattr(args, 'bootstrap_scenarios') and hasattr(args, 'stress_scenarios'):
        # Użyj bezpośrednich wartości
        normal_scenarios = args.normal_scenarios
        bootstrap_scenarios = args.bootstrap_scenarios
        stress_scenarios = args.stress_scenarios
    else:
        # Domyślne proporcje, jeśli podano tylko łączną liczbę scenariuszy
        total_scenarios = getattr(args, 'scenarios', 14)  # Domyślnie 14, jeśli nie podano
        normal_scenarios = int(total_scenarios * 0.714)  # Około 10/14
        bootstrap_scenarios = int(total_scenarios * 0.143)  # Około 2/14
        stress_scenarios = total_scenarios - normal_scenarios - bootstrap_scenarios  # Reszta
    
    # Generowanie konfiguracji scenariuszy
    configs = prepare_scenario_configs(
        base_days=args.days,
        normal_scenarios=normal_scenarios,
        bootstrap_scenarios=bootstrap_scenarios,
        stress_scenarios=stress_scenarios
    )
    
    # Dodanie konfiguracji analizy technicznej w trybie zaawansowanym
    if hasattr(args, 'advanced_mode') and args.advanced_mode:
        # W trybie zaawansowanym włączamy AT dla WSZYSTKICH scenariuszy
        for i in range(len(configs)):
            configs[i]['use_at'] = True
            configs[i]['at_config_file'] = args.at_config
    elif args.at_config and args.advanced > 0:
        # W trybie linii poleceń włączamy AT dla wszystkich scenariuszy, 
        # gdy podano plik konfiguracyjny i włączono scenariusze zaawansowane
        for i in range(len(configs)):
            configs[i]['use_at'] = True
            configs[i]['at_config_file'] = args.at_config
    
    # Generowanie scenariuszy (wyłącz równoległość dla lepszej widoczności logów)
    result_paths = simulator.generate_multiple_scenarios(configs, parallel=True)
    
    print(f"Wygenerowano {len(result_paths)} scenariuszy:")
    print(f" - Normalnych: {normal_scenarios}")
    print(f" - Bootstrap: {bootstrap_scenarios}")
    print(f" - Stresowych: {stress_scenarios}")
    for path in result_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()