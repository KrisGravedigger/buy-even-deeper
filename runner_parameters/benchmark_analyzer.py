#!/opt/homebrew/bin/python3.11
# -*- coding: utf-8 -*-

"""
Analizator benchmarków dla porównania z wynikami strategii tradingowych.

Ten moduł implementuje trzy benchmarki:
1. Odsetki: Procent składany 10% w skali roku
2. HODL BTC: Kupno i trzymanie BTC przez cały okres
3. HODL Token: Kupno i trzymanie tokena przez cały okres
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import os
from collections import defaultdict
import sys
import re

class BenchmarkAnalyzer:
    """
    Klasa do analizy benchmarków i porównania ich z wynikami strategii.
    
    Wykorzystywana przez FronttestAnalyzer do rozszerzenia raportu.
    """
    
    def __init__(self, fronttest_analyzer, output_dir=None, order_amount=None):
        """
        Inicjalizacja analizatora benchmarków.

        Args:
            fronttest_analyzer: Instancja FronttestAnalyzer
            output_dir: Katalog wyjściowy dla analiz benchmarków
            order_amount: Kwota pojedynczego zlecenia
        """
        self.logger = logging.getLogger(__name__)

        # Używamy fronttest_analyzer przekazanego jako argument
        self.fronttest_analyzer = fronttest_analyzer
        self.data = fronttest_analyzer.data

        # Używamy podkatalogu benchmarks w output_dir fronttest_analyzer
        self.output_dir = Path(output_dir) if output_dir else Path(fronttest_analyzer.output_dir) / 'benchmarks'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Inicjalizacja session_id
        self.session_id = fronttest_analyzer.session_id

        # Pobieramy maksymalną liczbę zleceń z parametrów strategii
        self.max_orders = self.extract_max_orders_from_parameters()

        # Pytamy o kwotę zlecenia, jeśli nie została podana
        if order_amount is None:
            try:
                # Użyj wartości z fronttest_analyzer, jeśli dostępna, jako domyślnej
                default_amount = getattr(self.fronttest_analyzer, 'order_amount_arg', 100)
                if default_amount is None: default_amount = 100 # Dodatkowy fallback
                self.order_amount = float(input(f"Podaj kwotę pojedynczego zlecenia (domyślnie {default_amount}): ") or default_amount)
            except ValueError:
                self.logger.warning(f"Nieprawidłowa kwota. Używam domyślnej wartości {default_amount}.")
                self.order_amount = default_amount
        else:
            self.order_amount = order_amount

        # Obliczamy całkowitą inwestycję
        self.total_investment = self.order_amount * self.max_orders

        # Inicjalizacja metryk benchmarków
        self.benchmark_results = {}
        self.scenario_data = {} # Initialize as empty dictionary

        # --- DODANY BLOK ---
        # Load scenario data immediately upon initialization
        self.logger.info("Attempting to load scenario data during BenchmarkAnalyzer initialization...")
        try:
            # Use a temporary variable to avoid modifying self.scenario_data directly if load fails
            loaded_data = self.load_scenario_data()
            if loaded_data:
                self.scenario_data = loaded_data
                self.logger.info(f"Successfully loaded data for {len(self.scenario_data)} scenarios during initialization.")
            else:
                self.logger.warning("Failed to load scenario data during initialization.")
        except Exception as e_init_load:
            self.logger.error(f"Error loading scenario data during initialization: {e_init_load}", exc_info=True) # Dodano exc_info
        # --- KONIEC DODANEGO BLOKU ---


        self.logger.info(f"Inicjalizacja analizatora benchmarków. "
                    f"Kwota zlecenia: {self.order_amount}, "
                    f"Max zleceń: {self.max_orders}, "
                    f"Kwota inwestycji: {self.total_investment}")
    
    def extract_max_orders_from_parameters(self) -> int:
        """
        Ekstrahuje maksymalną liczbę zleceń z parametrów strategii.
        
        Returns:
            int: Maksymalna liczba zleceń (domyślnie 5 jeśli nie znaleziono)
        """
        try:
            # Próbujemy znaleźć parametr max_open_orders w danych
            if hasattr(self.fronttest_analyzer, 'strategy_rankings') and self.fronttest_analyzer.strategy_rankings is not None:
                # Sprawdzamy najlepszą strategię
                top_strategy = self.fronttest_analyzer.strategy_rankings.iloc[0]
                
                # Pobieramy klucz parametrów
                param_key = top_strategy.get('param_key', '')
                
                if param_key:
                    # Obsługa nowego formatu param_key (nazwa_pliku|json_params)
                    if '|' in param_key:
                        # Rozdzielamy nazwę pliku od JSON parametrów
                        parts = param_key.split('|', 1)
                        if len(parts) > 1:
                            try:
                                params = json.loads(parts[1])  # Bierzemy tylko część JSON
                            except json.JSONDecodeError:
                                self.logger.warning("Niepoprawny format JSON w param_key")
                                params = {}
                        else:
                            params = {}
                    else:
                        # Stary format - próbujemy parsować całość jako JSON
                        try:
                            params = json.loads(param_key)
                        except json.JSONDecodeError:
                            self.logger.warning("Niepoprawny format JSON w param_key")
                            params = {}
                    
                    # Szukamy parametru max_open_orders
                    if 'max_open_orders' in params:
                        return int(params['max_open_orders'])
            
            # Jeśli nie znaleziono w strategy_rankings, szukamy w wynikach dla scenariuszy
            results_by_scenario = self.data.get('results_by_scenario', {})
            for scenario_name, results in results_by_scenario.items():
                if results:
                    # Bierzemy pierwszy wynik
                    first_result = results[0]
                    
                    # Sprawdzamy parametry
                    params = first_result.get('parameters', {})
                    
                    if 'max_open_orders' in params:
                        return int(params['max_open_orders'])
            
            # Jeśli nadal nie znaleziono, zwracamy domyślną wartość
            return 5
            
        except Exception as e:
            self.logger.warning(f"Błąd podczas ekstrakcji max_open_orders: {str(e)}. Używam domyślnej wartości 5.")
            return 5
    
    def load_scenario_data(self) -> Dict[str, pd.DataFrame]:
        """
        Wczytuje dane scenariuszy z plików CSV.
        
        Returns:
            Dict[str, pd.DataFrame]: Słownik DataFrame'ów z danymi scenariuszy
        """
        self.logger.info("Wczytywanie danych scenariuszy...")
        
        scenario_data = {}
        results_by_scenario = self.data.get('results_by_scenario', {})
        
        if not results_by_scenario:
            self.logger.error("Brak danych o wynikach dla scenariuszy")
            return {}
        
        # Wczytujemy ścieżki plików scenariuszy z danych wyjściowych
        # --- NOWA LOGIKA WCZYTYWANIA ---
        # Priorytet: Spróbuj wczytać ścieżki zapisane w pliku PKL
        pkl_scenario_paths = self.fronttest_analyzer.data.get('scenarios_processed_paths', None)

        if pkl_scenario_paths is not None:
            if not pkl_scenario_paths:
                self.logger.warning("Klucz 'scenarios_processed_paths' istnieje w PKL, ale lista ścieżek jest pusta.")
            else:
                self.logger.info(f"Znaleziono {len(pkl_scenario_paths)} ścieżek scenariuszy w kluczu 'scenarios_processed_paths' z pliku PKL.")

            successful_loads_from_pkl = 0
            for file_path_str in pkl_scenario_paths:
                file_path = Path(file_path_str)
                try:
                    if not file_path.exists():
                        self.logger.error(f"Plik scenariusza ze ścieżki '{file_path_str}' (z PKL) nie istnieje. Pomijam.")
                        continue

                    scenario_name = file_path.stem
                    df = pd.read_csv(file_path)

                    if df.empty:
                        self.logger.warning(f"Plik scenariusza {file_path.name} (z PKL) jest pusty. Pomijam.")
                        continue

                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    else:
                        self.logger.warning(f"Plik scenariusza {file_path.name} (z PKL) nie ma kolumny 'timestamp'. Benchmark odsetek może nie działać.")

                    # Dodaj pełną ścieżkę do DataFrame dla późniejszego użytku (np. przez generator raportów)
                    df.scenario_file_path = str(file_path.resolve())

                    scenario_data[scenario_name] = df
                    self.logger.debug(f"Wczytano dane scenariusza: {scenario_name} z pliku {file_path.name} (ścieżka z PKL)")
                    successful_loads_from_pkl += 1

                except Exception as e:
                    self.logger.error(f"Błąd podczas wczytywania pliku {file_path_str} (ścieżka z PKL): {str(e)}")

            self.logger.info(f"Pomyślnie wczytano dane dla {successful_loads_from_pkl} z {len(pkl_scenario_paths)} scenariuszy używając ścieżek z pliku PKL.")

        else:
            # Fallback: Jeśli klucz 'scenarios_processed_paths' nie istnieje w PKL
            self.logger.warning("Nie znaleziono klucza 'scenarios_processed_paths' w danych PKL. Próbuję metody fallback: szukanie plików na podstawie nazw scenariuszy.")

            scenario_dirs = ['csv/symulacje', 'data/symulacje', 'simulations']
            successful_loads_from_fallback = 0

            for scenario_name in results_by_scenario.keys():
                if scenario_name in scenario_data: # Sprawdź, czy już nie wczytano (nie powinno się zdarzyć, ale na wszelki wypadek)
                    continue

                found_in_fallback = False
                for dir_path in scenario_dirs:
                    if found_in_fallback: break
                    if not os.path.exists(dir_path): continue

                    # Szukaj pliku pasującego do nazwy scenariusza
                    # Używamy bardziej elastycznego wzorca, dopuszczając inne części nazwy
                    potential_files = list(Path(dir_path).glob(f"*{scenario_name}*.csv"))

                    if not potential_files:
                        # Spróbuj znaleźć plik tylko po części 'simulated_...' jeśli nazwa jest długa
                        match = re.search(r"(simulated_.*?_USDT_\d+m_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_.*)", scenario_name)
                        if match:
                            short_name_part = match.group(1)
                            potential_files = list(Path(dir_path).glob(f"{short_name_part}*.csv"))


                    if potential_files:
                        # Weź pierwszy pasujący plik (lub można dodać logikę wyboru, np. najnowszy)
                        file_path = potential_files[0]
                        if len(potential_files) > 1:
                            self.logger.warning(f"Znaleziono wiele plików pasujących do '{scenario_name}' w '{dir_path}'. Używam pierwszego: {file_path.name}")

                        try:
                            df = pd.read_csv(file_path)
                            if df.empty:
                                self.logger.warning(f"Plik scenariusza {file_path.name} (fallback) jest pusty. Pomijam.")
                                continue

                            if 'timestamp' in df.columns:
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                            else:
                                self.logger.warning(f"Plik scenariusza {file_path.name} (fallback) nie ma kolumny 'timestamp'.")

                            # Dodaj pełną ścieżkę
                            df.scenario_file_path = str(file_path.resolve())

                            scenario_data[scenario_name] = df
                            self.logger.info(f"Wczytano dane scenariusza: {scenario_name} z {file_path.name} (metoda fallback)")
                            successful_loads_from_fallback += 1
                            found_in_fallback = True
                            break # Znaleziono dla tego scenariusza, idź do następnego
                        except Exception as e:
                            self.logger.error(f"Błąd podczas wczytywania pliku {file_path} (metoda fallback): {str(e)}")
                    # else: # Opcjonalnie loguj, jeśli nie znaleziono pliku w danym katalogu
                    #      self.logger.debug(f"Nie znaleziono pliku dla '{scenario_name}' w katalogu '{dir_path}'.")

                if not found_in_fallback:
                    self.logger.error(f"Nie udało się znaleźć pliku CSV dla scenariusza '{scenario_name}' metodą fallback we wszystkich sprawdzanych katalogach.")

            self.logger.info(f"Pomyślnie wczytano dane dla {successful_loads_from_fallback} scenariuszy używając metody fallback.")

        # Końcowe sprawdzenie
        if not scenario_data:
            self.logger.error("Nie udało się wczytać DANYCH dla ŻADNEGO scenariusza (ani z PKL, ani metodą fallback). Implementacja benchmarków będzie niepełna.")
        
        self.scenario_data = scenario_data
        return scenario_data
    
    # === POCZĄTEK NOWEGO KODU ===
    def calculate_interest_benchmark(self) -> Dict[str, Dict]:
        """
        Oblicza benchmark "Odsetki" - 10% rocznie, procent składany.

        Returns:
            Dict[str, Dict]: Słownik z wynikami dla każdego scenariusza
        """
        self.logger.info("Obliczanie benchmarku 'Odsetki'...")

        # Najpierw sprawdź, czy dane były w ogóle wczytane
        if not hasattr(self, 'scenario_data') or not self.scenario_data:
            self.logger.warning("Atrybut 'scenario_data' nie istnieje lub jest pusty. Próbuję wczytać ponownie.")
            
        # Sprawdź ponownie po próbie wczytania
        if not self.scenario_data: # Sprawdź, czy słownik jest pusty
            self.logger.error("Brak danych scenariuszy (nawet po próbie wczytania) do obliczenia benchmarku 'Odsetki'.")
            # Zwróć pusty słownik, ale zapisz go też w self.benchmark_results, jeśli atrybut istnieje
            if hasattr(self, 'benchmark_results'):
                self.benchmark_results['interest'] = {}
            return {}

        interest_results = {}
        start_date = None
        end_date = None
        first_valid_scenario_name = None

        # Znajdź PIERWSZY scenariusz, który ma kolumnę 'timestamp' i nie jest pusty
        for scenario_name, df in self.scenario_data.items():
            if df is not None and not df.empty and 'timestamp' in df.columns:
                try:
                    # Sprawdź, czy timestampy są poprawne
                    temp_start = pd.to_datetime(df['timestamp'].min())
                    temp_end = pd.to_datetime(df['timestamp'].max())
                    # Sprawdź, czy daty nie są NaT (Not a Timestamp)
                    if pd.notna(temp_start) and pd.notna(temp_end):
                        start_date = temp_start
                        end_date = temp_end
                        first_valid_scenario_name = scenario_name
                        self.logger.info(f"Używam dat ze scenariusza '{first_valid_scenario_name}' do obliczenia benchmarku odsetek.")
                        break # Znaleziono pierwszy poprawny, wychodzimy z pętli
                except Exception as e_time:
                    self.logger.warning(f"Błąd konwersji timestamp w scenariuszu {scenario_name}: {e_time}. Szukam dalej.")
                    continue # Przejdź do następnego scenariusza

        if start_date is None or end_date is None:
            self.logger.error(f"Nie znaleziono żadnego scenariusza z poprawną kolumną 'timestamp' do obliczenia benchmarku 'Odsetki'.")
            # Zapisz pusty wynik również w self.benchmark_results
            if hasattr(self, 'benchmark_results'):
                self.benchmark_results['interest'] = {}
            return {}
    # === KONIEC NOWEGO KODU ===

        # Obliczamy liczbę dni w symulacji (używamy total_seconds dla precyzji)
        simulation_duration = end_date - start_date
        # Dodajemy małą wartość, aby uniknąć 0 dni, jeśli start == end
        simulation_days = max(simulation_duration.total_seconds() / (24 * 3600), 0.0001)
    
    def calculate_hodl_btc_benchmark(self) -> Dict[str, Dict]:
        """
        Oblicza benchmark "HODL BTC" - kupno i trzymanie BTC.
        Obsługuje scenariusz, gdy analizowana para to BTC/USDT.

        Returns:
            Dict[str, Dict]: Słownik z wynikami dla każdego scenariusza
        """
        self.logger.info("Obliczanie benchmarku 'HODL BTC'...")

        if not self.scenario_data:
            # Nie próbuj wczytywać ponownie, jeśli init się nie powiódł
            self.logger.error("Brak danych scenariuszy (self.scenario_data jest pusty) do obliczenia benchmarku 'HODL BTC'.")
            self.benchmark_results['hodl_btc'] = {} # Zapisz pusty wynik
            return {}

        hodl_btc_results = {}

        for scenario_name, df in self.scenario_data.items():
            if df is None or df.empty:
                self.logger.warning(f"Pusty DataFrame dla scenariusza {scenario_name}. Pomijam HODL BTC.")
                continue

            # Sprawdź timestamp
            if 'timestamp' not in df.columns:
                self.logger.warning(f"Brak kolumny 'timestamp' w danych scenariusza {scenario_name}. Pomijam HODL BTC.")
                continue

            btc_price_column_name = None
            # --- NOWA LOGIKA WYBORU KOLUMNY CENY BTC ---
            if 'btc_average_price' in df.columns:
                btc_price_column_name = 'btc_average_price'
                self.logger.debug(f"HODL BTC dla {scenario_name}: Używam kolumny 'btc_average_price'.")
            elif 'average_price' in df.columns:
                # Sprawdź, czy to scenariusz BTC/USDT
                # Użyjemy nazwy scenariusza jako wskazówki, jeśli nie ma kolumny 'main_symbol'
                is_btc_scenario = False
                if 'main_symbol' in df.columns and isinstance(df['main_symbol'].iloc[0], str) and 'BTC' in df['main_symbol'].iloc[0].upper():
                    is_btc_scenario = True
                elif 'BTC' in scenario_name.upper() and 'USDT' in scenario_name.upper(): # Sprawdź nazwę scenariusza
                    is_btc_scenario = True
                    self.logger.debug(f"HODL BTC dla {scenario_name}: Brak 'btc_average_price', ale nazwa wskazuje na BTC/USDT.")

                if is_btc_scenario:
                    btc_price_column_name = 'average_price'
                    self.logger.debug(f"HODL BTC dla {scenario_name}: Używam kolumny 'average_price' (scenariusz BTC/USDT).")
                else:
                    self.logger.warning(f"Brak kolumny 'btc_average_price' i scenariusz '{scenario_name}' nie wydaje się być BTC/USDT. Nie można obliczyć HODL BTC.")
                    continue # Przejdź do następnego scenariusza
            else:
                # Jeśli brakuje obu kolumn cenowych
                self.logger.warning(f"Brak kolumn 'btc_average_price' oraz 'average_price' w danych scenariusza {scenario_name}. Pomijam HODL BTC.")
                continue
            # --- KONIEC NOWEJ LOGIKI ---

            # Pobieramy ceny BTC z wybranej kolumny
            try:
                # Sprawdź czy kolumna istnieje (dodatkowe zabezpieczenie)
                if btc_price_column_name not in df.columns:
                    self.logger.error(f"Kolumna '{btc_price_column_name}' jednak nie istnieje w DF dla {scenario_name}? Dostępne: {list(df.columns)}. Pomijam.")
                    continue
                btc_prices = df[btc_price_column_name].values
                if len(btc_prices) == 0:
                    self.logger.warning(f"Kolumna '{btc_price_column_name}' jest pusta dla scenariusza {scenario_name}. Pomijam HODL BTC.")
                    continue
                # Upewnij się, że ceny są numeryczne
                if not np.issubdtype(btc_prices.dtype, np.number):
                    self.logger.warning(f"Dane w kolumnie '{btc_price_column_name}' nie są numeryczne dla {scenario_name}. Pomijam HODL BTC.")
                    continue

                # Usuń wartości NaN lub nieskończone, jeśli istnieją
                valid_prices = btc_prices[np.isfinite(btc_prices)]
                if len(valid_prices) < 2: # Potrzebujemy co najmniej ceny startowej i końcowej
                    self.logger.warning(f"Niewystarczająca liczba poprawnych cen w kolumnie '{btc_price_column_name}' dla {scenario_name}. Pomijam HODL BTC.")
                    continue
                btc_prices = valid_prices # Używaj tylko poprawnych cen

            except KeyError:
                self.logger.error(f"KeyError przy próbie dostępu do kolumny '{btc_price_column_name}' dla scenariusza {scenario_name}. Pomijam HODL BTC.")
                continue
            except Exception as e_price:
                self.logger.error(f"Nieoczekiwany błąd podczas pobierania cen BTC dla {scenario_name}: {e_price}. Pomijam HODL BTC.")
                continue

            # Obliczamy ilość zakupionego BTC
            initial_btc_price = btc_prices[0]
            if initial_btc_price <= 0:
                self.logger.warning(f"Początkowa cena BTC ({initial_btc_price}) jest nieprawidłowa dla {scenario_name}. Pomijam HODL BTC.")
                continue
            btc_amount = self.total_investment / initial_btc_price

            # Obliczamy końcowe metryki
            final_btc_price = btc_prices[-1]
            final_capital = btc_amount * final_btc_price
            final_profit = final_capital - self.total_investment
            final_profit_pct = (final_profit / self.total_investment) * 100.0

            # Obliczamy maksymalny drawdown
            peak = self.total_investment
            max_drawdown = 0.0

            for price in btc_prices:
                current_value = btc_amount * price
                peak = max(peak, current_value)
                drawdown = peak - current_value
                max_drawdown = max(max_drawdown, drawdown)

            hodl_btc_results[scenario_name] = {
                'final_capital': final_capital,
                'final_profit': final_profit,
                'final_profit_pct': final_profit_pct,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100.0 / self.total_investment if self.total_investment > 0 else 0.0
            }

            self.logger.info(f"Benchmark 'HODL BTC' dla scenariusza {scenario_name}: "
                        f"zysk {final_profit_pct:.2f}%, kapitał końcowy {final_capital:.2f} (użyto kolumny: {btc_price_column_name})") # Dodano info o kolumnie

        self.benchmark_results['hodl_btc'] = hodl_btc_results
        return hodl_btc_results
    
    def calculate_hodl_token_benchmark(self) -> Dict[str, Dict]:
        """
        Oblicza benchmark "HODL Token" - kupno i trzymanie tokena.
        
        Returns:
            Dict[str, Dict]: Słownik z wynikami dla każdego scenariusza
        """
        self.logger.info("Obliczanie benchmarku 'HODL Token'...")
        
        if not self.scenario_data:
            self.load_scenario_data()
        
        hodl_token_results = {}
        
        for scenario_name, df in self.scenario_data.items():
            if 'timestamp' not in df.columns or 'average_price' not in df.columns:
                self.logger.warning(f"Brak wymaganych kolumn w danych scenariusza {scenario_name}")
                continue
            
            # Pobieramy ceny tokena
            token_prices = df['average_price'].values
            
            # Obliczamy ilość zakupionego tokena
            initial_token_price = token_prices[0]
            token_amount = self.total_investment / initial_token_price
            
            # Obliczamy końcowe metryki
            final_token_price = token_prices[-1]
            final_capital = token_amount * final_token_price
            final_profit = final_capital - self.total_investment
            final_profit_pct = (final_profit / self.total_investment) * 100
            
            # Obliczamy maksymalny drawdown
            peak = self.total_investment
            max_drawdown = 0
            
            for price in token_prices:
                current_value = token_amount * price
                peak = max(peak, current_value)
                drawdown = peak - current_value
                max_drawdown = max(max_drawdown, drawdown)
            
            hodl_token_results[scenario_name] = {
                'final_capital': final_capital,
                'final_profit': final_profit,
                'final_profit_pct': final_profit_pct,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100 / self.total_investment
            }
            
            self.logger.info(f"Benchmark 'HODL Token' dla scenariusza {scenario_name}: "
                           f"zysk {final_profit_pct:.2f}%, kapitał końcowy {final_capital:.2f}")
        
        self.benchmark_results['hodl_token'] = hodl_token_results
        return hodl_token_results
    
    def extract_strategy_performance(self, top_n=10) -> Dict[str, Dict]:
        """
        Wyciąga wyniki strategii z analizatora fronttestingu.
        
        Args:
            top_n: Liczba najlepszych strategii do uwzględnienia
            
        Returns:
            Dict[str, Dict]: Słownik z wynikami dla każdego scenariusza
        """
        self.logger.info(f"Ekstrahowanie wyników dla top {top_n} strategii...")
        
        # Jeśli nie mamy jeszcze porównania strategii, obliczamy je
        if getattr(self.fronttest_analyzer, 'strategy_rankings', None) is None:
            self.fronttest_analyzer.compare_strategy_configurations()
        
        strategy_results = {}
        
        # Pobieramy top N strategii
        top_strategies = self.fronttest_analyzer.strategy_rankings.head(top_n)
        
        # Wyciągamy wyniki dla każdego scenariusza
        results_by_scenario = self.data.get('results_by_scenario', {})
        
        if not results_by_scenario:
            self.logger.error("Brak danych o wynikach dla scenariuszy")
            return {}
        
        # Pobieramy klucze parametrów dla top strategii
        top_param_keys = set(top_strategies['param_key'].values)
        
        # Agregujemy wyniki dla top strategii dla każdego scenariusza
        for scenario_name, results in results_by_scenario.items():
            scenario_strategy_results = []
            
            for result in results:
                # Pobieramy klucz parametrów
                params = result.get('parameters', {})
                param_key = json.dumps({k: params[k] for k in sorted(params.keys())})
                
                if param_key in top_param_keys:
                    scenario_strategy_results.append(result)
            
            if scenario_strategy_results:
                # Obliczamy średnie wyniki dla top strategii
                avg_profit = np.mean([r.get('avg_profit', 0) for r in scenario_strategy_results])
                max_drawdown_estimate = 0  # Brak informacji o drawdown w wynikach
                
                # Przygotowujemy dane dla strategii
                strategy_results[scenario_name] = {
                    'avg_profit': avg_profit,
                    'final_profit_pct': avg_profit,  # Zakładamy, że avg_profit to średni zysk procentowy
                    'max_drawdown_pct': max_drawdown_estimate  # Brak informacji o drawdown w wynikach
                }
                
                self.logger.info(f"Średni zysk dla top strategii w scenariuszu {scenario_name}: {avg_profit:.2f}%")
            else:
                self.logger.warning(f"Brak wyników dla top strategii w scenariuszu {scenario_name}")
        
        return strategy_results
    
    def run_benchmark_analysis(self) -> Dict[str, Any]:
        """
        Przeprowadza pełną analizę benchmarków.

        Returns:
            Dict[str, Any]: Słownik z wynikami
        """
        self.logger.info("Rozpoczynam pełną analizę benchmarków...")

        # Wczytywanie danych scenariuszy - USUNIĘTO WYWOŁANIE STĄD

        # --- DODANY BLOK SPRAWDZAJĄCY ---
        if not self.scenario_data:
            self.logger.error("Benchmark analysis cannot proceed: scenario data is missing (was not loaded during init).")
            # Spróbujmy załadować jeszcze raz jako ostatnią deskę ratunku
            self.logger.info("Attempting to load scenario data again before failing benchmark analysis...")
            try:
                loaded_data = self.load_scenario_data()
                if loaded_data:
                    self.scenario_data = loaded_data
                    self.logger.info(f"Successfully loaded data for {len(self.scenario_data)} scenarios on second attempt.")
                else:
                    self.logger.error("Second attempt to load scenario data failed.")
                    return {
                        'benchmark_results': {},
                        'benchmark_summary': "Error: Scenario data missing for benchmark analysis."
                    }
            except Exception as e_second_load:
                self.logger.error(f"Error during second attempt to load scenario data: {e_second_load}", exc_info=True)
                return {
                    'benchmark_results': {},
                    'benchmark_summary': "Error: Scenario data missing for benchmark analysis."
                }
        # --- KONIEC DODANEGO BLOKU SPRAWDZAJĄCEGO ---


        # Obliczanie benchmarków
        self.calculate_interest_benchmark()
        self.calculate_hodl_btc_benchmark()
        self.calculate_hodl_token_benchmark()

        # Generowanie podsumowania
        benchmark_summary = self.generate_benchmark_summary()

        return {
            'benchmark_results': self.benchmark_results,
            'benchmark_summary': benchmark_summary
        }
    
    def generate_benchmark_summary(self) -> str:
        """
        Generuje podsumowanie wyników benchmarków w formacie Markdown.
        
        Returns:
            str: Podsumowanie w formacie Markdown
        """
        summary = []
        
        # Przygotowujemy dane
        if not self.benchmark_results:
            return "Brak danych do wygenerowania podsumowania benchmarków.\n"
        
        # Obliczamy średnie wyniki dla każdego typu benchmarku
        avg_results = {
            'interest': {'final_profit_pct': [], 'max_drawdown_pct': []},
            'hodl_btc': {'final_profit_pct': [], 'max_drawdown_pct': []},
            'hodl_token': {'final_profit_pct': [], 'max_drawdown_pct': []}
        }
        
        for benchmark_type, results in self.benchmark_results.items():
            for scenario, result in results.items():
                avg_results[benchmark_type]['final_profit_pct'].append(result['final_profit_pct'])
                avg_results[benchmark_type]['max_drawdown_pct'].append(result['max_drawdown_pct'])
        
        # Porównanie z benchmarkiem "Odsetki"
        if 'interest' in avg_results and avg_results['interest']['final_profit_pct']:
            avg_interest_profit = sum(avg_results['interest']['final_profit_pct']) / len(avg_results['interest']['final_profit_pct'])
            summary.append("### Porównanie z benchmarkiem \"Odsetki\"\n")
            summary.append(f"Średni zysk z benchmarku \"Odsetki\" (10% rocznie): **{avg_interest_profit:.2f}%**\n")
            summary.append("Interpretacja:\n")
            summary.append("- Ten benchmark reprezentuje stosunkowo bezpieczną inwestycję o stałym tempie wzrostu.\n")
            summary.append("- Strategie tradingowe powinny osiągać wyższe stopy zwrotu, aby uzasadnić dodatkowe ryzyko.\n\n")
        
        # Porównanie z benchmarkiem "HODL BTC"
        if 'hodl_btc' in avg_results and avg_results['hodl_btc']['final_profit_pct']:
            avg_btc_profit = sum(avg_results['hodl_btc']['final_profit_pct']) / len(avg_results['hodl_btc']['final_profit_pct'])
            avg_btc_drawdown = sum(avg_results['hodl_btc']['max_drawdown_pct']) / len(avg_results['hodl_btc']['max_drawdown_pct'])
            
            summary.append("### Porównanie z benchmarkiem \"HODL BTC\"\n")
            summary.append(f"Średni zysk z benchmarku \"HODL BTC\": **{avg_btc_profit:.2f}%**\n")
            summary.append(f"Średni maksymalny drawdown: **{avg_btc_drawdown:.2f}%**\n")
            summary.append("Interpretacja:\n")
            summary.append("- Ten benchmark pokazuje, jak prosta strategia kupna i trzymania BTC wypada na tle aktywnego tradingu.\n")
            summary.append("- W okresach hossy HODL może dawać lepsze wyniki, natomiast w okresach bessy lub konsolidacji aktywne strategie często wypadają lepiej.\n\n")
        
        # Porównanie z benchmarkiem "HODL Token"
        if 'hodl_token' in avg_results and avg_results['hodl_token']['final_profit_pct']:
            avg_token_profit = sum(avg_results['hodl_token']['final_profit_pct']) / len(avg_results['hodl_token']['final_profit_pct'])
            avg_token_drawdown = sum(avg_results['hodl_token']['max_drawdown_pct']) / len(avg_results['hodl_token']['max_drawdown_pct'])
            
            summary.append("### Porównanie z benchmarkiem \"HODL Token\"\n")
            summary.append(f"Średni zysk z benchmarku \"HODL Token\": **{avg_token_profit:.2f}%**\n")
            summary.append(f"Średni maksymalny drawdown: **{avg_token_drawdown:.2f}%**\n")
            summary.append("Interpretacja:\n")
            summary.append("- Ten benchmark jest szczególnie istotny, gdyż pokazuje, czy aktywne tradowanie danym tokenem daje lepsze rezultaty niż proste trzymanie.\n")
            summary.append("- Wyższa zmienność altcoinów może dawać większe możliwości zysku, ale też wiąże się z większym ryzykiem.\n\n")
        
        # Ogólne wnioski
        summary.append("### Ogólne wnioski\n")
        summary.append("Na podstawie przeprowadzonej analizy benchmarków można wyciągnąć następujące wnioski:\n\n")
        
        # Podsumowanie
        summary.append("### Rekomendacje\n")
        summary.append("- **Dywersyfikacja**: Rozważ alokację środków między strategie aktywne i pasywne w zależności od warunków rynkowych.\n")
        summary.append("- **Dostosowanie parametrów**: Monitoruj i dostosowuj parametry strategii w oparciu o zmieniające się warunki rynkowe.\n")
        summary.append("- **Zarządzanie ryzykiem**: Zwróć szczególną uwagę na maksymalne drawdowny, które mogą być kluczowym czynnikiem w długoterminowej rentowności.\n")
        
        return "\n".join(summary)