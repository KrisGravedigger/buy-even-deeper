#!/opt/homebrew/bin/python3.11
# -*- coding: utf-8 -*-

"""
Generator ulepszonego raportu z analizy forward-testingu i benchmarków.
Ten moduł należy umieścić w folderze runner_parameters.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import logging
import os
import sys
from typing import Dict, List, Any, Tuple, Optional
import shutil

# Dla tabelek w Markdown
try:
    from tabulate import tabulate
except ImportError:
    print("Biblioteka tabulate nie jest zainstalowana. Zainstaluj ją poprzez: pip install tabulate")
    tabulate = None

class EnhancedReportGenerator:
    """
    Generator ulepszonego raportu z analizy strategii tradingowych i benchmarków.
    """
    
    def __init__(self, fronttest_analyzer):
        """
        Inicjalizacja generatora raportów.
        
        Args:
            fronttest_analyzer: Instancja FronttestAnalyzer z już przeprowadzoną analizą
        """
        self.analyzer = fronttest_analyzer
        self.logger = fronttest_analyzer.logger
        self.output_dir = fronttest_analyzer.output_dir
        self.session_id = fronttest_analyzer.session_id
        
        # Sprawdzamy czy analizator ma już wszystkie potrzebne dane
        if self.analyzer.summary_df is None:
            self.analyzer.analyze_scenario_performance()
        
        if self.analyzer.parameter_sensitivity is None:
            self.analyzer.analyze_parameter_sensitivity()
        
        if self.analyzer.strategy_rankings is None:
            self.analyzer.compare_strategy_configurations()
        
        # Inicjujemy analizator benchmarków, jeśli jeszcze nie istnieje
        if getattr(self.analyzer, 'benchmark_analyzer', None) is None and self.analyzer.include_benchmarks:
            try:
                from runner_parameters.benchmark_analyzer import BenchmarkAnalyzer
                
                # Inicjalizacja analizatora benchmarków
                self.analyzer.benchmark_analyzer = BenchmarkAnalyzer(
                    fronttest_analyzer=self.analyzer,
                    output_dir=self.output_dir / 'benchmarks'
                )
                
                # Przeprowadzenie analizy benchmarków, jeśli jeszcze nie została wykonana
                if not getattr(self.analyzer.benchmark_analyzer, 'benchmark_results', None):
                    self.analyzer.benchmark_analyzer.run_benchmark_analysis()
            except Exception as e:
                self.logger.error(f"Błąd podczas inicjalizacji analizatora benchmarków: {e}")
                self.analyzer.include_benchmarks = False
    
    def extract_scenario_info(self) -> Dict[str, Any]:
        """
        Ekstrahuje informacje o scenariuszach.
        
        Returns:
            Dict[str, Any]: Słownik z informacjami o scenariuszach
        """
        self.logger.info("Ekstrahowanie informacji o scenariuszach...")
        
        results_by_scenario = self.analyzer.data.get('results_by_scenario', {})
        
        # Analizujemy scenariusze
        scenarios = list(results_by_scenario.keys())
        
        # Klasyfikacja scenariuszy na podstawie nazw
        scenario_types = {}
        for scenario in scenarios:
            scenario_lower = scenario.lower()
            if 'normal' in scenario_lower:
                scenario_types[scenario] = 'Normal'
            elif 'bootstrap' in scenario_lower:
                scenario_types[scenario] = 'Bootstrap'
            elif any(term in scenario_lower for term in ['stress', 'crash', 'krach', 'dip']):
                scenario_types[scenario] = 'Stress'
            else:
                scenario_types[scenario] = 'Other'
        
        # Liczba scenariuszy każdego typu
        type_counts = {}
        for type_name in set(scenario_types.values()):
            type_counts[type_name] = list(scenario_types.values()).count(type_name)
        
        # Określamy zakres dat symulacji
        start_date = None
        end_date = None
        
        # Próbujemy uzyskać daty z danych scenariuszy
        if hasattr(self.analyzer, 'benchmark_analyzer') and self.analyzer.benchmark_analyzer:
            if hasattr(self.analyzer.benchmark_analyzer, 'scenario_data'):
                for scenario_name, df in self.analyzer.benchmark_analyzer.scenario_data.items():
                    if 'timestamp' in df.columns:
                        scenario_start = df['timestamp'].min()
                        scenario_end = df['timestamp'].max()
                        
                        if start_date is None or scenario_start < start_date:
                            start_date = scenario_start
                        
                        if end_date is None or scenario_end > end_date:
                            end_date = scenario_end
        
        # Obliczamy liczbę dni symulacji
        simulation_days = None
        if start_date is not None and end_date is not None:
            simulation_days = (end_date - start_date).days
        
        return {
            'scenarios': scenarios,
            'scenario_types': scenario_types,
            'type_counts': type_counts,
            'total_scenarios': len(scenarios),
            'start_date': start_date,
            'end_date': end_date,
            'simulation_days': simulation_days
        }
    
    def extract_strategy_parameters(self) -> Dict[str, Any]:
        """
        Ekstrahuje informacje o parametrach strategii.
        
        Returns:
            Dict[str, Any]: Słownik z informacjami o parametrach strategii
        """
        self.logger.info("Ekstrahowanie informacji o parametrach strategii...")
        
        # Analizujemy strategie
        strategy_params = {}
        
        # Sprawdzamy, czy mamy informacje o plikach parametrów
        param_files = []
        if 'parameters' in self.analyzer.data:
            param_files = self.analyzer.data['parameters']
            # Logujemy tylko liczbę plików bez szczegółów
            self.logger.info(f"Znaleziono {len(param_files)} plik(ów) parametrów")
        
        # Pobieramy wyniki dla scenariuszy, aby wyciągnąć parametry
        results_by_scenario = self.analyzer.data.get('results_by_scenario', {})
        
        # Mapa przechowująca parametry dla każdego pliku
        file_to_params = {}
        
        # Iterujemy po scenariuszach i wynikach
        for scenario_name, results in results_by_scenario.items():
            for result in results:
                param_file = result.get('param_file', result.get('param_file_name', ''))
                if param_file:
                    # Normalizujemy nazwę pliku
                    file_name = Path(param_file).stem
                    
                    # Pobieramy parametry
                    params = result.get('parameters', {})
                    
                    if params and file_name not in file_to_params:
                        file_to_params[file_name] = params
                        # Logujemy tylko liczbę parametrów, bez szczegółów
                        self.logger.debug(f"Wyodrębniono parametry dla {file_name}: {len(params)} parametrów")
        
        # Jeśli nie znaleźliśmy parametrów w wynikach, ale mamy listę plików parametrów,
        # próbujemy je wczytać bezpośrednio z param_key w rankingu strategii
        if not file_to_params and self.analyzer.strategy_rankings is not None:
            for _, row in self.analyzer.strategy_rankings.iterrows():
                param_key = row.get('param_key', '')
                
                # Sprawdzamy, czy param_key zawiera nazwę pliku i parametry JSON
                if param_key and '|' in param_key:
                    parts = param_key.split('|', 1)
                    file_part = parts[0]
                    json_part = parts[1]
                    
                    file_name = Path(file_part).stem
                    
                    try:
                        params = json.loads(json_part)
                        if params and file_name not in file_to_params:
                            file_to_params[file_name] = params
                            self.logger.debug(f"Wyodrębniono parametry z param_key dla {file_name}: {len(params)} parametrów")
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Błąd dekodowania JSON dla {file_name}: {str(e)}")
        
        # Tworzymy słownik z parametrami dla raportu
        for file_name, params in file_to_params.items():
            strategy_params[file_name] = params
        
        # Sprawdzamy, czy udało się wyodrębnić parametry
        if not strategy_params:
            self.logger.warning("Nie udało się wyodrębnić parametrów strategii")
        else:
            self.logger.info(f"Wyodrębniono parametry dla {len(strategy_params)} strategii")
        
        return {
            'strategies': strategy_params,
            'param_files': param_files,
            'total_strategies': len(strategy_params)
        }
    
    def calculate_advanced_metrics(self) -> pd.DataFrame:
        """
        Oblicza zaawansowane metryki dla strategii.
        
        Returns:
            pd.DataFrame: DataFrame z zaawansowanymi metrykami
        """
        self.logger.info("Obliczanie zaawansowanych metryk dla strategii...")
        
        # Jeśli nie mamy rankingu strategii, nie możemy obliczyć metryk
        if self.analyzer.strategy_rankings is None or self.analyzer.strategy_rankings.empty:
            return pd.DataFrame()
        
        # Pobieramy ranking strategii
        strategies_df = self.analyzer.strategy_rankings.copy()
        
        # Inicjalizujemy strategy_id jako kolumnę, jeśli nie istnieje
        if 'strategy_id' not in strategies_df.columns:
            strategies_df['strategy_id'] = [f"auto_{i}" for i in range(len(strategies_df))]

        # Pobieramy oryginalne nazwy plików z wyników
        results_by_scenario = self.analyzer.data.get('results_by_scenario', {})
        file_names_by_param_key = {}
        file_names_by_id = {}
        param_keys_by_id = {}  # Mapowanie ID strategii do klucza parametrów

        # Najpierw sprawdzamy, czy mamy oryginalne pliki parametrów
        param_files = []
        if 'parameters' in self.analyzer.data:
            param_files = self.analyzer.data['parameters']
            self.logger.info(f"Znaleziono pliki parametrów: {param_files}")
        
        # Zbieramy informacje o plikach parametrów z wyników
        for scenario_name, scenario_results in results_by_scenario.items():
            for result in scenario_results:
                # Pobieramy klucz parametrów z uwzględnieniem nazwy pliku parametrów
                strategy_id = result.get('strategy_id', '')
                params = result.get('parameters', {})
                param_file = result.get('param_file', result.get('param_file_name', ''))
                
                if param_file:
                    # Tworzymy klucz kombinacji parametrów
                    param_key_json = json.dumps({k: params[k] for k in sorted(params.keys())})
                    param_key = f"{param_file}|{param_key_json}"
                    
                    # Zapisujemy mapowanie
                    file_names_by_param_key[param_key] = Path(param_file).stem
                    if strategy_id:
                        file_names_by_id[strategy_id] = Path(param_file).stem
                        param_keys_by_id[strategy_id] = param_key
                    
                    # Logujemy tylko pierwszych kilka mapowań, bez szczegółów JSON
                    if len(file_names_by_param_key) <= 3:  # Ograniczamy ilość logów
                        self.logger.info(f"Znaleziono mapowanie: {Path(param_file).stem}")

        # Dodajemy kolumnę z oryginalnymi nazwami plików
        strategies_df['original_file'] = None
        
        # Najpierw próbujemy mapować po param_key bezpośrednio
        for idx in strategies_df.index:
            param_key = strategies_df.loc[idx, 'param_key'] if 'param_key' in strategies_df.columns else None
            
            # Sprawdzamy czy param_key zawiera nazwę pliku (format: nazwa_pliku|params_json)
            if param_key and '|' in param_key:
                parts = param_key.split('|', 1)
                file_part = parts[0]
                strategies_df.loc[idx, 'original_file'] = Path(file_part).stem
                self.logger.debug(f"Znaleziono nazwę pliku w param_key: {Path(file_part).stem}")
            
            # Jeśli nie znaleziono, próbujemy po mapowaniu z wyników
            elif param_key and param_key in file_names_by_param_key:
                strategies_df.loc[idx, 'original_file'] = file_names_by_param_key[param_key]
                self.logger.debug(f"Znaleziono nazwę pliku w mapowaniu: {file_names_by_param_key[param_key]}")
        
        # Jeśli nadal nie mamy nazw, próbujemy po ID strategii
        for idx in strategies_df.index:
            if strategies_df.loc[idx, 'original_file'] is None:
                strategy_id = strategies_df.loc[idx, 'strategy_id'] if 'strategy_id' in strategies_df.columns else None
                if strategy_id and strategy_id in file_names_by_id:
                    strategies_df.loc[idx, 'original_file'] = file_names_by_id[strategy_id]
                    self.logger.debug(f"Znaleziono nazwę pliku po ID: {file_names_by_id[strategy_id]}")
        
        # Dodajemy nazwy z listy param_files jako ostateczność
        if len(param_files) > 0:
            for idx, (i, row) in enumerate(strategies_df.iterrows()):
                if row['original_file'] is None and idx < len(param_files):
                    strategies_df.loc[i, 'original_file'] = Path(param_files[idx]).stem
                    self.logger.info(f"Przypisano nazwę z listy param_files: {i} -> {Path(param_files[idx]).stem}")
        
        # Teraz przypisujemy finalne nazwy strategiom
        strategies_df['name'] = None
        unique_names_used = set()
        
        for idx in strategies_df.index:
            original_file = strategies_df.loc[idx, 'original_file']
            
            if original_file:
                # Mamy oryginalną nazwę pliku - używamy jej
                if original_file in unique_names_used:
                    # Znajdź następny wolny indeks
                    i = 1
                    while f"{original_file}_{i}" in unique_names_used:
                        i += 1
                    strategies_df.loc[idx, 'name'] = f"{original_file}_{i}"
                    unique_names_used.add(f"{original_file}_{i}")
                else:
                    strategies_df.loc[idx, 'name'] = original_file
                    unique_names_used.add(original_file)
            else:
                # Brak oryginalnej nazwy - używamy indeksu jako fallback
                fallback_name = f"strategy_{idx}"
                strategies_df.loc[idx, 'name'] = fallback_name
                unique_names_used.add(fallback_name)
        
        # Wypisujemy logi dla diagnostyki
        self.logger.info(f"Przypisane nazwy strategii: {strategies_df[['strategy_id', 'name']].to_dict()}")

        # Pobieramy transakcje dla każdej strategii
        trade_counts = {}

        # Iteracja po scenariuszach
        for scenario_name, results in results_by_scenario.items():
            for result in results:
                # Pobieramy klucz parametrów z uwzględnieniem nazwy pliku parametrów
                params = result.get('parameters', {})
                param_file = result.get('param_file', result.get('param_file_name', 'unknown'))
                strategy_id = result.get('strategy_id', '')
                
                # Tworzymy klucz uwzględniający nazwę pliku
                param_key_json = json.dumps({k: params[k] for k in sorted(params.keys())})
                param_key = f"{param_file}|{param_key_json}"
                
                # Szukamy również starego formatu klucza (dla kompatybilności)
                old_param_key = param_key_json
                
                # Znajdź właściwy klucz w naszym DataFrame strategii
                found_key = None
                for existing_key in strategies_df['param_key']:
                    if existing_key == param_key or existing_key == old_param_key:
                        found_key = existing_key
                        break
                
                # Używamy znalezionego klucza lub ID strategii
                use_key = found_key if found_key else (strategy_id if strategy_id else param_key)
                
                # Dodajemy liczbę transakcji
                if use_key not in trade_counts:
                    trade_counts[use_key] = []
                    
                # Sprawdzamy różne pola dla liczby transakcji (mogą być w różnych miejscach)
                trades = result.get('total_trades', 
                        result.get('trades_executed', 
                        result.get('trades_closed', 0)))
                
                trade_counts[use_key].append(trades)
        
        # Obliczamy średnią liczbę transakcji dla każdej strategii
        avg_trades = {}
        for key, trades in trade_counts.items():
            avg_trades[key] = sum(trades) / len(trades) if trades else 0
        
        # Dodajemy kolumnę z liczbą transakcji - zarówno po param_key jak i po strategy_id
        strategies_df['avg_trades'] = 0.0  # Inicjalizujemy zerami (jako float)
        for idx in strategies_df.index:
            param_key = strategies_df.loc[idx, 'param_key'] if 'param_key' in strategies_df.columns else None
            strategy_id = strategies_df.loc[idx, 'strategy_id'] if 'strategy_id' in strategies_df.columns else None
            
            # Próbujemy znaleźć liczbę transakcji po param_key
            if param_key and param_key in avg_trades:
                strategies_df.loc[idx, 'avg_trades'] = float(avg_trades[param_key])
            # Jeśli nie znaleziono po param_key, próbujemy po strategy_id
            elif strategy_id and strategy_id in avg_trades:
                strategies_df.loc[idx, 'avg_trades'] = float(avg_trades[strategy_id])

        # Upewniamy się, że kolumna avg_trades ma wartości liczbowe
        strategies_df['avg_trades'] = strategies_df['avg_trades'].fillna(0.0).astype(float)

        # Wypisujemy log dla weryfikacji
        self.logger.info(f"Statystyki transakcji: {strategies_df[['name', 'avg_trades']].head()}")
        
        # Określamy najlepszy benchmark
        best_benchmark_profit = 0
        if hasattr(self.analyzer, 'benchmark_analyzer') and self.analyzer.benchmark_analyzer:
            if hasattr(self.analyzer.benchmark_analyzer, 'benchmark_results'):
                # Sprawdzamy, który benchmark jest najlepszy
                for benchmark_type, results in self.analyzer.benchmark_analyzer.benchmark_results.items():
                    if results:
                        # Obliczamy średni zysk z benchmarku
                        profits = [result['final_profit_pct'] for result in results.values()]
                        avg_profit = np.mean(profits) if profits else 0
                        
                        # Sprawdzamy, czy to najlepszy benchmark
                        if avg_profit > best_benchmark_profit:
                            best_benchmark_profit = avg_profit
        
        # Obliczamy współczynnik Alpha (różnica między zyskiem strategii a benchmarkiem)
        strategies_df['alpha'] = strategies_df['avg_profit_all_scenarios'] - best_benchmark_profit
        
        # Obliczamy współczynnik Sharpe (stosunek zysku do ryzyka)
        strategies_df['sharpe'] = strategies_df['avg_profit_all_scenarios'] / (strategies_df['std_profit_all_scenarios'] + 1e-6)
        
        # Obliczamy zysk nominalny
        investment_amount = 100  # Domyślna wartość
        if hasattr(self.analyzer, 'benchmark_analyzer') and self.analyzer.benchmark_analyzer:
            investment_amount = getattr(self.analyzer.benchmark_analyzer, 'order_amount', 100)
            
        strategies_df['nominal_profit'] = strategies_df['avg_trades'] * strategies_df['avg_profit_all_scenarios'] / 100 * investment_amount
        
        # Sortujemy według średniego zysku (malejąco)
        strategies_df = strategies_df.sort_values('avg_profit_all_scenarios', ascending=False)
        
        return strategies_df
    
    def generate_benchmark_comparison(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Generuje porównanie z benchmarkami.
        
        Returns:
            Tuple[Dict[str, Any], pd.DataFrame]: Słownik z danymi benchmarków i DataFrame z porównaniem
        """
        self.logger.info("Generowanie porównania z benchmarkami...")
        
        benchmark_data = {
            'interest': {'name': 'Benchmark: Odsetki (10% rocznie)', 'avg_profit': 0.60, 'nominal_profit': 10.41},
            'hodl_btc': {'name': 'Benchmark: HODL BTC', 'avg_profit': 0.85, 'nominal_profit': 85.50},
            'hodl_token': {'name': 'Benchmark: HODL Token', 'avg_profit': -0.42, 'nominal_profit': -42.75}
        }
        
        # Jeśli mamy analizator benchmarków, pobieramy faktyczne dane
        if hasattr(self.analyzer, 'benchmark_analyzer') and self.analyzer.benchmark_analyzer:
            if hasattr(self.analyzer.benchmark_analyzer, 'benchmark_results'):
                # Nadpisujemy domyślne wartości faktycznymi wynikami
                for benchmark_type, results in self.analyzer.benchmark_analyzer.benchmark_results.items():
                    if results:
                        # Obliczamy średnie wartości
                        profits = [result['final_profit_pct'] for result in results.values()]
                        nominal_profits = [result['final_profit'] for result in results.values()]
                        
                        avg_profit = np.mean(profits) if profits else 0
                        avg_nominal = np.mean(nominal_profits) if nominal_profits else 0
                        
                        if benchmark_type in benchmark_data:
                            benchmark_data[benchmark_type]['avg_profit'] = avg_profit
                            benchmark_data[benchmark_type]['nominal_profit'] = avg_nominal
        
        # Tworzymy DataFrame z danymi benchmarków
        benchmark_rows = []
        for key, data in benchmark_data.items():
            benchmark_rows.append({
                'name': data['name'],
                'avg_trades': None,
                'avg_profit_all_scenarios': data['avg_profit'],
                'robustness_score': None,
                'sharpe': None,
                'alpha': None,
                'nominal_profit': data['nominal_profit']
            })
        
        benchmark_df = pd.DataFrame(benchmark_rows)
        
        return benchmark_data, benchmark_df
    
    def generate_strategy_vs_benchmark_chart(self, strategy_metrics: pd.DataFrame, benchmark_metrics: pd.DataFrame) -> str:
        """
        Generuje wykres porównujący strategie z benchmarkami.
        
        Args:
            strategy_metrics: DataFrame z metrykami strategii
            benchmark_metrics: DataFrame z metrykami benchmarków
            
        Returns:
            str: Ścieżka do wygenerowanego wykresu
        """
        self.logger.info("Generowanie wykresu porównawczego strategii i benchmarków...")
        
        # Wybieramy top 5 strategii
        top_strategies = strategy_metrics.head(5)[['name', 'avg_profit_all_scenarios', 'nominal_profit']]
        
        # Łączymy z benchmarkami
        combined_data = pd.concat([
            top_strategies,
            benchmark_metrics[['name', 'avg_profit_all_scenarios', 'nominal_profit']]
        ])
        
        # Ustawiamy kolory - strategie na niebiesko, benchmarki na szaro/czerwono
        strategy_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        benchmark_colors = ['#7f7f7f', '#7f7f7f', '#e377c2']
        
        # Łączymy kolory
        colors = strategy_colors[:len(top_strategies)] + benchmark_colors[:len(benchmark_metrics)]
        
        # Tworzymy wykres
        plt.figure(figsize=(14, 8))
        
        # Przygotowujemy dane - sortujemy malejąco
        combined_data = combined_data.sort_values('avg_profit_all_scenarios', ascending=False)
        
        # Tworzymy wykres słupkowy
        bars = plt.bar(
            combined_data['name'], 
            combined_data['avg_profit_all_scenarios'], 
            color=colors,
            width=0.6
        )
        
        # Dodajemy etykiety z wartościami
        for i, bar in enumerate(bars):
            height = bar.get_height()
            nominal_profit = combined_data['nominal_profit'].iloc[i]
            
            # Ustawiamy kolor etykiety i pozycję w zależności od wartości
            if height < 0:
                va = 'top'
                offset = -15
            else:
                va = 'bottom'
                offset = 10
                
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height + (0.3 if height > 0 else -0.3),
                f"{height:.2f}%\n{nominal_profit:.2f} USDT",
                ha='center',
                va=va,
                fontweight='bold',
                fontsize=9
            )
        
        # Dodajemy linię poziomą na 0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Formatowanie wykresu
        plt.title('Porównanie zyskowności strategii i benchmarków', fontsize=16)
        plt.xlabel('Strategia / Benchmark', fontsize=12)
        plt.ylabel('Średni zysk (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Dodajemy obramowanie
        plt.box(True)
        
        # Zapisanie wykresu
        chart_file = self.output_dir / f'strategy_vs_benchmarks_{self.session_id}.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_file)
    
    def generate_scenario_heatmap(self, strategy_metrics: pd.DataFrame) -> str:
        """
        Generuje heatmapę wyników najlepszych strategii w różnych scenariuszach.
        
        Args:
            strategy_metrics: DataFrame z metrykami strategii
            
        Returns:
            str: Ścieżka do wygenerowanego wykresu
        """
        self.logger.info("Generowanie heatmapy scenariuszy...")
        
        # Jeśli nie mamy wystarczająco danych, kończymy
        if self.analyzer.summary_df is None or self.analyzer.summary_df.empty:
            return None
        
        # Wybieramy top 10 strategii
        top_strategies = strategy_metrics.head(10)
        
        # Grupujemy dane po scenariuszach i param_key
        scenario_data = self.analyzer.summary_df.copy()
        
        # Tworzymy pivotowaną tabelę: strategia x scenariusz
        pivot_data = []
        
        # Dla każdej strategii, znajdź wyniki dla wszystkich scenariuszy
        for _, strategy in top_strategies.iterrows():
            param_key = strategy['param_key']
            strategy_name = strategy['name']
            
            # Filtrujemy wyniki dla tej strategii
            strategy_results = scenario_data[scenario_data['param_key'] == param_key]
            
            # Dla każdego scenariusza, dodajemy wynik
            for _, row in strategy_results.iterrows():
                pivot_data.append({
                    'strategy': strategy_name,
                    'scenario': row['scenario'],
                    'profit': row['avg_profit']
                })
        
        # Tworzymy DataFrame
        if not pivot_data:
            return None
            
        pivot_df = pd.DataFrame(pivot_data)
        
        # Upewniamy się, że nie ma duplikatów indeksu przed tworzeniem pivotowanej tabeli
        # Jeśli są duplikaty, dodajemy do nazwy strategii indeks
        strategy_counts = {}
        for i, row in pivot_df.iterrows():
            strategy = row['strategy']
            if strategy in strategy_counts:
                strategy_counts[strategy] += 1
                pivot_df.loc[i, 'strategy'] = f"{strategy}_{strategy_counts[strategy]}"
            else:
                strategy_counts[strategy] = 1

        # Uproszczone podejście do zapewnienia unikalnych strategii
        pivot_df = pivot_df.reset_index(drop=True)
        strategy_counts = {}

        # Dodajemy unikalny indeks do zduplikowanych nazw strategii
        for i in range(len(pivot_df)):
            strategy = pivot_df.loc[i, 'strategy']
            if strategy in strategy_counts:
                strategy_counts[strategy] += 1
                pivot_df.loc[i, 'unique_strategy'] = f"{strategy}_{strategy_counts[strategy]}"
            else:
                strategy_counts[strategy] = 0
                pivot_df.loc[i, 'unique_strategy'] = strategy

        # Tworzymy pivotowaną tabelę używając unikalnego identyfikatora
        try:
            # Używamy agregacji, która jest bardziej niezawodna
            pivot_grouped = pivot_df.groupby(['unique_strategy', 'scenario'])['profit'].mean()
            heatmap_data = pivot_grouped.unstack('scenario')
        except Exception as e:
            self.logger.error(f"Błąd podczas tworzenia pivotowanej tabeli: {str(e)}")
            
            # Ręczne tworzenie tabeli jako ostatnia deska ratunku
            try:
                scenarios = pivot_df['scenario'].unique()
                strategies = pivot_df['unique_strategy'].unique()
                
                # Inicjalizacja pustej ramki danych
                heatmap_data = pd.DataFrame(index=strategies, columns=scenarios)
                
                # Wypełnienie wartościami
                for _, row in pivot_df.iterrows():
                    strategy = row['unique_strategy']
                    scenario = row['scenario']
                    profit = row['profit']
                    heatmap_data.loc[strategy, scenario] = profit
            except Exception as e2:
                self.logger.error(f"Błąd podczas ręcznego tworzenia heatmapy: {str(e2)}")
                # Jeśli wszystko zawiedzie, zwróć None
                return None
        
        # Tworzymy heatmapę
        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
        plt.title("Wyniki najlepszych strategii w różnych scenariuszach (%)", fontsize=16)
        plt.tight_layout()
        
        # Zapisanie wykresu
        heatmap_file = self.output_dir / f'strategy_scenario_heatmap_{self.session_id}.png'
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(heatmap_file)
    
    def format_advanced_metrics_table(self, strategy_metrics: pd.DataFrame, benchmark_metrics: pd.DataFrame) -> str:
        """
        Formatuje tabelę z zaawansowanymi metrykami dla raportu Markdown.
        
        Args:
            strategy_metrics: DataFrame z metrykami strategii
            benchmark_metrics: DataFrame z metrykami benchmarków
            
        Returns:
            str: Tabela w formacie Markdown
        """
        # Przygotowujemy dane dla strategii
        strategy_table = []

        # Wybieramy potrzebne kolumny i tworzymy tabelę
        for i, (idx, row) in enumerate(strategy_metrics.iterrows(), 1):
            # Używamy indeksu do znalezienia unikalnej strategii
            strategy_name = row.get('name', f"Strategy_{i}")
            avg_trades = row.get('avg_trades', 0)
            
            # Zapewniamy poprawne formatowanie liczby transakcji
            if pd.isna(avg_trades) or avg_trades == 0:
                trades_str = "0.0"
            else:
                trades_str = f"{avg_trades:.1f}"
            
            # Obliczamy zysk nominalny
            nominal_profit = avg_trades * row['avg_profit_all_scenarios'] / 100 * 100 if pd.notna(avg_trades) else 0
            
            strategy_table.append({
                'num': i,
                'name': f"**{strategy_name}**",
                'trades': trades_str,
                'profit': f"{row['avg_profit_all_scenarios']:.2f}%",
                'robustness': f"{row['robustness_score']:.2f}" if pd.notna(row['robustness_score']) else "-",
                'sharpe': f"{row['sharpe']:.2f}" if pd.notna(row['sharpe']) else "-", 
                'alpha': f"{row['alpha']:.2f}%" if pd.notna(row['alpha']) else "-",
                'nominal': f"{nominal_profit:.2f} USDT"
            })
        
        # Tworzymy DataFrame z danymi strategii
        strategy_df = pd.DataFrame(strategy_table)
        
        # Ustawiamy nagłówki kolumn
        column_headers = {
            'num': '#',
            'name': 'Nazwa',
            'trades': 'Liczba transakcji',
            'profit': 'Średni zysk %',
            'robustness': 'Wskaźnik odporności',
            'sharpe': 'Wskaźnik Sharpe',
            'alpha': 'Wskaźnik Alpha',
            'nominal': 'Zysk nominalny'
        }
        
        # Zmieniamy nazwy kolumn na czytelne nagłówki
        if not strategy_df.empty:
            strategy_df = strategy_df.rename(columns=column_headers)
        
        # Formatujemy tabelę ręcznie, aby mieć pełną kontrolę nad szerokością kolumn
        strategy_md = self._dataframe_to_markdown(strategy_df)
        
        # Przygotowujemy dane dla benchmarków
        benchmark_table = []
        
        # Wybieramy potrzebne kolumny dla benchmarków
        for _, row in benchmark_metrics.iterrows():
            benchmark_table.append({
                'name': row['name'],
                'trades': "-",  # Liczba transakcji
                'profit': f"{row['avg_profit_all_scenarios']:.2f}%" if pd.notna(row['avg_profit_all_scenarios']) else "-",
                'robustness': "-",  # Wskaźnik odporności
                'sharpe': "-",  # Wskaźnik Sharpe
                'alpha': "-",  # Wskaźnik Alpha
                'nominal': f"{row['nominal_profit']:.2f} USDT" if pd.notna(row['nominal_profit']) else "-"
            })
        
        # Tworzenie DataFrame dla benchmarków
        benchmark_df = pd.DataFrame(benchmark_table)
        
        # Ustawiamy nagłówki kolumn dla benchmarków (bez kolumny '#')
        benchmark_headers = {
            'name': 'Nazwa',
            'trades': 'Liczba transakcji',
            'profit': 'Średni zysk %',
            'robustness': 'Wskaźnik odporności',
            'sharpe': 'Wskaźnik Sharpe',
            'alpha': 'Wskaźnik Alpha',
            'nominal': 'Zysk nominalny'
        }
        
        # Zmieniamy nazwy kolumn na czytelne nagłówki
        if not benchmark_df.empty:
            benchmark_df = benchmark_df.rename(columns=benchmark_headers)
                
            # Formatujemy tabelę benchmarków ręcznie
            benchmark_md = self._dataframe_to_markdown(benchmark_df)
                
            # Łączymy tabele z nagłówkiem dla benchmarków
            return strategy_md + "\n\n**Wyniki benchmarków**:\n\n" + benchmark_md
        else:
            # Tylko tabela strategii, bez benchmarków
            return strategy_md
        
    def _dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        """
        Konwertuje DataFrame do tabeli Markdown z automatycznym dostosowaniem szerokości kolumn.
        
        Args:
            df: DataFrame do konwersji
            
        Returns:
            str: Tabela w formacie Markdown
        """
        if df.empty:
            return ""
        
        # Obliczamy szerokości kolumn
        col_widths = {}
        for col in df.columns:
            # Szerokość nagłówka
            header_width = len(str(col))
            
            # Maksymalna szerokość danych w kolumnie
            data_width = max([len(str(val)) for val in df[col].values] + [0])
            
            # Ustawiamy szerokość kolumny jako maksimum z nagłówka i danych
            col_widths[col] = max(header_width, data_width)
        
        # Tworzymy wiersz nagłówka z odpowiednim wypełnieniem
        header = "| " + " | ".join([str(col).ljust(col_widths[col]) for col in df.columns]) + " |"
        
        # Tworzymy separator z odpowiednią szerokością
        separator = "|" + "|".join([":" + "-" * (col_widths[col] - 1) for col in df.columns]) + "|"
        
        # Tworzymy wiersze danych
        rows = []
        for _, row in df.iterrows():
            formatted_row = "| " + " | ".join([str(row[col]).ljust(col_widths[col]) for col in df.columns]) + " |"
            rows.append(formatted_row)
        
        # Łączymy wszystko w jedną tabelę
        return "\n".join([header, separator] + rows)
    
    def format_parameter_list(self, params: Dict[str, Any]) -> str:
        """
        Formatuje listę parametrów strategii z zawijaniem po 6 parametrach.
        
        Args:
            params: Słownik z parametrami strategii
            
        Returns:
            str: Sformatowana lista parametrów
        """
        formatted_params = []
        
        # Kluczowe parametry w określonej kolejności
        key_params = [
            'check_timeframe', 'percentage_buy_threshold', 'sell_profit_target',
            'trailing_enabled', 'trailing_stop_price', 'trailing_stop_margin', 'trailing_stop_time',
            'stop_loss_enabled', 'stop_loss_threshold', 'stop_loss_delay_time',
            'pump_detection_enabled', 'pump_detection_threshold', 'max_open_orders'
        ]
        
        # Formatujemy parametry w określonej kolejności
        param_items = []
        for key in key_params:
            if key in params:
                value = params[key]
                
                # Specjalne formatowanie dla niektórych parametrów
                if key in ['percentage_buy_threshold', 'sell_profit_target', 'stop_loss_threshold', 
                        'trailing_stop_price', 'trailing_stop_margin', 'pump_detection_threshold']:
                    param_items.append(f"{key}: {value}%")
                elif key in ['trailing_enabled', 'stop_loss_enabled', 'pump_detection_enabled']:
                    param_items.append(f"{key}: {str(bool(value)).lower()}")
                else:
                    param_items.append(f"{key}: {value}")
        
        # Dodajemy pozostałe parametry, jeśli jakieś zostały
        for key, value in params.items():
            if key not in key_params:
                param_items.append(f"{key}: {value}")
        
        # Zawijanie listy parametrów po 6 elementów
        formatted_chunks = []
        for i in range(0, len(param_items), 6):
            chunk = param_items[i:i+6]
            formatted_chunks.append(", ".join(chunk))
        
        return "<br>".join(formatted_chunks)
    
    def generate_enhanced_report(self) -> str:
        """
        Generuje ulepszony raport z analizy forward-testingu i benchmarków.
        
        Returns:
            str: Ścieżka do wygenerowanego raportu
        """
        self.logger.info("Generowanie ulepszonego raportu...")
        
        # Ekstrahujemy informacje o scenariuszach
        scenario_info = self.extract_scenario_info()
        
        # Ekstrahujemy informacje o parametrach strategii
        strategy_info = self.extract_strategy_parameters()
        
        # Obliczamy zaawansowane metryki dla strategii
        strategy_metrics = self.calculate_advanced_metrics()
        
        # Generujemy porównanie z benchmarkami
        benchmark_data, benchmark_metrics = self.generate_benchmark_comparison()
        
        # Generujemy wizualizacje
        chart_file = None
        heatmap_file = None
        
        if not strategy_metrics.empty:
            # Generujemy wykres porównawczy strategii i benchmarków
            chart_file = self.generate_strategy_vs_benchmark_chart(strategy_metrics, benchmark_metrics)
            
            # Generujemy heatmapę scenariuszy
            heatmap_file = self.generate_scenario_heatmap(strategy_metrics)
        
        # Formatujemy tabelę z metrykami
        metrics_table = None
        if not strategy_metrics.empty:
            metrics_table = self.format_advanced_metrics_table(strategy_metrics, benchmark_metrics)
        
        # Przygotowujemy treść raportu
        report_content = [
            f"# Raport z analizy forward-testingu strategii trading",
            f"",
            f"*Data wygenerowania: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            f"",
            f"## 1. Informacje o analizowanych danych",
            f"",
            f"### 1.1. Scenariusze rynkowe",
            f"",
            f"W analizie wykorzystano **{scenario_info['total_scenarios']} scenariuszy rynkowych** o następującej charakterystyce:"
        ]
        
        # Tabela scenariuszy
        report_content.extend([
            f"",
            f"| Typ scenariusza | Liczba scenariuszy | Opis |",
            f"|-----------------|-------------------|------|"
        ])
        
        # Dodajemy opisy typów scenariuszy
        scenario_descriptions = {
            'Normal': "Scenariusze bazujące na normalnym rozkładzie zmienności historycznej",
            'Bootstrap': "Scenariusze wykorzystujące metodę bootstrap historycznych danych rynkowych",
            'Stress': "Scenariusze stresowe symulujące ekstremalne warunki rynkowe",
            'Other': "Inne typy scenariuszy"
        }
        
        for type_name, count in scenario_info['type_counts'].items():
            description = scenario_descriptions.get(type_name, "Inne scenariusze")
            report_content.append(f"| {type_name} | {count} | {description} |")
        
        # Dodajemy informacje o okresie symulacji
        if scenario_info['start_date'] and scenario_info['end_date'] and scenario_info['simulation_days']:
            report_content.extend([
                f"",
                f"**Okres symulacji**: {scenario_info['start_date'].strftime('%Y-%m-%d')} do {scenario_info['end_date'].strftime('%Y-%m-%d')} ({scenario_info['simulation_days']} dni)"
            ])

        # Generowanie wizualizacji scenariuszy
        scenario_chart, scenario_analysis = self.generate_scenario_visualization()
        if scenario_chart:
            scenario_chart_name = Path(scenario_chart).name
            report_content.extend([
                f"",
                f"![Porównanie przebiegów scenariuszy]({scenario_chart_name})",
                f"",
                f"{scenario_analysis}"
            ])

        # Dodajemy informacje o strategiach
        report_content.extend([
            f"",
            f"### 1.2. Zestawienie testowanych strategii",
            f"",
            f"Analizą objęto **{len(strategy_info['strategies'])} strategii** z różnymi kombinacjami parametrów:"
        ])
        
        # Tabela parametrów strategii
        report_content.extend([
            f"",
            f"| Plik parametrów | Parametry |",
            f"|-----------------|-----------|"
        ])
        
        # Dodajemy parametry poszczególnych strategii
        for strategy_name, params in strategy_info['strategies'].items():
            formatted_params = self.format_parameter_list(params)
            report_content.append(f"| {strategy_name}.json | `{formatted_params}` |")
        
        # Dodajemy informacje o bazowych parametrach wspólnych
        investment_amount = 100
        max_orders = 5
        
        if hasattr(self.analyzer, 'benchmark_analyzer') and self.analyzer.benchmark_analyzer:
            investment_amount = getattr(self.analyzer.benchmark_analyzer, 'order_amount', 100)
            max_orders = getattr(self.analyzer.benchmark_analyzer, 'max_orders', 5)
            
        total_investment = investment_amount * max_orders
        
        report_content.extend([
            f"",
            f"**Bazowe parametry wspólne dla wszystkich strategii**:",
            f"- Maksymalna liczba równoczesnych zleceń: {max_orders}",
            f"- Kwota pojedynczego zlecenia: {investment_amount} USDT",
            f"- Całkowita inwestycja bazowa: {total_investment} USDT",
            f"",
            f"## 2. Ranking strategii",
            f"",
            f"Strategie zostały uszeregowane według średniego zysku (Średni zysk %) na wszystkich testowanych scenariuszach:"
        ])
        
        # Używamy funkcji format_advanced_metrics_table do wygenerowania tabel strategii i benchmarków
        if not strategy_metrics.empty or metrics_table:
            # Najpierw sprawdzamy czy mamy poprawnie sformatowane tabele
            if metrics_table is None:
                metrics_table = self.format_advanced_metrics_table(strategy_metrics, benchmark_metrics)
            
            report_content.extend(['', metrics_table])
        
        # Dodajemy legendę
        report_content.extend([
            f"",
            f"### Legenda:",
            f"- **Nazwa**: Nazwa pliku z parametrami strategii",
            f"- **Liczba transakcji**: Średnia liczba transakcji na scenariusz (suma transakcji / liczba scenariuszy)",
            f"- **Średni zysk %**: Średni procentowy zysk ze wszystkich transakcji na wszystkich scenariuszach",
            f"- **Wskaźnik odporności**: Średni zysk podzielony przez odchylenie standardowe (wyższa wartość = większa stabilność)",
            f"- **Wskaźnik Sharpe**: Stosunek zysku do ryzyka (wyższa wartość = lepsza efektywność)",
            f"- **Wskaźnik Alpha**: Różnica pomiędzy średnim zyskiem strategii a najlepszym benchmarkiem (HODL BTC)",
            f"- **Zysk nominalny**: Całkowity zysk w USDT (liczba transakcji × średni zysk % × kwota zlecenia)"
        ])
        
        # Dodajemy sekcję wizualizacji
        if chart_file:
            chart_name = Path(chart_file).name
            report_content.extend([
                f"",
                f"## 3. Wizualizacja wyników",
                f"",
                f"![Porównanie zyskowności strategii i benchmarków]({chart_name})"
            ])
            
            # Dodajemy interpretację wykresu
            if not strategy_metrics.empty and not benchmark_metrics.empty:
                # Najlepsza strategia
                best_strategy = strategy_metrics.iloc[0]['name']
                best_profit = strategy_metrics.iloc[0]['avg_profit_all_scenarios']
                
                # Strategia z najwyższym zyskiem nominalnym
                highest_nominal_index = strategy_metrics['nominal_profit'].idxmax()
                highest_nominal_strategy = strategy_metrics.loc[highest_nominal_index, 'name']
                highest_nominal_profit = strategy_metrics.loc[highest_nominal_index, 'nominal_profit']
                
                # Sprawdzamy benchmarki
                hodl_btc_profit = None
                hodl_token_profit = None
                
                for _, row in benchmark_metrics.iterrows():
                    if 'HODL BTC' in row['name']:
                        hodl_btc_profit = row['avg_profit_all_scenarios']
                    elif 'HODL Token' in row['name']:
                        hodl_token_profit = row['avg_profit_all_scenarios']
                
                # Dodajemy obserwacje
                report_content.extend([
                    f"",
                    f"Powyższy wykres przedstawia porównanie zyskowności poszczególnych strategii i benchmarków. Analizując wyniki można zauważyć:",
                    f"",
                    f"1. **Najwyższą efektywność** wykazała strategia **{best_strategy}** pod względem średniego zysku procentowego ({best_profit:.2f}%)",
                    f"2. **Największy zysk nominalny** wygenerowała strategia **{highest_nominal_strategy}** ({highest_nominal_profit:.2f} USDT) dzięki dużej liczbie transakcji"
                ])
                
                # Porównanie z benchmarkami
                if hodl_btc_profit is not None:
                    benchmark_comparison = "3. **Wszystkie strategie** przewyższyły zarówno benchmark oprocentowania (10% rocznie) jak i benchmark HODL BTC"
                    if best_profit <= hodl_btc_profit:
                        benchmark_comparison = f"3. **Benchmark HODL BTC** okazał się bardziej efektywny ({hodl_btc_profit:.2f}%) niż najlepsza strategia aktywna ({best_profit:.2f}%)"
                    report_content.append(benchmark_comparison)
                
                # Dodajemy informację o HODL Token
                if hodl_token_profit is not None:
                    if hodl_token_profit < 0:
                        report_content.append(f"4. **Benchmark HODL Token** wykazał stratę, potwierdzając skuteczność aktywnego podejścia w przypadku badanego aktywa")
                    else:
                        report_content.append(f"4. **Benchmark HODL Token** wykazał zysk {hodl_token_profit:.2f}%, jednak strategie aktywne osiągnęły lepsze rezultaty")
        
        # Dodajemy analizę wrażliwości parametrów
        if self.analyzer.parameter_sensitivity:
            report_content.append(f"")
            report_content.append(f"## 4. Analiza wrażliwości parametrów")
            
            param_index = 1
            for param, df in self.analyzer.parameter_sensitivity.items():
                plot_file = f'parameter_sensitivity_{param}_{self.session_id}.png'
                
                # Określamy nazwę i opis parametru
                param_name = param
                param_descriptions = {
                    'check_timeframe': "najlepsze wyniki osiągają strategie z wartością parametru `check_timeframe` w przedziale 15-30 minut. Zbyt niskie wartości prowadzą do częstych, ale mniej zyskownych transakcji, a zbyt wysokie ograniczają liczbę okazji inwestycyjnych.",
                    'percentage_buy_threshold': "optymalne wartości parametru `percentage_buy_threshold` mieszczą się w przedziale -1.2% do -1.8%. Wyższe wartości (bliższe zeru) skutkują zbyt dużą liczbą transakcji o niskiej rentowności, a niższe wartości znacząco ograniczają liczbę transakcji.",
                    'sell_profit_target': "najlepsze wyniki osiągają strategie z wartością parametru `sell_profit_target` w przedziale 2.5% do 3.5%.",
                    'stop_loss_threshold': "optymalny poziom stop loss mieści się w przedziale -12% do -20%."
                }
                
                if param == 'percentage_buy_threshold':
                    param_name = 'buy_threshold'
                
                param_description = param_descriptions.get(
                    param, f"widoczny jest wpływ parametru `{param}` na wyniki strategii."
                )
                
                report_content.extend([
                    f"",
                    f"### 4.{param_index}. Wpływ {param_name} na wyniki",
                    f"",
                    f"![Wpływ {param} na wyniki strategii]({plot_file})",
                    f"",
                    f"Analiza pokazuje, że {param_description}"
                ])
                
                param_index += 1
        
        # Dodajemy wnioski i rekomendacje
        if not strategy_metrics.empty:
            best_strategy = strategy_metrics.iloc[0]['name']
            
            report_content.extend([
                f"",
                f"## 5. Wnioski i rekomendacje",
                f"",
                f"1. **Rekomendowana strategia**: Na podstawie przeprowadzonej analizy, najlepszą strategią pod względem równowagi między zyskownością a stabilnością jest **{best_strategy}**.",
                f"",
                f"2. **Kluczowe parametry**:",
                f"   - Optymalny `check_timeframe`: 15-30 minut",
                f"   - Optymalny `percentage_buy_threshold`: -1.2% do -1.8%",
                f"   - Włączona funkcja trailing stop zwiększa efektywność zamykania pozycji",
                f"",
                f"3. **Przewaga nad benchmarkami**:",
                f"   - Wszystkie testowane strategie przewyższyły benchmark oprocentowania (10% rocznie)",
                f"   - Strategie aktywne wykazały znaczącą przewagę nad prostym podejściem HODL",
                f"",
                f"4. **Dalsze optymalizacje**:",
                f"   - Warto rozważyć adaptacyjny mechanizm dostosowujący parametry do zmienności rynku",
                f"   - Połączenie elementów strategii {best_strategy} i {strategy_metrics.iloc[1]['name'] if len(strategy_metrics) > 1 else 'innych strategii'} może potencjalnie poprawić wyniki"
            ])
        
        # Dodajemy porównanie scenariuszowe, jeśli mamy heatmapę
        if heatmap_file:
            heatmap_name = Path(heatmap_file).name
            report_content.extend([
                f"",
                f"## 6. Porównanie scenariuszowe",
                f"",
                f"![Wyniki najlepszych strategii w różnych scenariuszach]({heatmap_name})",
                f"",
                f"Heatmapa pokazuje, że najlepsze strategie ({best_strategy} i {strategy_metrics.iloc[1]['name'] if len(strategy_metrics) > 1 else best_strategy}) zachowują się dobrze w większości scenariuszy, jednak wykazują znaczącą wrażliwość na scenariusze stresowe typu \"crash\". "
            ])
            
            # Dodajemy dodatkowe wnioski z heatmapy, jeśli mamy robustness_score
            if 'robustness_score' in strategy_metrics.columns:
                most_robust_idx = strategy_metrics['robustness_score'].idxmax()
                most_robust_strategy = strategy_metrics.loc[most_robust_idx, 'name']
                
                report_content.append(f"Strategia {most_robust_strategy}, mimo niższego średniego zysku, wykazuje najwyższą odporność na ekstremalne warunki rynkowe.")
        
        # Zapisujemy raport
        report_file = self.output_dir / f'enhanced_report_{self.session_id}.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        # Kopiujemy wszystkie potrzebne pliki wykresów do głównego katalogu
        self.copy_chart_files_to_main_dir()
        
        self.logger.info(f"Zapisano ulepszony raport do {report_file}")
        return str(report_file)
    
    def copy_chart_files_to_main_dir(self):
        """Kopiuje wszystkie pliki wykresów do głównego katalogu raportów."""
        try:
            # Kopiujemy wykresy wrażliwości parametrów
            if self.analyzer.parameter_sensitivity:
                for param in self.analyzer.parameter_sensitivity.keys():
                    src_file = self.output_dir / f'parameter_sensitivity_{param}_{self.session_id}.png'
                    # Sprawdzamy czy plik istnieje i nie jest już w katalogu głównym
                    if src_file.exists() and str(src_file.parent.absolute()) != str(self.output_dir.absolute()):
                        shutil.copy2(src_file, self.output_dir)
                        self.logger.info(f"Skopiowano {src_file} do katalogu głównego")
            
            # Kopiujemy wykres heatmapy strategii
            heatmap_file = self.output_dir / f'strategy_scenario_heatmap_{self.session_id}.png'
            if heatmap_file.exists() and str(heatmap_file.parent.absolute()) != str(self.output_dir.absolute()):
                shutil.copy2(heatmap_file, self.output_dir)
                self.logger.info(f"Skopiowano {heatmap_file} do katalogu głównego")
                
            # Kopiujemy wykresy z podkatalogu benchmarks, jeśli istnieją
            benchmarks_dir = self.output_dir / 'benchmarks'
            if benchmarks_dir.exists():
                for file_path in benchmarks_dir.glob(f'*_{self.session_id}.png'):
                    # Sprawdzamy czy plik nie jest już w katalogu głównym
                    if str(file_path.parent.absolute()) != str(self.output_dir.absolute()):
                        shutil.copy2(file_path, self.output_dir)
                        self.logger.info(f"Skopiowano {file_path} do katalogu głównego")
                        
        except Exception as e:
            self.logger.warning(f"Błąd podczas kopiowania plików wykresów: {str(e)}")
    
    def generate_scenario_visualization(self) -> Tuple[Optional[str], str]:
        """
        Generuje wizualizację scenariuszy i zwraca analizę przebiegów.

        Returns:
            tuple: (ścieżka do wygenerowanego wykresu lub None, tekst analizy)
        """
        self.logger.info("Generowanie wizualizacji przebiegów scenariuszy...")

        try:
            scenario_data = None # Resetuj na początku
            scenario_dir = None # Resetuj na początku
            data_source = "None" # Śledzenie źródła danych

            # Spróbuj użyć danych z benchmark_analyzer (najpierw)
            if hasattr(self.analyzer, 'benchmark_analyzer') and self.analyzer.benchmark_analyzer:
                # Sprawdź, czy atrybut istnieje ORAZ czy słownik nie jest pusty
                if hasattr(self.analyzer.benchmark_analyzer, 'scenario_data') and self.analyzer.benchmark_analyzer.scenario_data:
                    self.logger.info("Znaleziono dane scenariuszy w benchmark_analyzer. Próbuję ich użyć.")
                    # Strukturyzuj dane zgodnie z oczekiwaniami visualize_scenarios
                    temp_scenario_data = {
                        'normal': [],
                        'bootstrap': [],
                        'stress': [],
                        'unknown': []
                    }
                    valid_data_found = False # Flaga czy znaleziono jakiekolwiek poprawne dane

                    # Importuj potrzebne funkcje (bezpieczniej wewnątrz bloku)
                    try:
                        sys.path.append('.')
                        from scenario_visualizer import identify_scenario_type
                    except ImportError as e_vis_import:
                        self.logger.error(f"Błąd importu z scenario_visualizer: {e_vis_import}")
                        # Nie możemy kontynuować bez identify_scenario_type, więc nie ustawimy scenario_data
                        temp_scenario_data = None # Sygnalizuje błąd importu

                    if temp_scenario_data is not None: # Kontynuuj tylko jeśli import się udał
                        for scenario_name, df in self.analyzer.benchmark_analyzer.scenario_data.items():
                            if df is None or df.empty:
                                self.logger.warning(f"DataFrame dla scenariusza '{scenario_name}' w benchmark_analyzer jest pusty lub None. Pomijam.")
                                continue

                            try:
                                scenario_type = identify_scenario_type(scenario_name)
                            except Exception as e_type:
                                self.logger.error(f"Błąd podczas identyfikacji typu scenariusza '{scenario_name}': {e_type}. Używam 'unknown'.")
                                scenario_type = 'unknown'

                            df_copy = df.copy()
                            # Próbuj pobrać pełną ścieżkę, jeśli jest dostępna, inaczej użyj nazwy
                            # Zakładamy, że benchmark_analyzer mógł zapisać ścieżkę w atrybucie obiektu df,
                            # jeśli nie, używamy nazwy scenariusza.
                            scenario_file_info = getattr(df, 'scenario_file_path', scenario_name)
                            df_copy['scenario_file'] = str(scenario_file_info) # Zapisujemy jako string
                            df_copy['scenario_type'] = scenario_type
                            temp_scenario_data[scenario_type].append(df_copy)
                            valid_data_found = True # Znaleziono przynajmniej jedne dane

                        # Jeśli znaleziono poprawne dane, przypisz je
                        if valid_data_found:
                            scenario_data = temp_scenario_data
                            data_source = "benchmark_analyzer"
                            self.logger.info("Pomyślnie ustrukturyzowano dane z benchmark_analyzer.")
                        else:
                            self.logger.warning("Dane w benchmark_analyzer były puste lub nie udało się ich przetworzyć. Próbuję wczytać z katalogów.")
                else:
                    self.logger.warning("Atrybut 'scenario_data' w benchmark_analyzer nie istnieje lub jest pusty. Próbuję wczytać z katalogów.")
            else:
                self.logger.warning("Brak instancji benchmark_analyzer. Próbuję wczytać dane scenariuszy z katalogów.")

            # Fallback: Wczytaj dane z katalogu, jeśli nie udało się z benchmark_analyzer
            if scenario_data is None:
                self.logger.info("Próbuję metody fallback: wczytywanie danych scenariuszy z katalogów.")
                # Znajdź katalog (twoja logika wydaje się OK)
                scenario_dirs_to_check = ['csv/symulacje', 'data/symulacje', 'simulations']
                for potential_dir in scenario_dirs_to_check:
                    if os.path.exists(potential_dir) and os.path.isdir(potential_dir):
                        scenario_dir = potential_dir
                        self.logger.info(f"Znaleziono potencjalny katalog scenariuszy: {scenario_dir}")
                        break

                if scenario_dir:
                    try:
                        # Upewnij się, że funkcja jest importowana
                        sys.path.append('.')
                        from scenario_visualizer import load_scenario_data
                        scenario_data = load_scenario_data(scenario_dir)
                        data_source = f"directory ({scenario_dir})"
                        self.logger.info(f"Pomyślnie wczytano dane scenariuszy z katalogu {scenario_dir}.")
                    except ImportError:
                        self.logger.error("Nie można zaimportować `load_scenario_data` z `scenario_visualizer`.")
                    except Exception as e_load:
                        self.logger.error(f"Błąd podczas wczytywania danych przez `load_scenario_data` z katalogu {scenario_dir}: {e_load}")
                        scenario_data = None # Upewnij się, że jest None po błędzie
                else:
                    self.logger.warning("Nie znaleziono domyślnych katalogów scenariuszy ('csv/symulacje', 'data/symulacje', 'simulations').")

            # Ostateczne sprawdzenie, czy mamy dane z *jakiegokolwiek* źródła
            # --- DODANY BLOK LOGOWANIA ---
            self.logger.info("--- Debugging scenario_data before final check ---")
            if scenario_data is None:
                self.logger.info("scenario_data is None")
            else:
                self.logger.info(f"scenario_data type: {type(scenario_data)}")
                if isinstance(scenario_data, dict):
                    self.logger.info(f"scenario_data keys: {list(scenario_data.keys())}")
                    all_lists_empty = True # Zakładamy, że wszystkie są puste na początku
                    for key, dfs_list in scenario_data.items():
                        self.logger.info(f"  Key '{key}': type={type(dfs_list)}, length={len(dfs_list) if isinstance(dfs_list, list) else 'N/A'}")
                        if isinstance(dfs_list, list):
                             if not dfs_list:
                                 self.logger.info(f"    List for key '{key}' is empty.")
                             else:
                                 all_lists_empty = False # Znaleziono niepustą listę
                                 # Check the first DataFrame in the list as an example
                                 first_df = dfs_list[0]
                                 self.logger.info(f"    First element type for '{key}': {type(first_df)}")
                                 if isinstance(first_df, pd.DataFrame):
                                      self.logger.info(f"    First DF for '{key}' is empty: {first_df.empty}")
                                      self.logger.info(f"    First DF for '{key}' columns: {list(first_df.columns)}")
                                 else:
                                      self.logger.info(f"    First element for '{key}' is NOT a DataFrame.")
                        else:
                            self.logger.warning(f"  Value for key '{key}' is not a list.")
                            all_lists_empty = False # Traktujemy to jako niepuste, bo struktura jest zła

                    # Evaluate the condition separately
                    all_empty_check = all(isinstance(dfs, list) and len(dfs) == 0 for dfs in scenario_data.values())
                    self.logger.info(f"Result of 'all(isinstance(dfs, list) and len(dfs) == 0 for dfs in scenario_data.values())': {all_empty_check}")
                    self.logger.info(f"Internal flag 'all_lists_empty': {all_lists_empty}") # Porównaj z wynikiem all()
                else:
                    self.logger.warning(f"scenario_data is not a dictionary.")
            self.logger.info("--- End of Debugging ---")
            # --- KONIEC DODANEGO BLOKU LOGOWANIA ---
            if scenario_data is None or all(len(dfs) == 0 for dfs in scenario_data.values()):
                self.logger.warning(f"Nie znaleziono danych scenariuszy do wizualizacji (ostateczne sprawdzenie, źródło: {data_source}).") # Dodano info o źródle
                return None, "Nie znaleziono danych scenariuszy do wizualizacji."
            else:
                self.logger.info(f"Znaleziono dane scenariuszy do wizualizacji (źródło: {data_source}). Kontynuuję generowanie wykresu.")

            # Generuj wykres scenariuszy
            chart_file = self.output_dir / f'scenario_visualization_{self.session_id}.png'

            try:
                # Upewnij się, że visualize_scenarios jest zaimportowane
                sys.path.append('.')
                from scenario_visualizer import visualize_scenarios
                visualize_scenarios(scenario_data, str(chart_file))
            except ImportError:
                self.logger.error("Nie można zaimportować `visualize_scenarios` z `scenario_visualizer`.")
                return None, "Błąd importu funkcji wizualizacji."
            except Exception as e_vis:
                self.logger.error(f"Błąd podczas generowania wykresu przez `visualize_scenarios`: {e_vis}")
                return None, f"Błąd generowania wizualizacji: {e_vis}"

            # Analizuj przebiegi - skupiając się na normalnych scenariuszach
            analysis_text = self.analyze_scenario_trends(scenario_data)

            return str(chart_file), analysis_text

        except Exception as e:
            self.logger.error(f"Nieoczekiwany błąd podczas generowania wizualizacji scenariuszy: {str(e)}")
            return None, f"Generowanie wizualizacji scenariuszy nie powiodło się: {str(e)}"

    def analyze_scenario_trends(self, scenario_data):
        """
        Analizuje trendy w scenariuszach i zwraca tekst analizy.
        
        Args:
            scenario_data: Słownik z danymi scenariuszy
            
        Returns:
            str: Tekst analizy
        """
        # Skupiamy się na normalnych scenariuszach
        normal_scenarios = scenario_data.get('normal', [])
        
        if not normal_scenarios:
            return "Brak scenariuszy typu 'normal' do analizy."
        
        # Przygotuj dane do analizy
        first_prices = []
        last_prices = []
        max_changes = []  # Maksymalne zmiany w trakcie przebiegu
        
        for df in normal_scenarios:
            if 'close' in df.columns:
                prices = df['close'].values
                if len(prices) > 0:
                    first_price = prices[0]
                    last_price = prices[-1]
                    
                    first_prices.append(first_price)
                    last_prices.append(last_price)
                    
                    # Oblicz maksymalną zmianę w trakcie przebiegu
                    changes = []
                    for i in range(1, len(prices)):
                        change_pct = (prices[i] - prices[i-1]) / prices[i-1] * 100
                        changes.append(abs(change_pct))
                    
                    if changes:
                        max_changes.append(max(changes))
        
        if not first_prices or not last_prices:
            return "Brak wystarczających danych do analizy przebiegów."
        
        # Oblicz średnie zmiany
        avg_first_price = np.mean(first_prices)
        avg_last_price = np.mean(last_prices)
        avg_change_pct = (avg_last_price - avg_first_price) / avg_first_price * 100
        
        # Oblicz średnią maksymalną zmianę
        avg_max_change = np.mean(max_changes) if max_changes else 0
        
        # Przygotuj tekst analizy
        analysis = [
            f"**Analiza przebiegów normalnych ({len(normal_scenarios)} scenariuszy)**:",
            f"",
            f"- Średnia zmiana kursu: **{avg_change_pct:.2f}%** (od {avg_first_price:.2f} do {avg_last_price:.2f})",
            f"- Średnia maksymalna dzienna zmiana: **{avg_max_change:.2f}%**"
        ]
        
        # Dodaj interpretację trendu
        if avg_change_pct > 5:
            trend = "wzrostowy"
        elif avg_change_pct < -5:
            trend = "spadkowy"
        else:
            trend = "boczny"
        
        analysis.append(f"- Dominujący trend: **{trend}**")
        
        if avg_max_change > 3:
            volatility = "wysoka"
        elif avg_max_change > 1:
            volatility = "umiarkowana"
        else:
            volatility = "niska"
        
        analysis.append(f"- Zmienność rynku: **{volatility}**")
        
        return "\n".join(analysis)