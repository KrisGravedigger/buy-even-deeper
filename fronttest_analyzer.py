#!/opt/homebrew/bin/python3.11
# -*- coding: utf-8 -*-

"""
Analizator wyników forward-testingu strategii tradingowych.
Rozszerzony o benchmarki: odsetki, HODL BTC i HODL Token.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
import os
from collections import defaultdict
import sys

# Dodajemy import konfiguracji i modułu benchmarków
sys.path.append('.')
from utils.config import WYNIKI_DIR, WYNIKI_FRONTTEST_DIR, WYNIKI_FRONTTEST_ANALIZY_DIR, ensure_directory_structure, get_newest_file


class FronttestAnalyzer:
    def __init__(self, results_file: str = None, output_dir: str = None):
        """
        Inicjalizacja analizatora wyników forward-testingu.
        
        Args:
            results_file: Ścieżka do pliku PKL z wynikami z run_fronttest (opcjonalnie)
            output_dir: Katalog wyjściowy dla analiz (opcjonalnie)
        """
        # Upewniamy się, że struktura katalogów istnieje
        ensure_directory_structure()
        
        # Używamy podanego output_dir lub domyślnego z konfiguracji
        self.output_dir = Path(output_dir) if output_dir else WYNIKI_FRONTTEST_ANALIZY_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicjalizacja session_id PRZED inicjalizacją loggera
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Inicjalizacja loggera
        self.logger = self._setup_logger()
        
        # Używamy podanego pliku wyników lub znajdujemy najnowszy
        if results_file is None:
            results_file = get_newest_file(WYNIKI_FRONTTEST_DIR, pattern="*.pkl", recursive=True)
            if not results_file:
                raise FileNotFoundError("Nie znaleziono plików PKL z wynikami w katalogu wyniki/fronttesty ani jego podkatalogach")
            print(f"Automatycznie wybrano najnowszy plik wyników: {results_file}")
        
        # Wczytanie wyników
        with open(results_file, 'rb') as f:
            self.data = pickle.load(f)
            self.logger.info(f"Wczytano dane z PKL. Dostępne klucze: {list(self.data.keys())}")
        
        self.logger.info(f"Inicjalizacja analizatora. Plik wyników: {results_file}, Katalog wyjściowy: {self.output_dir}")
        
        # Dodatkowe pola
        self.summary_df = None
        self.parameter_sensitivity = None
        self.strategy_rankings = None
        
        # Inicjalizacja analizatora benchmarków jako None, zostanie utworzony później
        self.benchmark_analyzer = None
        self.include_benchmarks = True  # Domyślnie włączamy benchmarki
    
    def _setup_logger(self) -> logging.Logger:
        """Konfiguracja loggera"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Ustawienie formatowania
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # Handler konsoli
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # Handler pliku
            log_dir = self.output_dir / 'logs'
            log_dir.mkdir(exist_ok=True, parents=True)
            file_handler = logging.FileHandler(log_dir / f'fronttest_analyzer_{self.session_id}.log')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def analyze_scenario_performance(self) -> pd.DataFrame:
        """
        Analizuje wyniki dla różnych scenariuszy rynkowych.
        
        Returns:
            pd.DataFrame: DataFrame z podsumowaniem wyników
        """
        self.logger.info("Rozpoczynam analizę wyników dla różnych scenariuszy...")
        
        results_by_scenario = self.data.get('results_by_scenario', {})
        if not results_by_scenario:
            self.logger.error("Brak danych o wynikach dla scenariuszy")
            return pd.DataFrame()
        
        # Inicjalizacja pustych list na dane
        rows = []
        
        # Iteracja po scenariuszach
        for scenario_name, results in results_by_scenario.items():
            self.logger.info(f"Analizuję scenariusz: {scenario_name}")
            
            # Grupowanie wyników wg kombinacji parametrów
            strategy_results = defaultdict(list)
            
            for result in results:
                strategy_id = result.get('strategy_id', 'unknown')
                avg_profit = result.get('avg_profit', 0)
                total_trades = result.get('total_trades', 0)
                
                # Wyciągnięcie najważniejszych parametrów
                params = result.get('parameters', {})
                
                # Klucz unikatowo identyfikujący konfigurację parametrów (z dodaną nazwą pliku)
                param_file = result.get('param_file', result.get('param_file_name', 'unknown'))
                param_key = f"{param_file}|{json.dumps({k: params[k] for k in sorted(params.keys())})}"
                
                strategy_results[param_key].append({
                    'strategy_id': strategy_id,
                    'avg_profit': avg_profit,
                    'total_trades': total_trades,
                    'parameters': params
                })
            
            # Agregacja wyników dla każdej konfiguracji parametrów
            for param_key, strategy_data in strategy_results.items():
                avg_profits = [data['avg_profit'] for data in strategy_data]
                total_trades = [data['total_trades'] for data in strategy_data]
                
                # Reprezentatywne parametry (bierzemy pierwszy zestaw)
                params = strategy_data[0]['parameters']
                
                # Dodanie wiersza do DataFrame
                rows.append({
                    'scenario': scenario_name,
                    'avg_profit': np.mean(avg_profits),
                    'avg_trades': np.mean(total_trades),
                    'min_profit': np.min(avg_profits),
                    'max_profit': np.max(avg_profits),
                    'std_profit': np.std(avg_profits),
                    'check_timeframe': params.get('check_timeframe', 0),
                    'percentage_buy_threshold': params.get('percentage_buy_threshold', 0),
                    'sell_profit_target': params.get('sell_profit_target', 0),
                    'stop_loss_threshold': params.get('stop_loss_threshold', 0),
                    'trailing_enabled': params.get('trailing_enabled', 0),
                    'param_key': param_key,
                    'param_file': param_file  # Dodajemy nazwę pliku parametrów
                })
        
        # Tworzenie DataFrame
        summary_df = pd.DataFrame(rows)
        
        # Sortowanie wg średniego zysku
        if not summary_df.empty:
            summary_df = summary_df.sort_values('avg_profit', ascending=False)
        
        # Zapisanie do pliku
        summary_file = self.output_dir / f'scenario_performance_{self.session_id}.csv'
        summary_df.to_csv(summary_file, index=False)
        self.logger.info(f"Zapisano podsumowanie wyników dla scenariuszy do {summary_file}")
        
        # Zapisanie podsumowania do pola klasy
        self.summary_df = summary_df
        
        return summary_df
    
    def analyze_parameter_sensitivity(self) -> Dict[str, pd.DataFrame]:
        """
        Analizuje wpływ różnych parametrów na wyniki.
        
        Returns:
            Dict[str, pd.DataFrame]: Słownik z analizami dla poszczególnych parametrów
        """
        self.logger.info("Rozpoczynam analizę wrażliwości parametrów...")
        
        if self.summary_df is None:
            self.analyze_scenario_performance()
            
        if self.summary_df is None or self.summary_df.empty:
            self.logger.error("Brak danych do analizy wrażliwości parametrów")
            return {}
        
        # Parametry do analizy
        params_to_analyze = [
            'check_timeframe', 
            'percentage_buy_threshold', 
            'sell_profit_target', 
            'stop_loss_threshold'
        ]
        
        # Słownik na wyniki
        sensitivity_results = {}
        
        # Analiza dla każdego parametru
        for param in params_to_analyze:
            if param not in self.summary_df.columns:
                continue
                
            self.logger.info(f"Analizuję wrażliwość parametru: {param}")
            
            # Grupowanie po wartościach parametru i obliczanie statystyk
            param_df = self.summary_df.groupby(param).agg({
                'avg_profit': ['mean', 'std', 'min', 'max', 'count'],
                'avg_trades': 'mean'
            }).reset_index()
            
            # Zmiana nazw kolumn
            param_df.columns = [param, 'avg_profit', 'std_profit', 'min_profit', 'max_profit', 'count', 'avg_trades']
            
            # Sortowanie wg wartości parametru
            param_df = param_df.sort_values(param)
            
            # Zapisanie do pliku
            param_file = self.output_dir / f'parameter_sensitivity_{param}_{self.session_id}.csv'
            param_df.to_csv(param_file, index=False)
            
            # Dodanie do słownika wyników
            sensitivity_results[param] = param_df
            
            # Tworzenie wykresu
            plt.figure(figsize=(12, 6))
            
            # Wykres liniowy średniego zysku
            ax1 = plt.gca()
            line1 = ax1.plot(param_df[param], param_df['avg_profit'], 'b-', label='Średni zysk (%)')
            ax1.set_xlabel(f'Wartość parametru {param}')
            ax1.set_ylabel('Średni zysk (%)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Dodanie przedziału ufności
            ax1.fill_between(
                param_df[param],
                param_df['avg_profit'] - param_df['std_profit'],
                param_df['avg_profit'] + param_df['std_profit'],
                alpha=0.2,
                color='b'
            )
            
            # Druga oś Y dla liczby transakcji
            ax2 = ax1.twinx()
            line2 = ax2.plot(param_df[param], param_df['avg_trades'], 'r-', label='Średnia liczba transakcji')
            ax2.set_ylabel('Średnia liczba transakcji', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Legenda
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='best')
            
            plt.title(f'Wpływ parametru {param} na wyniki strategii')
            plt.grid(True, alpha=0.3)
            
            # Zapisanie wykresu
            plot_file = self.output_dir / f'parameter_sensitivity_{param}_{self.session_id}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Zapisano analizę wrażliwości parametru {param} do {param_file} i {plot_file}")
        
        # Zapisanie wyników do pola klasy
        self.parameter_sensitivity = sensitivity_results
        
        return sensitivity_results
    
    def compare_strategy_configurations(self) -> pd.DataFrame:
        """
        Porównuje różne konfiguracje strategii.
        
        Returns:
            pd.DataFrame: DataFrame z porównaniem konfiguracji
        """
        self.logger.info("Rozpoczynam porównanie konfiguracji strategii...")
        
        results_by_scenario = self.data.get('results_by_scenario', {})
        if not results_by_scenario:
            self.logger.error("Brak danych o wynikach dla scenariuszy")
            return pd.DataFrame()
        
        # Tworzenie słownika strategia -> wyniki dla różnych scenariuszy
        strategy_results = defaultdict(lambda: defaultdict(list))
        
        # Iteracja po scenariuszach
        for scenario_name, results in results_by_scenario.items():
            for result in results:
                # Parametry jako klucz strategii
                params = result.get('parameters', {})
                param_file = result.get('param_file', result.get('param_file_name', 'unknown'))
                param_key = f"{param_file}|{json.dumps({k: params[k] for k in sorted(params.keys())})}"
                
                # Dodanie wyniku do słownika
                strategy_results[param_key][scenario_name].append({
                    'avg_profit': result.get('avg_profit', 0),
                    'total_trades': result.get('total_trades', 0),
                    'parameters': params
                })
        
        # Przygotowanie danych do DataFrame
        rows = []
        
        for param_key, scenario_data in strategy_results.items():
            # Sprawdzenie czy strategia ma wyniki dla wszystkich scenariuszy
            if len(scenario_data) < len(results_by_scenario):
                continue
            
            # Obliczenie statystyk dla każdego scenariusza
            scenario_stats = {}
            for scenario_name, results in scenario_data.items():
                profits = [r['avg_profit'] for r in results]
                trades = [r['total_trades'] for r in results]
                
                scenario_stats[f'{scenario_name}_profit'] = np.mean(profits)
                scenario_stats[f'{scenario_name}_trades'] = np.mean(trades)
            
            # Agregacja ogólnych statystyk
            all_profits = [stats['avg_profit'] for results in scenario_data.values() for stats in results]
            
            # Pobranie reprezentatywnych parametrów (z pierwszego wyniku pierwszego scenariusza)
            first_scenario = list(scenario_data.keys())[0]
            params = scenario_data[first_scenario][0]['parameters']
            
            # Dodanie wiersza
            row = {
                'param_key': param_key,
                'param_file': param_file,  # Dodajemy nazwę pliku parametrów
                'avg_profit_all_scenarios': np.mean(all_profits),
                'min_profit_all_scenarios': np.min(all_profits),
                'max_profit_all_scenarios': np.max(all_profits),
                'std_profit_all_scenarios': np.std(all_profits),
                'robustness_score': np.mean(all_profits) / (np.std(all_profits) + 1e-6),  # Im wyższy, tym lepiej
                'check_timeframe': params.get('check_timeframe', 0),
                'percentage_buy_threshold': params.get('percentage_buy_threshold', 0),
                'sell_profit_target': params.get('sell_profit_target', 0),
                'stop_loss_threshold': params.get('stop_loss_threshold', 0),
                'trailing_enabled': params.get('trailing_enabled', 0)
            }
            
            # Dodanie statystyk dla poszczególnych scenariuszy
            row.update(scenario_stats)
            
            rows.append(row)
        
        # Tworzenie DataFrame
        comparison_df = pd.DataFrame(rows)
        
        # Sortowanie wg współczynnika robustness (odporności)
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('robustness_score', ascending=False)
        
        # Zapisanie do pliku
        comparison_file = self.output_dir / f'strategy_comparison_{self.session_id}.csv'
        comparison_df.to_csv(comparison_file, index=False)
        self.logger.info(f"Zapisano porównanie strategii do {comparison_file}")
        
        # Zapisanie do pola klasy
        self.strategy_rankings = comparison_df
        
        return comparison_df
    
    def generate_visualizations(self) -> List[str]:
        """
        Generuje wizualizacje wyników.
        
        Returns:
            List[str]: Lista ścieżek do wygenerowanych wykresów
        """
        self.logger.info("Generuję wizualizacje wyników...")
        
        generated_files = []
        
        # Upewnienie się, że mamy wszystkie potrzebne analizy
        if self.strategy_rankings is None:
            self.compare_strategy_configurations()
        
        if self.strategy_rankings is not None and not self.strategy_rankings.empty:
            # 1. Wykres heatmap dla top 10 strategii i ich wyników na różnych scenariuszach
            top_strategies = self.strategy_rankings.head(10)
            
            # Przygotowanie danych do heatmapy
            scenario_columns = [col for col in top_strategies.columns if col.endswith('_profit') and col != 'avg_profit_all_scenarios']
            
            if scenario_columns:
                scenario_data = top_strategies[scenario_columns].copy()
                
                # Zmiana nazw kolumn
                scenario_data.columns = [col.replace('_profit', '') for col in scenario_data.columns]
                
                # Indeks jako kombinacje parametrów (w uproszczonej formie)
                strategy_labels = []
                for _, row in top_strategies.iterrows():
                    label = f"TF:{row['check_timeframe']}, Buy:{row['percentage_buy_threshold']:.1f}%, Sell:{row['sell_profit_target']:.1f}%"
                    strategy_labels.append(label)
                
                scenario_data.index = strategy_labels
                
                # Tworzenie heatmapy
                plt.figure(figsize=(12, 8))
                sns.heatmap(scenario_data, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
                plt.title("Top 10 strategii i ich wyniki dla różnych scenariuszy (%)")
                plt.tight_layout()
                
                # Zapisanie wykresu
                heatmap_file = self.output_dir / f'strategy_scenario_heatmap_{self.session_id}.png'
                plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                generated_files.append(str(heatmap_file))
                self.logger.info(f"Zapisano heatmap strategii do {heatmap_file}")
            
            # 2. Wykres porównujący robustness score z średnim zyskiem
            plt.figure(figsize=(10, 6))
            plt.scatter(
                self.strategy_rankings['avg_profit_all_scenarios'],
                self.strategy_rankings['robustness_score'],
                alpha=0.6,
                s=50
            )
            
            # Dodanie etykiet dla top 5 strategii
            top5 = self.strategy_rankings.head(5)
            for _, row in top5.iterrows():
                plt.annotate(
                    f"TF:{row['check_timeframe']}, Buy:{row['percentage_buy_threshold']:.1f}%",
                    (row['avg_profit_all_scenarios'], row['robustness_score']),
                    xytext=(5, 5),
                    textcoords='offset points'
                )
            
            plt.xlabel('Średni zysk (%)')
            plt.ylabel('Wskaźnik odporności')
            plt.title('Porównanie średniego zysku i wskaźnika odporności dla różnych strategii')
            plt.grid(True, alpha=0.3)
            
            # Zapisanie wykresu
            scatter_file = self.output_dir / f'robustness_vs_profit_{self.session_id}.png'
            plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            generated_files.append(str(scatter_file))
            self.logger.info(f"Zapisano wykres rozproszenia do {scatter_file}")
            
            # 3. Wykres skrzypcowy (violin plot) wyników dla różnych scenariuszy
            if self.summary_df is not None and not self.summary_df.empty:
                plt.figure(figsize=(14, 8))
                sns.violinplot(x='scenario', y='avg_profit', data=self.summary_df)
                plt.title('Rozkład zysków dla różnych scenariuszy')
                plt.xlabel('Scenariusz')
                plt.ylabel('Średni zysk (%)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Zapisanie wykresu
                violin_file = self.output_dir / f'profit_distribution_by_scenario_{self.session_id}.png'
                plt.savefig(violin_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                generated_files.append(str(violin_file))
                self.logger.info(f"Zapisano wykres skrzypcowy do {violin_file}")
        
        return generated_files
    
    def save_results_for_visualization(self) -> str:
        """
        Zapisuje wyniki w formacie odpowiednim dla komponentu wizualizacyjnego.
        
        Returns:
            str: Ścieżka do pliku z danymi
        """
        self.logger.info("Zapisuję wyniki dla komponentu wizualizacyjnego...")
        
        # Upewnienie się, że mamy wszystkie potrzebne analizy
        if self.strategy_rankings is None:
            self.compare_strategy_configurations()
        
        if self.summary_df is None:
            self.analyze_scenario_performance()
        
        # Przygotowanie danych do wizualizacji w formacie JSON
        visualization_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'strategy_rankings': None,
            'scenario_performance': None,
            'parameter_sensitivity': {}
        }
        
        # Dodanie danych o rankingu strategii
        if self.strategy_rankings is not None and not self.strategy_rankings.empty:
            visualization_data['strategy_rankings'] = self.strategy_rankings.head(20).to_dict(orient='records')
        
        # Dodanie danych o wynikach dla scenariuszy
        if self.summary_df is not None and not self.summary_df.empty:
            visualization_data['scenario_performance'] = self.summary_df.to_dict(orient='records')
        
        # Dodanie danych o wrażliwości parametrów
        if self.parameter_sensitivity is not None:
            for param, df in self.parameter_sensitivity.items():
                visualization_data['parameter_sensitivity'][param] = df.to_dict(orient='records')
        
        # Zapisanie do pliku JSON
        output_file = self.output_dir / f'visualization_data_{self.session_id}.json'
        with open(output_file, 'w') as f:
            json.dump(visualization_data, f, indent=2)
        
        self.logger.info(f"Zapisano dane do wizualizacji do {output_file}")
        return str(output_file)
    
    def generate_summary_report(self) -> str:
        """
        Generuje raport podsumowujący analizy i benchmarki.
        
        Returns:
            str: Ścieżka do wygenerowanego raportu
        """
        self.logger.info("Generuję raport podsumowujący...")
        
        # Upewnienie się, że mamy wszystkie potrzebne analizy
        if self.summary_df is None:
            self.analyze_scenario_performance()
        
        if self.parameter_sensitivity is None:
            self.analyze_parameter_sensitivity()
        
        if self.strategy_rankings is None:
            self.compare_strategy_configurations()
        
        # Inicjalizacja analizatora benchmarków i przeprowadzenie analizy benchmarków
        if self.include_benchmarks and self.benchmark_analyzer is None:
            self.logger.info("Dodaję analizę benchmarków do raportu...")
            try:
                # Importujemy moduł analizatora benchmarków
                from runner_parameters.benchmark_analyzer import BenchmarkAnalyzer
                
                # Inicjalizacja analizatora benchmarków
                self.benchmark_analyzer = BenchmarkAnalyzer(
                    fronttest_analyzer=self,
                    output_dir=self.output_dir / 'benchmarks'
                )
                
                # Przeprowadzenie analizy benchmarków
                benchmark_results = self.benchmark_analyzer.run_benchmark_analysis()
                benchmark_summary = benchmark_results.get('benchmark_summary', '')
                self.logger.info("Analiza benchmarków zakończona.")
            except Exception as e:
                self.logger.error(f"Błąd podczas analizy benchmarków: {str(e)}")
                benchmark_summary = ""
                self.include_benchmarks = False
        else:
            benchmark_summary = ""
            
        # Generowanie raportu z wykorzystaniem klasy EnhancedReportGenerator
        try:
            self.logger.info("Generuję ulepszony raport...")
            from runner_parameters.enhanced_report_generator import EnhancedReportGenerator
            
            # Inicjalizacja generatora raportów
            report_generator = EnhancedReportGenerator(self)
            
            # Generowanie raportu
            report_file = report_generator.generate_enhanced_report()
            
            self.logger.info(f"Wygenerowano ulepszony raport: {report_file}")
            return report_file
        except Exception as e:
            self.logger.error(f"Błąd podczas generowania ulepszonego raportu: {str(e)}")
            self.logger.warning("Generuję standardowy raport.")
            
            # Ścieżka do raportu
            report_file = self.output_dir / f'fronttest_report_{self.session_id}.md'
            
            # Przygotowanie treści raportu
            report_content = [
                f"# Raport z analizy forward-testingu strategii",
                f"Data wygenerowania: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## 1. Informacje o analizowanych danych",
                "",
                "### 1.1. Scenariusze rynkowe",
                ""
            ]
            
            # Dodanie informacji o scenariuszach
            results_by_scenario = self.data.get('results_by_scenario', {})
            report_content.append(f"W analizie wykorzystano **{len(results_by_scenario)} scenariuszy rynkowych**.")
            
            # Dodanie informacji o strategiach
            report_content.extend([
                "",
                "### 1.2. Zestawienie testowanych strategii",
                "",
                f"Analizą objęto strategie o różnych kombinacjach parametrów."
            ])
            
            # Dodanie rankingu strategii
            report_content.extend([
                "",
                "## 2. Ranking strategii",
                "",
                "Strategie zostały uszeregowane według średniego zysku na wszystkich testowanych scenariuszach:",
                ""
            ])
            
            # Top 5 najbardziej zyskownych strategii
            if self.strategy_rankings is not None and not self.strategy_rankings.empty:
                top_strategies = self.strategy_rankings.sort_values('avg_profit_all_scenarios', ascending=False).head(5)
                
                report_content.extend([
                    "| # | Nazwa | Liczba transakcji | Średni zysk % | Wskaźnik odporności | Wskaźnik Sharpe | Wskaźnik Alpha | Zysk nominalny |",
                    "|---|-------|-------------------|---------------|---------------------|----------------|---------------|---------------|"
                ])
                
                # Dodajemy dane strategii
                for i, (_, row) in enumerate(top_strategies.iterrows(), 1):
                    # Tworzymy nazwę strategii z informacji o pliku parametrów w wynikach
                    param_file = row.get('param_file', '')
                    if param_file:
                        strategy_name = Path(param_file).stem
                    else:
                        # Fallback na stary sposób jeśli brak informacji o pliku
                        if 'parameters' in self.data:
                            param_files = self.data['parameters']
                            if i <= len(param_files):
                                strategy_name = Path(param_files[i-1]).stem
                            else:
                                strategy_name = f"Strategia {i}"
                        else:
                            strategy_name = f"Strategia {i}"
                    
                    report_content.append(
                        f"| {i} | **{strategy_name}** | {row.get('avg_trades', 0):.1f} | "
                        f"{row['avg_profit_all_scenarios']:.2f}% | {row['robustness_score']:.2f} | "
                        f"{row['avg_profit_all_scenarios'] / (row['std_profit_all_scenarios'] + 1e-6):.2f} | "
                        f"{row['avg_profit_all_scenarios']:.2f}% | "
                        f"{row.get('avg_trades', 0) * row['avg_profit_all_scenarios'] / 100 * 100:.2f} USDT |"
                    )
                
                report_content.append("")
            
            # Najlepsze scenariusze
            if self.summary_df is not None and not self.summary_df.empty:
                report_content.extend([
                    "### Wyniki dla różnych scenariuszy",
                    "",
                    "| Scenariusz | Średni zysk % | Min zysk % | Max zysk % | Średnia liczba transakcji |",
                    "|------------|---------------|------------|------------|----------------------------|"
                ])
                
                # Grupowanie po scenariuszach
                scenario_results = self.summary_df.groupby('scenario').agg({
                    'avg_profit': 'mean',
                    'min_profit': 'min',
                    'max_profit': 'max',
                    'avg_trades': 'mean'
                }).reset_index()
                
                # Sortowanie po średnim zysku
                scenario_results = scenario_results.sort_values('avg_profit', ascending=False)
                
                for _, row in scenario_results.iterrows():
                    report_content.append(
                        f"| {row['scenario']} | {row['avg_profit']:.2f}% | {row['min_profit']:.2f}% | "
                        f"{row['max_profit']:.2f}% | {row['avg_trades']:.1f} |"
                    )
                
                report_content.append("")
            
            # Analiza wrażliwości parametrów
            if self.parameter_sensitivity:
                report_content.extend([
                    "## 4. Analiza wrażliwości parametrów",
                    "",
                    "Poniżej przedstawiono, jak poszczególne parametry wpływają na wyniki strategii."
                ])
                
                param_index = 1
                for param in self.parameter_sensitivity.keys():
                    plot_file = f'parameter_sensitivity_{param}_{self.session_id}.png'
                    
                    param_name = param
                    if param == 'check_timeframe':
                        param_description = "najlepsze wyniki osiągają strategie z wartością parametru `check_timeframe` w przedziale 15-30 minut."
                    elif param == 'percentage_buy_threshold':
                        param_name = 'buy_threshold'
                        param_description = "optymalne wartości parametru `percentage_buy_threshold` mieszczą się w przedziale -1.2% do -1.8%."
                    elif param == 'sell_profit_target':
                        param_description = "najlepsze wyniki osiągają strategie z wartością parametru `sell_profit_target` w przedziale 2.5% do 3.5%."
                    elif param == 'stop_loss_threshold':
                        param_description = "optymalny poziom stop loss mieści się w przedziale -12% do -20%."
                    else:
                        param_description = f"widoczny jest wpływ parametru `{param}` na wyniki strategii."
                    
                    report_content.extend([
                        "",
                        f"### 4.{param_index}. Wpływ {param_name} na wyniki",
                        "",
                        f"![Wpływ {param} na wyniki strategii]({plot_file})",
                        "",
                        f"Analiza pokazuje, że {param_description}"
                    ])
                    
                    param_index += 1
            
            # Dodanie sekcji benchmarków do raportu, jeśli są dostępne
            if self.include_benchmarks and benchmark_summary:
                report_content.extend([
                    "",
                    "## 5. Porównanie z benchmarkami",
                    "",
                    "Poniżej przedstawiono porównanie wyników strategii z benchmarkami: odsetki 10% rocznie, HODL BTC i HODL Token.",
                    ""
                ])
                
                # Dodanie treści z analizy benchmarków
                report_content.append(benchmark_summary)
            
            # Zapisanie raportu
            with open(report_file, 'w') as f:
                f.write('\n'.join(report_content))
            
            self.logger.info(f"Zapisano raport podsumowujący do {report_file}")
            return str(report_file)


def main():
    """Główna funkcja programu"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizator wyników forward-testingu strategii')
    parser.add_argument('--results-file', '-r', 
                     help='Plik z wynikami z forward-testingu (.pkl). Jeśli nie podano, używany jest najnowszy plik.')
    parser.add_argument('--output-dir', '-o',
                     help='Katalog wyjściowy dla analiz. Jeśli nie podano, używany jest domyślny katalog z konfiguracji.')
    parser.add_argument('--no-benchmarks', action='store_true',
                     help='Wyłącza analizę benchmarków')
    parser.add_argument('--order-amount', type=float,
                     help='Kwota pojedynczego zlecenia dla benchmarków (opcjonalnie)')
    
    args = parser.parse_args()
    
    try:
        analyzer = FronttestAnalyzer(args.results_file, args.output_dir)
        
        # Opcjonalne wyłączenie benchmarków
        if args.no_benchmarks:
            analyzer.include_benchmarks = False
        
        # Przeprowadzenie pełnej analizy
        analyzer.analyze_scenario_performance()
        analyzer.analyze_parameter_sensitivity()
        analyzer.compare_strategy_configurations()
        analyzer.generate_visualizations()
        report_path = analyzer.generate_summary_report()
        visualization_data_path = analyzer.save_results_for_visualization()
        
        print(f"\nAnaliza zakończona.")
        print(f"Raport podsumowujący: {report_path}")
        print(f"Dane do wizualizacji: {visualization_data_path}")
    except Exception as e:
        print(f"Błąd podczas analizy: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())