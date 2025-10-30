#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moduł obsługujący generowanie i zapisywanie raportów z analizy strategii.
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from utils.logging_setup import get_strategy_logger
from utils.config import (
    WYNIKI_ANALIZA_DIR, 
    JSON_ANALYSIS_DIR,
    get_analysis_path
)

from analysis.cache import AnalysisCache

from analysis.metrics import (
    format_global_stats,
    format_metrics,
    format_parameter_value,
    are_metrics_similar
)

from analysis.parameters import (
    analyze_parameter_distributions,
    calculate_parameter_importance,
#    analyze_parameter_correlations
)

from analysis.parallel_processing import parallel_process_results

logger = get_strategy_logger('strategy_reports')

def save_analysis(
    ranked_results: List[Dict], 
    error_results: List[Dict], 
    analysis_stats: Dict, 
    source_file: str,
    importance_by_symbol: Dict[str, Dict[str, float]],
    cache: Optional[AnalysisCache] = None
) -> Path:
    """
    Zapisuje wyniki analizy do pliku z wykorzystaniem równoległego przetwarzania per symbol.
    
    Args:
        ranked_results: Lista wyników posortowanych według score
        error_results: Lista wyników z błędami
        analysis_stats: Statystyki analizy
        source_file: Nazwa pliku źródłowego
        importance_by_symbol: Słownik z wagami parametrów per symbol
        cache: Instancja cache'u analiz
        
    Returns:
        Path: Ścieżka do wygenerowanego pliku
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = WYNIKI_ANALIZA_DIR / f'analysis_results_{timestamp}.txt'
    
    logger.info("Rozpoczynam zapis analizy...")
    
    # Przetwarzanie równoległe wyników per symbol
    parallel_results = parallel_process_results(ranked_results, cache)
    
    # Buforowany zapis do pliku
    with open(output_file, 'w', encoding='utf-8') as f:
        # Nagłówek i statystyki globalne
        initial_total = analysis_stats.get('total_initial_results_in_file', 0)
        errors = analysis_stats.get('error_count', 0)
        unique_count = analysis_stats.get('total_unique', 0)
        # Użyj klucza opisującego filtrowanie przez klastrowanie
        filtered_global_count = analysis_stats.get('filtered_cluster_count', 0)
        # Liczba strategii na wejściu do klastrowania
        count_before_clustering = analysis_stats.get('total_after_prefiltering', 0)
        # Oblicz globalny procent odrzuconych przez klastrowanie
        global_filtered_percentage = (filtered_global_count / count_before_clustering * 100) if count_before_clustering > 0 else 0.0

        header = [
            "=== Analiza strategii tradingowych ===",
            f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Źródło danych: {source_file}\n",
            "=== Statystyki globalne ===",
            f"Całkowita liczba strategii (w pliku): {initial_total:,}",
            f"Liczba błędów (obliczanie metryk): {errors:,}",
            f"Liczba strategii po pre-filtrowaniu: {count_before_clustering:,}",
            f"Liczba unikalnych strategii (po klastrowaniu): {unique_count:,}",
            f"Liczba odrzuconych (klastrowanie): {filtered_global_count:,} "
            f"({global_filtered_percentage:.1f}% odrzuconych na tym etapie)\n"
        ]
        f.write('\n'.join(header))
        
        # Szczegółowe statystyki per symbol
        for symbol, sym_stats in analysis_stats.get('strategy_distribution', {}).items():
            f.write(f"\nStatystyki dla {symbol}:\n")
            f.write(f"Początkowa liczba strategii: {sym_stats['initial_count']:,}\n")
            f.write(f"Końcowa liczba strategii: {sym_stats['final_count']:,}\n")
            f.write(f"Odrzucono: {sym_stats['filtered_count']:,} "
                   f"({sym_stats['filtered_percentage']:.1f}%)\n")

            # Statystyki follow BTC dla symbolu
            btc_strategies = [r for r in ranked_results 
                            if r.get('symbol') == symbol and 
                            r['parameters'].get('follow_btc_price', False)]
            if btc_strategies:
                f.write("\nStatystyki follow BTC:\n")
                f.write(f"Liczba strategii z follow BTC: {len(btc_strategies):,}\n")
                avg_effectiveness = np.mean([s['metrics']['btc_block_effectiveness'] 
                                          for s in btc_strategies])
                avg_block_rate = np.mean([s['metrics']['btc_block_rate'] 
                                        for s in btc_strategies])
                f.write(f"Średnia skuteczność blokad: {avg_effectiveness:.2f}%\n")
                f.write(f"Średni procent czasu blokowania: {avg_block_rate:.2f}%\n")
            
            if symbol in analysis_stats.get('similarity_stats', {}):
                sim_stats = analysis_stats['similarity_stats'][symbol]
                f.write("\nStatystyki podobieństwa:\n")
                f.write(f"Średnie podobieństwo: {sim_stats['mean_similarity']:.3f}\n")
                f.write(f"Zakres podobieństwa: {sim_stats['min_similarity']:.3f} - "
                       f"{sim_stats['max_similarity']:.3f}\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Zapisywanie szczegółowych wyników per symbol
        for symbol in sorted(parallel_results.keys()):
            symbol_data = parallel_results[symbol]
            f.write(symbol_data['text'])
            f.write('\n\n' + '=' * 80 + '\n\n')
        
        # Informacje o ważności parametrów
        f.write("\n=== Ważność parametrów ===\n")
        for symbol in sorted(importance_by_symbol.keys()):
            f.write(f"\nParametry dla {symbol}:\n")
            importance = importance_by_symbol[symbol]
            sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            for param_name, importance_value in sorted_params:
                if importance_value > 0.01:  # Pokazuj tylko istotne parametry
                    f.write(f"  {param_name}: {importance_value:.3f}\n")
        
        # Sekcja błędów
        if error_results:
            f.write("\n=== Błędy analizy ===\n")
            for error in error_results:
                f.write(f"Strategy ID: {error.get('strategy_id', 'N/A')}\n")
                f.write(f"Error: {error.get('error', 'Unknown error')}\n\n")
    
    logger.info(f"Zapisano analizę do pliku: {output_file}")
    return output_file

def generate_analysis_json(
    ranked_results: List[Dict], 
    error_results: List[Dict],
    analysis_stats: Dict,
    param_distributions: Dict,
    output_folder: Path,
    top_n: int = 10) -> Path:
    """
    Generuje plik JSON z wynikami analizy z zachowaniem struktury grup.
    Wykorzystuje przetwarzanie równoległe dla przyspieszenia obliczeń.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = get_strategy_logger('strategy_reports')
        
        def process_symbol_data(symbol: str, all_results: List[Dict]) -> Tuple[str, Dict[str, Any]]:
            """Przetwarza dane dla pojedynczego symbolu"""
            symbol_results = [r for r in all_results if r.get('symbol') == symbol]
            return symbol, {
                "total_strategies": len(symbol_results),
                "total_groups": len(group_similar_strategies(symbol_results)),
                "avg_win_rate": float(np.mean([r['metrics']['win_rate'] for r in symbol_results])),
                "avg_profit": float(np.mean([r['metrics']['avg_profit'] for r in symbol_results])),
                "best_score": float(max([r.get('score', 0) for r in symbol_results])),
                "worst_score": float(min([r.get('score', 0) for r in symbol_results]))
            }

        # Zbieranie unikalnych symboli
        unique_symbols = {r.get('symbol', 'BTC/USDT') for r in ranked_results}
        logger.info(f"Rozpoczynam równoległe przetwarzanie {len(unique_symbols)} symboli...")

        # Równoległe przetwarzanie symboli
        symbols_data = {}
        with ThreadPoolExecutor(max_workers=min(8, len(unique_symbols))) as executor:
            process_func = partial(process_symbol_data, all_results=ranked_results)
            future_to_symbol = {executor.submit(process_func, symbol): symbol 
                              for symbol in unique_symbols}
            
            # Zbieranie wyników
            for future in future_to_symbol:
                try:
                    symbol, data = future.result()
                    symbols_data[symbol] = data
                except Exception as e:
                    logger.error(f"Błąd podczas przetwarzania symbolu: {str(e)}")

        # Przygotowanie głównej struktury danych (bez zmian)
        analysis_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "global_stats": format_global_stats(analysis_stats),
            "strategy_groups": format_strategies(ranked_results, analysis_stats, top_n),
            "parameters": format_parameters(param_distributions),
            "error_summary": {
                "count": len(error_results),
                "details": [{"id": err.get('strategy_id'), "error": str(err.get('error'))} 
                           for err in error_results]
            },
            "symbols_summary": symbols_data
        }

        # Generowanie nazwy pliku i zapis (bez zmian)
        output_file = output_folder / f'analysis_{timestamp}.json'
        logger.info(f"Zapisuję wyniki do: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        logger.info("Zakończono generowanie pliku JSON")
        return output_file
        
    except Exception as e:
        logger.error(f"Błąd podczas generowania pliku JSON: {str(e)}")
        raise

def group_similar_strategies(results: List[Dict]) -> List[Dict]:
    """
    Grupuje strategie na podstawie podobieństwa ich metryk za pomocą haszowania.

    Args:
        results: Lista wyników strategii

    Returns:
        List[Dict]: Lista grup strategii
    """
    if not results:
        return []

    # --- OPTYMALIZACJA: Grupowanie przez haszowanie metryk ---
    logger.debug(f"Grouping {len(results)} strategies by metrics hash...")
    metrics_groups = {} # Klucz: hash metryk, Wartość: lista strategii

    # Metryki kluczowe do haszowania (wybierz te, które definiują podobieństwo)
    # Można dostosować listę i precyzję zaokrąglenia
    HASH_METRICS = [
        'win_rate', 'avg_profit', 'total_trades', 'max_drawdown',
        'profit_factor', 'sharpe_ratio', 'sortino_ratio'
    ]
    HASH_PRECISION = 3 # Liczba miejsc po przecinku do zaokrąglenia

    valid_results_count = 0
    invalid_structure_count = 0
    for r in results:
        # Walidacja struktury
        if not isinstance(r, dict) or not all(k in r for k in ['metrics', 'parameters', 'strategy_id']) \
           or not isinstance(r['metrics'], dict):
            invalid_structure_count += 1
            continue

        metrics = r['metrics']
        try:
            # Tworzenie klucza hashującego na podstawie zaokrąglonych metryk
            metric_values_for_hash = []
            valid_hash_key = True
            for key in HASH_METRICS:
                val = metrics.get(key)
                if isinstance(val, (int, float)) and np.isfinite(val):
                    metric_values_for_hash.append(round(float(val), HASH_PRECISION))
                else:
                    # Jeśli brakuje kluczowej metryki lub jest niepoprawna, nie grupuj
                    valid_hash_key = False
                    break
            
            if not valid_hash_key:
                # Traktuj jako unikalną grupę (lub odrzuć, zależy od wymagań)
                # Tutaj tworzymy unikalny hash, aby zachować strategię
                metrics_hash = f"unique_{r['strategy_id']}"
            else:
                metrics_hash = tuple(metric_values_for_hash)

            if metrics_hash not in metrics_groups:
                metrics_groups[metrics_hash] = []
            metrics_groups[metrics_hash].append(r)
            valid_results_count += 1

        except Exception as e:
            logger.warning(f"Error hashing metrics for strategy {r.get('strategy_id', 'N/A')}: {e}")
            # Traktuj jako unikalną grupę
            metrics_hash = f"unique_{r['strategy_id']}"
            if metrics_hash not in metrics_groups: metrics_groups[metrics_hash] = []
            metrics_groups[metrics_hash].append(r)
            valid_results_count += 1


    if invalid_structure_count > 0:
        logger.warning(f"Skipped {invalid_structure_count} results due to invalid structure during grouping.")
    if not metrics_groups:
        logger.warning("No valid groups created after hashing metrics.")
        return []

    # Tworzenie finalnych grup z posortowanych list w każdej grupie hasha
    final_groups = []
    logger.debug(f"Processing {len(metrics_groups)} hashed metric groups...")
    for metrics_hash, group_strategies in metrics_groups.items():
        if not group_strategies: continue

        # Sortuj strategie wewnątrz grupy wg score (malejąco)
        group_strategies.sort(key=lambda x: x.get('metrics', {}).get('score', 0), reverse=True)
        base_strategy = group_strategies[0] # Najlepsza strategia w grupie

        # Identyfikacja parametrów wspólnych i zmiennych (jak poprzednio, ale na mniejszej grupie)
        common_params = {}
        varying_params = {}
        strategy_ids = [s['strategy_id'] for s in group_strategies]

        if len(group_strategies) > 1:
            param_names = list(base_strategy['parameters'].keys())
            for param_name in param_names:
                # Użyj set() do zbierania unikalnych wartości
                values_set = set()
                valid_value_type_found = False # Flaga, czy parametr ma sensowny typ
                for strategy in group_strategies:
                    param_value = strategy['parameters'].get(param_name)
                    # Sprawdzamy typy, które potrafimy porównać (liczby, bool, stringi)
                    if isinstance(param_value, (int, float, bool, str)):
                         # Zaokrąglij floaty dla porównania
                         if isinstance(param_value, float):
                             param_value = round(param_value, HASH_PRECISION + 1) # Trochę większa precyzja
                         values_set.add(param_value)
                         valid_value_type_found = True
                    # else: ignorujemy typy list/dict itp. dla porównania

                if valid_value_type_found:
                    if len(values_set) == 1:
                         common_params[param_name] = next(iter(values_set))
                    else:
                         # Spróbuj posortować, jeśli to możliwe (liczby)
                         try:
                             varying_params[param_name] = sorted(list(values_set))
                         except TypeError: # Jeśli typy mieszane lub nieporównywalne
                             varying_params[param_name] = list(values_set)
        else: # Grupa jednoelementowa
            common_params = base_strategy['parameters']

        final_groups.append({
            'base_strategy': base_strategy,
            'similar_count': len(group_strategies),
            'strategy_ids': strategy_ids,
            'common_params': common_params,
            'varying_params': varying_params,
            'metrics': base_strategy['metrics'] # Metryki najlepszej strategii w grupie
        })

    # Sortowanie finalnych grup wg score najlepszej strategii w grupie
    final_groups.sort(key=lambda g: g['base_strategy'].get('metrics', {}).get('score', 0), reverse=True)
    logger.debug(f"Finished grouping. Created {len(final_groups)} final groups.")
    return final_groups
    # --- KONIEC OPTYMALIZACJI ---

def format_parameters(param_distributions: Dict) -> Dict:
    """
    Formatuje analizę rozkładu parametrów.
    
    Args:
        param_distributions: Słownik z rozkładami parametrów
        
    Returns:
        Dict: Sformatowane rozkłady parametrów
    """
    formatted_params = {}
    
    for symbol, params in param_distributions.items():
        formatted_params[symbol] = {}
        
        for param_name, analysis in params.items():
            if isinstance(analysis, dict) and 'type' in analysis:
                if analysis['type'] == 'constant':
                    formatted_params[symbol][param_name] = {
                        "type": "constant",
                        "value": analysis['value']
                    }
                elif analysis['type'] == 'distribution':
                    param_data = {
                        "type": "distribution",
                        "statistics": {
                            "mean": round(analysis['mean'], 3),
                            "median": round(analysis['median'], 3),
                            "min": round(analysis['min'], 3),
                            "max": round(analysis['max'], 3),
                            "std": round(analysis['std'], 3)
                        }
                    }
                    
                    # Dodawanie klastrów jeśli istnieją
                    if 'clusters' in analysis:
                        param_data['clusters'] = [{
                            "center": round(cluster['center'], 3),
                            "size": cluster['size'],
                            "std": round(cluster['std'], 3)
                        } for cluster in analysis['clusters']]
                    
                    # Dodawanie wpływu na metryki jeśli istnieje
                    if 'metrics_impact' in analysis:
                        param_data['metrics_impact'] = {
                            metric: round(correlation, 3)
                            for metric, correlation in analysis['metrics_impact'].items()
                        }
                        
                    formatted_params[symbol][param_name] = param_data
                
    return formatted_params

def format_strategies(ranked_results: List[Dict], analysis_stats: Dict, top_n: int = 10) -> List[Dict]:
    """
    Formatuje wyniki najlepszych grup strategii.
    
    Args:
        ranked_results: Lista posortowanych wyników
        analysis_stats: Statystyki analizy
        top_n: Liczba najlepszych grup do uwzględnienia
        
    Returns:
        List[Dict]: Lista sformatowanych grup strategii
    """
    # Najpierw grupujemy strategie
    strategy_groups = group_similar_strategies(ranked_results)
    formatted_groups = []
    
    for group in strategy_groups[:top_n]:
        base_strategy = group['base_strategy']
        formatted_metrics = format_metrics(base_strategy['metrics'])
            
        formatted_group = {
            "group_id": len(formatted_groups) + 1,
            "similar_count": group['similar_count'],
            "base_strategy": {
                "strategy_id": base_strategy['strategy_id'],
                "symbol": base_strategy.get('symbol', 'BTC/USDT'),
                "metrics": formatted_metrics,
                "score": base_strategy.get('score', 0),
                "param_file_name": base_strategy.get('param_file_name', 'unknown')  # Dodane pole
            },
            "parameters": {
                "static": group['common_params'],
                "variable": {}
            }
        }
        
        # Dodanie informacji o zmiennych parametrach
        for param_name, values in group['varying_params'].items():
            formatted_group['parameters']['variable'][param_name] = {
                "values": sorted(values),
                "range": [min(values), max(values)]
            }
        
        formatted_groups.append(formatted_group)
    
    return formatted_groups