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
)

from analysis.parameters import (
    analyze_parameter_distributions,
    calculate_parameter_importance,
#    analyze_parameter_correlations
)
logger = get_strategy_logger('strategy_reports')

def save_analysis(
    ranked_results: List[Dict],
    error_results: List[Dict],
    analysis_stats: Dict,
    source_file: str,
    importance_by_symbol: Dict[str, Dict[str, float]],
    cache: Optional[AnalysisCache] = None # Cache nadal może być potrzebny dla metryk formatowanych
) -> Path:
    """
    Zapisuje wyniki analizy do pliku TXT.
    Raportuje unikalne strategie (reprezentantów klastrów) znalezione przez analyze_strategies.

    Args:
        ranked_results: Lista unikalnych strategii posortowanych według score (wynik z analyze_strategies).
        error_results: Lista wyników z błędami.
        analysis_stats: Statystyki analizy.
        source_file: Nazwa pliku źródłowego.
        importance_by_symbol: Słownik z wagami parametrów per symbol.
        cache: Opcjonalna instancja cache'u analiz (głównie dla formatowania metryk).

    Returns:
        Path: Ścieżka do wygenerowanego pliku.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Upewnijmy się, że katalog istnieje (przeniesione z utils.config dla pewności)
    WYNIKI_ANALIZA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = WYNIKI_ANALIZA_DIR / f'analysis_results_{timestamp}.txt'

    logger.info("Rozpoczynam zapis analizy do pliku TXT...")

    # Grupujemy unikalne wyniki wg symbolu (dla struktury raportu)
    results_by_symbol = {}
    for result in ranked_results:
        symbol = result.get('symbol', 'BTC/USDT')
        if symbol not in results_by_symbol:
            results_by_symbol[symbol] = []
        results_by_symbol[symbol].append(result)

    # Buforowany zapis do pliku
    with open(output_file, 'w', encoding='utf-8') as f:
        # --- Nagłówek i statystyki globalne (bez zmian) ---
        initial_total = analysis_stats.get('total_initial_results_in_file', 0)
        errors = analysis_stats.get('error_count', 0)
        unique_count = analysis_stats.get('total_unique', 0)
        filtered_global_count = analysis_stats.get('filtered_cluster_count', 0)
        count_before_clustering = analysis_stats.get('total_after_prefiltering', 0)
        global_filtered_percentage = (filtered_global_count / count_before_clustering * 100) if count_before_clustering > 0 else 0.0

        header = [
            "=== Analiza strategii tradingowych ===",
            f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Źródło danych: {source_file}\n",
            "=== Statystyki globalne ===",
            f"Całkowita liczba strategii (w pliku): {initial_total:,}",
            f"Liczba błędów (obliczanie metryk): {errors:,}",
            f"Liczba strategii po pre-filtrowaniu: {count_before_clustering:,}",
            f"Liczba unikalnych strategii (klastrów): {unique_count:,}", # Zmieniono opis
            f"Liczba odrzuconych (klastrowanie): {filtered_global_count:,} "
            f"({global_filtered_percentage:.1f}% odrzuconych na etapie klastrowania)\n"
        ]
        f.write('\n'.join(header))

        # --- Szczegółowe statystyki per symbol (bez zmian) ---
        for symbol in sorted(analysis_stats.get('strategy_distribution', {}).keys()):
            sym_stats = analysis_stats['strategy_distribution'][symbol]
            f.write(f"\nStatystyki dla {symbol}:\n")
            f.write(f"Początkowa liczba strategii (wejście do klastrowania): {sym_stats['initial_count']:,}\n")
            f.write(f"Liczba unikalnych strategii (klastrów): {sym_stats['final_count']:,}\n") # Zmieniono opis
            f.write(f"Odrzucono (klastrowanie): {sym_stats['filtered_count']:,} "
                   f"({sym_stats['filtered_percentage']:.1f}%)\n")

            # Statystyki follow BTC dla symbolu (bez zmian w logice, ale działa na ranked_results)
            btc_strategies = [r for r in ranked_results
                            if r.get('symbol') == symbol and
                            isinstance(r.get('parameters'), dict) and # Dodatkowe sprawdzenie
                            r['parameters'].get('follow_btc_price', False)]
            if btc_strategies:
                f.write("\nStatystyki follow BTC (dla unikalnych strategii):\n")
                f.write(f"Liczba unikalnych strategii z follow BTC: {len(btc_strategies):,}\n")
                # Sprawdź czy metryki istnieją przed obliczeniem średniej
                valid_btc_metrics = [s['metrics'] for s in btc_strategies if isinstance(s.get('metrics'), dict)]
                if valid_btc_metrics:
                    avg_effectiveness = np.mean([m.get('btc_block_effectiveness', 0.0) for m in valid_btc_metrics if m.get('btc_block_effectiveness') is not None])
                    avg_block_rate = np.mean([m.get('btc_block_rate', 0.0) for m in valid_btc_metrics if m.get('btc_block_rate') is not None])
                    f.write(f"Średnia skuteczność blokad: {avg_effectiveness:.2f}%\n")
                    f.write(f"Średni procent czasu blokowania: {avg_block_rate:.2f}%\n")

        f.write("\n" + "=" * 80 + "\n\n")

        # --- Zapisywanie szczegółowych wyników per symbol (NOWA LOGIKA) ---
        f.write("=== Najlepsze Unikalne Strategie (Reprezentanci Klastrów) ===\n")
        total_reported = 0
        MAX_REPORTED_PER_SYMBOL = 10 # Limit strategii raportowanych per symbol

        for symbol in sorted(results_by_symbol.keys()):
            f.write(f"\n\n--- Symbol: {symbol} ---\n")
            symbol_results = results_by_symbol[symbol]
            # Wyniki dla symbolu powinny być już posortowane wg score
            reported_count_symbol = 0
            for idx, strategy in enumerate(symbol_results):
                if reported_count_symbol >= MAX_REPORTED_PER_SYMBOL:
                    break

                strategy_id = strategy.get('strategy_id', 'N/A')
                score = strategy.get('score', 0.0)
                metrics = strategy.get('metrics', {})
                params = strategy.get('parameters', {})
                # Pobierz rozmiar klastra z analysis_info lub bezpośrednio, jeśli tam jest
                cluster_size = strategy.get('analysis_info', {}).get('cluster_size', strategy.get('cluster_size', 1))
                param_importance = importance_by_symbol.get(symbol, {})

                f.write(f"\n=== Unikalna Strategia {idx + 1} (Rank: {idx + 1}) ===")
                if cluster_size > 1:
                     f.write(f" [Reprezentant klastra {cluster_size} strategii]\n")
                else:
                     f.write(" [Unikalna strategia / Klaster rozmiaru 1]\n") # Lub pomiń dla klastrów=1

                f.write(f"ID strategii: {strategy_id}\n")
                f.write(f"Score: {score:.2f}\n")

                # Formatowanie metryk (używamy istniejącej funkcji, opcjonalnie cache)
                if cache:
                    formatted_metrics = cache.get_formatted_metrics(strategy_id, metrics)
                else:
                    formatted_metrics = format_metrics(metrics) # format_metrics jest w metrics.py

                f.write("\nMetryki:\n")
                for k, v in formatted_metrics.items():
                    if k != 'btc_metrics':
                        # Formatowanie nazw kluczy dla lepszej czytelności
                        key_name = k.replace('_', ' ').title()
                        f.write(f"  {key_name}: {v}\n")
                if 'btc_metrics' in formatted_metrics:
                    f.write("  Metryki BTC:\n")
                    for k_btc, v_btc in formatted_metrics['btc_metrics'].items():
                        key_name_btc = k_btc.replace('_', ' ').title()
                        f.write(f"    {key_name_btc}: {v_btc}\n")

                f.write("\nParametry:\n")
                if not params:
                    f.write("  (Brak)\n")
                for name, value in sorted(params.items()):
                    imp = param_importance.get(name, 0)
                    tag = f" [Ważność: {imp:.3f}]" if imp > 0.05 else ""
                    # Używamy format_parameter_value z metrics.py
                    f.write(f"  {name}: {format_parameter_value(name, value)}{tag}\n")

                f.write("-" * 50 + "\n")
                reported_count_symbol += 1
                total_reported += 1

            if reported_count_symbol == 0:
                f.write("(Brak unikalnych strategii do raportowania dla tego symbolu)\n")
            elif len(symbol_results) > MAX_REPORTED_PER_SYMBOL:
                f.write(f"\n... (pominięto {len(symbol_results) - MAX_REPORTED_PER_SYMBOL} mniej istotnych unikalnych strategii dla {symbol}) ...\n")


        f.write(f"\nŁącznie zaraportowano {total_reported} najlepszych unikalnych strategii.\n")
        f.write('\n\n' + '=' * 80 + '\n\n')

        # --- Informacje o ważności parametrów (bez zmian) ---
        f.write("\n=== Ważność parametrów ===\n")
        if not importance_by_symbol:
             f.write("(Brak danych o ważności parametrów)\n")
        else:
             for symbol in sorted(importance_by_symbol.keys()):
                 f.write(f"\nParametry dla {symbol}:\n")
                 importance = importance_by_symbol[symbol]
                 if not importance:
                     f.write("  (Brak istotnych parametrów)\n")
                     continue
                 sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)

                 for param_name, importance_value in sorted_params:
                     if importance_value > 0.01:  # Pokazuj tylko istotne parametry
                         f.write(f"  {param_name}: {importance_value:.3f}\n")

        # --- Sekcja błędów (bez zmian) ---
        if error_results:
            f.write("\n\n=== Błędy analizy ===\n")
            for error in error_results:
                f.write(f"Strategy ID: {error.get('strategy_id', 'N/A')}\n")
                f.write(f"Error: {error.get('error', 'Unknown error')}\n\n")

    logger.info(f"Zapisano analizę TXT do pliku: {output_file}")
    return output_file

def generate_analysis_json(
    ranked_results: List[Dict],
    error_results: List[Dict],
    analysis_stats: Dict,
    param_distributions: Dict, # Z cache lub obliczone gdzie indziej
    output_folder: Path,
    top_n: int = 100 # Można zwiększyć limit dla JSON
) -> Path:
    """
    Generuje plik JSON z wynikami analizy.
    Raportuje unikalne strategie (reprezentantów klastrów).

    Args:
        ranked_results: Lista unikalnych strategii posortowanych wg score.
        error_results: Lista wyników z błędami.
        analysis_stats: Statystyki analizy.
        param_distributions: Słownik z rozkładami parametrów per symbol.
        output_folder: Katalog docelowy dla pliku JSON.
        top_n: Liczba najlepszych unikalnych strategii do uwzględnienia w sekcji 'strategies'.

    Returns:
        Path: Ścieżka do wygenerowanego pliku JSON.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = get_strategy_logger('strategy_reports') # Użyj lokalnego loggera
        output_folder.mkdir(parents=True, exist_ok=True) # Upewnij się, że katalog istnieje

        logger.info(f"Generuję plik JSON z {len(ranked_results)} unikalnymi strategiami (top {top_n} w raporcie)...")

        # Przygotowanie podsumowania per symbol na podstawie unikalnych wyników
        symbols_summary = {}
        results_by_symbol = {}
        for r in ranked_results:
            symbol = r.get('symbol', 'BTC/USDT')
            if symbol not in results_by_symbol:
                results_by_symbol[symbol] = []
            results_by_symbol[symbol].append(r)

        for symbol, symbol_results in results_by_symbol.items():
             # Statystyki z analysis_stats dla tego symbolu
             dist_stats = analysis_stats.get('strategy_distribution', {}).get(symbol, {})
             # Oblicz średnie metryki dla unikalnych strategii
             valid_metrics_list = [r['metrics'] for r in symbol_results if isinstance(r.get('metrics'), dict)]
             avg_win_rate = float(np.mean([m.get('win_rate', 0.0) for m in valid_metrics_list])) if valid_metrics_list else 0.0
             avg_profit = float(np.mean([m.get('avg_profit', 0.0) for m in valid_metrics_list])) if valid_metrics_list else 0.0
             # Znajdź najlepszy i najgorszy score wśród unikalnych
             scores = [r.get('score', 0.0) for r in symbol_results]
             best_score = float(max(scores)) if scores else 0.0
             worst_score = float(min(scores)) if scores else 0.0

             symbols_summary[symbol] = {
                 "total_unique_strategies": len(symbol_results), # Liczba unikalnych dla symbolu
                 "clusters_found": dist_stats.get('final_count', len(symbol_results)), # Liczba klastrów = liczba unikalnych
                 "initial_strategies_before_clustering": dist_stats.get('initial_count', 0),
                 "filtered_by_clustering": dist_stats.get('filtered_count', 0),
                 "avg_win_rate_unique": round(avg_win_rate, 2), # Średnia dla unikalnych
                 "avg_profit_unique": round(avg_profit, 2),   # Średnia dla unikalnych
                 "best_score_unique": round(best_score, 2),
                 "worst_score_unique": round(worst_score, 2)
             }

        # Przygotowanie głównej struktury danych
        analysis_data = {
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_info": analysis_stats.get('source_info', 'N/A'), # Dodaj info o źródle jeśli jest
            "global_stats": format_global_stats(analysis_stats),
            # Użyj nowej funkcji formatującej dla unikalnych strategii
            "top_unique_strategies": format_unique_strategies_json(ranked_results, top_n),
            # Użyj istniejącej funkcji formatującej dla dystrybucji parametrów
            "parameter_distributions": format_parameters(param_distributions),
            "error_summary": {
                "count": len(error_results),
                "details": [{"id": err.get('strategy_id', 'N/A'), "error": str(err.get('error', 'Unknown error'))}
                           for err in error_results[:50]] # Limit błędów w JSON
            },
            "symbols_summary": symbols_summary
        }

        # Generowanie nazwy pliku i zapis
        output_file = output_folder / f'analysis_{timestamp}.json'
        logger.info(f"Zapisuję wyniki JSON do: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            # Użyjemy konwertera do obsługi typów numpy
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                        np.int16, np.int32, np.int64, np.uint8,
                                        np.uint16, np.uint32, np.uint64)):
                        return int(obj)
                    elif isinstance(obj, (np.float_, np.float16, np.float32,
                                          np.float64)):
                        # Sprawdź NaN/Inf przed konwersją
                        if np.isnan(obj): return None # lub 'NaN' jako string
                        if np.isinf(obj): return None # lub 'Infinity'/' -Infinity'
                        return float(obj)
                    elif isinstance(obj, (np.ndarray,)): # Obsługa tablic numpy
                        return obj.tolist() # Konwertuj do listy
                    elif isinstance(obj, (np.bool_)):
                        return bool(obj)
                    elif isinstance(obj, (np.void)): # Obsługa typów złożonych np. z pandas
                        return None
                    return json.JSONEncoder.default(self, obj)

            json.dump(analysis_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        logger.info("Zakończono generowanie pliku JSON")
        return output_file

    except Exception as e:
        logger.error(f"Błąd podczas generowania pliku JSON: {str(e)}", exc_info=True)
        # Zwróć None lub rzuć wyjątek dalej, aby zasygnalizować błąd
        return None # Zwracamy None w przypadku błędu

'''
def group_similar_strategies(results: List[Dict]) -> List[Dict]:
    """
    Grupuje strategie na podstawie podobieństwa ich metryk za pomocą haszowania.
    AKTUALIZACJA : FUNKCJA ZASTĄPIONA PRZEZ KLASTROWANIE W filter_unique_strategies. 
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
'''
    

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

'''
def format_strategies(ranked_results: List[Dict], analysis_stats: Dict, top_n: int = 10) -> List[Dict]:
    """
    Formatuje wyniki najlepszych grup strategii.
    AKTUALIZACJA: [FUNKCJA ZASTĄPIONA] Formatuje wyniki najlepszych GRUP strategii.
    
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
'''
    
def format_unique_strategies_json(unique_results: List[Dict], top_n: int) -> List[Dict]:
    """
    Formatuje listę unikalnych strategii (reprezentantów klastrów) dla raportu JSON.

    Args:
        unique_results: Lista unikalnych strategii posortowana wg score.
        top_n: Liczba najlepszych strategii do sformatowania.

    Returns:
        List[Dict]: Lista sformatowanych strategii.
    """
    formatted_strategies = []
    for rank, strategy in enumerate(unique_results[:top_n]):
        strategy_id = strategy.get('strategy_id', 'N/A')
        symbol = strategy.get('symbol', 'BTC/USDT')
        score = strategy.get('score', 0.0)
        metrics = strategy.get('metrics', {})
        params = strategy.get('parameters', {})
        cluster_size = strategy.get('analysis_info', {}).get('cluster_size', strategy.get('cluster_size', 1))
        param_file = strategy.get('param_file_name', 'unknown')

        # Użyj funkcji format_metrics do sformatowania kluczowych metryk
        formatted_metrics = format_metrics(metrics)

        formatted_strategy = {
            "rank": rank + 1,
            "strategy_id": strategy_id,
            "symbol": symbol,
            "score": round(score, 3) if isinstance(score, (int, float)) else score,
            "cluster_size": cluster_size,
            "metrics": formatted_metrics, # Sformatowane kluczowe metryki
            "parameters": params, # Pełne parametry reprezentanta
            "param_file_name": param_file
        }
        formatted_strategies.append(formatted_strategy)

    return formatted_strategies