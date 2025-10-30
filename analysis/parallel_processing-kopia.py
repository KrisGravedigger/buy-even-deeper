#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moduł zawierający funkcje do równoległego przetwarzania analizy per symbol.
"""

from typing import Dict, List, Tuple, Any, Optional
import multiprocessing as mp
from datetime import datetime
import numpy as np
import time
from analysis.cache import AnalysisCache
from utils.logging_setup import get_strategy_logger

logger = get_strategy_logger('parallel_processing')

__all__ = ['parallel_process_results']

def process_symbol_data(symbol: str, 
                       symbol_results: List[Dict], 
                       cache: AnalysisCache) -> Dict:
    """
    Przetwarza dane dla pojedynczego symbolu.
    
    Args:
        symbol: Symbol do przetworzenia
        symbol_results: Lista wyników dla danego symbolu
        cache: Instancja cache'u analiz
        
    Returns:
        Dict: Przetworzone dane dla symbolu
    """
    # Pobieramy grupy z cache'u lub obliczamy
    strategy_groups = cache.get_groups(symbol, symbol_results) if cache else []
    
    # Przygotowanie statystyk dla symbolu
    stats = {
        "total_strategies": len(symbol_results),
        "total_groups": len(strategy_groups),
        "avg_win_rate": np.mean([r['metrics']['win_rate'] for r in symbol_results]),
        "avg_profit": np.mean([r['metrics']['avg_profit'] for r in symbol_results]),
        "best_score": max([r.get('score', 0) for r in symbol_results]),
        "worst_score": min([r.get('score', 0) for r in symbol_results]),
        "strategy_groups": []
    }
    
    # Przetwarzanie każdej grupy
    for group in strategy_groups[:10]:  # TOP 10 grup
        base_strategy = group['base_strategy']
        formatted_metrics = {
            'trade_count': base_strategy['metrics'].get('trade_count', 0),
            'win_rate': base_strategy['metrics'].get('win_rate', 0),
            'avg_profit': base_strategy['metrics'].get('avg_profit', 0),
            'total_profit': base_strategy['metrics'].get('total_profit', 0),
            'max_drawdown': base_strategy['metrics'].get('max_drawdown', 0)
        }
        
        processed_group = {
            "base_strategy": {
                "strategy_id": base_strategy['strategy_id'],
                "metrics": formatted_metrics,
                "score": base_strategy.get('score', 0)
            },
            "similar_count": group['similar_count'],
            "common_params": group['common_params'],
            "varying_params": group['varying_params']
        }
        
        stats["strategy_groups"].append(processed_group)
    
    # Analiza parametrów
    if cache:
        param_distributions = cache.get_param_distributions(symbol, symbol_results)
    else:
        from analysis.parameters import analyze_parameter_distributions
        param_distributions = analyze_parameter_distributions(symbol_results)
    
    stats["param_distributions"] = param_distributions
    
    # Generowanie tekstu dla symbolu
    text_output = []
    text_output.append(f"\n=== Analiza dla pary {symbol} ===")
    text_output.append("=" * 50 + "\n")
    
    # Statystyki dla symbolu
    text_output.append("Statystyki strategii:")
    text_output.append(f"Całkowita liczba strategii: {stats['total_strategies']:,}")
    text_output.append(f"Liczba grup: {stats['total_groups']:,}")
    text_output.append(f"Średni win rate: {stats['avg_win_rate']:.2f}%")
    text_output.append(f"Średni zysk: {stats['avg_profit']:.2f}%")
    text_output.append(f"Zakres score: {stats['worst_score']:.2f} - {stats['best_score']:.2f}\n")
    
    # Prezentacja TOP 10 grup/strategii
    text_output.append("TOP 10 strategii:")
    for idx, group in enumerate(stats['strategy_groups'], 1):
        text_output.append(f"\n=== Grupa/Strategia {idx} ===")
        base_strategy = group['base_strategy']
        metrics = base_strategy['metrics']
        
        text_output.append(f"ID strategii: {base_strategy['strategy_id']}")
        text_output.append(f"Score: {base_strategy['score']:.2f}\n")
        
        text_output.append("Metryki:")
        text_output.append(f"  Liczba transakcji: {metrics['trade_count']}")
        text_output.append(f"  Win Rate: {metrics['win_rate']:.2f}%")
        text_output.append(f"  Średni zysk: {metrics['avg_profit']:.2f}%")
        text_output.append(f"  Całkowity zysk: {metrics['total_profit']:.2f}%")
        text_output.append(f"  Max drawdown: {metrics['max_drawdown']:.2f}%\n")
        
        if group['similar_count'] > 1:
            text_output.append(f"Liczba podobnych strategii: {group['similar_count']}\n")
        
        text_output.append("Parametry stałe:")
        for param, value in sorted(group['common_params'].items()):
            text_output.append(f"  {param}: {value}")
        
        if group['varying_params']:
            text_output.append("\nParametry zmienne:")
            for param, values in sorted(group['varying_params'].items()):
                if len(values) <= 5:
                    text_output.append(f"  {param}: {', '.join(map(str, values))}")
                else:
                    text_output.append(f"  {param}: {min(values)} - {max(values)}")
        
        text_output.append("-" * 50)
    
    # Analiza parametrów
    text_output.append("\nAnaliza rozkładu parametrów:")
    for param_name, analysis in sorted(stats['param_distributions'].items()):
        text_output.append(f"\nParametr: {param_name}")
        if isinstance(analysis, dict):
            if 'mean' in analysis:
                text_output.append(f"  Średnia: {analysis['mean']:.3f}")
                text_output.append(f"  Zakres: {analysis['min']:.3f} - {analysis['max']:.3f}")
            if 'clusters' in analysis:
                text_output.append("\n  Wykryte skupiska:")
                for cluster in analysis['clusters']:
                    text_output.append(f"    Centrum: {cluster['center']:.3f}, "
                                    f"Liczebność: {cluster['size']}")
            
            # Dodatkowa analiza dla parametrów follow BTC
            if param_name == 'follow_btc_price' and analysis.get('btc_metrics'):
                btc_metrics = analysis['btc_metrics']
                text_output.append("\n  Metryki follow BTC:")
                text_output.append(f"    Średnia skuteczność: {btc_metrics['avg_effectiveness']:.2f}%")
                text_output.append(f"    Średni procent blokowania: {btc_metrics['avg_block_rate']:.2f}%")
                text_output.append(f"    Wpływ na korelację: {btc_metrics['correlation_impact']:.3f}")
    
    # Dedykowana sekcja dla follow BTC
    btc_strategies = [r for r in symbol_results if r['parameters'].get('follow_btc_price', False)]
    if btc_strategies:
        text_output.append(f"\n=== Analiza strategii follow BTC ===")
        text_output.append("=" * 50 + "\n")
        
        avg_effectiveness = np.mean([s['metrics']['btc_block_effectiveness'] for s in btc_strategies])
        avg_block_rate = np.mean([s['metrics']['btc_block_rate'] for s in btc_strategies])
        avg_correlation = np.mean([s['metrics']['btc_correlation'] for s in btc_strategies])
        
        text_output.append(f"Liczba strategii z follow BTC: {len(btc_strategies)}")
        text_output.append(f"Średnia skuteczność blokad: {avg_effectiveness:.2f}%")
        text_output.append(f"Średni procent czasu blokowania: {avg_block_rate:.2f}%")
        text_output.append(f"Średnia korelacja z BTC: {avg_correlation:.3f}")
        
        # Analiza parametrów follow BTC
        threshold_values = [s['parameters'].get('follow_btc_threshold', 0) for s in btc_strategies]
        block_time_values = [s['parameters'].get('follow_btc_block_time', 0) for s in btc_strategies]
        
        text_output.append("\nParametry follow BTC w najlepszych strategiach:")
        text_output.append(f"  Threshold: {min(threshold_values):.2f}% - {max(threshold_values):.2f}%")
        text_output.append(f"  Block time: {min(block_time_values)} - {max(block_time_values)} min")
    
    return {
        "symbol": symbol,
        "stats": stats,
        "groups": strategy_groups,
        "text": "\n".join(text_output)
    }

def parallel_process_results(ranked_results: List[Dict], 
                           cache: Optional[AnalysisCache] = None,
                           processes: Optional[int] = None) -> Dict[str, Dict]:
    """
    Przetwarza wyniki równolegle dla każdego symbolu.
    
    Args:
        ranked_results: Lista posortowanych wyników
        cache: Opcjonalna instancja cache'u
        processes: Liczba procesów (domyślnie: liczba CPU)
        
    Returns:
        Dict[str, Dict]: Słownik z przetworzonymi danymi per symbol
    """
    if not processes:
        processes = mp.cpu_count()
    
    # Grupowanie wyników per symbol
    start_time = time.time()
    logger.info("\nPrzygotowuję dane do przetwarzania równoległego...")
    
    symbols = {r.get('symbol', 'BTC/USDT') for r in ranked_results}
    symbol_results_map = {
        symbol: [r for r in ranked_results if r.get('symbol', 'BTC/USDT') == symbol]
        for symbol in symbols
    }
    
    # Przygotowanie argumentów dla przetwarzania równoległego
    process_args = [
        (symbol, results, cache)
        for symbol, results in symbol_results_map.items()
    ]
    
    total_symbols = len(symbols)
    logger.info(f"Rozpoczynam równoległe przetwarzanie {total_symbols} symboli używając {processes} procesów")
    
    # Przetwarzanie równoległe
    with mp.Pool(processes) as pool:
        results = []
        for i, result in enumerate(pool.starmap(process_symbol_data, process_args), 1):
            results.append(result)
            
            # Logowanie postępu
            progress = (i / total_symbols) * 100
            elapsed_time = time.time() - start_time
            avg_time_per_symbol = elapsed_time / i
            remaining_symbols = total_symbols - i
            estimated_time = remaining_symbols * avg_time_per_symbol

            # Tworzenie paska postępu
            bar_length = 30
            filled_length = int(bar_length * i // total_symbols)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            
            logger.info(
                f"[{bar}] {progress:.1f}% ({i}/{total_symbols})\n"
                f"Przetworzono symbol: {result['symbol']}\n"
                f"Szacowany pozostały czas: {estimated_time/60:.1f} min"
            )
    
    total_time = time.time() - start_time
    logger.info(f"\nZakończono przetwarzanie równoległe w {total_time/60:.1f} min")
    
    # Konwersja wyników na słownik
    return {r["symbol"]: r for r in results}