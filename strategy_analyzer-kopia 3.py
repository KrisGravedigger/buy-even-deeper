#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skrypt analizujący wyniki strategii tradingowej z rozszerzoną analizą parametrów.
Generuje szczegółowe raporty i rekomendacje dotyczące optymalizacji parametrów.

Przepływ:
1. Wczytanie danych z .pkl (load_results).
2. Obliczenie metryk i score dla wszystkich strategii (calculate_initial_metrics).
3. Opcjonalne pre-filtrowanie (score, trades, profit_factor).
4. Analiza główna (analyze_strategies):
   - Grupowanie per symbol.
   - Wypełnianie cache (rozkłady, grupy).
   - Filtrowanie unikalnych strategii przez KLASTROWANIE (filter_unique_strategies).
   - Obliczanie finalnej ważności parametrów.
   - Zbieranie statystyk.
5. Zapis wyników (save_analysis_results).
6. Archiwizacja .pkl.
7. Opcjonalna analiza rekomendacji.
"""

from pathlib import Path
import pickle
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
import shutil
import numpy as np
import json
import gc
import time
import argparse
from analysis.cache import AnalysisCache
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Importy z utils
from utils.config import (
    WYNIKI_BACKTEST_DIR,
    WYNIKI_ANALIZA_DIR,
    JSON_ANALYSIS_DIR,
    ensure_directory_structure
)
from utils.logging_setup import get_strategy_logger

# Sprawdzamy strukturę katalogów
if not ensure_directory_structure():
    print("ERROR: Nie udało się utworzyć wymaganej struktury katalogów. Sprawdź uprawnienia.")
    exit(1)

# Konfiguracja loggera
logger = get_strategy_logger('strategy_analyzer')

# --- Importy modułów analizy ---
# (Importujemy wszystko na górze dla przejrzystości)
from analysis.metrics import (
    analyze_single_strategy, # Do obliczania metryk
    calculate_strategy_score # Używane wewnętrznie
)
from analysis.metrics_calculator import StrategyMetrics # Podstawowa klasa metryk
from analysis.parameters import (
    # analyze_parameter_distributions, # Używane w cache.py
    calculate_parameter_importance, # Do obliczania ważności na końcu
    filter_unique_strategies      # Kluczowa funkcja do filtrowania przez klastrowanie
)
from analysis.reports import (
    save_analysis,            # Do zapisu TXT
    generate_analysis_json,   # Do zapisu JSON
    # group_similar_strategies # Używane w cache.py
)
from analysis.parameter_recommendations import (
    apply_recommendations # Do analizy rekomendacji
)
# from analysis.similarity import calculate_strategy_similarity # NIE UŻYWANE w tej wersji


# --- Funkcje pomocnicze ---

def load_results(results_file: Path) -> Tuple[Optional[List[Dict]], Dict]:
    """
    Wczytuje dane z pliku .pkl.

    Args:
        results_file: Ścieżka do pliku .pkl.

    Returns:
        Tuple[Optional[List[Dict]], Dict]:
            - Lista strategii (słowników) lub None w przypadku błędu.
            - Słownik z informacjami o danych rynkowych (lub pusty).
    """
    logger.info(f"Wczytywanie pliku: {results_file}...")
    try:
        with open(results_file, 'rb') as f:
            data = pickle.load(f)
            if not isinstance(data, dict) or 'results' not in data or not isinstance(data['results'], list):
                 logger.error(f"Nieprawidłowa struktura danych w pliku {results_file.name}.")
                 return None, {}

            results_list = data['results']
            market_data_info = data.get('market_data_info', {})
            logger.info(f"Wczytano {len(results_list):,} wyników z pliku.")
            return results_list, market_data_info

    except FileNotFoundError:
        logger.error(f"Plik wyników nie został znaleziony: {results_file}")
        return None, {}
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania lub dekodowania pliku {results_file}: {e}", exc_info=True)
        return None, {}


def calculate_initial_metrics(results_list: List[Dict], num_processes: int, batch_size: int = 2000) -> Tuple[List[Dict], List[Dict]]:
    """
    Oblicza metryki i score dla podanej listy strategii równolegle.

    Args:
        results_list: Lista strategii (słowników) z pliku .pkl.
        num_processes: Liczba procesów do użycia.
        batch_size: Rozmiar batcha do przetwarzania.

    Returns:
        Tuple[List[Dict], List[Dict]]:
            - initial_valid_results: Lista poprawnych strategii z dodanymi 'metrics' i 'score'.
            - error_results: Lista strategii, dla których wystąpił błąd podczas obliczania metryk.
    """
    total_results = len(results_list)
    if total_results == 0:
        return [], []

    initial_valid_results = []
    error_results = []
    logger.info(f"Rozpoczynam obliczanie metryk dla {total_results:,} strategii (procesy={num_processes}, batch={batch_size})...")

    try:
        analysis_func = analyze_single_strategy # Używamy funkcji zaimportowanej globalnie

        with mp.Pool(num_processes) as pool:
            batches = [results_list[i:min(i + batch_size, total_results)]
                       for i in range(0, total_results, batch_size)]
            num_batches = len(batches)

            with tqdm(total=total_results, desc="Obliczanie metryk", unit="strat") as pbar:
                for i, batch_data in enumerate(batches):
                    if not batch_data: continue

                    try:
                        # Równoległe obliczanie metryk dla batcha
                        analyzed_batch = pool.map(analysis_func, batch_data)
                    except Exception as pool_err:
                        logger.error(f"Błąd w puli procesów (batch {i+1}/{num_batches}): {pool_err}", exc_info=True)
                        error_results.extend([{'error': f'Pool error: {pool_err}', 'strategy_id': r.get('strategy_id', 'N/A')} for r in batch_data])
                        pbar.update(len(batch_data))
                        continue

                    # Zbieranie wyników z batcha
                    for result in analyzed_batch:
                        if result and isinstance(result, dict):
                            if 'error' in result:
                                error_results.append(result)
                            elif 'metrics' in result and 'score' in result:
                                initial_valid_results.append(result)
                            else:
                                logger.warning(f"Otrzymano niekompletny wynik z analyze_single_strategy: {result.get('strategy_id', 'N/A')}")
                                error_results.append({'error': 'Incomplete result', 'data': result})
                        elif result:
                            logger.warning(f"Otrzymano nieoczekiwany typ wyniku: {type(result)}")
                            error_results.append({'error': f'Unexpected result type: {type(result)}', 'data': result})

                    pbar.update(len(batch_data))
                    pbar.set_postfix({
                        'valid': len(initial_valid_results),
                        'errors': len(error_results),
                        'batch': f"{i+1}/{num_batches}"
                    })

                    del analyzed_batch, batch_data
                    if i % 5 == 0: gc.collect()

    except Exception as e:
         logger.error(f"Krytyczny błąd podczas równoległego obliczania metryk: {e}", exc_info=True)
         logger.warning("Zwracam częściowe wyniki.")
    finally:
         gc.collect()

    logger.info(f"Zakończono obliczanie metryk. Poprawne: {len(initial_valid_results):,}, Błędy: {len(error_results):,}")
    return initial_valid_results, error_results


def analyze_strategies(results_with_metrics: List[Dict], processes: int, cache: AnalysisCache) -> Tuple[List[Dict], Dict, Dict]:
    """
    Wykonuje główną analizę: filtrowanie unikalnych strategii przez klastrowanie,
    wypełnianie cache, obliczanie finalnej ważności i zbieranie statystyk.

    Args:
        results_with_metrics: Lista strategii z obliczonymi 'metrics' i 'score'
                              (po ewentualnym pre-filtrowaniu).
        processes: Liczba procesów (używana głównie przez funkcje wewnątrz, np. importance).
        cache: Instancja AnalysisCache do wypełnienia i użycia.

    Returns:
        Tuple[List[Dict], Dict, Dict]:
            - final_unique_results: Lista unikalnych strategii (wynik klastrowania), posortowana.
            - analysis_stats: Słownik z finalnymi statystykami analizy.
            - importance_by_symbol: Słownik z ważnością parametrów dla unikalnych strategii.
    """
    if not results_with_metrics:
        logger.warning("analyze_strategies otrzymało pustą listę wyników.")
        return [], {'total_after_prefiltering': 0, 'total_unique': 0}, {}

    if cache is None:
        logger.error("Błąd krytyczny: analyze_strategies wymaga instancji AnalysisCache!")
        raise ValueError("AnalysisCache instance is required for analyze_strategies.")

    initial_count_for_analysis = len(results_with_metrics)
    logger.info(f"Rozpoczynam analizę i filtrowanie przez klastrowanie dla {initial_count_for_analysis:,} strategii.")

    analysis_stats = {
        'total_after_prefiltering': initial_count_for_analysis,
        'error_count': 0, # Błędy powinny być odfiltrowane wcześniej
        'filtered_cluster_count': 0,
        'total_unique': 0,
        'clusters_count': 0, # Może być ustawione przez filter_unique_strategies
        'cluster_stats': {},
        'similarity_stats': {'status': 'Not calculated (clustering used)'}, # Oznaczamy brak tych statystyk
        'strategy_distribution': {}
    }

    # Grupowanie wyników według symboli
    results_by_symbol = {}
    for result in results_with_metrics:
        if not isinstance(result, dict) or 'strategy_id' not in result: continue
        symbol = result.get('symbol', 'BTC/USDT')
        if symbol not in results_by_symbol: results_by_symbol[symbol] = []
        results_by_symbol[symbol].append(result)
        cache.add_symbol(symbol)

    logger.info(f"Znaleziono {len(results_by_symbol)} symboli do analizy i filtrowania.")

    final_unique_results_list = [] # Globalna lista na unikalne strategie

    # Przetwarzanie symbol po symbolu
    for symbol in sorted(results_by_symbol.keys()):
        symbol_results = results_by_symbol[symbol]
        initial_count_symbol = len(symbol_results)
        logger.info(f"\nAnalizuję i filtruję symbol: {symbol} ({initial_count_symbol:,} strategii)")
        symbol_start_time = time.time()

        # Sortowanie wg score - ważne dla cache i wyboru reprezentanta klastra
        symbol_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        # --- Operacje Cache (przed filtrowaniem) ---
        try:
            logger.debug(f"[{symbol}] Cache: Pobieram/Obliczam rozkłady parametrów...")
            cache.get_param_distributions(symbol, symbol_results) # Używa top N z posortowanych   
        except Exception as e:
            logger.error(f"[{symbol}] Błąd podczas operacji cache: {e}", exc_info=True)
            logger.warning(f"[{symbol}] Pomijam dalszą analizę dla tego symbolu.")
            continue # Przejdź do następnego symbolu

        # --- Filtrowanie przez Klastrowanie (PCA/KMeans) ---
        logger.info(f"[{symbol}] Uruchamiam filtrowanie unikalnych strategii przez klastrowanie...")
        filter_start_time = time.time()
        unique_symbol_results = [] # Inicjalizacja na wypadek błędu
        try:
            # Wywołujemy funkcję z parameters.py - kluczowy krok filtrowania
            # Dostosuj parametry `similarity_threshold` i `min_strategies` wg potrzeb
            unique_symbol_results = filter_unique_strategies(
                results=symbol_results,
                similarity_threshold=0.90, # Próg może wymagać dostosowania dla klastrowania
                min_strategies=max(10, int(initial_count_symbol * 0.05)) # Np. min 10 lub 5%
            )
            filter_time = time.time() - filter_start_time
            logger.info(f"[{symbol}] Filtrowanie przez klastrowanie zakończone w {filter_time:.2f}s. "
                        f"Liczba unikalnych: {len(unique_symbol_results)}")

        except ImportError:
            logger.error(f"[{symbol}] Błąd importu w filter_unique_strategies. Czy scikit-learn jest zainstalowany?")
            logger.warning(f"[{symbol}] Zwracam niefiltrowane wyniki dla tego symbolu.")
            unique_symbol_results = symbol_results
        except Exception as filter_err:
             logger.error(f"[{symbol}] Błąd podczas filtrowania przez klastrowanie: {filter_err}", exc_info=True)
             logger.warning(f"[{symbol}] Zwracam niefiltrowane wyniki dla tego symbolu.")
             unique_symbol_results = symbol_results

        filtered_count_symbol = initial_count_symbol - len(unique_symbol_results)

        # --- Zbieranie statystyk dla symbolu ---
        analysis_stats['strategy_distribution'][symbol] = {
            'initial_count': initial_count_symbol, # Liczba przed klastrowaniem
            'final_count': len(unique_symbol_results),
            'filtered_count': filtered_count_symbol,
            'filtered_percentage': (filtered_count_symbol / initial_count_symbol * 100) if initial_count_symbol > 0 else 0.0
        }
        # Można by dodać info o klastrach, jeśli filter_unique_strategies by je zwracało

        # Dodawanie informacji analitycznych (rankingi, metoda filtrowania, rozmiar klastra)
        for final_rank, strategy in enumerate(unique_symbol_results):
            original_rank_in_symbol_list = -1
            try: original_rank_in_symbol_list = next(i for i, r in enumerate(symbol_results) if r['strategy_id'] == strategy['strategy_id'])
            except StopIteration: logger.debug(f"[{symbol}] Nie znaleziono oryg. rankingu dla {strategy.get('strategy_id')}")

            # Pobierz rozmiar klastra, jeśli istnieje, domyślnie 1
            cluster_size = strategy.get('cluster_size', 1)

            if 'analysis_info' not in strategy: strategy['analysis_info'] = {}
            strategy['analysis_info'].update({
                'ranking_info': {
                    'initial_symbol_rank': original_rank_in_symbol_list,
                    'final_symbol_rank': final_rank
                },
                'filter_method': 'clustering_pca_kmeans', # Oznacz metodę
                'cluster_size': cluster_size # Dodaj rozmiar klastra do informacji analitycznych
            })

        # Dodanie unikalnych wyników symbolu do globalnej listy
        final_unique_results_list.extend(unique_symbol_results)

        logger.info(
            f"[{symbol}] Zakończono analizę i filtrowanie symbolu: "
            f"Pozostało unikalnych: {len(unique_symbol_results):,} / {initial_count_symbol:,}. "
            f"({analysis_stats['strategy_distribution'][symbol]['filtered_percentage']:.1f}% odrzucono). "
            f"Czas dla symbolu: {time.time() - symbol_start_time:.2f}s"
        )

        # Czyszczenie pamięci
        logger.debug(f"[{symbol}] Czyszczenie pamięci po symbolu...")
        del symbol_results, unique_symbol_results; gc.collect()
        logger.debug(f"[{symbol}] Czyszczenie pamięci zakończone.")

    # --- Finalizacja po wszystkich symbolach ---
    logger.info("\nFinalizowanie analizy...")
    final_unique_results_list.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    logger.info(f"Posortowano {len(final_unique_results_list):,} unikalnych strategii globalnie.")

    # Obliczenie finalnej ważności parametrów
    logger.info("Obliczam finalną ważność parametrów dla unikalnych strategii...")
    final_importance_by_symbol = {}
    if final_unique_results_list:
        try:
             final_importance_by_symbol = calculate_parameter_importance(final_unique_results_list)
             for symbol, importance_dict in final_importance_by_symbol.items():
                  cache._importance_by_symbol[symbol] = importance_dict
                  cache._last_update[f"importance_{symbol}"] = time.time()
             logger.info("Zakończono obliczanie ważności parametrów.")
        except ImportError:
             logger.error("Nie można obliczyć ważności parametrów - brak scikit-learn?")
        except Exception as e:
             logger.error(f"Błąd podczas obliczania ważności parametrów: {e}", exc_info=True)

    # Podsumowanie globalnych statystyk
    analysis_stats['total_unique'] = len(final_unique_results_list)
    analysis_stats['filtered_cluster_count'] = initial_count_for_analysis - analysis_stats['total_unique']

    logger.info("\n" + " Podsumowanie Filtrowania przez Klastrowanie ".center(80, "="))
    logger.info(f"Liczba strategii na wejściu analizy: {initial_count_for_analysis:,}")
    logger.info(f"Liczba unikalnych strategii na wyjściu: {analysis_stats['total_unique']:,}")
    logger.info(f"Liczba strategii odrzuconych przez klastrowanie: {analysis_stats['filtered_cluster_count']:,}")
    if initial_count_for_analysis > 0:
        percentage_removed = (analysis_stats['filtered_cluster_count'] / initial_count_for_analysis) * 100
        logger.info(f"Procent strategii odrzuconych (wzgl. wejścia do tego kroku): {percentage_removed:.1f}%")

    # Logowanie statystyk per symbol
    for symbol in sorted(results_by_symbol.keys()):
        dist_stats = analysis_stats['strategy_distribution'].get(symbol)
        if dist_stats:
            logger.info(f"\nStatystyki dla {symbol}:")
            logger.info(f"  Wejście (do klastrowania): {dist_stats['initial_count']:,}")
            logger.info(f"  Wyjście (unikalne): {dist_stats['final_count']:,}")
            logger.info(f"  Odrzucono (klastrowanie): {dist_stats['filtered_count']:,} ({dist_stats['filtered_percentage']:.1f}%)")
    logger.info("=" * 80)

    # Zwracamy unikalne wyniki, finalne statystyki i ważność
    return final_unique_results_list, analysis_stats, final_importance_by_symbol


def save_analysis_results(
    final_unique_results: List[Dict],
    error_results: List[Dict],
    final_analysis_stats: Dict,
    source_info: str,
    final_importance_by_symbol: Dict,
    cache: AnalysisCache,
    base_output_filename: str
) -> None:
    """
    Zapisuje finalne wyniki analizy do plików txt i json. Używa cache.

    Args:
        final_unique_results: Lista unikalnych strategii.
        error_results: Lista wyników z błędami.
        final_analysis_stats: Słownik ze statystykami analizy.
        source_info: Informacja o pochodzeniu danych.
        final_importance_by_symbol: Słownik z ważnością parametrów.
        cache: Instancja AnalysisCache z danymi.
        base_output_filename: Podstawowa nazwa pliku wejściowego (bez rozszerzenia)
                               do użycia jako prefix plików wynikowych.
    """
    num_results_to_save = len(final_unique_results)
    num_errors = len(error_results)
    logger.info(f"Rozpoczynam zapisywanie {num_results_to_save:,} unikalnych wyników i {num_errors:,} błędów...")

    if cache is None:
        logger.error("Błąd krytyczny: save_analysis_results wymaga instancji AnalysisCache!")
        raise ValueError("AnalysisCache instance is required.")

    logger.debug(f"Cache przed zapisem zawiera dane dla {len(cache.symbols)} symboli.")

    # --- Zapis do pliku TXT ---
    logger.info("Generuję raport tekstowy (plik TXT)...")
    txt_file = None
    try:
        txt_file = save_analysis(
            # Poprawione nazwy argumentów zgodnie z definicją w reports.py
            ranked_results=final_unique_results,
            error_results=error_results,
            analysis_stats=final_analysis_stats,
            source_file=source_info, # Użyj 'source_file' jak w definicji funkcji
            importance_by_symbol=final_importance_by_symbol,
            cache=cache
        )
        if txt_file and txt_file.exists():
             # Zmiana nazwy pliku TXT
             txt_file_renamed = txt_file.parent / f"{base_output_filename}_{txt_file.name}"
             txt_file.rename(txt_file_renamed)
             logger.info(f"Zapisano analizę TXT: {txt_file_renamed}")
        elif txt_file:
             logger.warning(f"Nie udało się zapisać lub znaleźć pliku TXT: {txt_file}")
    except Exception as e_txt:
         logger.error(f"Nie udało się wygenerować lub przemianować raportu TXT: {e_txt}", exc_info=True)

    # --- Zapis do pliku JSON ---
    logger.info("Generuję raport JSON (plik JSON)...")
    json_file = None
    param_distributions_for_json = {}
    try:
        param_distributions_for_json = cache.param_distributions # Pobierz z cache
        json_file = generate_analysis_json(
            ranked_results=final_unique_results, # Ta nazwa jest OK
            error_results=error_results,
            analysis_stats=final_analysis_stats,
            param_distributions=param_distributions_for_json,
            # Poprawiona nazwa argumentu zgodnie z definicją w reports.py
            output_folder=JSON_ANALYSIS_DIR
        )
        if json_file and json_file.exists():
            # Zmiana nazwy pliku JSON
            json_file_renamed = json_file.parent / f"{base_output_filename}_{json_file.name}"
            json_file.rename(json_file_renamed)
            logger.info(f"Zapisano analizę JSON: {json_file_renamed}")
        elif json_file:
            logger.warning(f"Nie udało się zapisać lub znaleźć pliku JSON: {json_file}")
    except Exception as e_json:
         logger.error(f"Nie udało się wygenerować lub przemianować raportu JSON: {e_json}", exc_info=True)

    logger.info("Zakończono zapisywanie plików wynikowych.")
    # Zwracamy distributions, bo może być potrzebne do rekomendacji
    # return param_distributions_for_json # Ta funkcja nie musi nic zwracać


# --- Główna funkcja programu ---
def main():
    """Główna funkcja programu, orkiestrująca proces analizy."""
    main_start_time = time.time()
    logger.info("="*80)
    logger.info(f" Rozpoczęcie Strategy Analyzer ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ".center(80, "="))
    logger.info("="*80)

    try:
        # Konfiguracja parsera argumentów
        parser = argparse.ArgumentParser(
            description='Analizator wyników strategii tradingowych z backtestów.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument('--processes', type=int, default=None,
                            help='Liczba procesów (domyślnie: CPU count).')
        parser.add_argument('--pkl', type=str, default=None,
                            help='Opcjonalnie: Nazwa konkretnego pliku .pkl do analizy. Jeśli nie podano, analizuje wszystkie.')
        parser.add_argument('--skip-prefiltering', action='store_true',
                            help='Pomiń krok wstępnego filtrowania (score, trades, profit factor).')
        parser.add_argument('--min-score', type=float, default=5.0,
                            help='Próg min score dla pre-filtrowania.')
        parser.add_argument('--min-trades', type=int, default=5,
                            help='Próg min trades dla pre-filtrowania.')
        parser.add_argument('--min-profit-factor', type=float, default=1.0,
                            help='Próg min profit factor dla pre-filtrowania.')
        # Można dodać argumenty do kontroli klastrowania, np. --cluster-threshold
        args = parser.parse_args()

        num_processes = args.processes if args.processes and args.processes > 0 else mp.cpu_count()
        logger.info(f"Używana liczba procesów: {num_processes}")

        archive_dir = WYNIKI_BACKTEST_DIR / 'archiwum'
        archive_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Katalog archiwum: {archive_dir}")

        # Wyszukiwanie plików
        if args.pkl:
            target_file = WYNIKI_BACKTEST_DIR / args.pkl
            if not target_file.is_file():
                logger.error(f"Podany plik '{args.pkl}' nie istnieje lub nie jest plikiem w {WYNIKI_BACKTEST_DIR}.")
                # Wypisz dostępne pliki
                available = sorted(list(WYNIKI_BACKTEST_DIR.glob('*.pkl')))
                logger.info(f"Dostępne pliki .pkl ({len(available)}): {', '.join(f.name for f in available[:10])}{'...' if len(available)>10 else ''}")
                return
            files_to_analyze = [target_file]
            logger.info(f"Wybrano do analizy plik: {target_file.name}")
        else:
            files_to_analyze = sorted(list(WYNIKI_BACKTEST_DIR.glob('*.pkl')))
            if not files_to_analyze:
                 logger.warning(f"Brak plików .pkl do analizy w {WYNIKI_BACKTEST_DIR}.")
                 return
            logger.info(f"Znaleziono {len(files_to_analyze)} plików .pkl do analizy w {WYNIKI_BACKTEST_DIR}.")


        # --- Główna pętla przetwarzania plików ---
        logger.info("\n" + " Rozpoczynam przetwarzanie plików ".center(80, "#"))
        processed_files_count = 0
        failed_files_count = 0
        all_param_distributions_for_recommendations = {} # Zbieramy dla finalnych rekomendacji

        with tqdm(total=len(files_to_analyze), desc="Przetwarzanie plików", unit="plik") as pbar:
            for i, results_file in enumerate(files_to_analyze):
                analysis_cache = None # Reset cache dla pliku
                file_processed_successfully = False
                file_start_time = time.time()
                try:
                    pbar.set_description(f"Plik {i+1}/{len(files_to_analyze)}")
                    pbar.set_postfix({'plik': results_file.name[:30]+'...'})
                    logger.info(f"\n[{i+1}/{len(files_to_analyze)}] Przetwarzanie: {results_file.name}")

                    # === KROK 1: Wczytanie Danych ===
                    logger.info("  Krok 1: Wczytywanie danych...")
                    results_list, _ = load_results(results_file) # Ignorujemy market_data_info na razie
                    if results_list is None: # Obsługa błędu wczytania
                         logger.warning(f"  Pominięto plik {results_file.name} z powodu błędu wczytania.")
                         failed_files_count += 1
                         # Nie przenosimy do archiwum, jeśli nie udało się wczytać
                         pbar.update(1)
                         continue

                    # === KROK 2: Obliczenie Wstępnych Metryk ===
                    logger.info("  Krok 2: Obliczanie wstępnych metryk...")
                    step2_start_time = time.time()
                    results_with_metrics, error_results = calculate_initial_metrics(
                        results_list, num_processes
                    )
                    del results_list; gc.collect() # Usuń oryginalną listę
                    logger.info(f"  Krok 2 zakończony w {time.time() - step2_start_time:.2f}s")

                    if not results_with_metrics:
                        logger.warning(f"  Brak poprawnych strategii po obliczeniu metryk dla pliku {results_file.name}.")
                        failed_files_count += 1
                        archive_path = archive_dir / results_file.name
                        shutil.move(str(results_file), str(archive_path))
                        logger.info(f"  Przeniesiono plik {results_file.name} (brak wyników) do archiwum.")
                        pbar.update(1)
                        continue

                    original_count_before_prefilter = len(results_with_metrics)
                    logger.info(f"  Liczba strategii z metrykami: {original_count_before_prefilter:,}")
                    logger.info(f"  Liczba błędów przy obliczaniu metryk: {len(error_results):,}")

                    # === KROK 3: Pre-Filtering ===
                    pre_filtered_results = []
                    rejected_count = 0
                    if args.skip_prefiltering:
                        logger.info("  Krok 3: Pomijam wstępne filtrowanie (--skip-prefiltering).")
                        pre_filtered_results = results_with_metrics
                    else:
                        logger.info("  Krok 3: Wstępne filtrowanie strategii...")
                        # Definiujemy progi przed pętlą, pobierając je z argumentów
                        MIN_SCORE = args.min_score
                        MIN_TRADES = args.min_trades
                        MIN_PF = args.min_profit_factor
                        logger.info(f"    Kryteria: score>={MIN_SCORE}, trades>={MIN_TRADES}, pf>={MIN_PF}")
                        step3_start_time = time.time()

                        for res in results_with_metrics:
                            # Pobierz wartości z bezpiecznym domyślnym
                            metrics = res.get('metrics', {})
                            # Score powinien być float lub None jeśli błąd obliczeń metryk (co nie powinno się zdarzyć tutaj)
                            score_val = res.get('score')
                            # Trade count powinien być int lub None
                            trade_count_val = metrics.get('trade_count')
                            # Profit factor może być float, int, 'inf', 'nan' lub None
                            pf_val = metrics.get('profit_factor')

                            # --- Bezpieczna logika odrzucania ---
                            reject_reason = [] # Lista powodów odrzucenia

                            # Sprawdź score
                            try:
                                if not isinstance(score_val, (int, float)) or float(score_val) < MIN_SCORE:
                                     reject_reason.append(f"score ({score_val}) < {MIN_SCORE}")
                            except (ValueError, TypeError):
                                reject_reason.append(f"score invalid type ({type(score_val)})")

                            # Sprawdź trade_count
                            try:
                                if not isinstance(trade_count_val, int) or int(trade_count_val) < MIN_TRADES:
                                    reject_reason.append(f"trades ({trade_count_val}) < {MIN_TRADES}")
                            except (ValueError, TypeError):
                                reject_reason.append(f"trades invalid type ({type(trade_count_val)})")

                            # Sprawdź profit_factor (pf)
                            if pf_val is None:
                                reject_reason.append("pf is None")
                            elif isinstance(pf_val, str):
                                if pf_val == 'inf':
                                    pass # Infinite profit factor is OK, nie odrzucaj
                                elif pf_val == 'nan':
                                    reject_reason.append("pf is 'nan'")
                                else:
                                    # Nieoczekiwany string, odrzuć dla bezpieczeństwa
                                    reject_reason.append(f"pf unexpected str '{pf_val}'")
                            elif isinstance(pf_val, (int, float)):
                                # Teraz bezpiecznie porównujemy numerycznie
                                try:
                                    if float(pf_val) < MIN_PF:
                                        reject_reason.append(f"pf ({pf_val:.2f}) < {MIN_PF}")
                                except (ValueError, TypeError):
                                     reject_reason.append(f"pf invalid numeric value ({pf_val})")
                            else:
                                # Nieoczekiwany typ, odrzuć
                                reject_reason.append(f"pf unexpected type {type(pf_val)}")

                            # Decyzja o odrzuceniu
                            if reject_reason:
                                rejected_count += 1
                                # Opcjonalnie loguj powód odrzucenia dla debugowania (może spowolnić)
                                # if rejected_count % 100 == 0: # Loguj co setny
                                #    logger.debug(f"Rejecting {res.get('strategy_id', 'N/A')}: {'; '.join(reject_reason)}")
                            else:
                                pre_filtered_results.append(res)
                        # --- Koniec bezpiecznej logiki ---

                        logger.info(f"  Krok 3 zakończony w {time.time() - step3_start_time:.2f}s")
                        if original_count_before_prefilter > 0:
                             perc_rejected = rejected_count / original_count_before_prefilter * 100
                             logger.info(f"  Pre-Filtering: Odrzucono {rejected_count:,} strategii ({perc_rejected:.1f}%).")
                        else:
                            logger.info("  Pre-Filtering: Brak strategii do przetworzenia.")

                    logger.info(f"  Strategii pozostałych do głównej analizy: {len(pre_filtered_results):,}")
                    del results_with_metrics; gc.collect() # Usuń listę wejściową do pre-filtrowania

                    # === KROK 4: Główna Analiza (Klastrowanie, Cache, Importance) ===
                    logger.info("  Krok 4: Główna analiza (klastrowanie, cache, ważność)...")
                    step4_start_time = time.time()
                    analysis_cache = AnalysisCache() # Stwórz cache DLA TEGO PLIKU
                    # analyze_strategies zwraca: final_unique_results, final_analysis_stats, final_importance
                    final_unique_results, final_analysis_stats, final_importance = analyze_strategies(
                        pre_filtered_results, num_processes, analysis_cache
                    )
                    logger.info(f"  Krok 4 zakończony w {time.time() - step4_start_time:.2f}s")

                    # Dodanie globalnych statystyk do słownika (dla raportu)
                    final_analysis_stats['total_initial_results_in_file'] = original_count_before_prefilter
                    final_analysis_stats['prefiltered_rejected_count'] = rejected_count
                    # 'total_after_prefiltering' i 'total_unique' są już w final_analysis_stats

                    del pre_filtered_results; gc.collect() # Usuń listę wejściową do klastrowania

                    if not final_unique_results:
                         logger.warning(f"  Brak unikalnych wyników po filtrowaniu przez klastrowanie. Analiza dla pliku przerwana.")
                         failed_files_count += 1
                         archive_path = archive_dir / results_file.name
                         shutil.move(str(results_file), str(archive_path))
                         logger.info(f"  Przeniesiono plik {results_file.name} (brak wyników po klastrowaniu) do archiwum.")
                         pbar.update(1)
                         continue

                    logger.info(f"  Główna analiza zakończona. Liczba unikalnych strategii: {len(final_unique_results):,}")

                    # === KROK 5: Zapis Wyników (TXT i JSON) ===
                    logger.info("  Krok 5: Zapisywanie wyników analizy...")
                    step5_start_time = time.time()
                    source_info = f"Wyniki z pliku: {results_file.name} (Data analizy: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
                    base_name = results_file.stem
                    # Funkcja save_analysis_results zapisuje i zmienia nazwy plików
                    save_analysis_results(
                        final_unique_results, error_results, final_analysis_stats,
                        source_info, final_importance, analysis_cache, base_name
                    )
                    logger.info(f"  Krok 5 zakończony w {time.time() - step5_start_time:.2f}s")

                    # Zbieranie dystrybucji dla finalnych rekomendacji
                    if analysis_cache.param_distributions:
                         for symbol, dists in analysis_cache.param_distributions.items():
                              if symbol not in all_param_distributions_for_recommendations:
                                   all_param_distributions_for_recommendations[symbol] = {}
                              for param, values in dists.items():
                                   if param not in all_param_distributions_for_recommendations[symbol]:
                                        all_param_distributions_for_recommendations[symbol][param] = []
                                   # Zakładamy, że 'values' to już słownik statystyk, potrzebujemy listy z cache?
                                   # Musimy sprawdzić, co zwraca cache.param_distributions i co potrzebuje apply_recommendations
                                   # Na razie zakładamy, że potrzebujemy listy wartości - to wymagałoby zmiany w cache lub tutaj
                                   # --- MIEJSCE DO EWENTUALNEJ POPRAWKI ZBIERANIA DANYCH DLA REKOMENDACJI ---
                                   # logger.warning("Logika zbierania danych dla rekomendacji może wymagać dostosowania.")
                                   pass # Na razie pomijamy zbieranie


                    # === KROK 6: Archiwizacja ===
                    logger.info("  Krok 6: Archiwizacja pliku wejściowego...")
                    archive_path = archive_dir / results_file.name
                    shutil.move(str(results_file), str(archive_path))
                    logger.info(f"  Przeniesiono plik {results_file.name} do archiwum: {archive_path}")

                    file_processed_successfully = True
                    processed_files_count += 1

                except KeyboardInterrupt:
                     logger.warning("\nPrzerwano przez użytkownika (Ctrl+C).")
                     raise # Przerwij całe przetwarzanie
                except Exception as e:
                    logger.error(f"Krytyczny błąd podczas analizy pliku {results_file.name}: {e}", exc_info=True)
                    failed_files_count += 1
                    try: # Próba archiwizacji mimo błędu
                        archive_path = archive_dir / results_file.name
                        if results_file.exists():
                             shutil.move(str(results_file), str(archive_path))
                             logger.info(f"  Przeniesiono plik {results_file.name} (z błędem) do archiwum.")
                    except Exception as move_err:
                        logger.error(f"  Nie udało się przenieść pliku {results_file.name} do archiwum po błędzie: {move_err}")

                finally:
                    # Czyszczenie pamięci po pliku
                    file_end_time = time.time()
                    logger.debug(f"  Czyszczenie pamięci po pliku {results_file.name}...")
                    if 'final_unique_results' in locals(): del final_unique_results
                    if 'error_results' in locals(): del error_results
                    if 'final_analysis_stats' in locals(): del final_analysis_stats
                    if 'final_importance' in locals(): del final_importance
                    if analysis_cache: analysis_cache.clear(); del analysis_cache
                    gc.collect()
                    logger.debug("  Czyszczenie pamięci zakończone.")

                    pbar.update(1)
                    status_msg = 'OK' if file_processed_successfully else 'FAIL'
                    pbar.set_postfix({'plik': results_file.name[:30]+'...', 'status': status_msg})
                    logger.info(f"Zakończono przetwarzanie pliku: {results_file.name} (Status: {status_msg}, Czas: {file_end_time - file_start_time:.2f}s)")
                    logger.info("-" * 80)


        # --- Podsumowanie po wszystkich plikach ---
        logger.info("\n" + " Zakończono przetwarzanie wszystkich plików ".center(80, "#"))
        logger.info(f"Liczba pomyślnie przetworzonych plików: {processed_files_count}")
        logger.info(f"Liczba plików zakończonych błędem: {failed_files_count}")


        # --- Opcjonalna analiza rekomendacji ---
        if processed_files_count > 0:
             run_recommendation = input("\nCzy chcesz przeanalizować rekomendowane zmiany parametrów? (t/n): ").strip().lower()
             if run_recommendation == 't':
                 logger.info("\n" + " Rozpoczynam Analizę Rekomendacji ".center(80, "-"))
                 # --- UWAGA: Logika wczytywania JSON i zbierania all_param_distributions ---
                 # Musi być spójna z tym, co faktycznie zapisuje generate_analysis_json
                 # i czego oczekuje apply_recommendations. Poniżej przykład, może wymagać adaptacji.
                 all_param_distributions = {}
                 json_pattern = "*_analysis_results_*.json"
                 json_files_found = sorted(list(JSON_ANALYSIS_DIR.glob(json_pattern)))
                 logger.info(f"Znaleziono {len(json_files_found)} plików JSON w {JSON_ANALYSIS_DIR} do analizy.")

                 if not json_files_found: logger.warning("Brak plików JSON do wczytania.")
                 else:
                     for json_file in tqdm(json_files_found, desc="Wczytywanie dystr. parametrów", unit="plik"):
                         try:
                             with open(json_file, 'r') as f: data = json.load(f)
                             # Załóżmy, że JSON zawiera klucz 'param_distributions' na top levelu
                             p_dist_data = data.get('param_distributions')
                             if isinstance(p_dist_data, dict):
                                 for symbol, dist_dict in p_dist_data.items():
                                     if symbol not in all_param_distributions: all_param_distributions[symbol] = {}
                                     if isinstance(dist_dict, dict):
                                         for param, param_data in dist_dict.items():
                                             # Sprawdź, jakiego formatu oczekuje apply_recommendations
                                             # Jeśli oczekuje listy wartości, a mamy słownik statystyk:
                                             if isinstance(param_data, dict) and 'mean' in param_data: # Przykładowe sprawdzenie
                                                  # Tu trzeba by odtworzyć listę lub dostosować apply_recommendations
                                                  # logger.warning(f"Format danych dla rekomendacji ({symbol}/{param}) wymaga weryfikacji.")
                                                  pass # Na razie pomijamy
                                             # Jeśli oczekuje słownika statystyk:
                                             elif isinstance(param_data, dict):
                                                  if param not in all_param_distributions[symbol]:
                                                      all_param_distributions[symbol][param] = [] # Lista słowników?
                                                  all_param_distributions[symbol][param].append(param_data)
                                             # Jeśli oczekuje listy (jak w starym kodzie)
                                             elif isinstance(param_data, list):
                                                 if param not in all_param_distributions[symbol]: all_param_distributions[symbol][param] = []
                                                 all_param_distributions[symbol][param].extend(param_data)


                         except Exception as e: logger.error(f"Błąd wczytywania {json_file}: {e}", exc_info=False)

                     if all_param_distributions:
                         logger.info("Uruchamiam funkcję apply_recommendations...")
                         apply_recommendations(all_param_distributions) # Przekaż zebrane dane
                         logger.info("Zakończono analizę rekomendacji.")
                     else: logger.warning("Nie załadowano żadnych danych dystrybucji parametrów.")
             else: logger.info("Pominięto analizę rekomendacji.")
        else: logger.info("Brak pomyślnie przetworzonych plików, pomijam analizę rekomendacji.")

    except KeyboardInterrupt:
        logger.warning("\nPrzerwano działanie skryptu przez użytkownika (Ctrl+C).")
    except Exception as e:
        logger.critical(f"Nieoczekiwany błąd krytyczny w funkcji main: {e}", exc_info=True)
    finally:
        total_time = time.time() - main_start_time
        logger.info("\n" + f" Zakończono działanie skryptu strategy_analyzer (Całkowity czas: {total_time:.2f}s) ".center(80, "#"))


if __name__ == "__main__":
    main()