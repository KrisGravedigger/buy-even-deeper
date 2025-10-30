#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moduł analizujący parametry strategii tradingowych, w tym filtrowanie
unikalnych strategii za pomocą klastrowania (PCA/KMeans).
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import time
from utils.logging_setup import get_strategy_logger

logger = get_strategy_logger('strategy_parameters')

# --- Warunkowy import Sklearn ---
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import pairwise_distances
    SKLEARN_AVAILABLE = True
    logger.info("Biblioteka scikit-learn została pomyślnie zaimportowana.")
except ImportError:
    SKLEARN_AVAILABLE = False
    StandardScaler = KMeans = PCA = RandomForestRegressor = pairwise_distances = object # Definicje zastępcze
    logger.warning("Biblioteka scikit-learn nie jest zainstalowana. Funkcje wymagające sklearn będą niedostępne lub ograniczone.")

# ================================================================
# === Funkcja Pomocnicza do Bezpiecznej Konwersji na Float ===
# ================================================================

def _try_convert_float_and_log(value: Any, param_name: str, strategy_id: Any) -> Optional[float]:
    """
    Próbuje bezpiecznie przekonwertować wartość na float.
    Obsługuje bool, int, float, stringi liczbowe. Loguje ostrzeżenie w razie błędu.
    Zwraca float lub None.
    """
    if isinstance(value, bool): return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        f_val = float(value)
        return f_val if np.isfinite(f_val) else None # Zwróć None dla NaN/Inf
    if isinstance(value, str):
        try:
            cleaned = value.strip().replace('%', '')
            if not cleaned: return None
            f_val = float(cleaned)
            return f_val if np.isfinite(f_val) else None
        except ValueError: return None # Nie można przekonwertować stringa
    return None # Inne typy (None, listy, etc.) nie są konwertowane

# ================================================================
# === Analiza Dystrybucji Parametrów ===
# ================================================================

def analyze_parameter_distributions(ranked_results: List[Dict], n_top: int = 100) -> Dict:
    """
    Analizuje rozkłady parametrów w najlepszych `n_top` strategiach dla każdego symbolu.
    """
    if not ranked_results: return {}
    top_strategies = ranked_results[:min(n_top, len(ranked_results))]
    if not top_strategies: return {}

    params_analysis = {}
    results_by_symbol = {}
    all_params_by_symbol = {}

    for r in top_strategies:
        symbol = r.get('symbol', 'BTC/USDT')
        if symbol not in results_by_symbol: results_by_symbol[symbol] = []; all_params_by_symbol[symbol] = set()
        results_by_symbol[symbol].append(r)
        if isinstance(r.get('parameters'), dict): all_params_by_symbol[symbol].update(r['parameters'].keys())

    logger.info(f"Analizuję dystrybucje parametrów dla {len(results_by_symbol)} symboli (top {len(top_strategies)} strategii).")

    for symbol, symbol_strategies in results_by_symbol.items():
        params_analysis[symbol] = {}
        param_names = sorted(list(all_params_by_symbol[symbol]))
        if not param_names: continue
        logger.debug(f"[{symbol}] Analizowane parametry dystrybucji: {len(param_names)}")

        param_data_for_df = []
        valid_indices = [] # Indeksy w symbol_strategies, które miały przynajmniej 1 poprawny parametr

        for idx, r in enumerate(symbol_strategies):
            strategy_id = r.get('strategy_id', f'dist_idx_{idx}')
            params = r.get('parameters', {})
            processed_params = {}
            valid_param_found = False
            for param_name in param_names:
                 raw_value = params.get(param_name)
                 converted_value = _try_convert_float_and_log(raw_value, param_name, strategy_id)
                 processed_params[param_name] = converted_value # Może być None
                 if converted_value is not None: valid_param_found = True

            if valid_param_found:
                 param_data_for_df.append(processed_params)
                 valid_indices.append(idx) # Not currently used, but could be

        if not param_data_for_df:
             logger.warning(f"[{symbol}] Brak poprawnych danych parametrów do analizy dystrybucji.")
             continue

        # Użycie Pandas do obliczeń statystycznych
        try:
            df = pd.DataFrame(param_data_for_df, columns=param_names) # Utwórz z pełną listą kolumn
        except Exception as e_df:
            logger.error(f"[{symbol}] Błąd tworzenia DataFrame dla dystrybucji: {e_df}")
            continue

        for param_name in param_names:
            # Analizuj tylko jeśli kolumna istnieje w df (na wypadek błędów)
            if param_name not in df.columns: continue

            valid_values = df[param_name].dropna() # Usuń None (które stały się NaN)
            count = len(valid_values)

            if count == 0:
                stats = {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan,
                         'max': np.nan, 'range': np.nan, 'relative_std': np.nan, 'count': 0}
            else:
                values_np = valid_values.to_numpy()
                mean_val = np.mean(values_np); median_val = np.median(values_np)
                std_val = np.std(values_np); min_val = np.min(values_np); max_val = np.max(values_np)
                range_val = max_val - min_val
                rel_std = (std_val / abs(mean_val)) if abs(mean_val) > 1e-9 else 0.0
                stats = {'mean': float(mean_val), 'median': float(median_val), 'std': float(std_val),
                         'min': float(min_val), 'max': float(max_val), 'range': float(range_val),
                         'relative_std': float(rel_std), 'count': count}

                # Wykrywanie skupisk (KMeans)
                if SKLEARN_AVAILABLE and count >= 5:
                    try:
                        values_2d = values_np.reshape(-1, 1); n_unique = len(np.unique(values_np))
                        n_clusters = min(3, n_unique) if n_unique > 1 else 1
                        if n_clusters > 1:
                             kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(values_2d)
                             clusters_info = []
                             for i in range(n_clusters):
                                 cluster_vals = values_np[kmeans.labels_ == i]
                                 if len(cluster_vals) > 0:
                                     clusters_info.append({'center': float(kmeans.cluster_centers_[i][0]),
                                                           'size': len(cluster_vals), 'std': float(np.std(cluster_vals))})
                             stats['clusters'] = sorted(clusters_info, key=lambda x: x['size'], reverse=True)
                    except Exception as kmeans_err:
                         logger.warning(f"[{symbol}/{param_name}] Błąd klastrowania dystrybucji: {kmeans_err}", exc_info=False)

            params_analysis[symbol][param_name] = stats

    return params_analysis


# ================================================================
# === Obliczanie Ważności Parametrów ===
# ================================================================

def calculate_parameter_importance(results: List[Dict], max_samples_for_importance: int = 10000) -> Dict:
    """
    Oblicza ważność parametrów używając RandomForestRegressor na próbce danych.
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("calculate_parameter_importance: Pomijam obliczenia - brak scikit-learn.")
        return {}
    if not results:
        logger.warning("calculate_parameter_importance: Pusta lista wyników.")
        return {}

    importance_by_symbol = {}
    results_by_symbol = {}

    # Podział wyników wg symboli
    for r in results:
        if not isinstance(r, dict) or not isinstance(r.get('parameters'), dict) or 'score' not in r: continue
        symbol = r.get('symbol', 'BTC/USDT')
        if symbol not in results_by_symbol: results_by_symbol[symbol] = []
        results_by_symbol[symbol].append(r)

    logger.info(f"Obliczam ważność parametrów dla {len(results_by_symbol)} symboli.")

    for symbol, symbol_results_full in results_by_symbol.items():
        n_results = len(symbol_results_full)
        if n_results < 5: # Potrzebujemy kilku próbek do sensownego modelu
            logger.warning(f"[{symbol}] Za mało wyników ({n_results}) do obliczenia ważności.")
            importance_by_symbol[symbol] = {}
            continue

        # --- Próbkowanie ---
        if n_results > max_samples_for_importance:
            logger.info(f"[{symbol}] Używam próbki {max_samples_for_importance:,} najlepszych strategii (z {n_results:,}) do obliczenia ważności.")
            # Zakładamy, że results są już posortowane wg score
            symbol_results_sampled = symbol_results_full[:max_samples_for_importance]
        else:
            # logger.info(f"[{symbol}] Używam wszystkich {n_results:,} strategii do obliczenia ważności.")
            symbol_results_sampled = symbol_results_full

        # --- Przygotowanie danych X, y ---
        potential_param_names = set().union(*(r['parameters'].keys() for r in symbol_results_sampled if isinstance(r.get('parameters'), dict)))
        param_names_list = sorted(list(potential_param_names))
        if not param_names_list: logger.warning(f"[{symbol}] Brak parametrów do analizy ważności."); importance_by_symbol[symbol] = {}; continue

        X_rows, y, valid_rows_count = [], [], 0
        for idx, r in enumerate(symbol_results_sampled):
            strategy_id = r.get('strategy_id', f'imp_idx_{idx}')
            score_float = _try_convert_float_and_log(r.get('score'), 'score', strategy_id)
            if score_float is None: continue # Pomiń, jeśli score nieprawidłowy

            params = r.get('parameters', {})
            features = []
            valid_row = True
            for name in param_names_list:
                converted_value = _try_convert_float_and_log(params.get(name), name, strategy_id)
                if converted_value is None: features.append(0.0) # Użyj 0 dla brakujących/niepoprawnych
                else: features.append(converted_value)

            X_rows.append(features); y.append(score_float); valid_rows_count += 1

        if valid_rows_count < 5: logger.warning(f"[{symbol}] Za mało poprawnych danych ({valid_rows_count}) do modelu ważności."); importance_by_symbol[symbol] = {}; continue

        X = np.array(X_rows); y = np.array(y)

        # --- Model RandomForest ---
        try:
            variances = np.var(X, axis=0)
            non_constant_indices = np.where(variances > 1e-9)[0]
            if len(non_constant_indices) == 0: logger.warning(f"[{symbol}] Wszystkie parametry stałe."); importance_by_symbol[symbol] = {name: 0.0 for name in param_names_list}; continue

            X_variable = X[:, non_constant_indices]
            param_names_variable = [param_names_list[i] for i in non_constant_indices]

            scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_variable)
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
            model.fit(X_scaled, y)

            importance_raw = model.feature_importances_
            importance_dict = {name: 0.0 for name in param_names_list}
            for i, idx_in_original in enumerate(non_constant_indices):
                importance_dict[param_names_list[idx_in_original]] = importance_raw[i]

            importance_by_symbol[symbol] = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
            logger.info(f"[{symbol}] Obliczono ważność dla {len(param_names_list)} parametrów.")
        except Exception as model_err:
             logger.error(f"[{symbol}] Błąd modelu ważności: {model_err}", exc_info=False)
             importance_by_symbol[symbol] = {}

    return importance_by_symbol


# =====================================================================
# === Filtrowanie Unikalnych Strategii przez Klastrowanie (PCA/KMeans) ===
# =====================================================================

def filter_unique_strategies(results: List[Dict],
                           similarity_threshold: float = 0.90, # Zmieniono próg dla klastrowania
                           min_strategies: int = 10,
                           pca_variance_threshold: float = 0.95,
                           param_weight: float = 0.7,
                           metric_weight: float = 0.3) -> List[Dict]:
    """
    Filtruje unikalne strategie używając PCA i K-Means. Zwraca najlepszą strategię z każdego klastra.
    """
    num_results = len(results)
    if num_results <= min_strategies:
        logger.debug(f"filter_unique_strategies: Liczba strategii ({num_results}) <= minimum ({min_strategies}). Pomijam.")
        return results
    if not SKLEARN_AVAILABLE:
         logger.warning("filter_unique_strategies: Pomijam filtrowanie - brak scikit-learn.")
         return results

    logger.info(f"Filtruję {num_results:,} strategii przez klastrowanie (PCA/KMeans)...")
    start_time = time.time()

    # --- Przygotowanie danych ---
    param_data_rows, metric_data_rows, valid_results_indices, original_indices_map = [], [], [], {}
    potential_params = set().union(*(r['parameters'].keys() for r in results[:200] if isinstance(r.get('parameters'), dict)))
    potential_metrics = set().union(*(r['metrics'].keys() for r in results[:200] if isinstance(r.get('metrics'), dict)))
    param_names_list = sorted(list(potential_params))
    metric_names_list = sorted([m for m in potential_metrics if m != 'score']) # Wyklucz score z metryk do PCA

    if not param_names_list and not metric_names_list: logger.warning("Brak parametrów/metryk do klastrowania."); return results
    logger.debug(f"Klastrowanie - Params ({len(param_names_list)}): {param_names_list[:5]}...")
    logger.debug(f"Klastrowanie - Metrics ({len(metric_names_list)}): {metric_names_list[:5]}...")

    for original_idx, result in enumerate(results):
        strategy_id = result.get('strategy_id', f'filter_idx_{original_idx}')
        if not isinstance(result, dict) or not isinstance(result.get('parameters'), dict) \
           or not isinstance(result.get('metrics'), dict) \
           or _try_convert_float_and_log(result.get('score'), 'score', strategy_id) is None: continue

        param_vector, metric_vector = [], []
        valid_param, valid_metric = False, False
        params = result.get('parameters', {})
        metrics = result.get('metrics', {})

        for name in param_names_list:
            val = _try_convert_float_and_log(params.get(name), name, strategy_id)
            param_vector.append(val if val is not None else 0.0); valid_param |= (val is not None)
        for name in metric_names_list:
             val = _try_convert_float_and_log(metrics.get(name), name, strategy_id)
             metric_vector.append(val if val is not None else 0.0); valid_metric |= (val is not None)

        if valid_param or valid_metric:
             new_idx = len(param_data_rows)
             param_data_rows.append(param_vector); metric_data_rows.append(metric_vector)
             valid_results_indices.append(original_idx); original_indices_map[new_idx] = original_idx

    num_valid = len(param_data_rows)
    if num_valid <= min_strategies: logger.warning(f"Po walidacji zostało {num_valid} strategii <= min. Zwracam je."); return [results[i] for i in valid_results_indices]
    logger.info(f"Przygotowano dane dla {num_valid} strategii do klastrowania.")
    param_data = np.array(param_data_rows); metric_data = np.array(metric_data_rows)

    # --- Normalizacja, PCA, KMeans ---
    try:
        if param_data.shape[1] > 0 and num_valid > 1: param_norm = StandardScaler().fit_transform(param_data)
        else: param_norm = param_data
        if metric_data.shape[1] > 0 and num_valid > 1: metric_norm = StandardScaler().fit_transform(metric_data)
        else: metric_norm = metric_data

        if param_norm.shape[1] > 1:
            n_comp_p = min(int(param_norm.shape[1]), 50); param_pca = PCA(n_components=min(pca_variance_threshold, n_comp_p)); param_reduced = param_pca.fit_transform(param_norm)
            logger.debug(f"PCA params: {param_pca.n_components_} comp. (var: {param_pca.explained_variance_ratio_.sum():.3f})")
        else: param_reduced = param_norm
        if metric_norm.shape[1] > 1:
            n_comp_m = min(int(metric_norm.shape[1]), 20); metric_pca = PCA(n_components=min(pca_variance_threshold, n_comp_m)); metric_reduced = metric_pca.fit_transform(metric_norm)
            logger.debug(f"PCA metrics: {metric_pca.n_components_} comp. (var: {metric_pca.explained_variance_ratio_.sum():.3f})")
        else: metric_reduced = metric_norm

        combined_parts = []
        if param_reduced.shape[1] > 0: combined_parts.append(param_reduced * param_weight)
        if metric_reduced.shape[1] > 0: combined_parts.append(metric_reduced * metric_weight)
        if not combined_parts: logger.error("Brak danych po PCA."); return [results[i] for i in valid_results_indices]
        combined_data = np.hstack(combined_parts)
        logger.debug(f"Combined data shape for KMeans: {combined_data.shape}")

        target_clusters = max(min_strategies, int(num_valid * (1.0 - similarity_threshold)))
        n_clusters = min(num_valid, target_clusters); n_clusters = max(1, n_clusters)
        logger.info(f"Uruchamiam KMeans (n_clusters = {n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(combined_data)
        logger.info("KMeans zakończone.")

        # Wybór najlepszej strategii z każdego klastra i zapisanie rozmiaru klastra
        unique_strategies_orig_indices = []
        processed_clusters = set()
        # Używamy słownika do przechowania indeksu najlepszej strategii i rozmiaru dla każdego klastra
        best_strategy_per_cluster = {} # { cluster_label: (best_original_idx, cluster_size) }

        # Najpierw znajdźmy najlepszą strategię i rozmiar dla każdego unikalnego klastra
        unique_labels = set(cluster_labels)
        for label in unique_labels:
            indices_in_cluster = [
                new_idx for new_idx, cl_label in enumerate(cluster_labels) if cl_label == label
            ]
            if not indices_in_cluster:
                continue

            cluster_size = len(indices_in_cluster)
            best_new_idx_in_cluster = -1
            best_score_in_cluster = -np.inf

            for new_idx in indices_in_cluster:
                original_idx = original_indices_map[new_idx]
                current_score = results[original_idx].get('score', -np.inf)
                if current_score > best_score_in_cluster:
                    best_score_in_cluster = current_score
                    best_new_idx_in_cluster = new_idx

            if best_new_idx_in_cluster != -1:
                best_original_idx = original_indices_map[best_new_idx_in_cluster]
                best_strategy_per_cluster[label] = (best_original_idx, cluster_size)

        # Teraz zbierz unikalne strategie i dodaj do nich informację o rozmiarze klastra
        final_unique_strategies_list_with_info = []
        seen_original_indices = set()
        for label, (best_original_idx, cluster_size) in best_strategy_per_cluster.items():
            if best_original_idx not in seen_original_indices:
                strategy_copy = results[best_original_idx].copy() # Pracuj na kopii
                strategy_copy['cluster_size'] = cluster_size # Dodaj rozmiar klastra
                final_unique_strategies_list_with_info.append(strategy_copy)
                seen_original_indices.add(best_original_idx)
            else:
                logger.warning(f"Strategia z index {best_original_idx} jest najlepsza w więcej niż jednym klastrze? To nie powinno się zdarzyć.")


        # Zwracamy listę unikalnych strategii (słowników), teraz z potencjalnym kluczem 'cluster_size'
        # Sortowanie przenosimy na koniec, po dodaniu informacji
        final_unique_strategies = sorted(
            final_unique_strategies_list_with_info,
            key=lambda x: x.get('score', 0.0),
            reverse=True
        )
        # final_unique_strategies = [results[i] for i in unique_strategies_orig_indices] # Stara linia
        # final_unique_strategies.sort(key=lambda x: x.get('score', 0.0), reverse=True) # Stara linia

        filtering_time = time.time() - start_time
        logger.info(f"Filtrowanie przez klastrowanie zakończone w {filtering_time:.2f}s. Wybrano {len(final_unique_strategies)} unikalnych strategii (reprezentantów klastrów).")
        return final_unique_strategies # Zwraca listę unikalnych strategii z dodanym 'cluster_size'

        final_unique_strategies = [results[i] for i in unique_strategies_orig_indices]
        final_unique_strategies.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        filtering_time = time.time() - start_time
        logger.info(f"Filtrowanie przez klastrowanie zakończone w {filtering_time:.2f}s. Wybrano {len(final_unique_strategies)} strategii.")
        return final_unique_strategies

    except Exception as e:
        logger.error(f"Błąd filtrowania (PCA/KMeans): {e}. Zwracam poprawne, niefiltrowane wyniki.", exc_info=True)
        return [results[i] for i in valid_results_indices]


# Funkcja select_diverse_strategies pozostaje bez zmian (z bezpieczną konwersją i warunkowym sklearn)
# Jest ona teraz nieużywana w filter_unique_strategies, ale zostawiamy ją.
def select_diverse_strategies(cluster_results: List[Dict],
                            param_data: np.ndarray, # Oczekuje ZNORMALIZOWANYCH danych
                            metric_data: np.ndarray, # Oczekuje ZNORMALIZOWANYCH danych
                            max_strategies: int = 2) -> List[Dict]:
    """
    Wybiera zróżnicowane strategie z klastra używając analizy odległości.
    (Funkcja pomocnicza - obecnie nieużywana w filter_unique_strategies).
    """
    num_available = len(cluster_results)
    if num_available <= max_strategies: return cluster_results
    if not SKLEARN_AVAILABLE:
         logger.warning("select_diverse_strategies: Pomijam - brak scikit-learn. Zwracam najlepsze wg score.")
         return sorted(cluster_results, key=lambda x: x.get('score', 0.0), reverse=True)[:max_strategies]
    try:
        data_parts = []
        if param_data is not None and param_data.shape[1] > 0: data_parts.append(param_data * 0.7)
        if metric_data is not None and metric_data.shape[1] > 0: data_parts.append(metric_data * 0.3)
        if not data_parts:
            logger.warning("select_diverse_strategies: Brak danych. Zwracam najlepsze wg score.")
            return sorted(cluster_results, key=lambda x: x.get('score', 0.0), reverse=True)[:max_strategies]
        combined_data = np.hstack(data_parts)
        distances = pairwise_distances(combined_data, metric='euclidean')
        np.fill_diagonal(distances, np.inf)
        best_idx_in_cluster = max(range(num_available), key=lambda i: cluster_results[i].get('score', 0.0))
        selected_indices_in_cluster = [best_idx_in_cluster]
        min_distances_to_selected = distances[best_idx_in_cluster].copy()
        while len(selected_indices_in_cluster) < max_strategies:
             farthest_idx = np.argmax(min_distances_to_selected)
             if farthest_idx not in selected_indices_in_cluster and np.isfinite(min_distances_to_selected[farthest_idx]):
                 selected_indices_in_cluster.append(farthest_idx)
                 min_distances_to_selected = np.minimum(min_distances_to_selected, distances[farthest_idx])
                 min_distances_to_selected[selected_indices_in_cluster] = -1.0
             else:
                 remaining_indices = np.argsort(min_distances_to_selected)[::-1]
                 found_new = False
                 for cand_idx in remaining_indices:
                     if cand_idx not in selected_indices_in_cluster and np.isfinite(min_distances_to_selected[cand_idx]):
                         selected_indices_in_cluster.append(cand_idx)
                         min_distances_to_selected = np.minimum(min_distances_to_selected, distances[cand_idx])
                         min_distances_to_selected[selected_indices_in_cluster] = -1.0; found_new = True; break
                 if not found_new: logger.warning("select_diverse_strategies: Nie można znaleźć więcej punktów."); break
        return [cluster_results[i] for i in selected_indices_in_cluster]
    except Exception as e:
        logger.error(f"Błąd w select_diverse_strategies: {e}. Zwracam najlepsze wg score.", exc_info=False)
        return sorted(cluster_results, key=lambda x: x.get('score', 0.0), reverse=True)[:max_strategies]


# Funkcja format_parameter_value pozostaje bez zmian
def format_parameter_value(param_name: str, value: Any) -> str:
    """
    Formatuje wartość parametru zgodnie z jego typem i charakterystyką.
    """
    # --- POCZĄTEK KODU Z POPRZEDNIEJ WERSJI ---
    if isinstance(value, bool): return str(value).lower()
    if value is None: return "None"
    if not isinstance(value, (int, float, str)): return str(value)
    try:
        if isinstance(value, str):
             try: value_float = float(value.replace('%','').strip())
             except ValueError: return str(value)
        else: value_float = float(value)
    except (TypeError, ValueError): return str(value)
    param_name_lower = param_name.lower()
    if 'enabled' in param_name_lower or 'active' in param_name_lower or param_name_lower.startswith('use_') or isinstance(value, bool):
        if abs(value_float - 1.0) < 1e-9: return "true"
        if abs(value_float - 0.0) < 1e-9: return "false"
    is_integer_like = abs(value_float - round(value_float)) < 1e-9
    if any(s in param_name_lower for s in ['time', 'period', 'candle', 'bar', 'delay', 'count', 'orders', 'size', 'lots', 'contracts', 'leverage', 'precision', 'limit', 'attempts']):
        if is_integer_like: return str(int(round(value_float)))
    if any(s in param_name_lower for s in ['%', 'threshold', 'profit', 'loss', 'margin', 'rate', 'ratio', 'factor', 'coeff', 'deviation', 'spread', 'slippage', 'fee']):
        if abs(value_float) < 1e-9: return "0.0%"
        if abs(value_float) < 0.0001: return f"{value_float:.4f}%"
        if abs(value_float) < 0.01: return f"{value_float:.3f}%"
        if abs(value_float) < 1: return f"{value_float:.2f}%"
        if abs(value_float) < 10: return f"{value_float:.1f}%"
        return f"{value_float:.0f}%" if is_integer_like else f"{value_float:.1f}%"
    if any(s in param_name_lower for s in ['price', 'cost', 'value', 'amount', 'level', 'target', 'stop', 'pip', 'point']):
        if abs(value_float) < 1e-9: return "0.00"
        if abs(value_float) < 0.0001: return f"{value_float:.6f}"
        if abs(value_float) < 0.01: return f"{value_float:.5f}"
        if abs(value_float) < 1: return f"{value_float:.4f}"
        if abs(value_float) < 100: return f"{value_float:.2f}"
        if abs(value_float) < 10000: return f"{value_float:.1f}"
        return f"{value_float:.0f}" if is_integer_like else f"{value_float:.1f}"
    if is_integer_like: return str(int(round(value_float)))
    if abs(value_float) < 1e-9: return "0.00"
    if abs(value_float) < 0.01: return f"{value_float:.4f}"
    if abs(value_float) < 1: return f"{value_float:.3f}"
    if abs(value_float) < 100: return f"{value_float:.2f}"
    return f"{value_float:.1f}"