#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moduł obsługujący obliczanie i analizę metryk strategii tradingowych.
"""

from typing import Dict, Any, List # Dodano List
import numpy as np
import pandas as pd # Dodano import pandas dla pd.notna
import traceback # Dodano traceback
from utils.logging_setup import get_strategy_logger
# Używamy kropki dla importu względnego w pakiecie 'analysis'
# POPRAWIONY IMPORT: Zmieniono z .metrics_calculator na analysis.metrics_calculator
from analysis.metrics_calculator import StrategyMetrics
import gc # Dodano import gc na wszelki wypadek, chociaz nie jest tu uzywany bezposrednio

logger = get_strategy_logger('strategy_metrics')

def format_parameter_value(param_name: str, value: Any) -> str:
    """
    Formatuje wartość parametru do wyświetlenia. (Bez zmian)
    """
    try:
        if isinstance(value, bool): return str(value)
        elif isinstance(value, (int, float)):
            if param_name in ['max_open_orders', 'max_open_orders_per_coin']: return str(int(value))
            # Poprawka: Użyj round zamiast f-string dla precyzji
            elif abs(value) >= 0.01: return str(round(value, 2))
            else: return str(round(value, 3))
        return str(value)
    except Exception as e:
        logger.error(f"Błąd formatowania parametru {param_name}: {e}")
        return str(value)

def calculate_strategy_score(metrics: Dict) -> float:
    """
    Oblicza znormalizowany wynik strategii. (Bez zmian)
    """
    # --- POCZĄTEK KODU Z POPRZEDNIEJ WERSJI ---
    if not isinstance(metrics, dict) or metrics.get('trade_count', 0) < 5:
         return 0.0 # Zwróć 0.0 jeśli metryki są niepoprawne lub za mało transakcji

    # Inicjalizuj wartości domyślne na 0.0, jeśli metryka nie istnieje
    expectancy = metrics.get('expectancy', 0.0)
    profit_per_trade = metrics.get('profit_per_trade', 0.0)
    win_rate = metrics.get('win_rate', 0.0)
    sharpe = metrics.get('sharpe_ratio', 0.0)
    sortino = metrics.get('sortino_ratio', 0.0)
    max_dd = metrics.get('max_drawdown', 100.0) # Domyślnie wysoki drawdown
    sl_blocks = metrics.get('stop_loss_blocks', 0) # To nie jest metryka z StrategyMetrics? Może nie istnieć.
    sl_eff = metrics.get('stop_loss_block_effectiveness', 0.0) # To nie jest metryka z StrategyMetrics?
    btc_blocks = metrics.get('btc_blocks', 0)
    btc_eff = metrics.get('btc_block_effectiveness', 0.0)
    btc_corr = abs(metrics.get('btc_correlation', 0.0))
    btc_rate = metrics.get('btc_block_rate', 0.0)
    trade_count = metrics.get('trade_count', 0)
    max_loss = metrics.get('max_loss', 0.0) # Max loss jest ujemny lub 0
    profit_factor = metrics.get('profit_factor', 0.0)
    risk_reward = metrics.get('risk_reward_ratio', 0.0)

    # Sprawdź typy przed operacjami arytmetycznymi
    if not all(isinstance(v, (int, float)) for v in [expectancy, profit_per_trade, win_rate, sharpe, sortino]):
        logger.warning(f"Niektóre podstawowe metryki nie są liczbami: {metrics}")
        return 0.0

    # Bezpieczniejsze obliczenia - unikaj błędów przy brakujących kluczach
    base_score = (expectancy * 15 + profit_per_trade * 10 + win_rate * 15 + sharpe * 5 + sortino * 5)
    # Normalizacja - upewnij się, że dzielnik nie jest 0
    base_score = (base_score / 50 * 50) if 50 != 0 else 0.0 # Normalizacja do 50 punktów

    risk_score = 0
    if isinstance(max_dd, (int, float)):
        if max_dd < 5: risk_score += 10
        elif max_dd < 10: risk_score += 5
    # Te metryki mogą nie istnieć, sprawdźmy
    if isinstance(sl_blocks, (int, float)) and sl_blocks > 0:
        if isinstance(sl_eff, (int, float)):
            if sl_eff > 70: risk_score += 10
            elif sl_eff > 50: risk_score += 5

    btc_score = 0
    if isinstance(btc_blocks, (int, float)) and btc_blocks > 0:
        if isinstance(btc_eff, (int, float)):
            if btc_eff > 70: btc_score += 10
            elif btc_eff > 50: btc_score += 5
        if isinstance(btc_corr, (int, float)):
            if btc_corr < 0.3: btc_score += 10
            elif btc_corr < 0.5: btc_score += 5
        if isinstance(btc_rate, (int, float)) and btc_rate > 50:
             btc_score *= max(0, (1 - (btc_rate - 50)/100)) # Kara, ale nie ujemny score

    total_score = base_score + risk_score + btc_score
    penalties = 1.0
    if isinstance(trade_count, (int, float)) and trade_count < 20: penalties *= 0.8
    if isinstance(max_loss, (int, float)) and max_loss < -10: penalties *= 0.7 # Max loss jest ujemny
    if isinstance(profit_factor, (int, float)) and profit_factor < 1.2 and not (np.isinf(profit_factor)): penalties *= 0.9 # Ignoruj inf PF
    if isinstance(risk_reward, (int, float)) and risk_reward < 1 and not (np.isinf(risk_reward)): penalties *= 0.8 # Ignoruj inf RR

    # Zwróć finalny score, upewniając się, że jest >= 0
    return max(0.0, total_score * penalties)
    # --- KONIEC KODU Z POPRZEDNIEJ WERSJI ---


# Funkcje formatujące (format_global_stats, format_metrics, format_strategy_group)
# i are_metrics_similar pozostają bez zmian (są używane w reports.py lub nieużywane)

def format_global_stats(analysis_stats: Dict) -> Dict:
    """ Formatuje globalne statystyki analizy. (Bez zmian) """
    # --- KOD BEZ ZMIAN ---
    return {
        "total_strategies": analysis_stats.get('total_initial_results_in_file', analysis_stats.get('total_results', 0)), # Użyj pełnej liczby jeśli dostępna
        "strategies_after_prefiltering": analysis_stats.get('total_after_prefiltering', 0),
        "unique_strategies": analysis_stats.get('total_unique', 0),
        "error_count": analysis_stats.get('error_count', 0),
        "prefiltered_rejected_count": analysis_stats.get('prefiltered_rejected_count', 0),
        "filtered_cluster_count": analysis_stats.get('filtered_cluster_count', 0),
        # Dodaj procenty, jeśli chcesz
        "strategy_distribution": analysis_stats.get('strategy_distribution', {}),
        "similarity_stats": analysis_stats.get('similarity_stats', {}) # Może zawierać tylko status
    }

def format_metrics(metrics: Dict) -> Dict:
    """ Formatuje metryki strategii. (Bez zmian) """
    # --- KOD BEZ ZMIAN ---
    if not isinstance(metrics, dict): return {} # Handle case where metrics is not a dict
    key_metrics = { k: metrics.get(k, np.nan) for k in [ # Użyj NaN dla brakujących
        "trade_count", "win_rate", "avg_profit", "total_profit",
        "max_drawdown", "score", "profit_factor", "sharpe_ratio", "sortino_ratio" # Dodano więcej kluczowych
    ]}
    # Zaokrąglenie tam, gdzie to ma sens
    for k in ["win_rate", "avg_profit", "total_profit", "max_drawdown", "score", "profit_factor", "sharpe_ratio", "sortino_ratio"]:
        if k in key_metrics and pd.notna(key_metrics[k]):
             try: key_metrics[k] = round(float(key_metrics[k]), 3) # Użyj 3 miejsc dla spójności
             except (ValueError, TypeError): pass # Ignoruj błędy konwersji/zaokrąglenia
    trade_count_val = key_metrics.get("trade_count")
    if pd.notna(trade_count_val):
        try: key_metrics["trade_count"] = int(trade_count_val)
        except (ValueError, TypeError): pass

    btc_metrics = {}
    for k in ["btc_block_effectiveness", "btc_correlation", "btc_block_rate"]:
         btc_val = metrics.get(k)
         if pd.notna(btc_val):
              try: btc_metrics[k] = round(float(btc_val), 3)
              except (ValueError, TypeError): pass
    if btc_metrics: key_metrics['btc_metrics'] = btc_metrics
    return key_metrics

'''
def format_strategy_group(group: Dict, importance_by_symbol: Dict[str, Dict[str, float]], symbol: str) -> str:
    AKTUALIZACJA: [FUNKCJA ZASTĄPIONA] Formatuje informacje o GRUPIE strategii.
    """ Formatuje informacje o grupie strategii. (Bez zmian - używa format_parameter_value z tego pliku) """
    # --- KOD BEZ ZMIAN ---
    base_strategy = group.get('base_strategy', {})
    variations = group.get('variations', []) # variations nie jest używane w obecnej logice?
    metrics = base_strategy.get('metrics', {})
    params = group.get('common_params', {})
    varying_params = group.get('varying_params', {})
    param_importance = importance_by_symbol.get(symbol, {})
    lines = []
    lines.append(f"Grupa strategii (Liczba podobnych: {group.get('similar_count', 1)})") # Użyj similar_count
    lines.append(f"ID Reprezentanta (najlepszy score): {base_strategy.get('strategy_id', 'N/A')}")

    # Formatowanie score z obsługą błędu
    score_val = base_strategy.get('score', 'N/A')
    try:
        lines.append(f"Score: {float(score_val):.2f}")
    except (ValueError, TypeError):
        lines.append(f"Score: {score_val}")

    lines.append("\nMetryki Reprezentanta:")
    fm = format_metrics(metrics) # Użyj sformatowanych metryk
    for k, v in fm.items():
        if k != 'btc_metrics': lines.append(f"  {k.replace('_', ' ').title()}: {v}")
    if 'btc_metrics' in fm:
        lines.append("  Metryki BTC:")
        for k_btc, v_btc in fm['btc_metrics'].items(): lines.append(f"    {k_btc.replace('_', ' ').title()}: {v_btc}")

    lines.append("\nParametry Stałe w Grupie:")
    if not params: lines.append("  (Brak)")
    for name, value in sorted(params.items()):
        imp = param_importance.get(name, 0)
        tag = f" [Ważność: {imp:.3f}]" if imp > 0.05 else ""
        lines.append(f"  {name}: {format_parameter_value(name, value)}{tag}") # Użyj lokalnej funkcji formatowania

    lines.append("\nParametry Zmienne w Grupie:")
    if not varying_params: lines.append("  (Brak)")
    for name, values in sorted(varying_params.items()):
        imp = param_importance.get(name, 0)
        tag = f" [Ważność: {imp:.3f}]" if imp > 0.05 else ""
        # Formatuj wartości zmienne
        try: # Spróbuj pokazać zakres dla liczb
            # Upewnij się, że values to lista przed iteracją
            if not isinstance(values, list):
                values_list = [values] # Przekształć w listę, jeśli nie jest
            else:
                values_list = values

            num_vals = sorted([v for v in values_list if isinstance(v, (int, float))])
            if num_vals:
                 min_v, max_v = num_vals[0], num_vals[-1]
                 range_str = f"{format_parameter_value(name, min_v)} ... {format_parameter_value(name, max_v)}"
                 if len(num_vals) <= 5: # Pokaż kilka wartości
                     values_str = ", ".join(format_parameter_value(name, v) for v in num_vals)
                     lines.append(f"  {name}: {range_str} (Wartości: {values_str}){tag}")
                 else: # Pokaż tylko zakres
                     lines.append(f"  {name}: {range_str} ({len(num_vals)} wartości){tag}")
            else: # Jeśli nie ma liczb, pokaż jako stringi
                 values_str = ", ".join(str(v) for v in values_list[:5]) + ('...' if len(values_list)>5 else '')
                 lines.append(f"  {name}: {values_str}{tag}")
        except Exception as e_format: # Fallback
             logger.warning(f"Błąd formatowania wartości zmiennych dla {name}: {e_format}")
             values_str = str(values) # Pokaż surową reprezentację
             lines.append(f"  {name}: {values_str[:100]}{'...' if len(values_str)>100 else ''}{tag}")

    lines.append("\n" + "-" * 50)
    return "\n".join(lines)
'''

# --- Funkcja analyze_single_strategy z dodaną obsługą błędów ---
def analyze_single_strategy(result: Dict) -> Dict:
    """
    Analizuje pojedynczą strategię, oblicza metryki i score.
    Dodano szczegółowe logowanie błędów i odporność.
    """
    strategy_id = result.get('strategy_id', 'N/A')
    # logger.debug(f"Analizuję strategię: {strategy_id}") # Zbyt szczegółowe dla milionów wywołań

    if 'error' in result: return result # Zwróć istniejący błąd

    try:
        # 1. Walidacja danych wejściowych
        trades = result.get('trades')
        params = result.get('parameters')
        if not isinstance(trades, list):
            raise ValueError("Brakujące lub niepoprawne dane 'trades'.")
        if not isinstance(params, dict):
            raise ValueError("Brakujące lub niepoprawne dane 'parameters'.")

        # 2. Obliczanie metryk
        metrics_calculator = StrategyMetrics()
        metrics_calculator.calculate_metrics(
            trades=trades,
            # Użyj .get() z domyślnymi wartościami, aby uniknąć KeyError
            btc_blocks=result.get('btc_blocks', 0),
            total_checks=result.get('trades_checked', 0), # Upewnij się, że ten klucz istnieje w .pkl
            btc_correlation=result.get('btc_correlation', 0.0)
        )
        # Konwertuj metryki na słownik, obsługując potencjalne nieserializowalne atrybuty
        metrics_dict = {}
        for k, v in metrics_calculator.__dict__.items():
            if not k.startswith('_'):
                 # Podstawowa próba serializacji - można rozszerzyć
                 if isinstance(v, (int, float, str, bool, list, dict, tuple)) or v is None:
                     # Sprawdź NaN/Inf w float
                     if isinstance(v, float) and not np.isfinite(v):
                          metrics_dict[k] = str(v) # Zapisz jako string 'inf', '-inf', 'nan'
                     else:
                          metrics_dict[k] = v
                 else:
                     try: metrics_dict[k] = str(v) # Spróbuj skonwertować na string
                     except: metrics_dict[k] = f"Unserializable type: {type(v).__name__}"


        # 3. Obliczanie score
        # calculate_strategy_score jest w tym samym module
        score = calculate_strategy_score(metrics_dict)
        metrics_dict['score'] = score # Dodaj score do metryk

        # 4. Tworzenie finalnego słownika
        final_result = {
            'parameters': params,
            'metrics': metrics_dict,
            'score': score,
            'strategy_id': strategy_id,
            'symbol': result.get('symbol', 'BTC/USDT'),
            'param_file_name': result.get('param_file_name', 'unknown')
        }
        # logger.debug(f"[{strategy_id}] Analiza OK.") # Zbyt szczegółowe
        return final_result

    except Exception as e:
        # Złap *wszelkie* wyjątki
        error_msg = f"Błąd analizy strat. {strategy_id}: {type(e).__name__}: {e}"
        # Zmniejszamy szczegółowość logowania błędu w normalnym trybie
        # Pełny traceback można włączyć ustawiając logger na DEBUG
        if logger.getEffectiveLevel() <= 10: # DEBUG LEVEL
             logger.error(f"[{strategy_id}] {error_msg}", exc_info=True)
        else:
             logger.error(f"[{strategy_id}] {error_msg}") # Tylko komunikat błędu
        # Zwróć słownik błędu
        return {
            'error': error_msg,
            'strategy_id': strategy_id,
            'parameters': result.get('parameters', {}), # Dołącz parametry jeśli możliwe
            # 'traceback': traceback.format_exc(), # Opcjonalnie dodaj traceback do wyniku
        }

'''
def are_metrics_similar(metrics1: Dict, metrics2: Dict) -> bool:
    """Sprawdza czy metryki dwóch strategii są podobne w kluczowych aspektach.
    AKTUALIZACJA: [FUNKCJA PRAWDOPODOBNIE ZBĘDNA PO WPROWADZENIU KLASTROWANIA]"""
    # Upewnij się, że oba argumenty to słowniki
    if not isinstance(metrics1, dict) or not isinstance(metrics2, dict):
        return False

    # Wybierz kluczowe metryki do porównania
    # Upewnij się, że score jest jedną z nich
    key_metrics = ['score', 'trade_count', 'win_rate', 'avg_profit', 'max_drawdown', 'profit_factor']
    threshold = 0.01 # Próg tolerancji dla float

    try:
        for metric in key_metrics:
            val1 = metrics1.get(metric)
            val2 = metrics2.get(metric)

            # Obsługa None lub brakujących wartości - jeśli jedna ma, druga nie, to nie są podobne
            if val1 is None and val2 is not None: return False
            if val2 is None and val1 is not None: return False
            if val1 is None and val2 is None: continue # Obie brakujące - przejdź dalej

            # Konwersja na float do porównania
            try:
                 f_val1 = float(val1)
                 f_val2 = float(val2)
            except (ValueError, TypeError):
                 # Jeśli nie da się skonwertować na float, porównaj bezpośrednio
                 if val1 != val2: return False
                 continue # Jeśli równe, przejdź do następnej metryki

            # Porównanie float z tolerancją, obsługa NaN i Inf
            if pd.isna(f_val1) and pd.isna(f_val2): continue # Oba NaN - uznaj za równe w tym kontekście
            if pd.isna(f_val1) or pd.isna(f_val2): return False # Jeden NaN, drugi nie
            if np.isinf(f_val1) and np.isinf(f_val2) and np.sign(f_val1) == np.sign(f_val2): continue # Oba Inf z tym samym znakiem
            if np.isinf(f_val1) or np.isinf(f_val2): return False # Jeden Inf, drugi nie

            # Standardowe porównanie float
            if abs(f_val1 - f_val2) > threshold:
                return False

        return True # Jeśli wszystkie kluczowe metryki są podobne

    except Exception as e:
        logger.warning(f"Błąd w are_metrics_similar: {e}. Strategie: {metrics1.get('strategy_id', 'N/A')} vs {metrics2.get('strategy_id', 'N/A')}")
        return False # W razie nieoczekiwanego błędu, uznaj za różne
'''