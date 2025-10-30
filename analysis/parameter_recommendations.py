#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moduł obsługujący rekomendacje i analizę parametrów strategii.
"""

from typing import Dict, List
import numpy as np
from utils.logging_setup import get_strategy_logger
from analysis.parameters import format_parameter_value

logger = get_strategy_logger('parameter_recommendations')

def analyze_parameter_type(param_name: str) -> str:
    """Określa typ parametru na podstawie nazwy i charakterystyki"""
    if 'timeframe' in param_name or 'delay' in param_name or 'time' in param_name or param_name == 'follow_btc_block_time':
        return 'TIME'
    elif 'threshold' in param_name or param_name == 'follow_btc_threshold':
        return 'THRESHOLD'
    elif 'margin' in param_name or 'profit' in param_name or 'price' in param_name:
        return 'PRICE'
    elif 'count' in param_name or 'orders' in param_name:
        return 'COUNT'
    return 'OTHER'

def get_parameter_constraints(param_type: str) -> dict:
    """Zwraca ograniczenia dla danego typu parametru"""
    constraints = {
        'TIME': {
            'min_step': 1,
            'max_range_multiplier': 5,
            'preferred_values': [1, 5, 10, 15, 30, 60, 120, 240, 360, 720, 1440],
            'min_values_count': 3,
            'max_values_count': 8
        },
        'THRESHOLD': {
            'min_step': 0.05,
            'max_range_multiplier': 3,
            'symmetry_preferred': True,
            'min_values_count': 5,
            'max_values_count': 10
        },
        'PRICE': {
            'min_step': 0.1,
            'max_range_multiplier': 2,
            'min_values_count': 4,
            'max_values_count': 8
        },
        'COUNT': {
            'min_step': 1,
            'max_range_multiplier': 2,
            'preferred_values': [1, 2, 3, 5, 8, 13, 21],
            'min_values_count': 2,
            'max_values_count': 5
        },
        'OTHER': {
            'min_step': 0.1,
            'max_range_multiplier': 2,
            'min_values_count': 3,
            'max_values_count': 8
        }
    }
    return constraints.get(param_type, constraints['OTHER'])

def adjust_parameter_range(param_name: str, 
                         current_config: Dict, 
                         param_analysis: Dict,
                         is_integer: bool = False) -> Dict:
    """
    Dostosowuje zakres parametru na podstawie analizy.
    
    Args:
        param_name: Nazwa parametru
        current_config: Obecna konfiguracja parametru
        param_analysis: Analiza parametru
        is_integer: Czy parametr jest całkowitoliczbowy
        
    Returns:
        Dict: Rekomendowane zmiany lub None jeśli brak zmian
    """
    if not param_analysis or 'clusters' not in param_analysis:
        return None
        
    clusters = param_analysis['clusters']
    if not clusters:
        return None
        
    best_cluster = clusters[0]
    cluster_center = best_cluster['center']
    cluster_std = best_cluster['std']
    
    # Jeśli parametr ma małe odchylenie w klastrze, proponuj stałą wartość
    if cluster_std < 0.1 * cluster_center:
        value = round(cluster_center) if is_integer else round(cluster_center, 3)
        return {
            'action': 'CONSTANT',
            'value': value,
            'reason': f'Wykryto stabilną wartość w najlepszym klastrze (std={cluster_std:.3f})'
        }
    
    # W przeciwnym razie proponuj zakres
    min_val = max(0, cluster_center - 2 * cluster_std)
    max_val = cluster_center + 2 * cluster_std
    
    if is_integer:
        min_val = round(min_val)
        max_val = round(max_val)
        step = 1
    else:
        min_val = round(min_val, 3)
        max_val = round(max_val, 3)
        step = round((max_val - min_val) / 10, 3)
        step = max(0.001, step)
    
    return {
        'action': 'ADJUST',
        'range': [min_val, max_val, step],
        'reason': f'Proponowany zakres na podstawie najlepszego klastra (centrum={cluster_center:.3f}, std={cluster_std:.3f})'
    }

def apply_recommendations(recommendations: Dict):
    """
    Aplikuje rekomendowane zmiany do pliku trading_parameters.json
    
    Args:
        recommendations: Słownik z rekomendacjami zmian
    """
    try:
        try:
            from parameter_configurator import ParameterManager
            param_manager = ParameterManager()
        except ImportError:
            logger.error("Nie można zaimportować ParameterManager - sprawdź czy parameter_configurator.py jest w bieżącym katalogu")
            return
            
        current_params = param_manager.load_parameters()
        
        if not current_params:
            logger.error("Nie udało się wczytać obecnych parametrów")
            return
            
        modified_params = current_params.copy()
        made_changes = False
        
        # Lista parametrów całkowitoliczbowych
        integer_params = [
        'max_open_orders_per_coin', 'max_open_orders', 'check_timeframe',
        'next_buy_delay', 'stop_loss_delay_time', 'trailing_stop_time',
        'pump_detection_disabled_time', 'follow_btc_block_time'  # dodajemy follow_btc_block_time
    ]
        
        print("\nAnaliza zalecanych modyfikacji parametrów:\n")
        
        for symbol, params_analysis in recommendations.items():
            print(f"\nAnalizuję rekomendacje dla symbolu: {symbol}")
            
            for param_name, param_analysis in params_analysis.items():
                print(f"\nAnalizuję parametr: {param_name}")
                
                if param_name not in modified_params:
                    print(f"Parametr {param_name} nie istnieje w konfiguracji - pomijam")
                    continue
                
                if 'value' in current_params[param_name]:
                    print(f"Parametr {param_name} ma obecnie stałą wartość: {current_params[param_name]['value']}")
                else:
                    print(f"Parametr {param_name} ma obecnie zdefiniowany zakres")
                
                is_integer = param_name in integer_params
                
                recommendation = adjust_parameter_range(
                    param_name,
                    current_params[param_name],
                    param_analysis,
                    is_integer
                )
                
                if recommendation:
                    print(f"\nProponowana zmiana dla {param_name}:")
                    print(f"Powód: {recommendation['reason']}")
                    
                    if recommendation['action'] == 'CONSTANT':
                        print(f"Obecny zakres: {current_params[param_name].get('range', [])} lub wartość: {current_params[param_name].get('value')}")
                        print(f"Proponowana stała wartość: {recommendation['value']}")
                        
                        if input("\nCzy chcesz zastosować tę zmianę? (t/n): ").lower() == 't':
                            modified_params[param_name] = {
                                'value': recommendation['value'],
                                'description': current_params[param_name].get('description', ''),
                                'type': 'numeric',
                                'numeric_type': 'int' if is_integer else 'float'
                            }
                            made_changes = True
                            
                    elif recommendation['action'] == 'ADJUST':
                        new_min, new_max, new_step = recommendation['range']
                        current_range = current_params[param_name].get('range', [])
                        
                        if current_range:
                            print(f"Obecny zakres: {current_range[0]} do {current_range[1]} (krok: {current_range[2]})")
                        else:
                            print(f"Obecna stała wartość: {current_params[param_name].get('value')}")
                            
                        print(f"Nowy zakres: {new_min} do {new_max} (krok: {new_step})")
                        
                        if input("\nCzy chcesz zastosować tę zmianę? (t/n): ").lower() == 't':
                            modified_params[param_name] = {
                                'range': [new_min, new_max, new_step],
                                'description': current_params[param_name].get('description', ''),
                                'type': 'numeric',
                                'numeric_type': 'int' if is_integer else 'float'
                            }
                            made_changes = True
                            print(f"Zastosowano zmiany dla parametru {param_name}")
                else:
                    print("Brak rekomendacji zmian dla tego parametru")
        
        if made_changes:
            print("\n" + "="*80)
            if input("\nCzy chcesz zapisać zatwierdzone zmiany do pliku trading_parameters.json? (t/n): ").lower() == 't':
                param_manager.save_parameters(modified_params)
                logger.info("Zaktualizowano plik trading_parameters.json zgodnie z zatwierdzonymi zmianami")
                print("\nZmiany zostały zapisane.")
            else:
                logger.info("Anulowano zapisywanie zmian")
                print("\nZmiany nie zostały zapisane.")
        else:
            logger.info("Nie wprowadzono żadnych zmian w parametrach")
            print("\nNie wprowadzono żadnych zmian w parametrach.")
            
    except Exception as e:
        logger.error(f"Błąd podczas aplikowania rekomendacji: {str(e)}")
        print(f"\nWystąpił błąd podczas modyfikacji parametrów: {str(e)}")

def determine_parameter_recommendation(
    param_name: str,
    param_analysis: Dict,
    param_groups: Dict,
    timing_corr: Dict,
    threshold_corr: Dict,
    is_integer: bool
) -> Dict:
    """
    Określa rekomendację dla parametru na podstawie analizy jego rozkładu
    i interakcji z innymi parametrami.
    
    Args:
        param_name: Nazwa parametru
        param_analysis: Analiza parametru
        param_groups: Grupy parametrów
        timing_corr: Korelacje czasowe
        threshold_corr: Korelacje progów
        is_integer: Czy parametr jest całkowitoliczbowy
        
    Returns:
        Dict: Rekomendacja dla parametru
    """
    recommendation = {}
    
    # Jeśli parametr ma stałą wartość
    if param_analysis.get('type') == 'constant':
        recommendation['action'] = "STAŁA"
        recommendation['value'] = param_analysis['value']
        recommendation['reason'] = "Parametr wykazuje stałą optymalną wartość"
        return recommendation
        
    # Jeśli brak danych numerycznych
    if param_analysis.get('type') == 'no_data':
        return None
    
    # Dla parametrów ze zmienną dystrybucją
    if 'clusters' in param_analysis and len(param_analysis['clusters']) > 0:
        main_cluster = param_analysis['clusters'][0]
        relative_std = param_analysis.get('relative_std', 1.0)
        
        # Analiza parametrów follow BTC
        if param_name in param_groups.get('follow_btc', []):
            if 'effective_mean' in param_analysis:
                recommendation['action'] = "STAŁA"
                recommendation['value'] = param_analysis['effective_mean']
                recommendation['formatted_value'] = format_parameter_value(
                    param_name, param_analysis['effective_mean']
                )
                recommendation['reason'] = "Wartość optymalna dla strategii z efektywnym follow BTC"
                return recommendation
        
        # Analiza skupień
        if relative_std < 0.1:
            recommendation['action'] = "STAŁA"
            recommendation['value'] = main_cluster['center']
            recommendation['formatted_value'] = main_cluster['formatted_center']
            recommendation['reason'] = "Silne skupienie wartości wokół optymalnego punktu"
            
        elif relative_std < 0.2:
            min_val = max(param_analysis['min'], 
                         main_cluster['center'] - 2*main_cluster['std'])
            max_val = min(param_analysis['max'], 
                         main_cluster['center'] + 2*main_cluster['std'])
            
            step = 1 if is_integer else (param_analysis['range'] / 20)
            
            recommendation['action'] = "ZAWĘŻENIE"
            recommendation['range'] = [min_val, max_val, step]
            recommendation['formatted_range'] = [
                format_parameter_value(param_name, min_val),
                format_parameter_value(param_name, max_val),
                format_parameter_value(param_name, step)
            ]
            recommendation['reason'] = "Wyraźny optymalny zakres wartości"
            
        else:
            step = 1 if is_integer else (param_analysis['range'] / 10)
            extended_min = param_analysis['min'] - param_analysis['range']*0.2
            extended_max = param_analysis['max'] + param_analysis['range']*0.2
            
            recommendation['action'] = "ROZSZERZENIE"
            recommendation['range'] = [extended_min, extended_max, step]
            recommendation['formatted_range'] = [
                format_parameter_value(param_name, extended_min),
                format_parameter_value(param_name, extended_max),
                format_parameter_value(param_name, step)
            ]
            recommendation['reason'] = "Potrzebne szersze testowanie parametru"
    
    return recommendation

def confirm_change() -> bool:
    """
    Pyta użytkownika o potwierdzenie rekomendowanych zmian parametrów.
    
    Returns:
        bool: True jeśli użytkownik potwierdził, False w przeciwnym razie
    """
    while True:
        response = input("\nCzy chcesz zastosować te zmiany? (t/n): ").lower()
        if response in ('t', 'tak', 'y', 'yes'):
            return True
        elif response in ('n', 'nie', 'no'):
            return False
        print("Proszę odpowiedzieć 't' lub 'n'")