#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import hashlib
import logging
from typing import Dict, List, Generator, Optional
from collections import defaultdict
from .models import TradingParameters
from .validation import validate_parameters

logger = logging.getLogger(__name__)

def generate_parameter_combinations(param_ranges: Dict, 
                               market_data: Optional[Dict] = None,
                               max_combinations: int = None, 
                               resume_from: str = None,
                               mode: str = "backtest",
                               param_file_name: str = "unknown") -> List[TradingParameters]:
    """
    Generuje kombinacje parametrów używając Latin Hypercube Sampling dla lepszego pokrycia przestrzeni.
    
    Args:
        param_ranges: Słownik z zakresami parametrów 
        market_data: Opcjonalne dane rynkowe do walidacji
        max_combinations: Opcjonalne ograniczenie liczby kombinacji (domyślnie: None = wszystkie)
        resume_from: ID ostatnio testowanej kombinacji (do wznawiania testów)
        mode: Tryb walidacji (backtest/fronttest), domyślnie 'backtest'
        param_file_name: Nazwa pliku parametrów (dla trackowalności wyników)
        
    Returns:
        List[TradingParameters]: Lista wygenerowanych kombinacji parametrów
    """
    param_values = {}
    param_dimensions = {}
    continuous_params = set()
    invalid_params_summary = []
    provided_params = set(param_ranges.keys())
    
    # Definiujemy zależności między parametrami enable/disable a ich parametrami
    enable_params = {
        'stop_loss_enabled': ['stop_loss_threshold', 'stop_loss_delay_time'],
        'trailing_enabled': ['trailing_stop_price', 'trailing_stop_margin', 'trailing_stop_time'],
        'trailing_buy_enabled': ['trailing_buy_threshold', 'trailing_buy_time_in_min'],
        'pump_detection_enabled': ['pump_detection_threshold', 'pump_detection_disabled_time']
    }
    
    # Analizujemy zakresy parametrów
    try:
        # Tworzymy zbiór już przetworzonych parametrów
        processed_params = set()
        
        # Najpierw przetwarzamy parametry enable/disable
        for enable_param, dependent_params in enable_params.items():
            if enable_param in param_ranges:
                enable_info = param_ranges[enable_param]
                processed_params.add(enable_param)
                
                # Specjalna obsługa stop_loss_enabled - zawsze włączony
                if enable_param == 'stop_loss_enabled':
                    param_values[enable_param] = [1.0]  # Wymuszamy włączenie
                    param_dimensions[enable_param] = 1
                    continue
                
                # Dla pozostałych parametrów enable/disable
                if enable_info.get('type') == 'boolean':
                    value = enable_info.get('value', False)
                    param_values[enable_param] = [float(value)]
                    param_dimensions[enable_param] = 1
                    
                    # Jeśli parametr jest wyłączony, dla zależnych parametrów ustawiamy tylko wartości domyślne
                    is_enabled = bool(enable_info.get('value', False))
                    if not is_enabled:
                        for dep_param in dependent_params:
                            if dep_param in param_ranges:
                                param_values[dep_param] = [0.0]
                                param_dimensions[dep_param] = 1

        # Następnie przetwarzamy pozostałe parametry
        for param_name, param_info in param_ranges.items():
            # Pomijamy już przetworzone parametry
            if param_name in processed_params:
                continue
                
            # Specjalna obsługa stop_loss_threshold
            if param_name == 'stop_loss_threshold':
                if 'range' in param_info:
                    min_val, max_val, step = param_info['range']
                    values = list(np.arange(min_val, max_val + step, step))
                    param_values[param_name] = values
                else:
                    param_values[param_name] = [-20.0]  # Domyślny stop loss
                param_dimensions[param_name] = len(param_values[param_name])
                continue
                
            # Sprawdzamy czy to jest parametr zależny od wyłączonego przełącznika
            skip_range = False
            for enable_param, dependent_params in enable_params.items():
                if param_name in dependent_params and enable_param != 'stop_loss_enabled':
                    enable_info = param_ranges.get(enable_param, {})
                    if not bool(enable_info.get('value', False)):
                        skip_range = True
                        break
            
            if skip_range:
                continue
                
            if not isinstance(param_info, dict):
                raise ValueError(f"Nieprawidłowy format parametru {param_name}")
                
            param_type = param_info.get('type')
            
            if param_type == 'boolean':
                value = param_info.get('value', False)
                param_values[param_name] = [float(value)]
                param_dimensions[param_name] = 1
            elif 'range' in param_info:
                min_val, max_val, step = param_info['range']
                is_integer = param_info.get('numeric_type') == 'int'
                
                if is_integer:
                    values = list(range(int(min_val), int(max_val) + 1, max(1, int(step))))
                else:
                    continuous_params.add(param_name)
                    num_points = int((max_val - min_val) / step) + 1
                    values = list(np.linspace(min_val, max_val, num_points))
                    
                param_values[param_name] = values
                param_dimensions[param_name] = len(values)
                
                if len(values) != param_dimensions[param_name]:
                    logger.error(f"Niespójność wymiarów dla {param_name}: "
                                f"długość wartości={len(values)}, "
                                f"przypisany wymiar={param_dimensions[param_name]}")
                    
            elif 'value' in param_info:
                param_values[param_name] = [param_info['value']]
                param_dimensions[param_name] = 1
            else:
                raise ValueError(f"Nieprawidłowa konfiguracja parametru {param_name}")

        # Obliczamy całkowitą liczbę możliwych kombinacji
        dimensions = [dim for dim in param_dimensions.values()]
        total_combinations = np.prod(dimensions)
        
        if max_combinations and max_combinations < total_combinations:
            total_combinations = max_combinations

        # Generujemy punkty Latin Hypercube Sampling dla parametrów ciągłych
        continuous_param_names = sorted(list(continuous_params))
        if continuous_param_names:
            n_samples = min(total_combinations, 1000) if max_combinations else total_combinations
            lhs_samples = _generate_lhs_samples(n_samples, len(continuous_param_names))
            
            # Mapujemy próbki LHS na rzeczywiste wartości parametrów
            for i, param_name in enumerate(continuous_param_names):
                min_val = min(param_values[param_name])
                max_val = max(param_values[param_name])
                lhs_values = min_val + lhs_samples[:, i] * (max_val - min_val)
                
                # Zaokrąglamy do najbliższej wartości z siatki
                step = param_ranges[param_name]['range'][2]
                lhs_values = np.round(lhs_values / step) * step
                param_values[param_name] = sorted(list(set(lhs_values)))
                param_dimensions[param_name] = len(param_values[param_name])

        # Jeśli podano resume_from, znajdujemy indeks od którego wznowić
        start_idx = 0
        if resume_from:
            for idx, params in enumerate(_generate_combinations(param_values, param_dimensions)):
                if _generate_combination_id(params) == resume_from:
                    start_idx = idx + 1
                    break

        # Generujemy kombinacje
        trading_params = []
        combinations_generated = 0
        combinations_invalid = 0
        combinations_warned = 0
        error_summary = defaultdict(int)
        warning_summary = defaultdict(int)
        
        for params in _generate_combinations(param_values, param_dimensions, start_idx):
            try:
                combinations_generated += 1
                param_id = _generate_combination_id(params)
                
                # Tworzymy obiekt parametrów
                params_copy = params.copy()
                if '__mode__' in params_copy:
                    del params_copy['__mode__']
                trading_param = TradingParameters(**params_copy)
                
                # Walidujemy parametry
                validation_result = validate_parameters(
                    trading_param, 
                    market_data,
                    param_id,
                    provided_params=provided_params,
                    mode=mode
                )
                
                # Agregujemy ostrzeżenia
                for warning in validation_result.warnings:
                    combinations_warned += 1
                    warning_summary[warning.message] += 1
                    
                # Jeśli są błędy (nie ostrzeżenia!), agregujemy je i pomijamy kombinację
                if validation_result.errors:
                    combinations_invalid += 1
                    for error in validation_result.errors:
                        error_summary[error.message] += 1
                        if error_summary[error.message] == 1:
                            invalid_params_summary.append(
                                f"Przykład - ID: {error.param_id}, Błąd: {error.message}"
                            )
                    continue
                
                trading_params.append(trading_param)
                
                if max_combinations and len(trading_params) >= max_combinations:
                    break
                    
            except KeyboardInterrupt:
                logger.info("\nOtrzymano sygnał przerwania podczas generowania parametrów")
                raise
            except Exception as e:
                logger.warning(f"Pominięto nieprawidłową kombinację: {str(e)}")
                combinations_invalid += 1
                continue

        # Logujemy podsumowanie
        if combinations_invalid > 0 or combinations_warned > 0:
            logger.info(f"\nPodsumowanie walidacji parametrów:")
            logger.info(f"- Wygenerowano kombinacji: {combinations_generated}")
            logger.info(f"- Odrzucono kombinacji: {combinations_invalid}")
            logger.info(f"- Kombinacji z ostrzeżeniami: {combinations_warned}")
            logger.info(f"- Zaakceptowano kombinacji: {len(trading_params)}")

            if error_summary:
                logger.error("\nWykryte błędy walidacji:")
                for message, count in error_summary.items():
                    logger.error(f"- {message} (wystąpił {count} razy)")
                
            if warning_summary:
                logger.warning("\nWykryte ostrzeżenia:")
                for message, count in warning_summary.items():
                    logger.warning(f"- {message} (wystąpiło {count} razy)")

            if invalid_params_summary:
                logger.info("\nPrzykłady odrzuconych kombinacji:")
                for summary in invalid_params_summary:
                    logger.info(f"- {summary}")

        if not trading_params:
            logger.error(f"Nie znaleziono żadnych poprawnych kombinacji parametrów")
            return []

        # Dodanie nazwy pliku parametrów do każdej kombinacji
        for trading_param in trading_params:
            setattr(trading_param, '__param_file_name', param_file_name)

        return trading_params

    except Exception as e:
        logger.error(f"Błąd podczas generowania kombinacji parametrów: {str(e)}")
        raise


def _generate_lhs_samples(n_samples: int, n_dimensions: int) -> np.ndarray:
    """
    Generuje próbki metodą Latin Hypercube Sampling.
    
    Args:
        n_samples: Liczba próbek do wygenerowania
        n_dimensions: Liczba wymiarów (parametrów)
        
    Returns:
        np.ndarray: Tablica próbek o wymiarach (n_samples, n_dimensions)
    """
    # Generujemy podstawową siatkę LHS
    samples = np.zeros((n_samples, n_dimensions))
    
    for i in range(n_dimensions):
        samples[:, i] = np.random.permutation(n_samples)
    
    # Dodajemy losowe zaburzenie w każdej komórce
    samples += np.random.uniform(0, 1, samples.shape)
    
    # Normalizujemy do przedziału [0, 1]
    return samples / n_samples

def _generate_combinations(param_values: Dict, param_dimensions: Dict, 
                         start_idx: int = 0) -> Generator[Dict, None, None]:
    """Generator kombinacji parametrów."""
    param_names = sorted(param_values.keys())
    dimensions = [param_dimensions[name] for name in param_names]
    total_combinations = np.prod(dimensions)
    
    for idx in range(start_idx, total_combinations):
        indices = _unravel_index(idx, dimensions)
        combination = {}
        
        for name, index in zip(param_names, indices):
            combination[name] = param_values[name][index]
            
        yield combination

def _unravel_index(idx: int, dimensions: List[int]) -> List[int]:
    """
    Konwertuje płaski indeks na indeksy wielowymiarowe.
    
    Args:
        idx: Płaski indeks
        dimensions: Lista wymiarów
        
    Returns:
        List[int]: Lista indeksów dla każdego wymiaru
    """
    indices = []
    for dim in reversed(dimensions):
        indices.append(idx % dim)
        idx //= dim
    return list(reversed(indices))

def _generate_combination_id(params: Dict) -> str:
    """
    Generuje unikalny identyfikator kombinacji parametrów.
    
    Args:
        params: Słownik z parametrami
        
    Returns:
        str: Unikalny identyfikator
    """
    sorted_items = sorted(params.items())
    param_str = '_'.join(f"{k}:{v}" for k, v in sorted_items)
    return hashlib.md5(param_str.encode()).hexdigest()[:8]