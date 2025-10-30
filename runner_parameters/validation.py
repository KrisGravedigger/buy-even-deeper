#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple, Set
from .models import TradingParameters

logger = logging.getLogger(__name__)

# Stałe ścieżek
PARAMETRY_DIR = Path('parametry')
CSV_DIR = Path('csv')
WYNIKI_DIR = Path('wyniki')
LOGI_DIR = Path('logi')
WYNIKI_BACKTEST_DIR = WYNIKI_DIR / 'backtesty'
WYNIKI_ANALIZA_DIR = WYNIKI_DIR / 'analizy'

# Stałe dla trybów
BACKTEST_MODE = 'backtest'
FRONTTEST_MODE = 'fronttest'
VALID_MODES = {BACKTEST_MODE, FRONTTEST_MODE}

# Parametry które muszą być pojedynczymi wartościami w trybie fronttest
FRONTTEST_SINGLE_VALUE_PARAMS = {
    'check_timeframe',
    'percentage_buy_threshold',
    'stop_loss_threshold',
    'sell_profit_target',
    'trailing_stop_price',
    'trailing_stop_margin',
    'follow_btc_threshold'
}

# Parametry które muszą być włączone/wyłączone w trybie fronttest
FRONTTEST_REQUIRED_ENABLED = {
    'stop_loss_enabled',  # Stop loss musi być włączony w fronttestach
}

FRONTTEST_REQUIRED_DISABLED = set()  # Na razie brak wymaganych wyłączonych parametrów

class ValidationMessage(NamedTuple):
    """Struktura do przechowywania wiadomości walidacyjnych"""
    message: str
    param_id: str
    details: Dict

class ValidationResult(NamedTuple):
    """Struktura wyniku walidacji"""
    is_valid: bool
    errors: List[ValidationMessage]
    warnings: List[ValidationMessage]
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Łączy dwa wyniki walidacji"""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings
        )

def is_enabled(value) -> bool:
    """
    Sprawdza czy parametr jest włączony, niezależnie od typu wartości (bool/int/float).
    
    Args:
        value: Wartość parametru (bool/int/float)
        
    Returns:
        bool: True jeśli parametr jest włączony (True/1/1.0), False w przeciwnym razie
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return False

def validate_csv_directory() -> Tuple[bool, Optional[Path]]:
    """
    Sprawdza, czy w folderze csv znajduje się dokładnie jeden plik CSV 
    odpowiadający wzorcowi (*_with_btc.csv lub binance_BTC_*.csv) 
    i czy zawiera wymagane kolumny.
    """
    # Szukamy plików altcoinów z danymi BTC i plików BTC/USDT
    altcoin_files = list(CSV_DIR.glob('*_with_btc.csv'))
    btc_files = list(CSV_DIR.glob('binance_BTC_*.csv'))
    
    # Łączymy listy i usuwamy potencjalne duplikaty (chociaż nie powinno ich być)
    candidate_files = list(set(altcoin_files + btc_files))
    
    if not candidate_files:
        logger.error(
            "Nie znaleziono odpowiednich plików CSV w katalogu csv/. "
            "Oczekiwano pliku *_with_btc.csv lub binance_BTC_*.csv."
        )
        return False, None
        
    if len(candidate_files) > 1:
        file_names = ', '.join([f.name for f in candidate_files])
        logger.error(
            f"W katalogu csv/ znajduje się więcej niż jeden potencjalny plik CSV do analizy: {file_names}. "
            "Pozostaw tylko jeden plik danych (np. przenieś nieużywane do csv/archiwum/)."
        )
        return False, None
    
    csv_file = candidate_files[0]
    
    try:
        df = pd.read_csv(csv_file)
        
        # --- POCZĄTEK ZMIANY - Warunkowe sprawdzanie kolumn ---
        # Sprawdzenie, czy plik to BTC/USDT
        is_btc_usdt_file = 'BTC_USDT' in csv_file.name 
        
        # Ustalenie wymaganych kolumn na podstawie typu pliku
        if is_btc_usdt_file:
            # Dla BTC/USDT wymagamy tylko podstawowych kolumn
            required_columns = ['timestamp', 'average_price']
            logger.debug(f"Plik {csv_file.name} rozpoznany jako BTC/USDT. Wymagane kolumny: {required_columns}")
        else:
            # Dla innych par (altcoinów) wymagamy również danych BTC
            required_columns = ['timestamp', 'average_price', 'btc_average_price']
            logger.debug(f"Plik {csv_file.name} rozpoznany jako Altcoin. Wymagane kolumny: {required_columns}")

        # Sprawdzenie brakujących kolumn
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(
                f"Brakujące kolumny w pliku CSV {csv_file.name}. Wymagane kolumny dla tego typu pliku: {', '.join(required_columns)}. Znalezione: {', '.join(df.columns)}"
            )
            return False, None
        # --- KONIEC ZMIANY ---

        # Usunęliśmy stąd check np.allclose, ponieważ jego miejsce jest w strategy_runner.py podczas ładowania danych

        logger.info(f"Znaleziono poprawny plik CSV: {csv_file.name}")
        return True, csv_file
        
    except Exception as e:
        logger.error(f"Błąd podczas walidacji pliku CSV {csv_file.name}: {str(e)}")
        return False, None

def get_parameter_files(param_dir: Optional[Path] = None) -> List[Path]: # <-- DODANY OPCJONALNY ARGUMENT param_dir
    """
    Znajduje pliki JSON z parametrami w podanym katalogu lub domyślnym.

    Args:
        param_dir: Opcjonalna ścieżka do katalogu z parametrami.
                   Jeśli None, używa PARAMETRY_DIR zdefiniowanego w tym module.

    Returns:
        Lista ścieżek do znalezionych plików .json posortowana alfabetycznie.
    """
    # Użyj podanego katalogu lub domyślnego, jeśli nie podano
    directory_to_check = param_dir if param_dir is not None else PARAMETRY_DIR # Używamy stałej z tego modułu

    if not directory_to_check.exists() or not directory_to_check.is_dir():
        logger.error(f"Katalog parametrów '{directory_to_check}' nie istnieje lub nie jest katalogiem.")
        return []

    # Szukaj plików .json w wybranym katalogu i sortuj
    parameter_files = sorted(list(directory_to_check.glob('*.json')))

    if not parameter_files:
        logger.warning(f"Nie znaleziono plików parametrów JSON w katalogu '{directory_to_check}'.")
        return []

    # Logowanie przeniesione do strategy_runner.py, gdzie jest kontekst
    # logger.info(f"Znaleziono {len(parameter_files)} plików z parametrami w '{directory_to_check}'")
    # for f in parameter_files:
    #     logger.info(f"- {f.name}")

    return parameter_files

def validate_btc_parameters(params: TradingParameters, 
                          market_data: Dict,
                          param_id: str) -> ValidationResult:
    """
    Walidacja parametrów związanych z BTC.
    Sprawdza czy dla pary BTC/USDT parametry BTC są ignorowane.
    """
    warnings = []
    
    if market_data.get('symbol') == 'BTCUSDT':
        btc_params = {
            'follow_btc_price': params.follow_btc_price,
            'follow_btc_threshold': params.follow_btc_threshold,
            'follow_btc_block_time': params.follow_btc_block_time
        }
        
        active_btc_params = [name for name, value in btc_params.items() 
                            if is_enabled(value) or (isinstance(value, (int, float)) and value != 0)]
        
        if active_btc_params:
            warnings.append(ValidationMessage(
                message="Parametry BTC są ignorowane dla pary BTC/USDT",
                param_id=param_id,
                details={'ignored_params': active_btc_params}
            ))
    
    return ValidationResult(is_valid=True, errors=[], warnings=warnings)

def validate_trailing_parameters(params: TradingParameters, 
                               param_id: str,
                               provided_params: Optional[Set[str]] = None) -> ValidationResult:
    """
    Walidacja parametrów związanych z trailing stop i trailing buy.
    Sprawdza zależności, konflikty i nadmiarowe parametry.
    
    Args:
        params: Obiekt z parametrami
        param_id: Identyfikator kombinacji parametrów
        provided_params: Opcjonalny zestaw parametrów faktycznie dostarczonych w pliku
    """
    errors = []
    warnings = []
    
    # Jeśli nie podano provided_params, zakładamy że wszystkie były w pliku (kompatybilność wsteczna)
    if provided_params is None:
        provided_params = set()
    
    # Sprawdzenie trailing stop
    if is_enabled(params.trailing_enabled):
        if params.sell_profit_target != 0:
            errors.append(ValidationMessage(
                message="Trailing stop wyklucza się z sell_profit_target",
                param_id=param_id,
                details={'trailing_enabled': True, 
                        'sell_profit_target': params.sell_profit_target}
            ))
        
        if params.trailing_stop_price <= 0:
            errors.append(ValidationMessage(
                message="Trailing stop wymaga ustawienia trailing_stop_price > 0",
                param_id=param_id,
                details={'trailing_stop_price': params.trailing_stop_price}
            ))
            
        if params.trailing_stop_margin <= 0:
            errors.append(ValidationMessage(
                message="Trailing stop wymaga ustawienia trailing_stop_margin > 0",
                param_id=param_id,
                details={'trailing_stop_margin': params.trailing_stop_margin}
            ))
    else:
        # Sprawdzenie nadmiarowych parametrów trailing stop - tylko jeśli były w pliku
        active_params = []
        if 'trailing_stop_price' in provided_params and params.trailing_stop_price != 0:
            active_params.append('trailing_stop_price')
        if 'trailing_stop_margin' in provided_params and params.trailing_stop_margin != 0:
            active_params.append('trailing_stop_margin')
        if 'trailing_stop_time' in provided_params and params.trailing_stop_time != 0:
            active_params.append('trailing_stop_time')
            
        if active_params:
            warnings.append(ValidationMessage(
                message="Parametry trailing stop są ignorowane gdy trailing_enabled=False",
                param_id=param_id,
                details={'ignored_params': active_params}
            ))
    
    # Sprawdzenie trailing buy
    if is_enabled(params.trailing_buy_enabled):
        if params.trailing_buy_threshold <= 0:
            errors.append(ValidationMessage(
                message="Trailing buy wymaga ustawienia trailing_buy_threshold > 0",
                param_id=param_id,
                details={'trailing_buy_threshold': params.trailing_buy_threshold}
            ))
            
        if params.trailing_buy_time_in_min <= 0:
            errors.append(ValidationMessage(
                message="Trailing buy wymaga ustawienia trailing_buy_time_in_min > 0",
                param_id=param_id,
                details={'trailing_buy_time_in_min': params.trailing_buy_time_in_min}
            ))
    else:
        # Sprawdzenie nadmiarowych parametrów trailing buy - tylko jeśli były w pliku
        active_params = []
        if 'trailing_buy_threshold' in provided_params and params.trailing_buy_threshold != 0:
            active_params.append('trailing_buy_threshold')
        if 'trailing_buy_time_in_min' in provided_params and params.trailing_buy_time_in_min != 0:
            active_params.append('trailing_buy_time_in_min')
            
        if active_params:
            warnings.append(ValidationMessage(
                message="Parametry trailing buy są ignorowane gdy trailing_buy_enabled=False",
                param_id=param_id,
                details={'ignored_params': active_params}
            ))
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_stop_loss_parameters(params: TradingParameters,
                                param_id: str) -> ValidationResult:
    """
    Walidacja parametrów związanych ze stop loss.
    Sprawdza zależności i konflikty między parametrami.
    """
    errors = []
    warnings = []
    
    if is_enabled(params.stop_loss_enabled):
        if params.stop_loss_threshold >= 0:
            errors.append(ValidationMessage(
                message="Stop loss wymaga ustawienia ujemnego stop_loss_threshold (np. -3.0 oznacza aktywację stop loss przy spadku o 3%)",
                param_id=param_id,
                details={'stop_loss_threshold': params.stop_loss_threshold}
            ))
            
        # Dodana walidacja delay time
        if params.stop_loss_delay_time < 0:
            errors.append(ValidationMessage(
                message="Stop loss delay time nie może być ujemny",
                param_id=param_id,
                details={'stop_loss_delay_time': params.stop_loss_delay_time}
            ))
    else:
        # Sprawdzamy czy parametry SL nie są ustawione mimo wyłączonego SL
        active_params = []
        if params.stop_loss_threshold != 0:
            active_params.append('stop_loss_threshold')
        if params.stop_loss_delay_time != 0:
            active_params.append('stop_loss_delay_time')
            
        if active_params:
            warnings.append(ValidationMessage(
                message="Parametry stop loss są ignorowane gdy stop_loss_enabled=False",
                param_id=param_id,
                details={'ignored_params': active_params}
            ))
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_pump_detection_parameters(params: TradingParameters,
                                     param_id: str) -> ValidationResult:
    """
    Walidacja parametrów związanych z wykrywaniem pump.
    """
    errors = []
    
    if is_enabled(params.pump_detection_enabled):
        if params.pump_detection_threshold <= 0:
            errors.append(ValidationMessage(
                message="Pump detection wymaga ustawienia pump_detection_threshold > 0",
                param_id=param_id,
                details={'pump_detection_threshold': params.pump_detection_threshold}
            ))
            
        if params.pump_detection_disabled_time <= 0:
            errors.append(ValidationMessage(
                message="Pump detection wymaga ustawienia pump_detection_disabled_time > 0",
                param_id=param_id,
                details={'pump_detection_disabled_time': params.pump_detection_disabled_time}
            ))
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=[]
    )

def validate_fronttest_parameters(
    params: TradingParameters,
    param_id: str,
    provided_params: Optional[Set[str]] = None
) -> ValidationResult:
    """
    Walidacja parametrów pod kątem zgodności z trybem fronttest.
    
    Args:
        params: Obiekt z parametrami
        param_id: Identyfikator kombinacji parametrów
        provided_params: Opcjonalny zestaw parametrów faktycznie dostarczonych w pliku
    """
    errors = []
    warnings = []
    
    # Jeśli nie podano provided_params, używamy pustego zbioru
    if provided_params is None:
        provided_params = set()
    
    # Sprawdzenie czy wymagane parametry są pojedynczymi wartościami (nie zakresami)
    for param_name in FRONTTEST_SINGLE_VALUE_PARAMS:
        # Sprawdzamy czy parametr jest faktycznie przekazany
        if hasattr(params, param_name):
            param_value = getattr(params, param_name)
            # W fronttestingu parametry muszą być pojedynczymi wartościami (nie zakresami)
            if isinstance(param_value, (list, tuple, dict)):
                errors.append(ValidationMessage(
                    message=f"Parametr {param_name} musi być pojedynczą wartością w trybie fronttest",
                    param_id=param_id,
                    details={'param': param_name, 'value': param_value}
                ))
    
    # Sprawdzenie wymaganych włączonych parametrów
    for param_name in FRONTTEST_REQUIRED_ENABLED:
        if hasattr(params, param_name) and not is_enabled(getattr(params, param_name)):
            errors.append(ValidationMessage(
                message=f"Parametr {param_name} musi być włączony w trybie fronttest",
                param_id=param_id,
                details={'param': param_name}
            ))
            
    # Sprawdzenie wymaganych wyłączonych parametrów
    for param_name in FRONTTEST_REQUIRED_DISABLED:
        if hasattr(params, param_name) and is_enabled(getattr(params, param_name)):
            errors.append(ValidationMessage(
                message=f"Parametr {param_name} musi być wyłączony w trybie fronttest",
                param_id=param_id,
                details={'param': param_name}
            ))
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_parameters(
    params: TradingParameters, 
    market_data: Optional[Dict] = None,
    param_id: str = "unknown",
    provided_params: Optional[Set[str]] = None,
    mode: str = BACKTEST_MODE  # Nowy parametr mode dodany na końcu z wartością domyślną
) -> ValidationResult:
    """
    Główna funkcja walidująca parametry.
    Wywołuje poszczególne funkcje walidacji i agreguje wyniki.
    
    Args:
        params: Obiekt z parametrami
        market_data: Opcjonalne dane rynkowe do walidacji
        param_id: Identyfikator kombinacji parametrów
        provided_params: Opcjonalny zestaw parametrów faktycznie dostarczonych w pliku
        mode: Tryb walidacji (backtest/fronttest), domyślnie 'backtest'
    """
    # Lista walidatorów zawsze zawiera podstawowe walidacje
    validators = [
        (validate_btc_parameters, [params, market_data, param_id]),
        (validate_trailing_parameters, [params, param_id, provided_params]),
        (validate_stop_loss_parameters, [params, param_id]),
        (validate_pump_detection_parameters, [params, param_id])
    ]
    
    # Dodajemy specyficzną walidację dla trybu fronttest
    if mode == FRONTTEST_MODE:
        validators.append((validate_fronttest_parameters, [params, param_id, provided_params]))
    
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    try:
        for validator, args in validators:
            validator_result = validator(*args)
            result = result.merge(validator_result)
            
            # Przerywamy tylko przy błędach, nie przy ostrzeżeniach
            if not validator_result.is_valid and validator_result.errors:
                break
                
    except Exception as e:
        logger.error(f"Błąd podczas walidacji parametrów: {str(e)}")
        return ValidationResult(
            is_valid=False,
            errors=[ValidationMessage(
                message=f"Błąd walidacji: {str(e)}",
                param_id=param_id,
                details={'error': str(e)}
            )],
            warnings=[]
        )
    
    return result

# Przy inicjalizacji modułu tworzymy wymagane katalogi
for dir_path in [PARAMETRY_DIR, CSV_DIR, WYNIKI_DIR, LOGI_DIR, 
                 WYNIKI_BACKTEST_DIR, WYNIKI_ANALIZA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)