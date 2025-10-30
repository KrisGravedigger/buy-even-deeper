#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Konfigurator parametrów strategii tradingowej z archiwizacją poprzednich konfiguracji.
"""

import json
from pathlib import Path
from datetime import datetime
import shutil
from typing import Dict, Optional, Tuple, Union, Set
import logging
import sys

# Import modułu validation z odpowiedniej ścieżki
try:
    # Absolutny import (dla modułu używanego jako część pakietu)
    from btd.runner_parameters.validation import BACKTEST_MODE, FRONTTEST_MODE, VALID_MODES, FRONTTEST_SINGLE_VALUE_PARAMS
except ImportError:
    try:
        # Relatywny import (dla bezpośredniego uruchomienia skryptu)
        from runner_parameters.validation import BACKTEST_MODE, FRONTTEST_MODE, VALID_MODES, FRONTTEST_SINGLE_VALUE_PARAMS
    except ImportError:
        # Fallback - definiujemy stałe lokalnie (kompatybilność wsteczna)
        BACKTEST_MODE = 'backtest'
        FRONTTEST_MODE = 'fronttest'
        VALID_MODES = {BACKTEST_MODE, FRONTTEST_MODE}
        FRONTTEST_SINGLE_VALUE_PARAMS = set()
        print("Ostrzeżenie: Nie można zaimportować modułu validation. Używanie wartości domyślnych.")

# Tworzenie katalogu logów jeśli nie istnieje
logs_dir = Path('logi')
logs_dir.mkdir(parents=True, exist_ok=True)

# Konfiguracja głównego loggera
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'parameter_configurator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ParameterManager:
    def __init__(self):
        """Inicjalizacja menedżera parametrów"""
        # Definicja ścieżek
        self.base_dir = Path('parametry')
        self.archive_dir = self.base_dir / 'archiwum'
        self.logs_dir = Path('logi')
        
        # Tworzenie katalogów jeśli nie istnieją
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.params_file = self.base_dir / 'trading_parameters.json'
        
        # Dodanie handlera do pliku dla bieżącej sesji
        session_log_file = self.logs_dir / f'parameter_configurator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(session_log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
       
    def save_parameters(self, param_ranges: Dict, mode: str = BACKTEST_MODE) -> None:
        """
        Zapisuje parametry do pliku z archiwizacją poprzedniej wersji.
        
        Args:
            param_ranges: Słownik z parametrami do zapisania
            mode: Tryb konfiguracji (backtest/fronttest)
        """
        try:
            # Upewniamy się, że katalog archiwum istnieje
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            
            if self.params_file.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Dodajemy prefiks do nazwy pliku w trybie fronttest
                prefix = "fronttest_" if mode == FRONTTEST_MODE else ""
                archive_file = self.archive_dir / f"{prefix}trading_parameters_{timestamp}.json"
                
                # Kopiowanie bez używania .absolute()
                shutil.copy2(self.params_file, archive_file)
                logger.info(f"Poprzednia konfiguracja zarchiwizowana jako: {archive_file}")
        
            # Zaokrąglamy wszystkie wartości numeryczne odpowiednio do ich typu
            rounded_params = {}
            for param_name, param_info in param_ranges.items():
                rounded_info = param_info.copy()
                is_integer = param_info.get('numeric_type') == 'int'
            
                if 'value' in param_info and isinstance(param_info['value'], (int, float)):
                    if is_integer:
                        rounded_info['value'] = int(round(float(param_info['value'])))
                    else:
                        rounded_info['value'] = round(float(param_info['value']), 3)
                elif 'range' in param_info:
                    if is_integer:
                        rounded_info['range'] = [int(round(float(x))) for x in param_info['range']]
                    else:
                        rounded_info['range'] = [round(float(x), 3) for x in param_info['range']]
                rounded_params[param_name] = rounded_info
            
            # Dodajemy informację o trybie
            rounded_params['__mode__'] = {
                'value': mode,
                'description': 'Tryb konfiguracji (backtest/fronttest)',
                'type': 'string'
            }
            
            # Zapisujemy nową konfigurację
            output_filename = 'trading_parameters.json'
            if mode == FRONTTEST_MODE:
                output_filename = f"fronttest_{output_filename}"
                
            output_file = self.base_dir / output_filename
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(rounded_params, f, indent=4, ensure_ascii=False)
            
            # Aktualizujemy ścieżkę parametrów tylko dla trybu backtest
            # W trybie fronttest zachowujemy standardową ścieżkę dla kompatybilności
            if mode == BACKTEST_MODE:
                self.params_file = output_file
                
            logger.info(f"Zapisano nową konfigurację do: {output_file} w trybie: {mode}")
                
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania parametrów: {str(e)}")
            raise
           
    def load_parameters(self) -> Tuple[Optional[Dict], str]:
        """
        Wczytuje parametry z pliku.
        
        Returns:
            Tuple[Dict, str]: Słownik z parametrami i tryb lub (None, BACKTEST_MODE) jeśli nie znaleziono pliku
        """
        mode = BACKTEST_MODE  # Domyślny tryb
        params = None
        
        if self.params_file.exists():
            with open(self.params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
                
                # Wyciągamy informację o trybie jeśli istnieje
                if '__mode__' in params:
                    mode_info = params.pop('__mode__')
                    mode = mode_info.get('value', BACKTEST_MODE)
                    
                    # Sprawdzamy czy tryb jest poprawny
                    if mode not in VALID_MODES:
                        logger.warning(f"Nieznany tryb: {mode}, używam domyślnego: {BACKTEST_MODE}")
                        mode = BACKTEST_MODE
                
        return params, mode

    def get_parameter_history(self) -> list:
        """
        Zwraca listę poprzednich konfiguracji.
        
        Returns:
            list: Lista plików z archiwum posortowana po dacie
        """
        return sorted(self.archive_dir.glob('*trading_parameters_*.json'), 
                     key=lambda x: x.stat().st_mtime, reverse=True)

def get_parameter_value(param_name: str, param_info: Dict, 
                       existing_value: Optional[Dict] = None,
                       confirm_mode: bool = False,
                       mode: str = BACKTEST_MODE) -> Optional[Dict]:
    """
    Pobiera od użytkownika wartość parametru.
    
    Args:
        param_name: Nazwa parametru
        param_info: Informacje o parametrze
        existing_value: Obecna wartość parametru
        confirm_mode: Czy jesteśmy w trybie potwierdzania istniejących wartości
        mode: Tryb konfiguracji (backtest/fronttest)
        
    Returns:
        Dict: Słownik z konfiguracją parametru lub None
    """
    print(f"\nParametr: {param_name}")
    print(f"Opis: {param_info['description']}")
    
    if existing_value is not None and confirm_mode:
        print(f"Obecne wartości: {existing_value}")
        confirm = input("Czy chcesz zachować te wartości? (t/n): ").lower()
        if confirm == 't':
            # W trybie fronttest sprawdzamy, czy parametr ma stałą wartość
            if mode == FRONTTEST_MODE and 'range' in existing_value:
                print(f"UWAGA: W trybie {FRONTTEST_MODE} wszystkie parametry muszą być stałymi wartościami.")
                print("Wykryto zakres wartości. Musisz zdefiniować stałą wartość dla tego parametru.")
            else:
                return existing_value
    
    while True:
        try:
            include = input("Czy chcesz konfigurować ten parametr? (t/n): ").lower() == 't'
            if not include:
                return None
            
            if param_info['type'] == 'boolean':
                value = input("Wprowadź wartość (t/n): ").lower() == 't'
                return {
                    'value': value,
                    'description': param_info['description'],
                    'type': 'boolean'
                }
            else:
                # W trybie fronttest wszystkie parametry muszą być stałymi wartościami
                force_constant = (mode == FRONTTEST_MODE)
                
                if force_constant:
                    print(f"W trybie {FRONTTEST_MODE} parametr {param_name} musi być stałą wartością.")
                    is_constant = True
                else:
                    # Pytamy czy ma być stała wartość czy zakres
                    is_constant = input("Czy chcesz ustawić stałą wartość (t) czy zakres (n)? ").lower() == 't'
                
                if is_constant:
                    value = float(input("Wprowadź wartość: "))
                    return {
                        'value': value,
                        'description': param_info['description'],
                        'type': 'numeric',
                        'numeric_type': 'int' if param_name in [
                            'max_open_orders_per_coin', 'max_open_orders', 'check_timeframe',
                            'next_buy_delay', 'stop_loss_delay_time', 'trailing_stop_time',
                            'pump_detection_disabled_time'
                        ] else 'float'
                    }
                else:
                    min_val = float(input("Minimalna wartość: "))
                    max_val = float(input("Maksymalna wartość: "))
                    step = float(input("Krok: "))
                    return {
                        'range': [min_val, max_val, step],
                        'description': param_info['description'],
                        'type': 'numeric',
                        'numeric_type': 'int' if param_name in [
                            'max_open_orders_per_coin', 'max_open_orders', 'check_timeframe',
                            'next_buy_delay', 'stop_loss_delay_time', 'trailing_stop_time',
                            'pump_detection_disabled_time'
                        ] else 'float'
                    }
                
        except ValueError:
            print("Nieprawidłowe wartości. Spróbuj ponownie.")

def collect_parameters(param_manager: ParameterManager) -> Dict:
    """
    Zbiera parametry od użytkownika z uwzględnieniem istniejących parametrów.
    
    Args:
        param_manager: Instancja ParameterManager
        
    Returns:
        Dict: Słownik z zebranymi parametrami
    """
    existing_params, mode = param_manager.load_parameters()
    param_ranges = {}
    confirm_mode = False
    
    # Wybór trybu konfiguracji (backtest/fronttest)
    print("\nWybierz tryb konfiguracji:")
    print(f"1) {BACKTEST_MODE.capitalize()} - testowanie wielu kombinacji parametrów na danych historycznych")
    print(f"2) {FRONTTEST_MODE.capitalize()} - symulacja przyszłych ruchów metodą Monte Carlo dla pojedynczego zestawu parametrów")
    
    mode_choice = input(f"Twój wybór (1-2, Enter dla {BACKTEST_MODE}): ")
    if mode_choice == "2":
        mode = FRONTTEST_MODE
        print(f"\nWybrany tryb: {FRONTTEST_MODE.upper()}")
        print("W tym trybie wszystkie parametry strategii muszą być pojedynczymi wartościami (nie zakresami).")
    else:
        mode = BACKTEST_MODE
        print(f"\nWybrany tryb: {BACKTEST_MODE.upper()}")
    
    if existing_params:
        print("\nZnaleziono zapisane parametry.")
        print("Wybierz opcję:")
        print("1) Przejrzyj i modyfikuj parametry")
        print("2) Zignoruj zapisane i skonfiguruj od nowa")
        print("3) Przejrzyj historię parametrów")
        
        choice = input("Twój wybór (1-3): ")
        
        if choice == "3":
            history = param_manager.get_parameter_history()
            if history:
                print("\nHistoria konfiguracji:")
                for i, file in enumerate(history, 1):
                    mod_time = datetime.fromtimestamp(file.stat().st_mtime)
                    print(f"{i}. {file.name} (zmodyfikowano: {mod_time})")
                    
                hist_choice = input("\nWybierz numer konfiguracji do wczytania (Enter aby pominąć): ")
                if hist_choice.isdigit() and 1 <= int(hist_choice) <= len(history):
                    with open(history[int(hist_choice)-1], 'r', encoding='utf-8') as f:
                        loaded_params = json.load(f)
                        
                        # Sprawdzamy czy w wczytanym pliku jest informacja o trybie
                        if '__mode__' in loaded_params:
                            mode_info = loaded_params.pop('__mode__')
                            loaded_mode = mode_info.get('value', BACKTEST_MODE)
                            
                            # Informujemy o trybie wczytanej konfiguracji
                            print(f"Wczytana konfiguracja jest w trybie: {loaded_mode.upper()}")
                            
                            # Sprawdzamy czy tryb wczytanej konfiguracji jest zgodny z wybranym
                            if loaded_mode != mode:
                                print(f"UWAGA: Wybrany tryb ({mode.upper()}) różni się od trybu wczytanej konfiguracji ({loaded_mode.upper()}).")
                                if input("Czy chcesz kontynuować? (t/n): ").lower() != 't':
                                    print("Anulowano wczytywanie konfiguracji.")
                                    existing_params = None
                                else:
                                    existing_params = loaded_params
                                    
                                    # Sprawdzamy czy wszystkie parametry są zgodne z wybranym trybem
                                    if mode == FRONTTEST_MODE:
                                        has_ranges = any('range' in param for param in existing_params.values())
                                        if has_ranges:
                                            print(f"UWAGA: Wczytana konfiguracja zawiera parametry z zakresami.")
                                            print(f"W trybie {FRONTTEST_MODE} wszystkie parametry muszą być stałymi wartościami.")
                                            print("Będziesz musiał zaktualizować parametry z zakresami.")
                            else:
                                existing_params = loaded_params
                                
                                # Sprawdzamy czy wszystkie parametry są zgodne z trybem
                                if mode == FRONTTEST_MODE:
                                    has_ranges = any('range' in param for param in existing_params.values())
                                    if has_ranges:
                                        print(f"UWAGA: Wczytana konfiguracja zawiera parametry z zakresami.")
                                        print(f"W trybie {FRONTTEST_MODE} wszystkie parametry muszą być stałymi wartościami.")
                                        print("Będziesz musiał zaktualizować parametry z zakresami.")
                        else:
                            # Brak informacji o trybie, zakładamy backtest
                            existing_params = loaded_params
                            print("Wczytana konfiguracja nie zawiera informacji o trybie, zakładam BACKTEST.")
                            
                            # Sprawdzamy czy wszystkie parametry są zgodne z wybranym trybem
                            if mode == FRONTTEST_MODE:
                                has_ranges = any('range' in param for param in existing_params.values())
                                if has_ranges:
                                    print(f"UWAGA: Wczytana konfiguracja zawiera parametry z zakresami.")
                                    print(f"W trybie {FRONTTEST_MODE} wszystkie parametry muszą być stałymi wartościami.")
                                    print("Będziesz musiał zaktualizować parametry z zakresami.")
                        
                        print("Wczytano parametry z archiwum")
            else:
                print("Brak historii konfiguracji")
                existing_params = None
        elif choice == "2":
            existing_params = None
        
        confirm_mode = (choice == "1")
    
    # Definicja parametrów i ich opisów
    param_definitions = {
        'check_timeframe': {
            'description': 'Określa zakres czasowy/świeczkę do analizy. Przykład: wartość 1 oznacza analiza 1-minutowych świeczek, '
                         'wartość 30 oznacza analiza 30-minutowych świeczek. Im większy przedział czasowy, tym większa szansa na '
                         'złapanie spadku, ale głównym celem jest łapanie krótkich spadków z szybkim odbiciem.',
            'type': 'numeric'
        },
        'percentage_buy_threshold': {
            'description': 'Określa procentowy spadek względem wybranego timeframe\'u. Należy dostosować wielkość spadku do '
                         'wybranego czasu - większy timeframe pozwala na większy spadek.',
            'type': 'numeric'
        },
        'max_allowed_usd': {
            'description': 'Maksymalny obrót jaki może wykonać aplikacja. Po przekroczeniu tej wartości aplikacja przestaje '
                         'dokonywać zakupów.',
            'type': 'numeric'
        },
        'add_to_limit_order': {
            'description': 'Zabezpieczenie przy zakupie - określa maksymalną różnicę procentową powyżej ceny zakupu. '
                         'Przy sprzedaży określa maksymalną różnicę poniżej ceny sprzedaży. Sugerowana wartość: 2%.',
            'type': 'numeric'
        },
        'sell_profit_target': {
            'description': 'Określa próg procentowy zysku, przy którym pozycja zostanie sprzedana w trybie zwykłej sprzedaży.',
            'type': 'numeric'
        },
        'trailing_enabled': {
            'description': 'Włącza/wyłącza funkcję trailing stop (podążający stop loss). Trailing śledzi wzrost ceny i '
                         'automatycznie podnosi poziom stop loss.',
            'type': 'boolean'
        },
        'trailing_stop_price': {
            'description': 'Określa próg procentowy zysku, przy którym zostanie aktywowany trailing stop.',
            'type': 'numeric'
        },
        'trailing_stop_margin': {
            'description': 'Określa o ile procent poniżej aktualnej ceny ma być ustawiony trailing stop loss. '
                         'Sugerowana minimalna wartość: 0.5%.',
            'type': 'numeric'
        },
        'trailing_stop_time': {
            'description': 'Czas w minutach, przez który cena musi utrzymać się powyżej progu trailing stop, '
                         'zanim zostanie on aktywowany.',
            'type': 'numeric'
        },
        'stop_loss_enabled': {
            'description': 'Włącza/wyłącza funkcję stop loss (automatyczna sprzedaż ze stratą w celu jej ograniczenia).',
            'type': 'boolean'
        },
        'stop_loss_threshold': {
            'description': 'Określa próg procentowy poniżej ceny zakupu, przy którym zostanie aktywowany stop loss.',
            'type': 'numeric'
        },
        'stop_loss_delay_time': {
            'description': 'Czas w minutach przed aktywacją stop loss. Przydatne przy większych spadkach/flash crashach - '
                         'chroni przed zbyt szybkim "wycięciem" przez stop loss. Używane głównie przy wysokich progach spadkowych.',
            'type': 'numeric'
        },
        'max_open_orders_per_coin': {
            'description': 'Maksymalna liczba jednocześnie otwartych pozycji dla jednej kryptowaluty.',
            'type': 'numeric'
        },
        'next_buy_delay': {
            'description': 'Minimalny czas w minutach, jaki musi upłynąć przed kolejnym zakupem tej samej kryptowaluty.',
            'type': 'numeric'
        },
        'next_buy_price_lower': {
            'description': 'O ile procent niżej od poprzedniego zakupu musi być cena, aby system mógł dokonać kolejnego '
                         'zakupu tej samej kryptowaluty (po upływie czasu next_buy_delay).',
            'type': 'numeric'
        },
        'pump_detection_enabled': {
            'description': 'Włącza/wyłącza funkcję wykrywania pump\'ów (gwałtownych wzrostów ceny) - '
                         'chroni przed zakupami na szczytach.',
            'type': 'boolean'
        },
        'pump_detection_threshold': {
            'description': 'Określa próg procentowy wzrostu ceny (w czasie określonym przez check_timeframe), '
                         'powyżej którego system uzna ruch za pump i wstrzyma zakupy.',
            'type': 'numeric'
        },
        'pump_detection_disabled_time': {
            'description': 'Minimalny czas w minutach, na jaki zostanie wyłączone kupowanie danej kryptowaluty po '
                         'wykryciu pump\'a. Jeśli wzrost będzie kontynuowany, czas może się wydłużyć.',
            'type': 'numeric'
        },
        'follow_btc_price': {
            'description': 'Włącza/wyłącza funkcję śledzenia ceny BTC. Gdy włączona, system wstrzyma zakupy altcoinów '
                         'jeśli BTC spadnie mocniej i szybciej niż dany altcoin - chroni przed silnymi spadkami rynku.',
            'type': 'boolean'
        },
        'max_open_orders': {
            'description': 'Maksymalna liczba wszystkich jednocześnie otwartych pozycji dla jednej strategii.',
            'type': 'numeric'
        },
        'stop_loss_disable_buy': {
            'description': 'Włącza/wyłącza funkcję modyfikacji kolejnych zakupów po aktywacji stop loss dla danej kryptowaluty.',
            'type': 'boolean'
        },
        'stop_loss_disable_buy_all': {
            'description': 'Włącza/wyłącza funkcję czasowego wstrzymania zakupów wszystkich kryptowalut po aktywacji '
                         'stop loss - funkcja chroniąca przed silnymi spadkami rynku.',
            'type': 'boolean'
        },
        'stop_loss_next_buy_lower': {
            'description': 'Określa o ile procent niżej od ceny sprzedaży po stop loss system będzie próbował kupić '
                         'ponownie tę samą kryptowalutę.',
            'type': 'numeric'
        },
        'stop_loss_no_buy_delay': {
            'description': 'Czas w minutach, który musi upłynąć od aktywacji stop loss zanim system sprawdzi cenę i '
                         'spróbuje dokonać ponownego zakupu. W przypadku włączonej opcji stop_loss_disable_buy_all '
                         'określa czas wstrzymania wszystkich zakupów.',
            'type': 'numeric'
        },
        'trailing_buy_enabled': {  
            'description': 'Włącza/wyłącza funkcję trailing buy (oczekiwanie na dalszy spadek ceny przed zakupem). '
                         'Wartość 1.0 oznacza włączone, 0.0 oznacza wyłączone.',
            'type': 'boolean'
        },
        'trailing_buy_threshold': {
            'description': 'Określa próg procentowy dla funkcji trailing buy - o ile procent cena musi spaść '
                         'dodatkowo od punktu aktywacji, żeby wykonać zakup.',
            'type': 'numeric',
            'numeric_type': 'float'
        },
        'trailing_buy_time_in_min': {
            'description': 'Określa maksymalny czas w minutach, przez który bot będzie czekał na dodatkowy spadek '
                         'przy aktywnej funkcji trailing buy.',
            'type': 'numeric',
            'numeric_type': 'int'
        },
        'follow_btc_threshold': {
            'description': 'Określa próg spadku ceny BTC w procentach. Jeśli BTC spadnie o więcej niż zadany próg, '
                        'system wstrzyma zakupy altcoinów - chroni przed silnymi spadkami rynku.',
            'type': 'numeric'
        },
        'follow_btc_block_time': {  
            'description': 'Czas w minutach, przez który system będzie wstrzymywał zakupy po wykryciu silnego spadku BTC.',
            'type': 'numeric'
        }
    }
    
    # W trybie fronttest musimy upewnić się, że stop_loss_enabled jest włączone
    if mode == FRONTTEST_MODE:
        print("\nUWAGA: W trybie FRONTTEST stop loss musi być włączony.")
        print("W trybie FRONTTEST wszystkie parametry muszą mieć stałe wartości (nie zakresy).")
    
    for param, param_info in param_definitions.items():
        existing_value = existing_params.get(param) if existing_params else None
        
        value = get_parameter_value(param, param_info, existing_value, confirm_mode, mode)
        if value is not None:
            param_ranges[param] = value
    
    # Sprawdzenie czy wszystkie parametry w trybie fronttest mają stałe wartości
    if mode == FRONTTEST_MODE:
        params_with_ranges = [param for param, info in param_ranges.items() if 'range' in info]
        if params_with_ranges:
            print("\nBŁĄD: W trybie FRONTTEST następujące parametry mają zakresy zamiast stałych wartości:")
            for param in params_with_ranges:
                print(f" - {param}")
            print("Musisz poprawić te parametry przed zapisaniem.")
            
            if input("Czy chcesz teraz poprawić te parametry? (t/n): ").lower() == 't':
                for param in params_with_ranges:
                    print(f"\nNaprawianie parametru: {param}")
                    param_info = param_definitions[param]
                    value = get_parameter_value(param, param_info, None, False, mode)
                    if value is not None:
                        param_ranges[param] = value
            else:
                print("Anulowano zapisywanie parametrów.")
                return param_ranges
    
    if input("\nCzy chcesz zapisać wprowadzone parametry? (t/n): ").lower() == 't':
        param_manager.save_parameters(param_ranges, mode)
    
    return param_ranges

def main():
    """Główna funkcja programu"""
    try:
        logger.info("Uruchamianie konfiguratora parametrów")
        param_manager = ParameterManager()
        params = collect_parameters(param_manager)
        logger.info("Konfiguracja zakończona pomyślnie")
        
    except Exception as e:
        logger.error(f"Błąd w konfiguratorze: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Błąd: {str(e)}")
        sys.exit(1)