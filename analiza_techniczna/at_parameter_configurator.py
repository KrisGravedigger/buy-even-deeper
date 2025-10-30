"""
Konfigurator parametrów analizy technicznej.
Umożliwia tworzenie, zapisywanie i wczytywanie konfiguracji dla orkiestratora analizy technicznej.
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

class ATParameterConfigurator:
    """
    Klasa do zarządzania parametrami analizy technicznej.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Inicjalizacja konfiguratora parametrów analizy technicznej.
        
        Args:
            config_file: Ścieżka do pliku konfiguracyjnego (jeśli None, używa domyślnej lokalizacji)
        """
        self.logger = self._setup_logger()
        
        # Ustalenie ścieżki do pliku konfiguracyjnego
        if config_file is None:
            config_dir = Path("configs")
            config_dir.mkdir(exist_ok=True, parents=True)
            self.config_file = config_dir / "at_config.json"
        else:
            self.config_file = Path(config_file)
        
        # Inicjalizacja konfiguracji - najpierw domyślne, potem z pliku jeśli istnieje
        self.config = self.get_default_config()
        
        if self.config_file.exists():
            self.logger.info(f"Wczytywanie konfiguracji z pliku: {self.config_file}")
            loaded_config = self.load_config()
            
            # Aktualizacja konfiguracji z pliku
            self._update_nested_dict(self.config, loaded_config)
            
    def _setup_logger(self) -> logging.Logger:
        """
        Konfiguracja loggera.
        
        Returns:
            logging.Logger: Skonfigurowany logger
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Ustawienie formatowania
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # Handler konsoli
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _update_nested_dict(self, base_dict: Dict, update_dict: Dict) -> None:
        """
        Aktualizuje słownik zagnieżdżony, zachowując strukturę.
        
        Args:
            base_dict: Słownik bazowy do aktualizacji
            update_dict: Słownik z nowymi wartościami
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                # Rekurencyjne aktualizowanie zagnieżdżonego słownika
                self._update_nested_dict(base_dict[key], value)
            else:
                # Bezpośrednia aktualizacja wartości
                base_dict[key] = value
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Zwraca domyślną konfigurację analizy technicznej.
        
        Returns:
            Dict[str, Any]: Domyślna konfiguracja
        """
        return {
            "technical_indicators": {
                "rsi": {
                    "enabled": True,
                    "weight": 25,
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30
                },
                "ema": {
                    "enabled": True,
                    "weight": 20,
                    "period": 20
                },
                "obv": {
                    "enabled": True,
                    "weight": 15
                }
            },
            "trend_detection": {
                "enabled": True,
                "weight": 25,
                "method": "ema",
                "period": 20
            },
            "support_resistance": {
                "enabled": True,
                "weight": 15,
                "method": "peaks",
                "sensitivity": 0.05
            },
            "advanced_options": {
                "main_token_influence": {
                    "weight": 70  # Waga wpływu AT na główny token (0-100)
                },
                "price_reaction_to_indicators": {
                    "rsi_overbought_effect": -0.5,
                    "rsi_oversold_effect": 0.5,
                    "trend_following_effect": 0.3,
                    "support_bounce_effect": 0.4,
                    "resistance_bounce_effect": -0.4
                },
                "volatility_adjustment": {
                    "trend_strength_effect": -0.2,
                    "near_support_resistance_effect": 0.3
                }
            },
            "indicator_weights": {
                "main_token": {
                    "rsi": 30,          # Waga RSI (0-100)
                    "trend": 30,        # Waga trendu (0-100)
                    "support_resistance": 20,  # Waga poziomów wsparcia/oporu (0-100)
                    "volume": 20        # Waga wskaźników opartych na wolumenie (0-100)
                },
                "btc": {
                    "rsi": 30,          # Waga RSI (0-100)
                    "trend": 40,        # Waga trendu (0-100)
                    "support_resistance": 20,  # Waga poziomów wsparcia/oporu (0-100)
                    "volume": 10        # Waga wskaźników opartych na wolumenie (0-100)
                }
            },
            "btc_influence": {
                "enabled": True,
                "weight": 30,  # Waga wpływu BTC (0-100)
                "use_dynamic_correlation": True  # Czy używać dynamicznej korelacji zamiast progu
            },
            "base_scenario": "normal",
            "version": "1.0",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def save_config(self, file_path: Optional[str] = None) -> str:
        """
        Zapisuje aktualną konfigurację do pliku JSON.
        
        Args:
            file_path: Opcjonalna ścieżka do pliku (jeśli None, użyje domyślnej)
            
        Returns:
            str: Ścieżka do zapisanego pliku
        """
        save_path = Path(file_path) if file_path else self.config_file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Aktualizacja daty ostatniej modyfikacji
        self.config["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4)
        
        self.logger.info(f"Zapisano konfigurację do pliku: {save_path}")
        return str(save_path)
    
    def load_config(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Wczytuje konfigurację z pliku JSON.
        
        Args:
            file_path: Opcjonalna ścieżka do pliku (jeśli None, użyje domyślnej)
            
        Returns:
            Dict[str, Any]: Wczytana konfiguracja
        """
        load_path = Path(file_path) if file_path else self.config_file
        
        if not load_path.exists():
            self.logger.warning(f"Plik konfiguracyjny nie istnieje: {load_path}. Używam domyślnej konfiguracji.")
            return {}
        
        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.logger.info(f"Wczytano konfigurację z pliku: {load_path}")
            return config
        except Exception as e:
            self.logger.error(f"Błąd wczytywania konfiguracji z pliku {load_path}: {e}")
            return {}
    
    def get_config(self) -> Dict[str, Any]:
        """
        Zwraca aktualną konfigurację.
        
        Returns:
            Dict[str, Any]: Aktualna konfiguracja
        """
        return self.config
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Ustawia nową konfigurację.
        
        Args:
            config: Nowa konfiguracja
        """
        self.config = config
        self.logger.info("Zaktualizowano konfigurację")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Aktualizuje aktualną konfigurację.
        
        Args:
            updates: Słownik z aktualizacjami
        """
        self._update_nested_dict(self.config, updates)
        self.logger.info("Zaktualizowano konfigurację")
    
    def create_template(self, file_path: Optional[str] = None) -> str:
        """
        Tworzy szablon pliku konfiguracyjnego z domyślnymi wartościami i komentarzami.
        
        Args:
            file_path: Opcjonalna ścieżka do pliku (jeśli None, użyje 'at_config_template.json')
            
        Returns:
            str: Ścieżka do zapisanego szablonu
        """
        if file_path is None:
            template_path = self.config_file.parent / "at_config_template.json"
        else:
            template_path = Path(file_path)
        
        template_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Dodanie komentarzy do pliku szablonu
        template_with_comments = {
            "_comment": "Szablon konfiguracji analizy technicznej. Możesz dostosować wartości wg potrzeb.",
            "technical_indicators": {
                "_comment": "Konfiguracja wskaźników technicznych",
                "rsi": {
                    "enabled": True,
                    "weight": 25,
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30,
                    "_comment": "RSI - Relative Strength Index. Weight określa wagę wskaźnika w analizie."
                },
                "ema": {
                    "enabled": True,
                    "weight": 20,
                    "period": 20,
                    "_comment": "EMA - Exponential Moving Average. Period określa okres średniej."
                },
                "obv": {
                    "enabled": True,
                    "weight": 15,
                    "_comment": "OBV - On-Balance Volume. Wskaźnik łączący cenę i wolumen."
                }
            },
            "trend_detection": {
                "enabled": True,
                "weight": 25,
                "method": "ema",
                "period": 20,
                "_comment": "Wykrywanie trendu. Method może być 'ema' lub 'sma'."
            },
            "support_resistance": {
                "enabled": True,
                "weight": 15,
                "method": "peaks",
                "sensitivity": 0.05,
                "_comment": "Wykrywanie poziomów wsparcia/oporu. Method może być 'peaks' lub 'fractals'."
            },
            "advanced_options": {
                "_comment": "Zaawansowane opcje wpływu analizy technicznej na symulację",
                "price_reaction_to_indicators": {
                    "rsi_overbought_effect": -0.5,
                    "rsi_oversold_effect": 0.5,
                    "trend_following_effect": 0.3,
                    "support_bounce_effect": 0.4,
                    "resistance_bounce_effect": -0.4,
                    "_comment": "Wartości określają wpływ wskaźników na średnią zwrotu. Zakres: -1.0 do 1.0."
                },
                "volatility_adjustment": {
                    "trend_strength_effect": -0.2,
                    "near_support_resistance_effect": 0.3,
                    "_comment": "Wartości określają wpływ wskaźników na zmienność. Zakres: -1.0 do 1.0."
                }
            },
            "base_scenario": "normal",
            "_comment_base_scenario": "Bazowy scenariusz: 'normal', 'bootstrap' lub 'stress'",
            "version": "1.0",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(template_path, 'w', encoding='utf-8') as f:
            json.dump(template_with_comments, f, indent=4)
        
        self.logger.info(f"Utworzono szablon konfiguracji: {template_path}")
        return str(template_path)
    
    def validate_config(self) -> bool:
        """
        Sprawdza poprawność konfiguracji.
        
        Returns:
            bool: True jeśli konfiguracja jest poprawna, False w przeciwnym razie
        """
        try:
            # Sprawdzenie podstawowych elementów konfiguracji
            required_sections = ['technical_indicators', 'trend_detection', 'support_resistance', 'base_scenario']
            for section in required_sections:
                if section not in self.config:
                    self.logger.error(f"Brak wymaganej sekcji konfiguracji: {section}")
                    return False
            
            # Sprawdzenie wskaźników
            indicators = self.config['technical_indicators']
            required_indicators = ['rsi', 'ema', 'obv']
            for indicator in required_indicators:
                if indicator not in indicators:
                    self.logger.error(f"Brak wymaganego wskaźnika: {indicator}")
                    return False
                
                if 'enabled' not in indicators[indicator]:
                    self.logger.error(f"Brak parametru 'enabled' dla wskaźnika: {indicator}")
                    return False
                
                if 'weight' not in indicators[indicator]:
                    self.logger.error(f"Brak parametru 'weight' dla wskaźnika: {indicator}")
                    return False
                
                # Sprawdzenie typów danych
                if not isinstance(indicators[indicator]['enabled'], bool):
                    self.logger.error(f"Parametr 'enabled' dla wskaźnika {indicator} musi być typu bool")
                    return False
                
                if not isinstance(indicators[indicator]['weight'], (int, float)):
                    self.logger.error(f"Parametr 'weight' dla wskaźnika {indicator} musi być typu numerycznego")
                    return False
            
            # Sprawdzenie trendu
            trend_detection = self.config['trend_detection']
            if not isinstance(trend_detection.get('enabled'), bool):
                self.logger.error("Parametr 'enabled' dla trend_detection musi być typu bool")
                return False
            
            if 'method' in trend_detection and trend_detection['method'] not in ['ema', 'sma', 'linear']:
                self.logger.error(f"Nieprawidłowa metoda wykrywania trendu: {trend_detection['method']}. Dozwolone: ema, sma, linear")
                return False
            
            # Sprawdzenie wsparcia/oporu
            support_resistance = self.config['support_resistance']
            if not isinstance(support_resistance.get('enabled'), bool):
                self.logger.error("Parametr 'enabled' dla support_resistance musi być typu bool")
                return False
            
            if 'method' in support_resistance and support_resistance['method'] not in ['peaks', 'fractals']:
                self.logger.error(f"Nieprawidłowa metoda wsparcia/oporu: {support_resistance['method']}. Dozwolone: peaks, fractals")
                return False
            
            # Sprawdzenie bazowego scenariusza
            valid_scenarios = ['normal', 'bootstrap', 'stress']
            if self.config['base_scenario'] not in valid_scenarios:
                self.logger.error(f"Nieprawidłowy bazowy scenariusz: {self.config['base_scenario']}. Dozwolone wartości: {valid_scenarios}")
                return False
            
            # Sprawdzenie advanced_options jeśli istnieje
            if 'advanced_options' in self.config:
                adv_options = self.config['advanced_options']
                
                # Sprawdzenie main_token_influence
                if 'main_token_influence' in adv_options:
                    main_token = adv_options['main_token_influence']
                    if 'weight' in main_token and not isinstance(main_token['weight'], (int, float)):
                        self.logger.error("Parametr 'weight' dla main_token_influence musi być typu numerycznego")
                        return False
                
                # Sprawdzenie price_reaction_to_indicators
                if 'price_reaction_to_indicators' in adv_options:
                    reactions = adv_options['price_reaction_to_indicators']
                    for key, value in reactions.items():
                        if not isinstance(value, (int, float)):
                            self.logger.error(f"Wartość '{key}' w price_reaction_to_indicators musi być typu numerycznego")
                            return False
            
            # Sprawdzenie btc_influence jeśli istnieje
            if 'btc_influence' in self.config:
                btc_inf = self.config['btc_influence']
                if 'enabled' in btc_inf and not isinstance(btc_inf['enabled'], bool):
                    self.logger.error("Parametr 'enabled' dla btc_influence musi być typu bool")
                    return False
                
                if 'weight' in btc_inf and not isinstance(btc_inf['weight'], (int, float)):
                    self.logger.error("Parametr 'weight' dla btc_influence musi być typu numerycznego")
                    return False
                
                if 'use_dynamic_correlation' in btc_inf and not isinstance(btc_inf['use_dynamic_correlation'], bool):
                    self.logger.error("Parametr 'use_dynamic_correlation' dla btc_influence musi być typu bool")
                    return False
            
            # Sprawdzenie indicator_weights jeśli istnieje
            if 'indicator_weights' in self.config:
                weights = self.config['indicator_weights']
                
                for token_type in ['main_token', 'btc']:
                    if token_type in weights:
                        token_weights = weights[token_type]
                        for key, value in token_weights.items():
                            if not isinstance(value, (int, float)):
                                self.logger.error(f"Waga '{key}' dla {token_type} musi być typu numerycznego")
                                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Błąd walidacji konfiguracji: {e}")
            return False


def main():
    """Główna funkcja demonstrująca użycie konfiguratora parametrów"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Konfigurator parametrów analizy technicznej')
    parser.add_argument('--create-template', '-t', action='store_true', help='Utwórz szablon konfiguracji')
    parser.add_argument('--output', '-o', help='Ścieżka wyjściowa dla szablonu lub zapisanych parametrów')
    parser.add_argument('--load', '-l', help='Wczytaj konfigurację z pliku')
    parser.add_argument('--validate', '-v', action='store_true', help='Sprawdź poprawność konfiguracji')
    
    args = parser.parse_args()
    
    # Inicjalizacja konfiguratora
    configurator = ATParameterConfigurator(
        config_file=args.load if args.load else None
    )
    
    # Tworzenie szablonu jeśli wymagane
    if args.create_template:
        template_path = configurator.create_template(args.output)
        print(f"Utworzono szablon konfiguracji: {template_path}")
        return
    
    # Walidacja konfiguracji jeśli wymagana
    if args.validate:
        is_valid = configurator.validate_config()
        if is_valid:
            print("Konfiguracja jest poprawna.")
        else:
            print("Konfiguracja jest niepoprawna.")
            return
    
    # Zapis konfiguracji jeśli podano ścieżkę wyjściową
    if args.output and not args.create_template:
        save_path = configurator.save_config(args.output)
        print(f"Zapisano konfigurację do pliku: {save_path}")


if __name__ == "__main__":
    main()