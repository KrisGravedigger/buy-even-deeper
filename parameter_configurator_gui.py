#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Konfigurator parametrów strategii tradingowej z archiwizacją poprzednich konfiguracji - WERSJA GUI z Tkinter.
"""

import json
from pathlib import Path
from datetime import datetime
import shutil
from typing import Dict, Optional, Tuple, Union, Set, List
import logging
import sys
import tkinter as tk
from tkinter import ttk  # Używamy ładniejszych widgetów ttk
from tkinter import messagebox
from tkinter import simpledialog # Do prostych okien dialogowych
from tkinter import filedialog # Do wyboru pliku z archiwum

# --- Twoja istniejąca logika (ParameterManager, stałe, logging) ---
# Można ją umieścić w osobnym pliku np. `parameter_logic.py` i importować
# lub zostawić tutaj, ale oddzielić od kodu GUI.

# Zakładam, że kod ParameterManager i definicje są dostępne
# (poniżej wklejam je dla kompletności, ale można je zaimportować)

# Import modułu validation z odpowiedniej ścieżki
try:
    from btd.runner_parameters.validation import BACKTEST_MODE, FRONTTEST_MODE, VALID_MODES, FRONTTEST_SINGLE_VALUE_PARAMS
except ImportError:
    try:
        from runner_parameters.validation import BACKTEST_MODE, FRONTTEST_MODE, VALID_MODES, FRONTTEST_SINGLE_VALUE_PARAMS
    except ImportError:
        BACKTEST_MODE = 'backtest'
        FRONTTEST_MODE = 'fronttest'
        VALID_MODES = {BACKTEST_MODE, FRONTTEST_MODE}
        FRONTTEST_SINGLE_VALUE_PARAMS = set() # Pusty set, jeśli nie ma specyficznych wymagań
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
        logging.StreamHandler() # Można usunąć jeśli nie chcemy logów w konsoli GUI
    ]
)
logger = logging.getLogger(__name__)

# Definicja parametrów (przeniesiona tutaj dla łatwiejszego dostępu w GUI)
PARAM_DEFINITIONS = {
    'check_timeframe': {
        'description': 'Okres świeczki do analizy (np. 1, 30 min).',
        'type': 'numeric', 'numeric_type': 'int'
    },
    'percentage_buy_threshold': {
        'description': 'Procentowy spadek do zakupu.',
        'type': 'numeric', 'numeric_type': 'float'
    },
    'max_allowed_usd': {
        'description': 'Maksymalny obrót USD (0 = bez limitu).',
        'type': 'numeric', 'numeric_type': 'float'
    },
    'add_to_limit_order': {
        'description': 'Margines (%) dla zleceń limit.',
        'type': 'numeric', 'numeric_type': 'float'
    },
    'sell_profit_target': {
        'description': 'Docelowy zysk (%) dla sprzedaży.',
        'type': 'numeric', 'numeric_type': 'float'
    },
    'trailing_enabled': {
        'description': 'Włączony Trailing Stop.',
        'type': 'boolean'
    },
    'trailing_stop_price': {
        'description': 'Próg aktywacji Trailing Stop (%).',
        'type': 'numeric', 'numeric_type': 'float'
    },
    'trailing_stop_margin': {
        'description': 'Margines Trailing Stop (%).',
        'type': 'numeric', 'numeric_type': 'float'
    },
    'trailing_stop_time': {
        'description': 'Czas utrzymania ceny dla aktywacji Trailing (min).',
        'type': 'numeric', 'numeric_type': 'int'
    },
    'stop_loss_enabled': {
        'description': 'Włączony Stop Loss.',
        'type': 'boolean'
    },
    'stop_loss_threshold': {
        'description': 'Próg Stop Loss (%).',
        'type': 'numeric', 'numeric_type': 'float'
    },
    'stop_loss_delay_time': {
        'description': 'Opóźnienie aktywacji Stop Loss (min).',
        'type': 'numeric', 'numeric_type': 'int'
    },
    'max_open_orders_per_coin': {
        'description': 'Max otwartych zleceń na monetę.',
        'type': 'numeric', 'numeric_type': 'int'
    },
    'next_buy_delay': {
        'description': 'Min. czas do nast. zakupu tej samej monety (min).',
        'type': 'numeric', 'numeric_type': 'int'
    },
    'next_buy_price_lower': {
        'description': 'Wymagany spadek (%) do nast. zakupu tej samej monety.',
        'type': 'numeric', 'numeric_type': 'float'
    },
    'pump_detection_enabled': {
        'description': 'Włączone wykrywanie Pump.',
        'type': 'boolean'
    },
    'pump_detection_threshold': {
        'description': 'Próg wykrycia Pump (%).',
        'type': 'numeric', 'numeric_type': 'float'
    },
    'pump_detection_disabled_time': {
        'description': 'Czas blokady zakupów po Pump (min).',
        'type': 'numeric', 'numeric_type': 'int'
    },
    'follow_btc_price': {
        'description': 'Włączone śledzenie ceny BTC.',
        'type': 'boolean'
    },
    'max_open_orders': {
        'description': 'Max wszystkich otwartych zleceń.',
        'type': 'numeric', 'numeric_type': 'int'
    },
    'stop_loss_disable_buy': {
        'description': 'Modyfikuj nast. zakup po SL.',
        'type': 'boolean'
    },
    'stop_loss_disable_buy_all': {
        'description': 'Wstrzymaj wszystkie zakupy po SL.',
        'type': 'boolean'
    },
    'stop_loss_next_buy_lower': {
        'description': 'Wymagany spadek (%) do ponownego zakupu po SL.',
        'type': 'numeric', 'numeric_type': 'float'
    },
    'stop_loss_no_buy_delay': {
        'description': 'Czas wstrzymania zakupów po SL (min).',
        'type': 'numeric', 'numeric_type': 'int'
    },
    'trailing_buy_enabled': {
        'description': 'Włączony Trailing Buy.',
        'type': 'boolean'
    },
    'trailing_buy_threshold': {
        'description': 'Dodatkowy spadek dla Trailing Buy (%).',
        'type': 'numeric', 'numeric_type': 'float'
    },
    'trailing_buy_time_in_min': {
        'description': 'Max czas oczekiwania Trailing Buy (min).',
        'type': 'numeric', 'numeric_type': 'int'
    },
    'follow_btc_threshold': {
        'description': 'Próg spadku BTC do blokady zakupów (%).',
        'type': 'numeric', 'numeric_type': 'float'
    },
    'follow_btc_block_time': {
        'description': 'Czas blokady zakupów po spadku BTC (min).',
        'type': 'numeric', 'numeric_type': 'int'
    }
}
# Upewnijmy się, że wszystkie klucze z FRONTTEST_SINGLE_VALUE_PARAMS są w PARAM_DEFINITIONS
# (chociaż w twoim oryginalnym kodzie FRONTTEST_SINGLE_VALUE_PARAMS był pusty po fallbacku)
# Jeśli validation.py zawierałby jakieś parametry, trzeba by je tu uwzględnić
# np. przy walidacji zapisu w trybie fronttest.


class ParameterManager:
    # ... (Twój kod ParameterManager bez zmian) ...
    def __init__(self):
        """Inicjalizacja menedżera parametrów"""
        self.base_dir = Path('parametry')
        self.archive_dir = self.base_dir / 'archiwum'
        self.logs_dir = Path('logi')
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.params_file = self.base_dir / 'trading_parameters.json'

        # Dodaj logger sesji jeśli potrzeba, ale może być zarządzany globalnie
        session_log_file = self.logs_dir / f'parameter_configurator_gui_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(session_log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.info("ParameterManager zainicjalizowany.")

    def _get_output_file(self, mode: str) -> Path:
        """Zwraca ścieżkę do pliku wyjściowego na podstawie trybu."""
        if mode == FRONTTEST_MODE:
            return self.base_dir / f"fronttest_trading_parameters.json"
        else:
            return self.base_dir / 'trading_parameters.json'

    def save_parameters(self, param_ranges: Dict, mode: str = BACKTEST_MODE) -> None:
        """Zapisuje parametry do pliku z archiwizacją poprzedniej wersji."""
        try:
            output_file = self._get_output_file(mode)
            # Archiwizujemy *istniejący* plik odpowiadający *nowemu trybowi*
            if output_file.exists():
                self.archive_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prefix = "fronttest_" if mode == FRONTTEST_MODE else ""
                archive_file = self.archive_dir / f"{prefix}trading_parameters_{timestamp}.json"
                shutil.copy2(output_file, archive_file)
                logger.info(f"Poprzednia konfiguracja ({output_file.name}) zarchiwizowana jako: {archive_file.name}")

            # Zaokrąglamy wartości
            rounded_params = {}
            for param_name, param_info in param_ranges.items():
                # Sprawdzamy, czy parametr jest zdefiniowany w PARAM_DEFINITIONS
                definition = PARAM_DEFINITIONS.get(param_name, {})
                numeric_type = definition.get('numeric_type', 'float') # Domyślnie float
                is_integer = (numeric_type == 'int')

                rounded_info = param_info.copy()

                if 'value' in param_info and isinstance(param_info['value'], (int, float, str)): # Akceptujemy też stringi z pól Entry
                    try:
                        val = float(param_info['value'])
                        if is_integer:
                           rounded_info['value'] = int(round(val))
                        else:
                           rounded_info['value'] = round(val, 3) # Zaokrąglenie do 3 miejsc po przecinku dla float
                    except (ValueError, TypeError):
                         # Jeśli to nie liczba (np. boolean), zostawiamy jak jest
                         pass # Lub logujemy ostrzeżenie jeśli typ jest numeric
                         if definition.get('type') == 'numeric':
                             logger.warning(f"Nie można przekonwertować wartości '{param_info['value']}' dla numerycznego parametru {param_name}")


                elif 'range' in param_info:
                    try:
                        vals = [float(x) for x in param_info['range']]
                        if is_integer:
                            rounded_info['range'] = [int(round(x)) for x in vals]
                        else:
                            rounded_info['range'] = [round(x, 3) for x in vals] # Zaokrąglenie
                    except (ValueError, TypeError):
                         logger.warning(f"Nie można przekonwertować zakresu '{param_info['range']}' dla numerycznego parametru {param_name}")
                         # Można by tu usunąć ten parametr lub zgłosić błąd

                # Kopiujemy dodatkowe metadane (description, type) z definicji
                rounded_info['description'] = definition.get('description', param_info.get('description', ''))
                rounded_info['type'] = definition.get('type', param_info.get('type', 'unknown'))
                if definition.get('type') == 'numeric':
                    rounded_info['numeric_type'] = numeric_type

                rounded_params[param_name] = rounded_info

            # Dodajemy informację o trybie
            rounded_params['__mode__'] = {
                'value': mode,
                'description': 'Tryb konfiguracji (backtest/fronttest)',
                'type': 'string'
            }

            # Zapisujemy nową konfigurację
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(rounded_params, f, indent=4, ensure_ascii=False, sort_keys=True) # sort_keys dla lepszej czytelności

            logger.info(f"Zapisano nową konfigurację do: {output_file} w trybie: {mode}")

        except Exception as e:
            logger.error(f"Błąd podczas zapisywania parametrów: {str(e)}", exc_info=True)
            raise

    def load_parameters_from_file(self, file_path: Path) -> Tuple[Optional[Dict], str]:
        """Wczytuje parametry z podanego pliku."""
        mode = BACKTEST_MODE
        params = None
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                    if '__mode__' in params:
                        mode_info = params.pop('__mode__')
                        loaded_mode = mode_info.get('value', BACKTEST_MODE)
                        if loaded_mode in VALID_MODES:
                            mode = loaded_mode
                        else:
                            logger.warning(f"Nieznany tryb '{loaded_mode}' w pliku {file_path.name}, używam domyślnego: {BACKTEST_MODE}")
                    else:
                         # Jeśli plik nie ma trybu, sprawdzamy nazwę pliku
                         if file_path.name.startswith("fronttest_"):
                             mode = FRONTTEST_MODE
                         else:
                             mode = BACKTEST_MODE
                         logger.info(f"Brak informacji o trybie w pliku {file_path.name}, zgaduję na podstawie nazwy: {mode}")

                    logger.info(f"Wczytano parametry z pliku: {file_path} (Tryb: {mode})")

            except json.JSONDecodeError as e:
                 logger.error(f"Błąd dekodowania JSON w pliku {file_path}: {e}")
                 return None, BACKTEST_MODE # Zwracamy None jeśli plik jest uszkodzony
            except Exception as e:
                 logger.error(f"Nieoczekiwany błąd podczas wczytywania pliku {file_path}: {e}", exc_info=True)
                 return None, BACKTEST_MODE
        else:
            logger.warning(f"Plik parametrów nie istnieje: {file_path}")

        return params, mode

    def load_parameters(self, mode: str = BACKTEST_MODE) -> Tuple[Optional[Dict], str]:
        """Wczytuje parametry z domyślnego pliku dla danego trybu."""
        file_to_load = self._get_output_file(mode)
        # Jeśli plik dla danego trybu nie istnieje, spróbuj wczytać domyślny (backtestowy)
        if not file_to_load.exists():
            logger.info(f"Plik dla trybu '{mode}' ({file_to_load.name}) nie istnieje. Próbuję wczytać domyślny plik {self.params_file.name}.")
            file_to_load = self.params_file # Domyślny plik to trading_parameters.json

        return self.load_parameters_from_file(file_to_load)


    def get_parameter_history(self) -> List[Path]:
        """Zwraca listę poprzednich konfiguracji."""
        try:
             # Upewniamy się, że katalog archiwum istnieje
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            # Globbing szuka plików pasujących do wzorca
            files = list(self.archive_dir.glob('*trading_parameters_*.json'))
            # Sortujemy po dacie modyfikacji, malejąco
            return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
        except Exception as e:
            logger.error(f"Błąd podczas pobierania historii parametrów: {e}", exc_info=True)
            return []


# --- Klasa GUI ---

class ParameterGUI(tk.Tk):
    def __init__(self, param_manager: ParameterManager, param_definitions: Dict):
        super().__init__()
        self.param_manager = param_manager
        self.param_definitions = param_definitions
        self.param_widgets = {}  # Słownik do przechowywania widgetów dla każdego parametru

        self.title("Konfigurator Parametrów Strategii")
        self.geometry("800x600") # Dostosuj rozmiar

        # Zmienna do przechowywania aktualnego trybu
        self.mode_var = tk.StringVar(value=BACKTEST_MODE)
        self.mode_var.trace_add("write", self._on_mode_change) # Obserwuj zmiany trybu

        self._create_widgets()
        self._populate_parameters()
        self.load_last_parameters() # Wczytaj ostatnie parametry przy starcie

    def _create_widgets(self):
        # --- Ramka górna (Tryb i Przyciski) ---
        top_frame = ttk.Frame(self, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # Wybór trybu
        ttk.Label(top_frame, text="Tryb:").pack(side=tk.LEFT, padx=5)
        rb_backtest = ttk.Radiobutton(top_frame, text=BACKTEST_MODE.capitalize(), variable=self.mode_var, value=BACKTEST_MODE)
        rb_backtest.pack(side=tk.LEFT, padx=5)
        rb_fronttest = ttk.Radiobutton(top_frame, text=FRONTTEST_MODE.capitalize(), variable=self.mode_var, value=FRONTTEST_MODE)
        rb_fronttest.pack(side=tk.LEFT, padx=5)

        # Przyciski akcji
        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="Wczytaj ostatnie", command=self.load_last_parameters).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Wczytaj z archiwum", command=self.load_from_archive).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Zapisz", command=self.save_parameters).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Wyczyść", command=self.clear_parameters).pack(side=tk.LEFT, padx=3)


        # --- Ramka główna na parametry (Scrollable) ---
        main_frame = ttk.Frame(self, padding="5")
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas) # Ramka wewnątrz canvas

        # Konfiguracja scrollowania
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        # Bindowanie kółka myszy (działa na różnych systemach)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel) # Windows/macOS
        self.canvas.bind_all("<Button-4>", self._on_mousewheel) # Linux góra
        self.canvas.bind_all("<Button-5>", self._on_mousewheel) # Linux dół


        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Pasek statusu ---
        self.status_var = tk.StringVar(value="Gotowy.")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding="2 5")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _on_mousewheel(self, event):
        """Obsługa scrollowania myszą."""
        if event.num == 5 or event.delta < 0: # Linux dół / Windows/macOS dół
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0: # Linux góra / Windows/macOS góra
            self.canvas.yview_scroll(-1, "units")

    def _populate_parameters(self):
        """Tworzy wiersze dla każdego parametru w scrollable_frame."""
        # Czyścimy poprzednie widgety jeśli są
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.param_widgets = {} # Resetujemy słownik widgetów

        row_num = 0
        current_mode = self.mode_var.get()

        for name, definition in self.param_definitions.items():
            param_frame = ttk.Frame(self.scrollable_frame, padding="5 2")
            param_frame.grid(row=row_num, column=0, sticky="ew", pady=2)
            self.scrollable_frame.grid_columnconfigure(0, weight=1) # Rozciąganie kolumny

            # --- Nazwa i Opis ---
            label_text = f"{name}:"
            ttk.Label(param_frame, text=label_text, width=25, anchor="w").grid(row=0, column=0, sticky="w", padx=5)
            ttk.Label(param_frame, text=definition['description'], foreground="grey", wraplength=450, justify=tk.LEFT).grid(row=1, column=1, columnspan=3, sticky="w", padx=5)

            widgets = {'frame': param_frame} # Przechowujemy widgety dla tego parametru

            # --- Pola wprowadzania danych ---
            if definition['type'] == 'boolean':
                var = tk.BooleanVar()
                cb = ttk.Checkbutton(param_frame, variable=var)
                cb.grid(row=0, column=1, sticky="w", padx=5)
                widgets['value_var'] = var
                widgets['widget'] = cb

            elif definition['type'] == 'numeric':
                widgets['value_type'] = tk.StringVar(value='value') # Domyślnie pojedyncza wartość
                widgets['value_var'] = tk.StringVar()
                widgets['range_min_var'] = tk.StringVar()
                widgets['range_max_var'] = tk.StringVar()
                widgets['range_step_var'] = tk.StringVar()

                # Ramka na kontrolki numeryczne
                numeric_controls_frame = ttk.Frame(param_frame)
                numeric_controls_frame.grid(row=0, column=1, columnspan=3, sticky="ew")

                # Opcje: Stała wartość / Zakres (tylko w backtest)
                if current_mode == BACKTEST_MODE:
                     rb_value = ttk.Radiobutton(numeric_controls_frame, text="Wartość:", variable=widgets['value_type'], value='value', command=lambda w=widgets: self._update_numeric_fields(w))
                     rb_value.grid(row=0, column=0, sticky="w", padx=(5,0))
                     rb_range = ttk.Radiobutton(numeric_controls_frame, text="Zakres:", variable=widgets['value_type'], value='range', command=lambda w=widgets: self._update_numeric_fields(w))
                     rb_range.grid(row=1, column=0, sticky="w", padx=(5,0))
                else: # W fronttest tylko stała wartość
                    ttk.Label(numeric_controls_frame, text="Wartość:").grid(row=0, column=0, sticky="w", padx=(5,0))
                    widgets['value_type'].set('value') # Ustawiamy na stałe 'value'

                # Pole dla pojedynczej wartości
                entry_value = ttk.Entry(numeric_controls_frame, textvariable=widgets['value_var'], width=10)
                entry_value.grid(row=0, column=1, sticky="w", padx=2)
                widgets['entry_value'] = entry_value

                # Pola dla zakresu
                range_frame = ttk.Frame(numeric_controls_frame)
                range_frame.grid(row=1, column=1, sticky="w")
                ttk.Label(range_frame, text="Min:").pack(side=tk.LEFT)
                entry_min = ttk.Entry(range_frame, textvariable=widgets['range_min_var'], width=8)
                entry_min.pack(side=tk.LEFT, padx=(0,5))
                ttk.Label(range_frame, text="Max:").pack(side=tk.LEFT)
                entry_max = ttk.Entry(range_frame, textvariable=widgets['range_max_var'], width=8)
                entry_max.pack(side=tk.LEFT, padx=(0,5))
                ttk.Label(range_frame, text="Krok:").pack(side=tk.LEFT)
                entry_step = ttk.Entry(range_frame, textvariable=widgets['range_step_var'], width=8)
                entry_step.pack(side=tk.LEFT)

                widgets['entry_min'] = entry_min
                widgets['entry_max'] = entry_max
                widgets['entry_step'] = entry_step

                # Inicjalne ustawienie widoczności pól
                self._update_numeric_fields(widgets)

            else: # Inne typy (np. string - jeśli dodasz)
                 var = tk.StringVar()
                 entry = ttk.Entry(param_frame, textvariable=var, width=30)
                 entry.grid(row=0, column=1, sticky="ew", padx=5)
                 widgets['value_var'] = var
                 widgets['widget'] = entry

            self.param_widgets[name] = widgets
            row_num += 1

        # Dodajemy pusty wiersz na końcu dla marginesu
        ttk.Frame(self.scrollable_frame, height=10).grid(row=row_num, column=0)

        # Po dodaniu elementów, zaktualizuj region scrollowania
        self.scrollable_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))


    def _update_numeric_fields(self, widgets: Dict):
        """Aktualizuje stan pól wprowadzania dla parametrów numerycznych."""
        is_range = widgets['value_type'].get() == 'range'
        is_backtest = self.mode_var.get() == BACKTEST_MODE

        # Włącz/Wyłącz pola na podstawie wyboru Wartość/Zakres
        widgets['entry_value'].config(state=tk.NORMAL if not is_range else tk.DISABLED)
        state_range = tk.NORMAL if (is_range and is_backtest) else tk.DISABLED
        widgets['entry_min'].config(state=state_range)
        widgets['entry_max'].config(state=state_range)
        widgets['entry_step'].config(state=state_range)

        # W trybie fronttest, opcja zakresu jest całkowicie wyłączona
        # (obsłużone przez brak Radiobuttona 'Zakres' i ustawienie value_type='value')


    def _on_mode_change(self, *args):
        """Wywoływana przy zmianie trybu (Backtest/Fronttest)."""
        current_mode = self.mode_var.get()
        self.status_var.set(f"Zmieniono tryb na: {current_mode.upper()}. Wczytaj lub skonfiguruj parametry.")
        logger.info(f"Zmieniono tryb GUI na: {current_mode}")

        # Przeładuj kontrolki parametrów, aby dostosować opcje (np. ukryć zakres w fronttest)
        self._populate_parameters()
        # Opcjonalnie: Automatycznie wczytaj parametry dla nowego trybu
        self.load_last_parameters()


    def clear_parameters(self):
        """Czyści wszystkie pola w GUI."""
        for name, widgets in self.param_widgets.items():
            param_type = self.param_definitions[name]['type']
            if param_type == 'boolean':
                widgets['value_var'].set(False)
            elif param_type == 'numeric':
                 widgets['value_type'].set('value') # Wróć do pojedynczej wartości
                 widgets['value_var'].set("")
                 widgets['range_min_var'].set("")
                 widgets['range_max_var'].set("")
                 widgets['range_step_var'].set("")
                 self._update_numeric_fields(widgets) # Zaktualizuj stan pól
            else:
                 widgets['value_var'].set("")
        self.status_var.set("Pola wyczyszczone. Skonfiguruj parametry.")
        logger.info("Wyczyszczono pola parametrów w GUI.")


    def load_parameters(self, params_dict: Optional[Dict], mode: str):
         """Wypełnia pola GUI na podstawie wczytanego słownika parametrów."""
         self.clear_parameters() # Najpierw czyścimy
         if not params_dict:
             self.status_var.set(f"Nie znaleziono parametrów do wczytania dla trybu {mode}.")
             return

         # Ustawiamy tryb w GUI zgodnie z wczytanym
         self.mode_var.set(mode)
         # Ponieważ zmiana mode_var wywołuje _populate_parameters i load_last_parameters,
         # musimy upewnić się, że dane zostaną poprawnie załadowane po repopulacji.
         # Najprościej jest załadować dane *po* zmianie trybu i repopulacji.
         # Czasami tkinter wymaga `update_idletasks` do przetworzenia zdarzeń.
         self.update_idletasks()

         loaded_count = 0
         for name, info in params_dict.items():
             if name in self.param_widgets:
                 widgets = self.param_widgets[name]
                 param_type = self.param_definitions[name]['type']

                 try:
                     if param_type == 'boolean':
                         widgets['value_var'].set(bool(info.get('value', False)))
                     elif param_type == 'numeric':
                         if 'value' in info:
                             widgets['value_type'].set('value')
                             widgets['value_var'].set(str(info['value']))
                         elif 'range' in info:
                             widgets['value_type'].set('range')
                             widgets['range_min_var'].set(str(info['range'][0]))
                             widgets['range_max_var'].set(str(info['range'][1]))
                             widgets['range_step_var'].set(str(info['range'][2]))
                         else:
                              logger.warning(f"Brak 'value' lub 'range' dla numerycznego parametru: {name}")
                         self._update_numeric_fields(widgets) # Zaktualizuj stan pól
                     else: # Inne typy
                         widgets['value_var'].set(str(info.get('value', '')))
                     loaded_count += 1
                 except Exception as e:
                      logger.error(f"Błąd podczas ładowania wartości dla parametru {name}: {e}", exc_info=True)
                      self.status_var.set(f"Błąd ładowania parametru: {name}")


         self.status_var.set(f"Wczytano {loaded_count} parametrów dla trybu {mode.upper()}.")
         logger.info(f"Załadowano {loaded_count} parametrów do GUI dla trybu {mode}.")
         # Po załadowaniu danych, zaktualizuj scrollregion
         self.scrollable_frame.update_idletasks()
         self.canvas.config(scrollregion=self.canvas.bbox("all"))


    def load_last_parameters(self):
        """Wczytuje ostatnio zapisane parametry dla BIEŻĄCEGO trybu."""
        current_mode = self.mode_var.get()
        try:
            params, mode_loaded = self.param_manager.load_parameters(current_mode)
            if params:
                 # Upewnij się, że tryb GUI pasuje do wczytanego pliku
                 # (load_parameters zwraca tryb *z pliku*, który może się różnić od current_mode)
                 if mode_loaded != current_mode:
                     logger.warning(f"Wczytywany plik ({self.param_manager._get_output_file(current_mode).name}) "
                                    f"pochodzi z trybu '{mode_loaded}', ale GUI jest w trybie '{current_mode}'. "
                                    f"Zmieniam tryb GUI na '{mode_loaded}'.")
                     self.mode_var.set(mode_loaded) # To wywoła _on_mode_change -> _populate -> load_last
                     # Potrzebujemy rekurencji lub lepszej logiki, aby uniknąć pętli.
                     # Prostsze rozwiązanie: po prostu załaduj dane bez zmiany trybu w GUI
                     # i poinformuj użytkownika. Albo po prostu zaufaj load_parameters.

                 # Po ewentualnej zmianie trybu (i repopulacji), ładujemy dane
                 self.load_parameters(params, mode_loaded)
                 self.status_var.set(f"Wczytano ostatnią konfigurację dla trybu {mode_loaded.upper()}.")
            else:
                 self.status_var.set(f"Nie znaleziono pliku konfiguracyjnego dla trybu {current_mode.upper()}.")
                 self.clear_parameters() # Wyczyść pola, jeśli nie ma pliku

        except Exception as e:
            logger.error(f"Błąd podczas wczytywania ostatnich parametrów: {e}", exc_info=True)
            messagebox.showerror("Błąd wczytywania", f"Nie udało się wczytać parametrów:\n{e}")
            self.status_var.set("Błąd wczytywania parametrów.")


    def load_from_archive(self):
        """Pozwala wybrać i wczytać plik konfiguracyjny z archiwum."""
        history = self.param_manager.get_parameter_history()
        if not history:
            messagebox.showinfo("Brak historii", "Nie znaleziono żadnych zarchiwizowanych konfiguracji.")
            return

        # Prosty dialog wyboru pliku (można zrobić bardziej zaawansowany)
        # initialdir wskazuje na katalog archiwum
        archive_dir = self.param_manager.archive_dir
        filepath = filedialog.askopenfilename(
            title="Wybierz plik z archiwum",
            initialdir=str(archive_dir),
            filetypes=[("JSON files", "*.json")]
        )

        if filepath:
            try:
                selected_path = Path(filepath)
                params, mode_loaded = self.param_manager.load_parameters_from_file(selected_path)
                if params:
                     self.load_parameters(params, mode_loaded) # Użyjemy funkcji ładującej do GUI
                     self.status_var.set(f"Wczytano konfigurację z archiwum: {selected_path.name} (Tryb: {mode_loaded.upper()})")
                else:
                     messagebox.showwarning("Błąd wczytywania", f"Nie udało się wczytać danych z pliku {selected_path.name} lub plik jest pusty.")
                     self.status_var.set(f"Błąd wczytywania pliku: {selected_path.name}")
            except Exception as e:
                 logger.error(f"Błąd podczas wczytywania pliku z archiwum {filepath}: {e}", exc_info=True)
                 messagebox.showerror("Błąd wczytywania", f"Nie udało się wczytać pliku z archiwum:\n{e}")
                 self.status_var.set("Błąd wczytywania z archiwum.")


    def save_parameters(self):
        """Zbiera dane z GUI, waliduje i zapisuje używając ParameterManager."""
        param_ranges = {}
        current_mode = self.mode_var.get()
        errors = []

        for name, widgets in self.param_widgets.items():
            definition = self.param_definitions[name]
            param_data = {}

            try:
                if definition['type'] == 'boolean':
                    param_data['value'] = widgets['value_var'].get()

                elif definition['type'] == 'numeric':
                    value_type = widgets['value_type'].get()

                    if value_type == 'value':
                        val_str = widgets['value_var'].get().strip()
                        if not val_str: # Pomijamy puste wartości numeryczne
                            logger.debug(f"Pominięto pusty parametr numeryczny: {name}")
                            continue
                        # Sprawdzamy czy jest liczbą
                        try:
                             val = float(val_str) # Test konwersji
                             param_data['value'] = val_str # Zapisujemy jako string, ParameterManager skonwertuje i zaokrągli
                        except ValueError:
                            errors.append(f"Parametr '{name}': Nieprawidłowa wartość liczbowa '{val_str}'.")
                            continue

                    elif value_type == 'range':
                         if current_mode == FRONTTEST_MODE:
                             errors.append(f"Parametr '{name}': Zakresy nie są dozwolone w trybie {FRONTTEST_MODE}.")
                             continue # Idź do następnego parametru

                         min_str = widgets['range_min_var'].get().strip()
                         max_str = widgets['range_max_var'].get().strip()
                         step_str = widgets['range_step_var'].get().strip()

                         if not min_str or not max_str or not step_str: # Wymagane wszystkie 3
                             errors.append(f"Parametr '{name}': Wszystkie pola zakresu (Min, Max, Krok) muszą być wypełnione.")
                             continue

                         try:
                             # Test konwersji
                             min_val = float(min_str)
                             max_val = float(max_str)
                             step_val = float(step_str)

                             if min_val >= max_val:
                                 errors.append(f"Parametr '{name}': Wartość Min musi być mniejsza niż Max.")
                                 continue
                             if step_val <= 0:
                                  errors.append(f"Parametr '{name}': Krok musi być większy od zera.")
                                  continue

                             # Zapisujemy jako stringi, ParameterManager skonwertuje
                             param_data['range'] = [min_str, max_str, step_str]

                         except ValueError:
                             errors.append(f"Parametr '{name}': Nieprawidłowe wartości liczbowe w zakresie.")
                             continue
                else: # Inne typy
                    param_data['value'] = widgets['value_var'].get()

                # Jeśli parametr został poprawnie przetworzony, dodajemy go
                if param_data:
                   # Dodajemy opis i typ dla pewności (choć ParameterManager też to robi)
                   param_data['description'] = definition['description']
                   param_data['type'] = definition['type']
                   if definition['type'] == 'numeric':
                        param_data['numeric_type'] = definition.get('numeric_type', 'float')
                   param_ranges[name] = param_data

            except Exception as e:
                # Złapanie nieoczekiwanych błędów podczas zbierania danych
                error_msg = f"Niespodziewany błąd przy przetwarzaniu parametru '{name}': {e}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)


        # Walidacja specyficzna dla trybu FRONTTEST (ponownie, dla pewności)
        if current_mode == FRONTTEST_MODE:
            for name, data in param_ranges.items():
                if 'range' in data:
                     errors.append(f"Parametr '{name}' nie może być zakresem w trybie {FRONTTEST_MODE}.")
                # Można dodać sprawdzanie FRONTTEST_SINGLE_VALUE_PARAMS jeśli jest używane

        if errors:
            error_message = "Wystąpiły błędy walidacji:\n\n" + "\n".join(f"- {e}" for e in errors)
            messagebox.showerror("Błąd zapisu", error_message)
            self.status_var.set(f"Błędy walidacji ({len(errors)}). Popraw i spróbuj ponownie.")
            return

        # Jeśli nie ma błędów, zapisujemy
        if not param_ranges:
             messagebox.showwarning("Brak danych", "Nie wprowadzono żadnych parametrów do zapisania.")
             self.status_var.set("Brak parametrów do zapisania.")
             return

        try:
            # Potwierdzenie zapisu
            if messagebox.askyesno("Potwierdzenie zapisu", f"Czy na pewno chcesz zapisać parametry dla trybu {current_mode.upper()}? Spowoduje to archiwizację poprzedniej wersji (jeśli istnieje)."):
                self.param_manager.save_parameters(param_ranges, current_mode)
                self.status_var.set(f"Parametry zapisane pomyślnie dla trybu {current_mode.upper()}.")
                logger.info(f"Parametry zapisane pomyślnie z GUI dla trybu {current_mode}.")
                messagebox.showinfo("Zapisano", f"Konfiguracja dla trybu {current_mode.upper()} została zapisana.")
            else:
                self.status_var.set("Zapis anulowany przez użytkownika.")
                logger.info("Zapis parametrów anulowany przez użytkownika.")

        except Exception as e:
            logger.error(f"Błąd podczas zapisywania parametrów z GUI: {e}", exc_info=True)
            messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać parametrów:\n{e}")
            self.status_var.set("Błąd podczas zapisywania parametrów.")


# --- Główna funkcja uruchamiająca ---

def main_gui():
    """Główna funkcja uruchamiająca GUI."""
    try:
        logger.info("Uruchamianie konfiguratora parametrów (GUI)")
        param_manager = ParameterManager()
        app = ParameterGUI(param_manager, PARAM_DEFINITIONS)
        app.mainloop()
        logger.info("Konfigurator parametrów (GUI) zakończył działanie.")
    except Exception as e:
        logger.critical(f"Krytyczny błąd podczas uruchamiania GUI: {e}", exc_info=True)
        # Wyświetl błąd również w prostym oknie, jeśli GUI padło
        try:
            root = tk.Tk()
            root.withdraw() # Ukryj główne okno tkinter
            messagebox.showerror("Krytyczny błąd", f"Wystąpił poważny błąd aplikacji:\n{e}\n\nSprawdź plik logu: {logs_dir / 'parameter_configurator.log'}")
        except tk.TclError: # Jeśli nawet tkinter nie działa
             print(f"KRYTYCZNY BŁĄD APLIKACJI (nie można wyświetlić okna): {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main_gui() # Uruchom wersję GUI zamiast konsolowej