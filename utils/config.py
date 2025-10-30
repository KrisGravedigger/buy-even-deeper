#!/opt/homebrew/bin/python3.11
# -*- coding: utf-8 -*-

"""
Moduł konfiguracyjny dla analizatora strategii.
Zawiera definicje ścieżek i podstawową konfigurację.
"""

from pathlib import Path
import os
import shutil
from datetime import datetime
import glob
from typing import Optional

# Definicja ścieżek
PARAMETRY_DIR = Path('parametry')
WYNIKI_DIR = Path('wyniki')
LOGI_DIR = Path('logi')
WYNIKI_BACKTEST_DIR = WYNIKI_DIR / 'backtesty'
WYNIKI_ANALIZA_DIR = WYNIKI_DIR / 'analizy'
JSON_ANALYSIS_DIR = WYNIKI_DIR / 'analizy_json'

# Ścieżki do obsługi symulacji
CSV_DIR = Path('csv')
SYMULACJE_DIR = CSV_DIR / 'symulacje'
PARAMETRY_SYMULACJE_DIR = PARAMETRY_DIR / 'symulacje'

# Nowe ścieżki do fronttest
WYNIKI_FRONTTEST_DIR = WYNIKI_DIR / 'fronttesty'
WYNIKI_FRONTTEST_ANALIZY_DIR = WYNIKI_DIR / 'fronttest_analizy'
AT_CSV_DIR = CSV_DIR / 'AT'

# Nowe ścieżki dla wizualizacji
WIZUALIZACJE_CSV_DIR = CSV_DIR / 'wizualizacje'
WIZUALIZACJE_PARAMETRY_DIR = PARAMETRY_DIR / 'wizualizacje'
WYNIKI_WIZUALIZACJE_DIR = WYNIKI_DIR / 'wizualizacje'

def create_directories():
    """Tworzy wymagane katalogi jeśli nie istnieją"""
    for dir_path in [PARAMETRY_DIR, WYNIKI_DIR, LOGI_DIR, 
                     WYNIKI_BACKTEST_DIR, WYNIKI_ANALIZA_DIR, JSON_ANALYSIS_DIR,
                     CSV_DIR, SYMULACJE_DIR, PARAMETRY_SYMULACJE_DIR,
                     WYNIKI_FRONTTEST_DIR, WYNIKI_FRONTTEST_ANALIZY_DIR,
                     WIZUALIZACJE_CSV_DIR, WIZUALIZACJE_PARAMETRY_DIR, WYNIKI_WIZUALIZACJE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

def ensure_directory_structure():
    """
    Sprawdza i tworzy strukturę katalogów.
    Zwraca True jeśli struktura jest poprawna.
    """
    try:
        create_directories()
        return True
    except Exception as e:
        print(f"Błąd podczas tworzenia struktury katalogów: {str(e)}")
        return False

def get_analysis_path(filename: str) -> Path:
    """Zwraca pełną ścieżkę do pliku w katalogu analiz"""
    return WYNIKI_ANALIZA_DIR / filename

def get_backtest_path(filename: str) -> Path:
    """Zwraca pełną ścieżkę do pliku w katalogu backtestów"""
    return WYNIKI_BACKTEST_DIR / filename

def get_log_path(filename: str) -> Path:
    """Zwraca pełną ścieżkę do pliku w katalogu logów"""
    return LOGI_DIR / filename

def get_fronttest_output_dir() -> Path:
    """
    Tworzy i zwraca ścieżkę do podkatalogu dla wyników fronttestów z datą i godziną.
    
    Returns:
        Path: Ścieżka do katalogu wyjściowego dla fronttestów
    """
    # Tworzenie katalogu z datą i godziną
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = WYNIKI_FRONTTEST_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def get_fronttest_analysis_dir() -> Path:
    """Zwraca ścieżkę do katalogu z analizami fronttestów"""
    ensure_directory_structure()  # Upewniamy się, że katalog istnieje
    return WYNIKI_FRONTTEST_ANALIZY_DIR

def get_newest_file(directory: str, pattern: str = "*", recursive: bool = False) -> str:
    """
    Zwraca ścieżkę do najnowszego pliku w katalogu (i opcjonalnie podkatalogach) pasującego do wzorca.
    
    Args:
        directory: Ścieżka do katalogu
        pattern: Wzorzec filtru plików (np. "*.pkl")
        recursive: Czy szukać również w podkatalogach (domyślnie False)
        
    Returns:
        str: Ścieżka do najnowszego pliku lub None jeśli nie znaleziono plików
    """
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        return None
    
    # Wyszukanie wszystkich plików pasujących do wzorca
    if recursive:
        # Używamy glob.glob z rekursywnym parametrem
        matching_files = list(glob.glob(f"{directory_path}/**/{pattern}", recursive=True))
    else:
        # Jeśli nie rekursywnie, używamy tylko glob z Path
        matching_files = list(directory_path.glob(pattern))
    
    if not matching_files:
        return None
    
    # Wybieramy najnowszy plik na podstawie daty modyfikacji
    newest_file = max(matching_files, key=os.path.getmtime)
    return str(newest_file)

def get_simulation_output_dir(source_data_path: str) -> Path:
    """
    Tworzy i zwraca ścieżkę do unikalnego podkatalogu dla wyników symulacji.
    
    Args:
        source_data_path: Ścieżka do pliku źródłowego CSV
        
    Returns:
        Path: Ścieżka do katalogu wyjściowego dla symulacji
    """
    # Tworzenie unikalnego podkatalogu dla każdego wywołania
    base_filename = os.path.basename(source_data_path).split('.')[0]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_subfolder = f"{base_filename}_{current_time}"
    
    output_dir = SYMULACJE_DIR / run_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def get_simulation_parameters_dir() -> Path:
    """Zwraca ścieżkę do katalogu z parametrami symulacji"""
    return PARAMETRY_SYMULACJE_DIR

def get_newest_csv_file() -> str:
    """
    Zwraca ścieżkę do najnowszego pliku CSV w głównym katalogu csv/
    
    Returns:
        str: Ścieżka do najnowszego pliku CSV lub None jeśli nie znaleziono plików
    """
    return get_newest_file(CSV_DIR, "*.csv")

def get_newest_visualization_csv() -> str:
    """Zwraca ścieżkę do najnowszego pliku CSV w katalogu wizualizacji"""
    return get_newest_file(WIZUALIZACJE_CSV_DIR, "*.csv")

def get_newest_visualization_params() -> str:
    """Zwraca ścieżkę do najnowszego pliku JSON z parametrami wizualizacji"""
    return get_newest_file(WIZUALIZACJE_PARAMETRY_DIR, "*.json")

def get_visualization_output_dir() -> Path:
    """Tworzy i zwraca ścieżkę do podkatalogu dla wyników wizualizacji z datą i godziną"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = WYNIKI_WIZUALIZACJE_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

# Ścieżki dla analizy technicznej
AT_PARAMETERS_DIR = PARAMETRY_DIR / 'at_parameters'

def ensure_at_directories():
    """Tworzy katalogi potrzebne dla analizy technicznej"""
    AT_PARAMETERS_DIR.mkdir(parents=True, exist_ok=True)
    AT_CSV_DIR.mkdir(parents=True, exist_ok=True)

def get_at_parameters_dir() -> Path:
    """Zwraca ścieżkę do katalogu z parametrami analizy technicznej"""
    ensure_at_directories()
    return AT_PARAMETERS_DIR

def get_newest_at_config_file() -> str:
    """
    Zwraca ścieżkę do najnowszego pliku JSON z konfiguracją analizy technicznej
    
    Returns:
        str: Ścieżka do najnowszego pliku JSON lub None jeśli nie znaleziono plików
    """
    ensure_at_directories()
    return get_newest_file(AT_PARAMETERS_DIR, "*.json")

def get_newest_at_csv_file() -> Optional[str]:
    """
    Zwraca ścieżkę do najnowszego pliku CSV w katalogu csv/AT.
    
    Returns:
        Optional[str]: Ścieżka do najnowszego pliku CSV lub None jeśli nie znaleziono
    """
    # Upewnij się, że katalog istnieje
    AT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    
    return get_newest_file(AT_CSV_DIR, "*.csv", recursive=False)

# Funkcja pomocnicza do tworzenia ścieżek z datą i godziną
def create_timestamped_dir(base_dir: Path, prefix: str = "") -> Path:
    """
    Tworzy podkatalog z datą i godziną w katalogu bazowym.
    
    Args:
        base_dir: Katalog bazowy
        prefix: Prefiks nazwy katalogu (opcjonalnie)
        
    Returns:
        Path: Ścieżka do utworzonego katalogu
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{prefix}_{timestamp}" if prefix else timestamp
    output_dir = base_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir