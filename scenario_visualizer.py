#!/opt/homebrew/bin/python3.11
# -*- coding: utf-8 -*-

"""
Wizualizator scenariuszy rynkowych wygenerowanych przez market_simulator.py
Generuje wykres porównujący przebieg cen dla różnych symulowanych scenariuszy.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
import re
import argparse
from datetime import datetime

def identify_scenario_type(filename):
    """
    Identyfikuje typ scenariusza na podstawie nazwy pliku.
    
    Args:
        filename: Nazwa pliku CSV z danymi scenariusza
        
    Returns:
        str: Typ scenariusza ('normal', 'bootstrap', 'stress' lub 'unknown')
    """
    filename = os.path.basename(filename).lower()
    
    if 'normal' in filename:
        return 'normal'
    elif 'bootstrap' in filename:
        return 'bootstrap'
    elif 'stress' in filename:
        return 'stress'
    else:
        # Próba identyfikacji na podstawie innych wzorców
        if 'mean' in filename and 'vol' in filename:
            return 'normal'  # Prawdopodobnie scenariusz normalny z parametrami
        if 'crash' in filename or 'factor' in filename:
            return 'stress'  # Prawdopodobnie scenariusz stresowy
            
    return 'unknown'

def load_scenario_data(directory):
    """
    Wczytuje wszystkie scenariusze z podanego katalogu.
    
    Args:
        directory: Ścieżka do katalogu zawierającego pliki CSV scenariuszy
        
    Returns:
        dict: Słownik z wczytanymi danymi, pogrupowany według typów scenariuszy
    """
    # Znajdź wszystkie pliki CSV w katalogu
    csv_files = glob.glob(os.path.join(directory, "simulated_*_USDT_*.csv"))
    
    # Inicjalizacja słownika na dane
    scenario_data = {
        'normal': [],
        'bootstrap': [],
        'stress': [],
        'unknown': []
    }
    
    # Wczytaj każdy plik i przypisz do odpowiedniego typu
    for file_path in csv_files:
        scenario_type = identify_scenario_type(file_path)
        
        try:
            # Wczytaj dane z CSV
            df = pd.read_csv(file_path)
            
            # Konwersja timestampów na datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Dodanie informacji o pliku źródłowym i typie
            df['scenario_file'] = os.path.basename(file_path)
            df['scenario_type'] = scenario_type
            
            # Dodanie do odpowiedniej kategorii
            scenario_data[scenario_type].append(df)
            
            print(f"Wczytano scenariusz {scenario_type}: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Błąd wczytywania pliku {file_path}: {e}")
    
    return scenario_data

def visualize_scenarios(scenario_data, output_file=None, figsize=(12, 8)):
    """
    Generuje wykres porównujący wszystkie scenariusze.
    
    Args:
        scenario_data: Słownik z danymi scenariuszy
        output_file: Ścieżka do pliku wyjściowego (jeśli None, wykres zostanie wyświetlony)
        figsize: Rozmiar wykresu
    """
    # Inicjalizacja wykresu
    plt.figure(figsize=figsize)
    
    # Kolory dla różnych typów scenariuszy
    colors = {
        'normal': 'blue',
        'bootstrap': 'green',
        'stress': 'red',
        'unknown': 'gray'
    }
    
    # Zliczanie typów scenariuszy
    scenario_counts = {
        'normal': len(scenario_data['normal']),
        'bootstrap': len(scenario_data['bootstrap']),
        'stress': len(scenario_data['stress']),
        'unknown': len(scenario_data['unknown'])
    }
    
    # Rysowanie linii dla każdego scenariusza
    first_scenarios = {}  # Przechowanie pierwszego scenariusza każdego typu dla legendy
    
    # Iteracja po typach scenariuszy
    for scenario_type, dataframes in scenario_data.items():
        # Pomiń typy bez danych
        if not dataframes:
            continue
            
        # Określenie przezroczystości linii w zależności od liczby scenariuszy
        alpha = max(0.2, min(0.8, 5.0 / len(dataframes))) if len(dataframes) > 5 else 0.8
        
        # Rysowanie linii dla każdego scenariusza
        for i, df in enumerate(dataframes):
            # Sprawdź czy mamy kolumny timestamp i close
            if 'timestamp' in df.columns and 'close' in df.columns:
                line = plt.plot(
                    df['timestamp'], 
                    df['close'],
                    color=colors[scenario_type],
                    alpha=alpha,
                    linewidth=1
                )
                
                # Zapisz pierwszy scenariusz dla legendy
                if i == 0:
                    first_scenarios[scenario_type] = line[0]
    
    # Ustawienia osi X - daty
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gcf().autofmt_xdate()
    
    # Dodanie opisów wykresu
    plt.title('Porównanie symulowanych scenariuszy cenowych', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Cena', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Dodanie legendy z liczbą scenariuszy każdego typu
    legend_items = []
    legend_labels = []
    
    for scenario_type, count in scenario_counts.items():
        if count > 0 and scenario_type in first_scenarios:
            legend_items.append(first_scenarios[scenario_type])
            legend_labels.append(f"{scenario_type.capitalize()} ({count})")
    
    plt.legend(legend_items, legend_labels, loc='best')
    
    # Dodanie dodatkowych informacji
    plt.figtext(
        0.02, 0.02, 
        f"Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
        fontsize=8
    )
    
    # Zapisanie lub wyświetlenie wykresu
    if output_file:
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Zapisano wykres do pliku: {output_file}")
    else:
        plt.tight_layout()
        plt.show()

def get_newest_subdir(parent_dir):
    """
    Znajduje najnowszy podkatalog w podanej lokalizacji.
    
    Args:
        parent_dir: Katalog nadrzędny
        
    Returns:
        str: Ścieżka do najnowszego podkatalogu lub None jeśli nie znaleziono
    """
    if not os.path.exists(parent_dir):
        return None
        
    # Pobierz wszystkie podkatalogi
    subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) 
              if os.path.isdir(os.path.join(parent_dir, d))]
    
    if not subdirs:
        # Jeśli nie ma podkatalogów, zwróć sam katalog nadrzędny
        return parent_dir if glob.glob(os.path.join(parent_dir, "simulated_*.csv")) else None
    
    # Sortuj według daty modyfikacji (najnowsze na końcu)
    subdirs.sort(key=os.path.getmtime)
    
    # Sprawdź czy najnowszy katalog zawiera pliki CSV
    newest_dir = subdirs[-1]
    if glob.glob(os.path.join(newest_dir, "simulated_*.csv")):
        return newest_dir
    
    # Jeśli najnowszy nie zawiera plików CSV, sprawdź pozostałe od najnowszego
    for dir_path in reversed(subdirs[:-1]):
        if glob.glob(os.path.join(dir_path, "simulated_*.csv")):
            return dir_path
    
    return None

def main():
    """Główna funkcja do uruchomienia z linii komend"""
    parser = argparse.ArgumentParser(description='Wizualizacja symulowanych scenariuszy cenowych')
    parser.add_argument('--directory', '-d', 
                        help='Katalog zawierający pliki CSV z symulowanymi scenariuszami')
    parser.add_argument('--output', '-o',
                        help='Ścieżka do pliku wyjściowego z wykresem (opcjonalnie)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8],
                        help='Rozmiar wykresu w calach (szerokość wysokość, domyślnie: 12 8)')
    
    args = parser.parse_args()
    
    # Jeśli nie podano katalogu, użyj domyślnej lokalizacji
    if args.directory is None:
        # Najpierw sprawdź najnowszy podkatalog w csv/symulacje
        symulations_dir = 'csv/symulacje'
        newest_dir = get_newest_subdir(symulations_dir)
        
        if newest_dir:
            args.directory = newest_dir
            print(f"Automatycznie wybrano najnowszy katalog: {args.directory}")
        else:
            # Jeśli nie znaleziono w csv/symulacje, sprawdź inne standardowe lokalizacje
            possible_dirs = [
                'csv/symulacje',
                'data/symulacje',
                'simulations'
            ]
            
            for dir_path in possible_dirs:
                if os.path.exists(dir_path) and len(glob.glob(os.path.join(dir_path, "simulated_*.csv"))) > 0:
                    args.directory = dir_path
                    print(f"Automatycznie wybrano katalog: {args.directory}")
                    break
    
    if args.directory is None:
        print("Błąd: Nie znaleziono katalogu z symulacjami. Użyj parametru --directory.")
        return
    
    # Wczytanie danych
    scenario_data = load_scenario_data(args.directory)
    
    # Wyświetlenie podsumowania
    total_scenarios = sum(len(dfs) for dfs in scenario_data.values())
    print(f"\nWczytano łącznie {total_scenarios} scenariuszy:")
    for scenario_type, dfs in scenario_data.items():
        if dfs:
            print(f" - {scenario_type.capitalize()}: {len(dfs)}")
    
    # Generowanie wizualizacji
    visualize_scenarios(scenario_data, args.output, tuple(args.figsize))

if __name__ == "__main__":
    main()