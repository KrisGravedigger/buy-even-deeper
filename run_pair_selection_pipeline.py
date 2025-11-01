#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kompletny pipeline selekcji par dla strategii BTD.
1. Pobiera top N par z Binance
2. Bulk download danych historycznych
3. Analizuje jakość każdej pary
4. Generuje ranking
"""

from binance_data_fetcher import BinanceDataFetcherBtcFollow
from pair_quality_analyzer import PairQualityAnalyzer
import sys


def main():
    """Główna funkcja pipeline"""
    print("="*80)
    print("PIPELINE SELEKCJI PAR DLA STRATEGII BTD")
    print("="*80)
    
    try:
        # KROK 1: Pobierz top pary
        print("\n[KROK 1/3] Pobieranie listy top par według volume...")
        fetcher = BinanceDataFetcherBtcFollow()
        top_pairs = fetcher.get_top_pairs_by_volume(n=50, quote='USDC')
        print(f"OK Znaleziono {len(top_pairs)} par")
        
        if len(top_pairs) == 0:
            print("BLAD: Nie znaleziono zadnych par USDC")
            sys.exit(1)
        
        # KROK 2: Bulk download
        print("\n[KROK 2/3] Pobieranie danych historycznych (to moze zajac ~30-60 min)...")
        print("UWAGA: To moze potrwac dlugo - pipeline respektuje rate limits API")
        data = fetcher.fetch_multiple_pairs(pairs=top_pairs, days=60, save_csv=True)
        print(f"OK Pobrano dane dla {len(data)} par")
        
        if len(data) == 0:
            print("BLAD: Nie udalo sie pobrac zadnych danych")
            sys.exit(1)
        
        # KROK 3: Analiza
        print("\n[KROK 3/3] Analiza jakosci par...")
        analyzer = PairQualityAnalyzer(csv_folder='./CSV/')
        results = analyzer.analyze_all_pairs()
        
        if results.empty:
            print("BLAD: Brak wynikow analizy")
            sys.exit(1)
        
        # Zapis wyników
        analyzer.save_results(results, output_file='pair_rankings.csv')
        analyzer.print_summary(results)
        
        print("\n" + "="*80)
        print("OK PIPELINE ZAKONCZONY SUKCESEM")
        print(f"OK Wyniki zapisane w: pair_rankings.csv i pair_rankings.json")
        print("="*80)
        print("\nNastepne kroki:")
        print("1. Sprawdz pair_rankings.csv - wybierz top 10-15 par")
        print("2. Skopiuj wybrane CSV do folderu dla strategy_runner.py")
        print("3. Uruchom backtesty na wybranych parach")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nPrzerwano przez uzytkownika")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nBLAD: {e}")
        print("Sprawdz logi w folderze ./logi/ po wiecej informacji")
        sys.exit(1)


if __name__ == '__main__':
    main()
