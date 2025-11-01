#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moduł do analizy jakości par kryptowalutowych dla strategii BTD (Buy The Dip).
Analizuje każdą parę pod kątem przydatności dla strategii i generuje ranking.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from multiprocessing import Pool, cpu_count
import logging
from datetime import datetime

# ============================================================================
# KONFIGURACJA - MODYFIKUJ TUTAJ PARAMETRY
# ============================================================================

# Okres analizy
ANALYSIS_DAYS = 60

# Ważenie kwartylowe
QUARTERLY_WEIGHTING = True

# Wagi dla kwartyli (jeśli QUARTERLY_WEIGHTING = True)
# Q1 = najnowsze dane (dni 1-15), Q4 = najstarsze (dni 46-60)
QUARTERLY_WEIGHTS = {
    'Q1': 0.4,  # Największa waga dla najnowszych danych
    'Q2': 0.3,
    'Q3': 0.2,
    'Q4': 0.1   # Najmniejsza waga dla najstarszych danych
}

# Próg dip detection
DIP_THRESHOLD = -1.5  # % spadku od high

# Okno wykrywania dipu
DIP_DETECTION_WINDOW_MIN = 15  # minuty
DIP_DETECTION_WINDOW_MAX = 30  # minuty

# Cel recovery
RECOVERY_TARGET = 1.0  # % odbicia

# Timeframe na recovery
RECOVERY_TIMEFRAME_MIN = 30  # minuty
RECOVERY_TIMEFRAME_MAX = 60  # minuty

# Minimalna płynność
MIN_DAILY_VOLUME_USD = 1000000

# Minimalna wymagana historia
MIN_HISTORY_DAYS = 7  # Obniżone dla testów

# Wagi metryk dla overall score
METRIC_WEIGHTS = {
    'dip_frequency': 0.25,
    'recovery_rate': 0.25,
    'avg_recovery_time': 0.15,
    'daily_drawdown': 0.10,
    'volatility': 0.10,
    'btc_correlation': 0.10,
    'liquidity': 0.05
}

# ============================================================================


class PairQualityAnalyzer:
    def __init__(self, csv_folder: str = './CSV/'):
        """Inicjalizacja analizatora"""
        self.csv_folder = Path(csv_folder)
        self.setup_logging()
    
    def setup_logging(self):
        """Konfiguracja loggera"""
        logs_dir = Path('logi')
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = logs_dir / f'pair_analyzer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_pair_data(self, csv_file: Path) -> pd.DataFrame:
        """Wczytuje dane dla pary z CSV"""
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def calculate_quarterly_weights(self, df: pd.DataFrame) -> pd.Series:
        """
        Oblicza wagi kwartylowe dla każdego wiersza w DataFrame.
        Returns: Series z wagami (0.4 dla Q1, 0.3 dla Q2, etc.)
        """
        if not QUARTERLY_WEIGHTING:
            return pd.Series(1.0, index=df.index)
        
        # Sortujemy od najnowszych do najstarszych
        df_sorted = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
        total_rows = len(df_sorted)
        
        # Tworzymy serie z wagami
        weights = pd.Series(index=df_sorted.index, dtype=float)
        
        # Obliczamy granice kwartyli
        q1_end = int(total_rows * 0.25)
        q2_end = int(total_rows * 0.50)
        q3_end = int(total_rows * 0.75)
        
        # Przypisujemy wagi
        weights.iloc[:q1_end] = QUARTERLY_WEIGHTS['Q1']
        weights.iloc[q1_end:q2_end] = QUARTERLY_WEIGHTS['Q2']
        weights.iloc[q2_end:q3_end] = QUARTERLY_WEIGHTS['Q3']
        weights.iloc[q3_end:] = QUARTERLY_WEIGHTS['Q4']
        
        # Mapujemy z powrotem do oryginalnego porządku
        weights_original = pd.Series(index=df.index, dtype=float)
        weights_original.loc[df_sorted['timestamp'].index] = weights.values
        
        return weights_original
    
    def detect_dips(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wykrywa dips zgodnie z logiką strategii BTD.
        Returns: DataFrame z kolumnami: timestamp, dip_size, recovery_time, recovered, weight
        """
        dips_list = []
        weights = self.calculate_quarterly_weights(df)
        
        # Iterujemy po DataFrame z oknem dla wykrywania HIGH
        for i in range(DIP_DETECTION_WINDOW_MAX, len(df)):
            # Okno dla wykrywania high (15-30 min wstecz)
            window_start = i - DIP_DETECTION_WINDOW_MAX
            window_end = i - DIP_DETECTION_WINDOW_MIN
            
            if window_end < window_start:
                continue
            
            # Znajdź HIGH w oknie
            window_high = df.iloc[window_start:window_end]['high'].max()
            current_price = df.iloc[i]['close']
            
            # Oblicz % spadku od high
            pct_change = ((current_price - window_high) / window_high) * 100
            
            # Czy to dip?
            if pct_change <= DIP_THRESHOLD:
                dip_timestamp = df.iloc[i]['timestamp']
                dip_weight = weights.iloc[i]
                
                # Sprawdź recovery w następnych 30-60 minutach
                recovery_end_min = min(i + RECOVERY_TIMEFRAME_MIN, len(df))
                recovery_end_max = min(i + RECOVERY_TIMEFRAME_MAX, len(df))
                
                recovered = False
                recovery_minutes = None
                
                for j in range(recovery_end_min, recovery_end_max):
                    future_price = df.iloc[j]['close']
                    recovery_pct = ((future_price - current_price) / current_price) * 100
                    
                    if recovery_pct >= RECOVERY_TARGET:
                        recovered = True
                        recovery_minutes = (df.iloc[j]['timestamp'] - dip_timestamp).total_seconds() / 60
                        break
                
                dips_list.append({
                    'timestamp': dip_timestamp,
                    'dip_size': abs(pct_change),
                    'recovery_time': recovery_minutes,
                    'recovered': recovered,
                    'weight': dip_weight
                })
        
        return pd.DataFrame(dips_list)
    
    def calculate_dip_frequency_score(self, df: pd.DataFrame, dips: pd.DataFrame) -> float:
        """Metryka 1: Dip Frequency Score"""
        if len(dips) == 0:
            return 0.0
        
        # Zlicz ważone dipy
        if QUARTERLY_WEIGHTING:
            weighted_dips = dips['weight'].sum()
        else:
            weighted_dips = len(dips)
        
        # Normalizuj do liczby dni
        days_in_data = (df['timestamp'].max() - df['timestamp'].min()).days
        if days_in_data == 0:
            return 0.0
        
        dips_per_day = weighted_dips / days_in_data
        
        # Normalizuj do 0-100 (zakładamy że 5 dipów dziennie to 100)
        score = min((dips_per_day / 5.0) * 100, 100.0)
        
        return score
    
    def calculate_recovery_rate_score(self, dips: pd.DataFrame) -> float:
        """Metryka 2: Recovery Rate Score"""
        if len(dips) == 0:
            return 0.0
        
        if QUARTERLY_WEIGHTING:
            # Ważona recovery rate
            total_weight = dips['weight'].sum()
            recovered_weight = dips[dips['recovered']]['weight'].sum()
            
            if total_weight == 0:
                return 0.0
            
            recovery_rate = recovered_weight / total_weight
        else:
            recovery_rate = dips['recovered'].sum() / len(dips)
        
        return recovery_rate * 100
    
    def calculate_avg_recovery_time_score(self, dips: pd.DataFrame) -> float:
        """Metryka 3: Average Recovery Time Score"""
        recovered_dips = dips[dips['recovered'] & dips['recovery_time'].notna()]
        
        if len(recovered_dips) == 0:
            return 0.0
        
        if QUARTERLY_WEIGHTING:
            # Ważona średnia recovery time
            avg_recovery_time = (recovered_dips['recovery_time'] * recovered_dips['weight']).sum() / recovered_dips['weight'].sum()
        else:
            avg_recovery_time = recovered_dips['recovery_time'].mean()
        
        # Im szybszy recovery, tym lepszy score
        # Zakładamy że max oczekiwany czas to RECOVERY_TIMEFRAME_MAX
        score = 100 - (avg_recovery_time / RECOVERY_TIMEFRAME_MAX) * 100
        score = max(0.0, score)  # Nie może być ujemny
        
        return score
    
    def calculate_daily_drawdown_score(self, df: pd.DataFrame) -> float:
        """Metryka 4: Daily Drawdown Score"""
        # Grupuj po dniach
        df['date'] = df['timestamp'].dt.date
        daily_stats = df.groupby('date').agg({
            'high': 'max',
            'low': 'min'
        }).reset_index()
        
        # Oblicz dzienny drawdown
        daily_stats['drawdown_pct'] = ((daily_stats['low'] - daily_stats['high']) / daily_stats['high']) * 100
        
        avg_drawdown = abs(daily_stats['drawdown_pct'].mean())
        
        # Im mniejszy drawdown, tym lepszy score
        # Zakładamy że 10% daily drawdown to 0 punktów, 0% to 100 punktów
        score = max(0, 100 - (avg_drawdown / 10.0) * 100)
        
        return score
    
    def calculate_volatility_score(self, df: pd.DataFrame) -> float:
        """Metryka 5: Volatility Score"""
        # Używamy true_range i returns
        avg_true_range = df['true_range'].mean()
        avg_price = df['close'].mean()
        
        # ATR jako % ceny
        atr_pct = (avg_true_range / avg_price) * 100 if avg_price > 0 else 0
        
        # Std dev returns
        returns_std = df['returns'].std() * 100  # Konwertuj do %
        
        # Optymalna volatility: ~2-3% (bell curve)
        optimal_volatility = 2.5
        
        # Użyj Gaussian scoring - peak przy optymalnej volatility
        volatility_metric = (atr_pct + returns_std) / 2  # Średnia z obu metryk
        
        # Bell curve: exp(-((x - optimal)^2) / (2 * sigma^2))
        sigma = 2.0
        score = 100 * np.exp(-((volatility_metric - optimal_volatility) ** 2) / (2 * sigma ** 2))
        
        return score
    
    def calculate_btc_correlation_score(self, df: pd.DataFrame) -> float:
        """Metryka 6: BTC Correlation Score"""
        # Sprawdź czy mamy dane BTC
        if 'btc_returns' not in df.columns:
            return 50.0  # Neutral score jeśli brak danych BTC
        
        # Usuń NaN
        valid_data = df[['returns', 'btc_returns']].dropna()
        
        if len(valid_data) < 100:  # Za mało danych
            return 50.0
        
        # Oblicz korelację
        correlation = valid_data['returns'].corr(valid_data['btc_returns'])
        
        # Niższa korelacja (zarówno dodatnia jak ujemna) = lepsza dywersyfikacja
        score = (1 - abs(correlation)) * 100
        
        return score
    
    def calculate_liquidity_score(self, df: pd.DataFrame) -> float:
        """Metryka 7: Liquidity Score"""
        # Średni dzienny volume w USD (volume * close)
        df['volume_usd'] = df['volume'] * df['close']
        df['date'] = df['timestamp'].dt.date
        
        daily_volume = df.groupby('date')['volume_usd'].sum()
        avg_daily_volume = daily_volume.mean()
        
        # Relative volume stability
        relative_volume_std = df['relative_volume'].std()
        
        # Score bazowany na volume
        if avg_daily_volume < MIN_DAILY_VOLUME_USD:
            volume_score = (avg_daily_volume / MIN_DAILY_VOLUME_USD) * 50
        else:
            # Logarytmiczna skala powyżej minimum
            volume_score = 50 + 50 * min(1.0, np.log10(avg_daily_volume / MIN_DAILY_VOLUME_USD))
        
        # Penalty za niestabilny volume
        stability_penalty = min(20, relative_volume_std * 10)
        
        score = max(0, volume_score - stability_penalty)
        
        return score
    
    def _calculate_overall_score(self, scores: Dict) -> float:
        """Oblicza overall score jako weighted average używając METRIC_WEIGHTS"""
        overall = 0.0
        
        overall += scores['dip_frequency_score'] * METRIC_WEIGHTS['dip_frequency']
        overall += scores['recovery_rate_score'] * METRIC_WEIGHTS['recovery_rate']
        overall += scores['avg_recovery_time_score'] * METRIC_WEIGHTS['avg_recovery_time']
        overall += scores['daily_drawdown_score'] * METRIC_WEIGHTS['daily_drawdown']
        overall += scores['volatility_score'] * METRIC_WEIGHTS['volatility']
        overall += scores['btc_correlation_score'] * METRIC_WEIGHTS['btc_correlation']
        overall += scores['liquidity_score'] * METRIC_WEIGHTS['liquidity']
        
        return overall
    
    def analyze_single_pair(self, csv_file: Path) -> Optional[Dict]:
        """
        Analizuje pojedynczą parę.
        Returns: Dict z wszystkimi metrykami + overall score
        """
        try:
            df = self.load_pair_data(csv_file)
            
            # Walidacja danych
            if len(df) < MIN_HISTORY_DAYS * 1440:  # 1440 min w dniu
                self.logger.warning(f"Za mało danych dla {csv_file.name} (wymagane: {MIN_HISTORY_DAYS} dni)")
                return None
            
            # Wykryj dips
            dips = self.detect_dips(df)
            
            # Oblicz wszystkie metryki
            scores = {
                'pair': df['main_symbol'].iloc[0],
                'dip_frequency_score': self.calculate_dip_frequency_score(df, dips),
                'recovery_rate_score': self.calculate_recovery_rate_score(dips),
                'avg_recovery_time_score': self.calculate_avg_recovery_time_score(dips),
                'daily_drawdown_score': self.calculate_daily_drawdown_score(df),
                'volatility_score': self.calculate_volatility_score(df),
                'btc_correlation_score': self.calculate_btc_correlation_score(df),
                'liquidity_score': self.calculate_liquidity_score(df)
            }
            
            # Overall score
            scores['overall_score'] = self._calculate_overall_score(scores)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Błąd podczas analizy {csv_file}: {e}")
            return None
    
    def analyze_all_pairs(self) -> pd.DataFrame:
        """
        Analizuje wszystkie pary w folderze CSV.
        Używa multiprocessing dla wydajności.
        Returns: DataFrame z rankingiem par
        """
        csv_files = list(self.csv_folder.glob('*.csv'))
        self.logger.info(f"Znaleziono {len(csv_files)} plików CSV do analizy")
        
        if len(csv_files) == 0:
            self.logger.error(f"Brak plików CSV w folderze {self.csv_folder}")
            return pd.DataFrame()
        
        # Multiprocessing
        results = []
        
        # Windows wymaga if __name__ == '__main__' dla multiprocessing
        # Używamy Pool tylko jeśli więcej niż 1 plik
        if len(csv_files) > 1:
            try:
                with Pool(processes=max(1, cpu_count() - 1)) as pool:
                    for i, result in enumerate(pool.imap(self.analyze_single_pair, csv_files), 1):
                        if result:
                            results.append(result)
                            self.logger.info(f"[{i}/{len(csv_files)}] Przeanalizowano {result['pair']}")
                        else:
                            self.logger.info(f"[{i}/{len(csv_files)}] Pominięto (brak danych)")
            except Exception as e:
                self.logger.warning(f"Multiprocessing nie działa, używam single-thread: {e}")
                # Fallback do single-thread
                for i, csv_file in enumerate(csv_files, 1):
                    result = self.analyze_single_pair(csv_file)
                    if result:
                        results.append(result)
                        self.logger.info(f"[{i}/{len(csv_files)}] Przeanalizowano {result['pair']}")
        else:
            # Single file - nie potrzeba multiprocessing
            result = self.analyze_single_pair(csv_files[0])
            if result:
                results.append(result)
        
        if len(results) == 0:
            self.logger.error("Brak wyników analizy")
            return pd.DataFrame()
        
        # Tworzenie DataFrame z wynikami
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('overall_score', ascending=False).reset_index(drop=True)
        
        return df_results
    
    def save_results(self, df_results: pd.DataFrame, output_file: str = 'pair_rankings.csv'):
        """Zapisuje wyniki do CSV i JSON"""
        if df_results.empty:
            self.logger.warning("Brak wyników do zapisania")
            return
        
        # CSV
        df_results.to_csv(output_file, index=False)
        self.logger.info(f"Zapisano wyniki do {output_file}")
        
        # JSON (bardziej szczegółowy)
        json_file = output_file.replace('.csv', '.json')
        df_results.to_json(json_file, orient='records', indent=2)
        self.logger.info(f"Zapisano wyniki do {json_file}")
    
    def print_summary(self, df_results: pd.DataFrame):
        """Wyświetla podsumowanie wyników"""
        if df_results.empty:
            print("\nBrak wyników do wyświetlenia")
            return
        
        print("\n" + "="*80)
        print("RANKING PAR DLA STRATEGII BTD")
        print("="*80)
        print(f"\nPrzeanalizowano {len(df_results)} par")
        print(f"\nTop 10 par:\n")
        
        top_10 = df_results.head(10)
        display_cols = ['pair', 'overall_score', 'dip_frequency_score', 'recovery_rate_score', 'liquidity_score']
        print(top_10[display_cols].to_string(index=False))
        
        if len(df_results) > 10:
            print(f"\n\nNajgorsze 5 par:\n")
            bottom_5 = df_results.tail(5)
            print(bottom_5[display_cols].to_string(index=False))
        
        print("\n" + "="*80)
        print("\nPełne wyniki zapisane w: pair_rankings.csv i pair_rankings.json")
        print("="*80 + "\n")


def main():
    """Główna funkcja programu"""
    analyzer = PairQualityAnalyzer(csv_folder='./CSV/')
    
    print("Rozpoczynam analizę par...")
    results = analyzer.analyze_all_pairs()
    
    if not results.empty:
        analyzer.save_results(results)
        analyzer.print_summary(results)
    else:
        print("Nie udało się przeanalizować żadnych par")


if __name__ == '__main__':
    main()
