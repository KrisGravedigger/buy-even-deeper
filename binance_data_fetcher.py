#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moduł do pobierania zsynchronizowanych danych historycznych z Binance 
dla pary głównej oraz BTC/USDC jako referencji.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
from typing import Optional, Tuple, Dict, List
import logging
from pathlib import Path

class BinanceDataFetcherBtcFollow:
    def __init__(self):
        """Inicjalizacja fetchera danych z Binance"""
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })
        
        # Tworzenie katalogów
        self.csv_dir = Path('csv')
        self.csv_archive = self.csv_dir / 'archiwum'
        self.logs_dir = Path('logi')
        
        for directory in [self.csv_dir, self.csv_archive, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Konfiguracja loggera
        log_file = self.logs_dir / f'binance_fetcher_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _convert_dates_to_timestamps(
        self,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Tuple[int, int]:
        """Konwertuje daty w formacie string na timestampy"""
        if start_date:
            start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        else:
            start_timestamp = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
            
        if end_date:
            end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        else:
            end_timestamp = int(datetime.now().timestamp() * 1000)
            
        return start_timestamp, end_timestamp

    def _calculate_additional_metrics(self, df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
        """
        Oblicza dodatkowe metryki dla danej pary
        
        Args:
            df: DataFrame z danymi
            prefix: Prefiks dla nazw kolumn (np. 'btc_' dla danych BTC)
        """
        # Podstawowe metryki cenowe
        df[f'{prefix}average_price'] = (df[f'{prefix}high'] + df[f'{prefix}low']) / 2
        df[f'{prefix}typical_price'] = (df[f'{prefix}high'] + df[f'{prefix}low'] + df[f'{prefix}close']) / 3
        df[f'{prefix}price_range'] = df[f'{prefix}high'] - df[f'{prefix}low']
        
        # Metryki zmienności
        df[f'{prefix}returns'] = df[f'{prefix}close'].pct_change()
        df[f'{prefix}volatility'] = df[f'{prefix}returns'].rolling(window=20).std()
        
        # Metryki wolumenu
        df[f'{prefix}volume_ma'] = df[f'{prefix}volume'].rolling(window=20).mean()
        df[f'{prefix}relative_volume'] = df[f'{prefix}volume'] / df[f'{prefix}volume_ma']
        
        # Wskaźniki momentum
        df[f'{prefix}price_momentum'] = df[f'{prefix}close'].pct_change(periods=5)
        
        # True Range
        df[f'{prefix}true_range'] = np.maximum(
            df[f'{prefix}high'] - df[f'{prefix}low'],
            np.maximum(
                abs(df[f'{prefix}high'] - df[f'{prefix}close'].shift(1)),
                abs(df[f'{prefix}low'] - df[f'{prefix}close'].shift(1))
            )
        )
        
        return df

    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Sprawdza jakość i kompletność pobranych danych"""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            self.logger.error(f"Brakujące kolumny w danych dla {symbol}")
            return False
            
        if df.isnull().any().any():
            self.logger.warning(f"Wykryto braki w danych dla {symbol}")
            return False
            
        if not all(df['high'] >= df['low']) or not all(df['high'] >= df['close']) or not all(df['high'] >= df['open']):
            self.logger.error(f"Niepoprawne relacje między cenami dla {symbol}")
            return False
            
        return True

    

    def _fetch_single_symbol_data(
        self,
        symbol: str,
        timeframe: str,
        start_timestamp: int,
        end_timestamp: int
    ) -> pd.DataFrame:
        """Pobiera dane dla pojedynczego symbolu"""
        all_candles = []
        current_timestamp = start_timestamp
        retry_count = 0
        max_retries = 5
        
        self.logger.info(f"Rozpoczynam pobieranie danych {symbol} od {datetime.fromtimestamp(start_timestamp/1000)}")
        
        while current_timestamp < end_timestamp:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_timestamp,
                    limit=1000
                )
                
                if not candles:
                    break
                    
                all_candles.extend(candles)
                current_timestamp = candles[-1][0] + 1
                
                self.logger.info(f"Pobrano dane {symbol} do: {datetime.fromtimestamp(current_timestamp/1000)}")
                retry_count = 0
                
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Błąd podczas pobierania danych {symbol}: {e}")
                
                if retry_count >= max_retries:
                    self.logger.error(f"Przekroczono maksymalną liczbę prób dla {symbol}")
                    break
                    
                wait_time = 60 * retry_count
                self.logger.info(f"Czekam {wait_time} sekund przed ponowną próbą...")
                time.sleep(wait_time)
        
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def _merge_and_prepare_data(
        self,
        main_df: pd.DataFrame,
        btc_df: pd.DataFrame,
        main_symbol: str
    ) -> pd.DataFrame:
        """Łączy i przygotowuje dane z obu par"""
        # Dla BTC/USDC zwracamy tylko przetworzone główne dane
        if main_symbol == 'BTC/USDC':
            merged_df = self._calculate_additional_metrics(main_df, '')
            merged_df['main_symbol'] = main_symbol
            return merged_df

        # Zmiana nazw kolumn w btc_df przed mergem
        btc_df_renamed = btc_df.rename(columns={
            'open': 'btc_open',
            'high': 'btc_high',
            'low': 'btc_low',
            'close': 'btc_close',
            'volume': 'btc_volume'
        })

        # Synchronizacja timestampów
        merged_df = pd.merge(
            main_df,
            btc_df_renamed,
            on='timestamp',
            how='inner'
        )
        
        # Obliczanie metryk dla głównej pary (bez prefiksu)
        merged_df = self._calculate_additional_metrics(merged_df, '')
        
        # Obliczanie metryk dla BTC z już zmienionymi nazwami kolumn
        merged_df['btc_average_price'] = (merged_df['btc_high'] + merged_df['btc_low']) / 2
        merged_df['btc_typical_price'] = (merged_df['btc_high'] + merged_df['btc_low'] + merged_df['btc_close']) / 3
        merged_df['btc_price_range'] = merged_df['btc_high'] - merged_df['btc_low']
        merged_df['btc_returns'] = merged_df['btc_close'].pct_change()
        merged_df['btc_volatility'] = merged_df['btc_returns'].rolling(window=20).std()
        merged_df['btc_volume_ma'] = merged_df['btc_volume'].rolling(window=20).mean()
        merged_df['btc_relative_volume'] = merged_df['btc_volume'] / merged_df['btc_volume_ma']
        merged_df['btc_price_momentum'] = merged_df['btc_close'].pct_change(periods=5)
        merged_df['btc_true_range'] = np.maximum(
            merged_df['btc_high'] - merged_df['btc_low'],
            np.maximum(
                abs(merged_df['btc_high'] - merged_df['btc_close'].shift(1)),
                abs(merged_df['btc_low'] - merged_df['btc_close'].shift(1))
            )
        )
        
        # Dodatkowe metryki porównawcze (z obsługą wartości NaN)
        merged_df['price_change_ratio'] = (merged_df['returns'] / merged_df['btc_returns'].replace({0: np.nan})).fillna(0)
        merged_df['volatility_ratio'] = (merged_df['volatility'] / merged_df['btc_volatility']).fillna(1)
        
        # Dodanie informacji o parach
        merged_df['main_symbol'] = main_symbol
        merged_df['btc_symbol'] = 'BTC/USDC'
        
        return merged_df

    def _archive_existing_csv(self, main_symbol: str, timeframe: str):
        """Archiwizuje istniejące pliki CSV"""
        pattern = f'binance_{main_symbol.replace("/", "_")}_{timeframe}_*.csv'
        existing_files = list(self.csv_dir.glob(pattern))
        
        if existing_files:
            for file in existing_files:
                archive_path = self.csv_archive / f"archived_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.name}"
                file.rename(archive_path)
                self.logger.info(f"Zarchiwizowano stary plik: {file} -> {archive_path}")

    def fetch_historical_data(
        self,
        main_symbol: str = 'BTC/USDC',
        timeframe: str = '1m',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_csv: bool = True
    ) -> pd.DataFrame:
        """
        Pobiera zsynchronizowane dane historyczne dla głównej pary oraz BTC/USDC
        
        Args:
            main_symbol: Główna para tradingowa (domyślnie BTC/USDC)
            timeframe: Interwał czasowy
            start_date: Data początkowa (YYYY-MM-DD)
            end_date: Data końcowa (YYYY-MM-DD)
            save_csv: Czy zapisać dane do CSV
        """
        # Ustawienie domyślnych dat jeśli nie podano
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        start_timestamp, end_timestamp = self._convert_dates_to_timestamps(start_date, end_date)
        
        # Pobieranie danych dla głównej pary
        main_df = self._fetch_single_symbol_data(main_symbol, timeframe, start_timestamp, end_timestamp)
        
        # Walidacja danych głównej pary
        if not self._validate_data(main_df, main_symbol):
            self.logger.warning(f"Dane dla {main_symbol} mogą być niekompletne lub niepoprawne")
        
        # Dla BTC/USDC nie pobieramy dodatkowych danych referencyjnych
        if main_symbol == 'BTC/USDC':
            merged_df = self._calculate_additional_metrics(main_df, '')
            merged_df['main_symbol'] = main_symbol
        else:
            # Pobieranie danych BTC jako referencji
            btc_df = self._fetch_single_symbol_data('BTC/USDC', timeframe, start_timestamp, end_timestamp)
            if not self._validate_data(btc_df, 'BTC/USDC'):
                self.logger.warning("Dane BTC mogą być niekompletne lub niepoprawne")
            
            # Łączenie i przygotowanie danych
            merged_df = self._merge_and_prepare_data(main_df, btc_df, main_symbol)
        
        if save_csv:
            self._archive_existing_csv(main_symbol, timeframe)
            
            # Modyfikacja nazwy pliku - dodanie "_with_btc" tylko dla par innych niż BTC/USDC
            filename = (f"binance_{main_symbol.replace('/', '_')}_{timeframe}_"
                    f"{start_date}_{end_date}")
            if main_symbol != 'BTC/USDC':
                filename += "_with_btc"
            filename += ".csv"
            
            file_path = self.csv_dir / filename
            merged_df.to_csv(file_path, index=False)
            self.logger.info(f"Zapisano dane do pliku: {file_path}")
            self.logger.info(f"Liczba pobranych świeczek: {len(merged_df)}")
            
            csv_files = list(self.csv_dir.glob('*.csv'))
            if len(csv_files) > 1:
                self.logger.warning(
                    "W katalogu csv/ znajduje się więcej niż jeden plik CSV. "
                    "Przenieś nieużywane pliki do folderu csv/archiwum/."
                )
        
        return merged_df

    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """Zwraca podstawowe informacje o pobranych danych"""
        symbol = df['main_symbol'].iloc[0]
        info = {
            'period_start': df['timestamp'].min(),
            'period_end': df['timestamp'].max(),
            'candle_count': len(df),
            'data_completeness': (len(df) / ((df['timestamp'].max() - df['timestamp'].min())
                                        .total_seconds() / 60)) * 100,
            'main_pair': {
                'avg_volume': df['volume'].mean(),
                'avg_volatility': df['volatility'].mean(),
                'price_range': {
                    'min': df['low'].min(),
                    'max': df['high'].max(),
                    'avg': df['average_price'].mean()
                }
            }
        }
        
        # Dodaj statystyki BTC tylko jeśli to nie jest para BTC/USDC
        if symbol != 'BTC/USDC':
            info['btc_pair'] = {
                'avg_volume': df['btc_volume'].mean(),
                'avg_volatility': df['btc_volatility'].mean(),
                'price_range': {
                    'min': df['btc_low'].min(),
                    'max': df['btc_high'].max(),
                    'avg': df['btc_average_price'].mean()
                }
            }
            info['correlation_metrics'] = {
                'avg_price_change_ratio': df['price_change_ratio'].mean(),
                'avg_volatility_ratio': df['volatility_ratio'].mean()
            }
        
        return info
    
    def get_top_pairs_by_volume(
        self,
        n: int = 50,
        quote: str = 'USDC',
        min_days_listed: int = 60
    ) -> List[str]:
        """
        Pobiera top N par według 24h volume z Binance.
        
        Args:
            n: Liczba par do pobrania
            quote: Quote currency (USDC)
            min_days_listed: Minimalna liczba dni od listingu (pomija młode pary)
        
        Returns:
            Lista symboli par, np. ['BTC/USDC', 'ETH/USDC', ...]
        """
        try:
            self.logger.info(f"Rozpoczynam pobieranie top {n} par {quote} według volume...")
            
            # Krok 1: Ładowanie rynków
            self.logger.info("Ładowanie dostępnych rynków...")
            markets = self.exchange.load_markets()
            
            # Krok 2: Filtrowanie par z quote currency
            usdc_pairs = [
                symbol for symbol, market in markets.items()
                if market.get('quote') == quote and market.get('active', True)
            ]
            self.logger.info(f"Znaleziono {len(usdc_pairs)} aktywnych par {quote}")
            
            # Krok 3: Pobieranie 24h ticker data
            self.logger.info("Pobieranie danych volume dla wszystkich par...")
            tickers = self.exchange.fetch_tickers(usdc_pairs)
            
            # Krok 4: Sortowanie według volume
            pairs_with_volume = [
                (symbol, ticker.get('quoteVolume', 0))
                for symbol, ticker in tickers.items()
                if ticker.get('quoteVolume') is not None
            ]
            pairs_with_volume.sort(key=lambda x: x[1], reverse=True)
            
            self.logger.info(f"Posortowano {len(pairs_with_volume)} par według volume")
            
            # Krok 5: Sprawdzanie wieku pary (czy ma wystarczająco danych historycznych)
            self.logger.info(f"Sprawdzanie dostępności danych historycznych (min. {min_days_listed} dni)...")
            valid_pairs = []
            days_ago_timestamp = int((datetime.now() - timedelta(days=min_days_listed)).timestamp() * 1000)
            
            for i, (symbol, volume) in enumerate(pairs_with_volume[:n*2], 1):  # Sprawdzamy 2x więcej na wypadek młodych par
                if len(valid_pairs) >= n:
                    break
                    
                try:
                    # Próba pobrania jednej świeczki sprzed min_days_listed dni
                    test_candles = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe='1d',
                        since=days_ago_timestamp,
                        limit=1
                    )
                    
                    if test_candles and len(test_candles) > 0:
                        valid_pairs.append(symbol)
                        self.logger.info(
                            f"[{len(valid_pairs)}/{n}] ✓ {symbol} "
                            f"(volume: {volume:,.0f} {quote})"
                        )
                    else:
                        self.logger.warning(
                            f"[{i}] ✗ {symbol} - brak danych sprzed {min_days_listed} dni (pomijam)"
                        )
                    
                    time.sleep(self.exchange.rateLimit / 1000 * 0.5)  # Conservative rate limiting
                    
                except Exception as e:
                    self.logger.warning(f"[{i}] ✗ {symbol} - błąd sprawdzania: {e} (pomijam)")
                    continue
            
            if len(valid_pairs) < n:
                self.logger.warning(
                    f"Znaleziono tylko {len(valid_pairs)} par spełniających kryteria "
                    f"(wymagano {n})"
                )
            
            self.logger.info(f"Zakończono: zwracam {len(valid_pairs)} par")
            return valid_pairs
            
        except Exception as e:
            self.logger.error(f"Błąd podczas pobierania top par: {e}")
            raise

    def fetch_multiple_pairs(
        self,
        pairs: List[str],
        days: int = 60,
        timeframe: str = '1m',
        save_csv: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Pobiera dane historyczne dla wielu par jednocześnie (bulk download).
        
        Args:
            pairs: Lista par do pobrania, np. ['BTC/USDC', 'ETH/USDC']
            days: Liczba dni wstecz
            timeframe: Interwał czasowy (domyślnie 1m)
            save_csv: Czy zapisać do CSV
        
        Returns:
            Dict z kluczami: symbol pary, wartościami: DataFrame z danymi
        """
        results = {}
        success_count = 0
        fail_count = 0
        
        # Obliczanie dat
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"BULK DOWNLOAD: {len(pairs)} par")
        self.logger.info(f"Okres: {start_date} do {end_date} ({days} dni)")
        self.logger.info(f"Timeframe: {timeframe}")
        self.logger.info(f"{'='*60}\n")
        
        for i, pair in enumerate(pairs, 1):
            try:
                self.logger.info(f"[{i}/{len(pairs)}] Pobieranie {pair}...")
                
                # Wywołanie istniejącej funkcji fetch_historical_data
                df = self.fetch_historical_data(
                    main_symbol=pair,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    save_csv=save_csv
                )
                
                if df is not None and len(df) > 0:
                    results[pair] = df
                    success_count += 1
                    self.logger.info(
                        f"[{i}/{len(pairs)}] ✓ {pair} - {len(df):,} świeczek"
                    )
                else:
                    fail_count += 1
                    self.logger.warning(
                        f"[{i}/{len(pairs)}] ✗ {pair} - brak danych"
                    )
                
                # Safety sleep co 10 par
                if i % 10 == 0:
                    self.logger.info("Przerwa techniczna (rate limiting safety)...")
                    time.sleep(5)
                    
            except Exception as e:
                fail_count += 1
                self.logger.error(
                    f"[{i}/{len(pairs)}] ✗ {pair} - błąd: {e}"
                )
                continue
        
        # Podsumowanie
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"PODSUMOWANIE BULK DOWNLOAD:")
        self.logger.info(f"Sukces: {success_count}/{len(pairs)} par")
        self.logger.info(f"Niepowodzenia: {fail_count}/{len(pairs)} par")
        self.logger.info(f"{'='*60}\n")
        
        return results

def main():
    """Główna funkcja programu"""
    try:
        fetcher = BinanceDataFetcherBtcFollow()
        
        main_symbol = input("Podaj symbol (np. TON/USDC) [domyślnie: BTC/USDC]: ") or 'BTC/USDC'
        timeframe = input("Podaj timeframe (np. 1m) [domyślnie: 1m]: ") or '1m'
        start_date = input("Podaj datę początkową (YYYY-MM-DD) [domyślnie: 30 dni wstecz]: ")
        end_date = input("Podaj datę końcową (YYYY-MM-DD) [domyślnie: dziś]: ")
        
        data = fetcher.fetch_historical_data(
            main_symbol=main_symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        info = fetcher.get_data_info(data)
        print("\nInformacje o pobranych danych:")
        print("\nOgólne:")
        print(f"Okres: {info['period_start']} - {info['period_end']}")
        print(f"Liczba świeczek: {info['candle_count']}")
        print(f"Kompletność danych: {info['data_completeness']:.2f}%")
        
        print(f"\nPara {data['main_symbol'].iloc[0]}:")
        print(f"Średni wolumen: {info['main_pair']['avg_volume']:.2f}")
        print(f"Średnia zmienność: {info['main_pair']['avg_volatility']:.4f}")
        print("Zakres cen:")
        print(f"  Min: {info['main_pair']['price_range']['min']:.4f}")
        print(f"  Max: {info['main_pair']['price_range']['max']:.4f}")
        print(f"  Średnia: {info['main_pair']['price_range']['avg']:.4f}")
        
        # Wyświetlanie informacji o BTC tylko dla par innych niż BTC/USDC
        if data['main_symbol'].iloc[0] != 'BTC/USDC':
            print("\nPara BTC/USDC:")
            print(f"Średni wolumen: {info['btc_pair']['avg_volume']:.2f}")
            print(f"Średnia zmienność: {info['btc_pair']['avg_volatility']:.4f}")
            print("Zakres cen:")
            print(f"  Min: {info['btc_pair']['price_range']['min']:.2f}")
            print(f"  Max: {info['btc_pair']['price_range']['max']:.2f}")
            print(f"  Średnia: {info['btc_pair']['price_range']['avg']:.2f}")
            
            print("\nMetryki korelacji:")
            print(f"Średni stosunek zmian cen: {info['correlation_metrics']['avg_price_change_ratio']:.4f}")
            print(f"Średni stosunek zmienności: {info['correlation_metrics']['avg_volatility_ratio']:.4f}")
            
    except Exception as e:
        logging.error(f"Błąd podczas pobierania danych: {str(e)}")
        raise

if __name__ == "__main__":
    main()