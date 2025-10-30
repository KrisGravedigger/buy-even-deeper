#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moduł implementujący prosty mechanizm cache'owania dla analizy strategii.
"""

from typing import Dict, List, Set, Optional # Dodano Optional
from utils.logging_setup import get_strategy_logger
import time
import gc

logger = get_strategy_logger('strategy_cache')

class AnalysisCache:
    """Cache dla wyników analizy strategii."""

    def __init__(self):
        """Inicjalizacja pustego cache'u."""
        self._param_distributions: Dict[str, Dict] = {} # { symbol: { param: { stats... } } }
        self._importance_by_symbol: Dict[str, Dict[str, float]] = {} # { symbol: { param: importance } }
        self._symbols: Set[str] = set()
        self._formatted_metrics: Dict[str, Dict] = {} # { strategy_id: { formatted_metrics } }
        self._last_update: Dict[str, float] = {} # { cache_key: timestamp }

    @property
    def param_distributions(self) -> Dict:
        """Zwraca słownik z rozkładami parametrów."""
        return self._param_distributions

    @property
    def importance_by_symbol(self) -> Dict:
        """Zwraca słownik z ważnością parametrów."""
        return self._importance_by_symbol

    def add_symbol(self, symbol: str):
        """Dodaje symbol do zbioru znanych symboli."""
        if symbol not in self._symbols:
            logger.debug(f"Dodano symbol '{symbol}' do cache.")
            self._symbols.add(symbol)

    @property
    def symbols(self) -> List[str]:
        """Zwraca posortowaną listę wszystkich symboli w cache."""
        return sorted(list(self._symbols))

    def get_param_distributions(self, symbol: str, results: List[Dict]) -> Dict:
        """
        Zwraca (i oblicza jeśli trzeba) rozkłady parametrów dla symbolu.
        """
        cache_key = f"param_distributions_{symbol}"
        if symbol not in self._param_distributions:
            logger.debug(f"[{symbol}] Cache MISS: Obliczam rozkłady parametrów...")
            start_time = time.time()
            # Import lokalny, aby uniknąć problemów z zależnościami cyklicznymi
            from analysis.parameters import analyze_parameter_distributions
            # Funkcja analyze_parameter_distributions powinna zwracać { symbol: { param: { stats } } }
            # lub pusty słownik jeśli analiza dla symbolu się nie powiedzie
            dist_data = analyze_parameter_distributions(results) # Zakładamy, że results są już dla TEGO symbolu
            # Sprawdź, czy funkcja zwróciła dane dla naszego symbolu
            if symbol in dist_data:
                 self._param_distributions[symbol] = dist_data[symbol]
                 self._last_update[cache_key] = time.time()
                 logger.debug(f"[{symbol}] Cache SET: Zapisano rozkłady parametrów (czas: {time.time() - start_time:.2f}s)")
            else:
                 logger.warning(f"[{symbol}] Funkcja analyze_parameter_distributions nie zwróciła danych dla symbolu.")
                 # Zapisz pusty słownik, aby uniknąć ponownego obliczania
                 self._param_distributions[symbol] = {}
                 self._last_update[cache_key] = time.time()

        else:
             logger.debug(f"[{symbol}] Cache HIT: Zwracam rozkłady parametrów.")
        # Zawsze zwracaj dane dla symbolu, nawet jeśli są puste
        return self._param_distributions.get(symbol, {})

    # --- ZMODYFIKOWANA get_importance ---
    def get_importance(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Zwraca *zapisaną w cache* ważność parametrów dla symbolu.
        Nie wykonuje obliczeń - obliczenia są robione w analyze_strategies.
        """
        cache_key = f"importance_{symbol}"
        if symbol in self._importance_by_symbol:
            logger.debug(f"[{symbol}] Cache HIT: Zwracam ważność parametrów.")
            return self._importance_by_symbol[symbol]
        else:
            logger.debug(f"[{symbol}] Cache MISS: Brak zapisanej ważności parametrów (oczekuje na obliczenie w analyze_strategies).")
            return None # Zwróć None, jeśli nie ma w cache


    def get_formatted_metrics(self, strategy_id: str, metrics: Dict) -> Dict:
        """
        Zwraca (i formatuje jeśli trzeba) metryki dla strategii.
        """
        cache_key = f"metrics_{strategy_id}"
        if strategy_id not in self._formatted_metrics:
            # logger.debug(f"Cache MISS: Formatuję metryki dla strategii {strategy_id}") # Zbyt szczegółowe
            from analysis.metrics import format_metrics # Import lokalny
            self._formatted_metrics[strategy_id] = format_metrics(metrics)
            self._last_update[cache_key] = time.time()
        # else: logger.debug(f"Cache HIT: Zwracam sformatowane metryki dla {strategy_id}")
        return self._formatted_metrics[strategy_id]

    def clear(self):
        """Czyści wszystkie dane z cache'u."""
        logger.info("Czyszczę instancję AnalysisCache...")
        start_size = (len(self._param_distributions) + 
                      len(self._importance_by_symbol) + len(self._formatted_metrics))
        self._param_distributions.clear()
        self._importance_by_symbol.clear()
        self._symbols.clear()
        self._formatted_metrics.clear()
        self._last_update.clear()
        # Aktualizacja logu:
        logger.info(f"Cache wyczyszczony.") # Uproszczony log