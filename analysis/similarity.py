#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moduł odpowiedzialny za obliczanie podobieństwa między strategiami.
"""

from typing import Dict, Optional, TYPE_CHECKING, Union, Tuple
import numpy as np # Dodaj import numpy

# usunięto, bo nie jest już potrzebny jako argument
# if TYPE_CHECKING:
#     from analysis.cache import AnalysisCache

def calculate_strategy_similarity(strategy1: Dict,
                                strategy2: Dict,
                                param_weights: Optional[Dict[str, float]] = None, # Zmieniono Optional
                                detailed: bool = True) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    Oblicza podobieństwo między dwiema strategiami na podstawie ich parametrów,
    z uwzględnieniem opcjonalnych wag. Obsługuje brakujące parametry, typy bool
    oraz próbuje konwersji do float dla porównań numerycznych.

    Args:
        strategy1: Słownik reprezentujący pierwszą strategię (oczekuje klucza 'parameters').
        strategy2: Słownik reprezentujący drugą strategię (oczekuje klucza 'parameters').
        param_weights: Opcjonalny słownik mapujący nazwy parametrów na ich wagi (float).
                       Jeśli None, używane są wagi domyślne. Parametry nie wymienione
                       w wagach otrzymują domyślną niską wagę.
        detailed: Jeśli True, zwraca krotkę (całkowite podobieństwo, słownik podobieństw per parametr).
                  Jeśli False, zwraca tylko całkowite podobieństwo (float).

    Returns:
        Union[float, Tuple[float, Dict[str, float]]]:
            - float: Wartość podobieństwa (0.0 - 1.0) gdy detailed=False.
            - Tuple[float, Dict[str, float]]: Krotka (całkowite podobieństwo, szczegóły per parametr)
              gdy detailed=True. Zwraca (0.0, {}) w przypadku braku wspólnych parametrów z wagami.
    """
    # Sprawdzenie podstawowej struktury wejściowej
    params1 = strategy1.get('parameters')
    params2 = strategy2.get('parameters')
    if not isinstance(params1, dict) or not isinstance(params2, dict):
        # logger.warning("Invalid strategy structure passed to calculate_strategy_similarity.") # Można dodać logger
        return (0.0, {}) if detailed else 0.0

    # Ustalenie wag parametrów
    if param_weights is None:
        # Domyślne wagi - można je dostosować lub przenieść do konfiguracji
        param_weights = {
            # Ważniejsze parametry (większy wpływ na logikę)
            'check_timeframe': 1.0, # Zakładając, że to kluczowy interwał
            'percentage_buy_threshold': 1.0,
            'stop_loss_threshold': 0.9,
            'trailing_stop_margin': 0.9,
            # Średnio ważne
            'stop_loss_enabled': 0.6,
            'trailing_enabled': 0.6,
            'follow_btc_price': 0.5, # Czy w ogóle śledzić BTC
            'follow_btc_threshold': 0.5, # Jaki próg dla BTC
            # Mniej ważne (często optymalizowane w wąskich zakresach lub mniej krytyczne)
            'max_open_orders': 0.3,
            'max_open_orders_per_coin': 0.3,
            'trading_volume_threshold': 0.2,
            'follow_btc_block_time': 0.2, # Czas blokady BTC może być mniej ważny niż sam fakt śledzenia
            'min_profit_threshold': 0.3, # Próg minimalnego zysku
            # Można dodać inne znane parametry z odpowiednimi wagami
        }
    default_weight = 0.1 # Niska waga dla parametrów nieznanych w `param_weights`

    param_similarity_sum = 0.0
    total_weight = 0.0
    similarities_details = {}  # Słownik na szczegółowe podobieństwa per parametr

    # Zbierz unikalne klucze parametrów z obu strategii
    all_param_keys = set(params1.keys()) | set(params2.keys())

    for param in all_param_keys:
        weight = param_weights.get(param, default_weight)
        if weight <= 0: continue # Pomiń parametry z wagą 0 lub ujemną

        value1_raw = params1.get(param)
        value2_raw = params2.get(param)

        similarity = 0.0 # Domyślne podobieństwo, jeśli nie uda się porównać

        # 1. Obsługa przypadków, gdy parametr istnieje tylko w jednej strategii
        if value1_raw is None and value2_raw is None:
            # Obie strategie nie mają tego parametru - uznajemy za identyczne pod tym względem
            similarity = 1.0
        elif value1_raw is None or value2_raw is None:
            # Jedna strategia ma parametr, druga nie - całkowicie różne
            similarity = 0.0
        else:
            # 2. Obie strategie mają parametr - próba porównania
            # Sprawdzenie typu bool
            is_bool1 = isinstance(value1_raw, bool)
            is_bool2 = isinstance(value2_raw, bool)

            if is_bool1 or is_bool2:
                # Traktuj jako bool (True=1, False=0)
                v1_bool = 1.0 if value1_raw else 0.0
                v2_bool = 1.0 if value2_raw else 0.0
                similarity = 1.0 if abs(v1_bool - v2_bool) < 1e-9 else 0.0
            else:
                # 3. Próba porównania numerycznego (jako float)
                try:
                    v1_num = float(value1_raw)
                    v2_num = float(value2_raw)

                    # Sprawdzenie NaN lub Inf
                    if not np.isfinite(v1_num) or not np.isfinite(v2_num):
                         similarity = 1.0 if str(v1_num) == str(v2_num) else 0.0 # Porównaj jako stringi ("inf" == "inf")
                    else:
                        # Normalizacja różnicy
                        range_val = max(abs(v1_num), abs(v2_num))
                        if abs(range_val) < 1e-9: # Obie wartości bliskie zero
                            similarity = 1.0
                        else:
                            diff = abs(v1_num - v2_num)
                            normalized_diff = diff / range_val
                            # Podobieństwo = 1 - znormalizowana różnica (ograniczone do [0, 1])
                            similarity = max(0.0, 1.0 - normalized_diff)

                except (ValueError, TypeError):
                    # 4. Nie da się porównać numerycznie - porównanie bezpośrednie (np. dla stringów)
                    similarity = 1.0 if value1_raw == value2_raw else 0.0

        similarities_details[param] = similarity # Zapisz szczegółowe podobieństwo
        param_similarity_sum += similarity * weight
        total_weight += weight

    # Obliczenie finalnego, ważonego podobieństwa
    if total_weight == 0:
        final_similarity = 0.0
    else:
        final_similarity = param_similarity_sum / total_weight
        # Upewnij się, że wynik jest w zakresie [0, 1] (choć powinien być z definicji)
        final_similarity = max(0.0, min(1.0, final_similarity))


    if detailed:
        return final_similarity, similarities_details
    else:
        return final_similarity