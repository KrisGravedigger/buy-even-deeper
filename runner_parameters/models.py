#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import numpy as np
from typing import Dict, Optional

@dataclass
class TradingParameters:
    check_timeframe: int
    percentage_buy_threshold: float
    max_allowed_usd: float = 1000.0
    add_to_limit_order: float = 2.0
    sell_profit_target: float = 1.0
    trailing_enabled: float = 0.0 
    trailing_stop_price: float = 1.0
    trailing_stop_margin: float = 0.5
    trailing_stop_time: int = 1
    stop_loss_enabled: bool = False
    stop_loss_threshold: float = 2.0
    stop_loss_delay_time: int = 1
    max_open_orders_per_coin: int = 1
    next_buy_delay: int = 1
    next_buy_price_lower: float = 1.0
    pump_detection_enabled: bool = False
    pump_detection_threshold: float = 5.0
    pump_detection_disabled_time: int = 30
    follow_btc_price: bool = False
    follow_btc_threshold: float = 1.0
    follow_btc_block_time: int = 30
    max_open_orders: int = 1
    stop_loss_disable_buy: bool = False
    stop_loss_disable_buy_all: bool = False
    stop_loss_next_buy_lower: float = 1.0
    stop_loss_no_buy_delay: int = 1
    trailing_buy_enabled: bool = False
    trailing_buy_threshold: float = 2.0
    trailing_buy_time_in_min: int = 15

    def validate(self, market_data: Optional[Dict] = None) -> bool:
        """
        Waliduje parametry strategii.
        
        Args:
            market_data: Opcjonalne dane rynkowe do dodatkowej walidacji
            
        Returns:
            bool: True jeśli parametry są poprawne
        """
        from .validation import validate_parameters
        validation_result = validate_parameters(self, market_data)
        return validation_result.is_valid

    def to_array(self) -> np.ndarray:
        """Konwertuje parametry na tablicę numpy dla użycia z njit"""
        return np.array([
            self.check_timeframe,
            self.percentage_buy_threshold,
            self.max_allowed_usd,
            self.sell_profit_target,
            float(self.trailing_enabled),
            self.trailing_stop_price,
            self.trailing_stop_margin,
            self.trailing_stop_time,
            float(self.stop_loss_enabled),
            self.stop_loss_threshold,
            self.stop_loss_delay_time,
            self.max_open_orders,
            self.next_buy_delay,
            self.next_buy_price_lower,
            float(self.pump_detection_enabled),
            self.pump_detection_threshold,
            float(self.follow_btc_price),
            self.follow_btc_threshold,
            self.follow_btc_block_time,
            float(self.stop_loss_disable_buy),
            float(self.stop_loss_disable_buy_all),
            self.stop_loss_next_buy_lower,
            self.stop_loss_no_buy_delay,
            float(self.trailing_buy_enabled),
            self.trailing_buy_threshold,
            self.trailing_buy_time_in_min
        ], dtype=np.float64)