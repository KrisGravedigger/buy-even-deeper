#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moduł odpowiedzialny za obliczanie metryk strategii tradingowych.
"""

from typing import List, Dict
import numpy as np
import pandas as pd
from utils.logging_setup import get_strategy_logger

logger = get_strategy_logger('metrics_calculator')

class StrategyMetrics:
    """Klasa obliczająca metryki dla strategii tradingowej."""
    
    def __init__(self):
        """Inicjalizacja metryk."""
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.win_rate = 0.0
        self.total_profit = 0.0
        self.avg_profit = 0.0
        self.max_loss = 0.0
        self.max_profit = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.profit_factor = 0.0
        self.risk_reward_ratio = 0.0
        self.expectancy = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.max_drawdown = 0.0
        self.profit_per_trade = 0.0
        self.btc_blocks = 0
        self.btc_block_rate = 0.0
        self.btc_correlation = 0.0
        self.btc_block_effectiveness = 0.0

    def calculate_metrics(self, trades: List[float], 
                        btc_blocks: int, 
                        total_checks: int,
                        btc_correlation: float) -> None:
        """
        Oblicza metryki na podstawie listy wyników transakcji.
        
        Args:
            trades: Lista zysków/strat z transakcji
            btc_blocks: Liczba blokad przez BTC
            total_checks: Całkowita liczba sprawdzeń
            btc_correlation: Korelacja z BTC
        """
        if not trades:
            return
            
        self.trade_count = len(trades)
        winning_trades = [t for t in trades if t > 0]
        losing_trades = [t for t in trades if t < 0]
        
        self.win_count = len(winning_trades)
        self.loss_count = len(losing_trades)
        self.win_rate = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0
        
        self.total_profit = sum(trades)
        self.avg_profit = np.mean(trades) if trades else 0
        self.max_loss = min(trades) if trades else 0
        self.max_profit = max(trades) if trades else 0
        
        self.avg_win = np.mean(winning_trades) if winning_trades else 0
        self.avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
        
        total_wins = sum(winning_trades) if winning_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0
        self.profit_factor = total_wins / total_losses if total_losses != 0 else float('inf')
        
        self.risk_reward_ratio = self.avg_win / self.avg_loss if self.avg_loss != 0 else float('inf')
        win_probability = self.win_count / self.trade_count if self.trade_count > 0 else 0
        self.expectancy = (win_probability * self.avg_win) - ((1 - win_probability) * self.avg_loss)

        returns = pd.Series(trades)
        self.sharpe_ratio = returns.mean() / returns.std() if len(returns) > 1 else 0
        
        downside_returns = returns[returns < 0]
        self.sortino_ratio = returns.mean() / downside_returns.std() if len(downside_returns) > 1 else 0
        
        cumulative_returns = returns.cumsum()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns - rolling_max
        self.max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
        
        self.profit_per_trade = self.total_profit / self.trade_count if self.trade_count > 0 else 0
        
        # Metryki follow BTC
        self.btc_blocks = btc_blocks
        self.btc_block_rate = (btc_blocks / total_checks * 100) if total_checks > 0 else 0
        self.btc_correlation = btc_correlation
        
        # Skuteczność blokad
        if self.btc_blocks > 0:
            avg_loss_with_blocks = self.avg_loss
            theoretical_loss = self.max_loss * (self.btc_blocks / total_checks) 
            self.btc_block_effectiveness = ((theoretical_loss - avg_loss_with_blocks) / abs(theoretical_loss)) * 100 if theoretical_loss != 0 else 0 