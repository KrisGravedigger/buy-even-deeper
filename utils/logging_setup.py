#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moduł konfiguracji logowania dla analizatora strategii.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.config import get_log_path

def setup_logger(name: str = None, 
                log_file: Optional[Path] = None, 
                level: int = logging.INFO) -> logging.Logger:
    """
    Konfiguruje i zwraca logger z obsługą pliku i konsoli.
    
    Args:
        name: Nazwa loggera
        log_file: Opcjonalna ścieżka do pliku logów
        level: Poziom logowania
        
    Returns:
        logging.Logger: Skonfigurowany logger
    """
    if not name:
        name = 'strategy_analyzer'
        
    if not log_file:
        log_file = get_log_path(f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
    # Tworzenie formattera
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Konfiguracja handlera pliku
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Konfiguracja handlera konsoli
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Konfiguracja loggera
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Usuwamy istniejące handlery (jeśli są)
    logger.handlers.clear()
    
    # Dodajemy nowe handlery
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_strategy_logger(name: str = 'strategy_analyzer') -> logging.Logger:
    """
    Zwraca skonfigurowany logger dla analizatora strategii.
    
    Args:
        name: Nazwa loggera
        
    Returns:
        logging.Logger: Skonfigurowany logger
    """
    return setup_logger(name) 