"""Logger factory."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "vpt", level: int = logging.INFO,
               log_file: Optional[str] = None) -> logging.Logger:
    """
    Create and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if logger already has them
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_run_logger(run_id: str, log_dir: str = "logs") -> logging.Logger:
    """
    Set up a logger for a specific training run.
    
    Args:
        run_id: Unique identifier for the run
        log_dir: Directory to store log files
        
    Returns:
        Configured logger for the run
    """
    log_file = f"{log_dir}/run_{run_id}.log"
    return get_logger(name=f"vpt.{run_id}", log_file=log_file)
