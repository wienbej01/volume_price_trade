"""Logger factory."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "vpt", level: int = logging.DEBUG,
               log_file: Optional[str] = None) -> logging.Logger:
    """
    Create and configure a logger. Idempotent and ensures file handler if requested.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Ensure a console handler exists
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                      for h in logger.handlers)
    if not has_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Ensure file handler if requested
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        has_file = False
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                try:
                    if getattr(h, 'baseFilename', None) == str(log_path.resolve()):
                        has_file = True
                        break
                except Exception:
                    continue
        if not has_file:
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
