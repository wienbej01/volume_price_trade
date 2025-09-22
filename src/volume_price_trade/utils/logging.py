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
    Set up a logger for a specific training run and attach a file handler to root
    so all module loggers (volume_price_trade.*) propagate into this run log.
    """
    log_file = f"{log_dir}/run_{run_id}.log"
    run_logger = get_logger(name=f"vpt.{run_id}", log_file=log_file)

    # Also attach the same file handler to the root logger for propagation
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Avoid duplicate handlers
    existing = False
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if getattr(h, 'baseFilename', None) == str(Path(log_file).resolve()):
                    existing = True
                    break
            except Exception:
                continue
    if not existing:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root.addHandler(fh)

    return run_logger
