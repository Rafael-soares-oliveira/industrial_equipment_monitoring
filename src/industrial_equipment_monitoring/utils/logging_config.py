import logging
import sys
from datetime import UTC, datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Personalized logger with module-specific date-based filenames.

    Args:
        name (str): Pipeline name
        log_dir (str, optional): Directory to store the logs. Defaults to "logs"

    Returns:
        logging.Logger: Logger and path to the log file

    Raises:
        ValueError: If name is empty or invalid
    """
    if not name or not isinstance(name, str):
        raise ValueError("Nome atribuído ao logger não deve estar vazio.")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Store every logger level

    date_str = datetime.now(UTC).strftime("%Y%m%d")  # 20250331 -> 2025/03/31
    log_filename = f"{date_str}.log"  # Name of each log file.
    log_path_dir = Path(log_dir)
    log_path_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_path_dir / log_filename

    # datetime - [LEVEL] Name - Starting pipeline
    formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] %(name)s - %(message)s"
    )

    # Terminal logger
    has_file_handler = any(
        isinstance(h, TimedRotatingFileHandler) for h in logger.handlers
    )
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, TimedRotatingFileHandler)
        for h in logger.handlers
    )

    # Adiciona StreamHandler (Terminal) apenas se não existir
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Adiciona FileHandler (Arquivo) apenas se não existir
    if not has_file_handler:
        file_handler = TimedRotatingFileHandler(
            filename=str(log_path),
            when="MIDNIGHT",
            interval=1,
            backupCount=30,
            encoding="utf-8",
            utc=True,
        )
        file_handler.setFormatter(formatter)
        file_handler.suffix = "%Y%m%d.log"
        logger.addHandler(file_handler)

    return logger
