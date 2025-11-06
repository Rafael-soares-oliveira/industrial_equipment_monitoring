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
        raise ValueError("Logger name must be a non-empty string")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    date_str = datetime.now(UTC).strftime("%Y%m%d")
    log_filename = f"{date_str}.log"
    log_path = Path(log_dir) / log_filename

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] %(name)s - %(message)s"
        )

        # Terminal
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Ensure that the log directory exists
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Daily logs with automatic rotation
        file_handler = TimedRotatingFileHandler(
            filename=str(log_path),
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
            utc=True,
        )
        file_handler.setFormatter(formatter)
        file_handler.suffix = "%Y%m%d.log"
        logger.addHandler(file_handler)

    return logger
