import logging
from logging.handlers import TimedRotatingFileHandler  # noqa
from unittest.mock import MagicMock, patch

import pytest

from industrial_equipment_monitoring.utils.logging_config import setup_logger


def test_setup_logger_basic():
    """Teste básico de criação do logger"""
    with patch("logging.handlers.TimedRotatingFileHandler") as mock_file_handler:
        mock_file_handler.return_value = MagicMock()

        logger = setup_logger("test_pipeline")

        assert logger.name == "test_pipeline"
        assert logger.level == logging.INFO


def test_setup_logger_invalid_name():
    """Teste com nome inválido"""
    with pytest.raises(ValueError):
        setup_logger("")


def test_setup_logger_handlers():
    """Teste se os handlers são adicionados"""
    with patch("logging.handlers.TimedRotatingFileHandler") as mock_file_handler:
        mock_file_handler.return_value = MagicMock()

        logger = setup_logger("test_pipeline")
        handlers = 2

        # Verifica se tem 2 handlers (stream + file)
        assert len(logger.handlers) == handlers

        # Verifica se um é StreamHandler
        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) == handlers


def test_setup_logger_multiple_calls():
    """Teste chamadas múltiplas não duplicam handlers"""
    with patch("logging.handlers.TimedRotatingFileHandler"):
        logger1 = setup_logger("same_pipeline")
        handler_count_first = len(logger1.handlers)

        logger2 = setup_logger("same_pipeline")
        handler_count_second = len(logger2.handlers)

        # Mesmo logger instance e mesma quantidade de handlers
        assert logger1 is logger2
        assert handler_count_first == handler_count_second


def test_setup_logger_different_names():
    """Teste loggers com nomes diferentes"""
    with patch("logging.handlers.TimedRotatingFileHandler"):
        logger_a = setup_logger("pipeline_a")
        logger_b = setup_logger("pipeline_b")

        assert logger_a is not logger_b
        assert logger_a.name == "pipeline_a"
        assert logger_b.name == "pipeline_b"
