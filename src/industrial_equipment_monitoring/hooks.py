import uuid
from datetime import UTC, datetime
from typing import Any

from kedro.framework.hooks import hook_impl
from kedro.pipeline.node import Node

from .utils.logging_config import setup_logger


class LoggingHooks:
    def __init__(self):
        self.loggers = {}
        self.pipeline_logger = setup_logger("pipeline")
        self._current_run_id: str | None = None

    def _make_run_id(self, raw_run_id: Any) -> str:
        if raw_run_id:
            return str(raw_run_id)
        # fallback: timestamp + short uuid
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        short_uuid = uuid.uuid4().hex[:8]
        return f"{ts}-{short_uuid}"

    @hook_impl
    def before_pipeline_run(self, run_params: dict[str, Any]) -> None:
        pipeline_name = run_params.get("pipeline_name") or "Main"
        # cria/normaliza run_id e guarda para uso posterior
        self._current_run_id = self._make_run_id(run_params.get("run_id"))
        self.pipeline_logger.info(f"🚀 Starting pipeline: {pipeline_name}")
        self.pipeline_logger.info(f"Run ID: {self._current_run_id}")

    @hook_impl
    def after_pipeline_run(self, run_params: dict[str, Any]) -> None:
        pipeline_name = run_params.get("pipeline_name") or "Main"
        run_id = self._current_run_id or self._make_run_id(run_params.get("run_id"))
        self.pipeline_logger.info(
            f"✅ Pipeline completed successfully: {pipeline_name}"
        )
        self.pipeline_logger.info(f"Run ID: {run_id}")
        # limpa estado
        self._current_run_id = None

    @hook_impl
    def on_pipeline_error(self, error: Exception, run_params: dict[str, Any]) -> None:
        pipeline_name = run_params.get("pipeline_name") or "Main"
        run_id = self._current_run_id or self._make_run_id(run_params.get("run_id"))
        self.pipeline_logger.error(f"❌ Pipeline failed: {pipeline_name}")
        self.pipeline_logger.error(f"Run ID: {run_id}")
        self.pipeline_logger.error(f"Error: {error}")
        self._current_run_id = None

    @hook_impl
    def before_node_run(self, node: Node, inputs: dict[str, Any]) -> None:
        node_name = node.name or node.func.__name__
        if node_name not in self.loggers:
            self.loggers[node_name] = setup_logger(f"node.{node_name}")
        self.loggers[node_name].info(f"Starting node: {node_name}")

    @hook_impl
    def after_node_run(self, node: Node, outputs: dict[str, Any]) -> None:
        node_name = node.name or node.func.__name__
        self.loggers[node_name].info(f"Finished node: {node_name}")

    @hook_impl
    def on_node_error(self, error: Exception, node: Node) -> None:
        node_name = node.name or node.func.__name__
        self.loggers[node_name].error(f"Error in node {node_name}: {error}")
