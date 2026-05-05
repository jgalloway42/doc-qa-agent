import logging

import mlflow

logger = logging.getLogger(__name__)


def setup_mlflow() -> None:
    """Set tracking URI and create experiment if not exists. Call once at startup."""
    from config.settings import settings

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def log_ingestion_run(
    filename: str,
    chunks_created: int,
    embedding_time_s: float,
    total_time_s: float,
    skipped: bool,
    file_hash: str,
    embedding_provider: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    """Start and end an MLflow run for a single ingestion. Return run_id."""
    from config.settings import settings

    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name=f"ingest:{filename}") as run:
        mlflow.log_param("filename", filename)
        mlflow.log_param("file_hash", file_hash)
        mlflow.log_param("skipped", str(skipped))
        mlflow.log_param("embedding_provider", embedding_provider)
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_metric("chunks_created", chunks_created)
        mlflow.log_metric("embedding_time_s", embedding_time_s)
        mlflow.log_metric("total_time_s", total_time_s)
        return run.info.run_id


def log_agent_turn(
    run_id: str,
    turn_index: int,
    user_message: str,
    response_preview: str,
    tool_calls: list[str],
    tool_results: list[str],
    latency_s: float,
    grounded: bool,
) -> None:
    """Log metrics and params for one agent turn to an existing MLflow run."""
    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_metric("latency_s", latency_s, step=turn_index)
        mlflow.log_metric("grounded", 1.0 if grounded else 0.0, step=turn_index)
        mlflow.log_param(f"turn_{turn_index}_user_message", user_message[:500])
        mlflow.log_param(f"turn_{turn_index}_response_preview", response_preview[:500])
        mlflow.log_param(f"turn_{turn_index}_tool_calls", ",".join(tool_calls))
        for i, result in enumerate(tool_results):
            mlflow.log_param(f"turn_{turn_index}_tool_result_{i}", result[:500])


def log_retrieval_quality(
    run_id: str,
    query: str,
    top_k_scores: list[float],
) -> None:
    """Log retrieval hit quality: mean, max, min of top-k scores."""
    if not top_k_scores:
        return
    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_metric("retrieval_score_mean", sum(top_k_scores) / len(top_k_scores))
        mlflow.log_metric("retrieval_score_max", max(top_k_scores))
        mlflow.log_metric("retrieval_score_min", min(top_k_scores))
