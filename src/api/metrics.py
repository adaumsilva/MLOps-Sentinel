"""
Prometheus instrumentation for the MLOps-Sentinel API.

Exposes:
  - sentinel_http_requests_total          (Counter)   — requests by endpoint & status
  - sentinel_prediction_latency_seconds   (Histogram) — end-to-end predict latency
  - sentinel_model_predictions_total      (Counter)   — predictions by outcome label
"""

from prometheus_client import Counter, Histogram, CollectorRegistry, REGISTRY

# Re-use the default global registry so /metrics picks everything up automatically.

REQUEST_COUNT = Counter(
    name="sentinel_http_requests_total",
    documentation="Total HTTP requests handled by the API",
    labelnames=["endpoint", "http_status"],
)

PREDICTION_LATENCY = Histogram(
    name="sentinel_prediction_latency_seconds",
    documentation="Latency of /predict calls in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

PREDICTION_COUNT = Counter(
    name="sentinel_model_predictions_total",
    documentation="Total model predictions broken down by outcome",
    labelnames=["outcome"],  # "high_risk" | "low_risk"
)
