"""NeuroUI Judge scoring service."""

# Side-effect import: loads the project's .env into os.environ before any
# downstream module reads its config. Keep this first.
from . import env_loader  # noqa: F401

from . import llm_client
from .deterministic_audit import run_audit
from .neural_proxy_mock import predict_neural_proxy, synthesize_roi_timeseries
from .reward_model import DEFAULT_WEIGHTS, score_candidate
from .preference_model import (
    METRIC_ORDER,
    fit_preference_weights,
    predict_pairwise_probability,
)
from .schemas import (
    AgentRun,
    CandidateReport,
    DeterministicAudit,
    NeuralProxyFeatures,
    PairwisePreference,
    Subscores,
    UICandidate,
)
from .storage import Storage
from . import tribe_adapter

__all__ = [
    "AgentRun",
    "CandidateReport",
    "DEFAULT_WEIGHTS",
    "DeterministicAudit",
    "METRIC_ORDER",
    "NeuralProxyFeatures",
    "PairwisePreference",
    "Storage",
    "Subscores",
    "UICandidate",
    "fit_preference_weights",
    "llm_client",
    "predict_neural_proxy",
    "predict_pairwise_probability",
    "run_audit",
    "score_candidate",
    "synthesize_roi_timeseries",
    "tribe_adapter",
]
