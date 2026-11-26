"""
FastAPI scoring service for Hybrid Fraud Intelligence.

.. note::
   The service currently exposes the **interface** of the scoring pipeline. Scoring
   itself is a transparent amount-based *heuristic placeholder* (``mode="placeholder"``);
   wiring in Neo4j subgraph extraction and trained ``HybridFraudModel`` weights is on the
   roadmap. The placeholder never reports fusion weights or component scores.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from hybrid_fraud import __version__
from hybrid_fraud.config import get_thresholds, load_config
from hybrid_fraud.scoring.risk_engine import risk_to_decision

app = FastAPI(
    title="Hybrid Fraud Intelligence API",
    description="RB-GNN + TGNN hybrid real-time fraud scoring (placeholder scoring; see README)",
    version=__version__,
)

CONFIG = load_config()
THRESHOLDS = get_thresholds(CONFIG)
MAX_BATCH_SIZE = 1000


class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    amount: float = Field(ge=0, description="Transaction amount (non-negative).")
    merchant_id: Optional[str] = None
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    card_id: Optional[str] = None
    timestamp: str = Field(description="ISO-8601 timestamp, e.g. 2026-11-27T00:00:00Z")
    features: Dict[str, Any] = Field(default_factory=dict)


class ScoreResponse(BaseModel):
    transaction_id: str
    risk_score: float
    decision: Literal["approve", "review", "decline"]
    mode: Literal["placeholder", "model"] = "placeholder"
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    fusion_weights: Dict[str, float] = Field(default_factory=dict)
    components: Dict[str, float] = Field(default_factory=dict)


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]
    count: int


def _placeholder_risk(amount: float) -> float:
    """Heuristic stand-in for the model: linear in amount, clipped to [0.05, 0.95]."""
    return min(0.95, max(0.05, (amount / 1000.0) * 0.3 + 0.15))


def _score(tx: TransactionRequest) -> ScoreResponse:
    risk = _placeholder_risk(tx.amount)
    return ScoreResponse(
        transaction_id=tx.transaction_id,
        risk_score=round(risk, 4),
        decision=risk_to_decision(risk, THRESHOLDS["approve"], THRESHOLDS["review"]),
        mode="placeholder",
        explanation=(
            "Placeholder heuristic based on transaction amount only; the trained "
            "hybrid model (graph + temporal + static) is not wired in yet."
        ),
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy", "service": "hybrid-fraud-intelligence", "version": __version__}


@app.post("/score", response_model=ScoreResponse)
async def score(tx: TransactionRequest) -> ScoreResponse:
    """
    Score a single transaction.

    Target pipeline (roadmap): feature engineering → Neo4j subgraph extraction →
    RB-GNN (G) → TGNN (T) → static encoder (S) → attention fusion (H) → risk head →
    decision (+ optional explanation). Currently returns a placeholder heuristic score.
    """
    return _score(tx)


@app.post("/batch_score", response_model=BatchScoreResponse)
async def batch_score(transactions: List[TransactionRequest]) -> BatchScoreResponse:
    """Batch scoring endpoint (up to ``MAX_BATCH_SIZE`` transactions per call)."""
    if len(transactions) > MAX_BATCH_SIZE:
        from fastapi import HTTPException

        raise HTTPException(status_code=413, detail=f"Batch too large (max {MAX_BATCH_SIZE}).")
    results = [_score(tx) for tx in transactions]
    return BatchScoreResponse(results=results, count=len(results))
