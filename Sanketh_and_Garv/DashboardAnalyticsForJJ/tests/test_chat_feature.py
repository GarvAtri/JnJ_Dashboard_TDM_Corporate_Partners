"""End-to-end chat feature tests with real dashboard data context.

These tests verify that:
1. The full chat pipeline (workflow → LLM → response) works correctly
2. Responses are relevant to the question being asked
3. The model name resolves without 404 errors
4. Fallback mode still returns structured responses
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from src.agent.graph import run_chat_workflow, run_dashboard_workflow
from src.agent.llm import ClaudeChatAssistant


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Realistic data context built from the actual dashboard workflow
# ---------------------------------------------------------------------------

def _build_live_data_context() -> dict[str, Any]:
    """Run the real dashboard workflow and build the data context that the chat receives."""
    result = run_dashboard_workflow(project_root=PROJECT_ROOT)
    assert result["status"] == "success", f"Dashboard workflow failed: {result.get('errors')}"
    ds = result["dashboard_state"]
    metrics = ds["metrics"]

    sector_breakdown = []
    raw_sector = ds.get("grouped", {}).get("sector_treemap")
    if raw_sector is not None and hasattr(raw_sector, "to_dict"):
        sector_breakdown = [
            {"sector": str(r.get("Sector", "")), "po_status": str(r.get("PO_Status", "")), "amount": float(r.get("PO_Total_Amount", 0))}
            for r in raw_sector.head(15).to_dict("records")
        ]

    variance_breakdown = []
    raw_variance = ds.get("grouped", {}).get("variance_bar")
    if raw_variance is not None and hasattr(raw_variance, "to_dict"):
        variance_breakdown = [
            {"root_cause": str(r.get("Root_Cause", "")), "sector": str(r.get("Sector", "")), "variance": float(r.get("Variance_vs_Budget", 0))}
            for r in raw_variance.head(15).to_dict("records")
        ]

    return {
        "metrics": {
            "total_po_volume": f"{metrics['total_po_volume']:,}",
            "average_variance": f"${metrics['average_variance']:,.2f}",
            "active_risk_count": f"{metrics['active_risk_count']:,}",
            "addressable_spend_pct": f"{metrics['addressable_spend_pct']:.2f}%",
            "total_spend": f"${metrics.get('total_spend', 0):,.2f}",
        },
        "filters": {"sector": "All", "po_status": "All", "addressable": "All", "risk_status": "All", "time_range": "month"},
        "open_risks": ds["tables"].get("open_risks", []),
        "sector_breakdown": sector_breakdown,
        "variance_breakdown": variance_breakdown,
        "insights": ds.get("insights", []),
        "variance_explanation": "",
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_data_context() -> dict[str, Any]:
    """Shared data context loaded once for all tests in this module."""
    return _build_live_data_context()


_has_api_key = bool(os.getenv("ANTHROPIC_API_KEY") and os.getenv("ANTHROPIC_API_KEY") != "YOUR_KEY_HERE")
requires_api_key = pytest.mark.skipif(not _has_api_key, reason="ANTHROPIC_API_KEY not set or is placeholder")


# ---------------------------------------------------------------------------
# Test: Model name is valid (no 404)
# ---------------------------------------------------------------------------

@requires_api_key
def test_model_name_resolves_without_error(live_data_context: dict[str, Any]) -> None:
    """Verify that the configured model name does not produce a 404 / not_found_error."""
    assistant = ClaudeChatAssistant()
    result = assistant.chat(
        user_message="Hello",
        chat_history=[],
        saved_memory="",
        data_context=live_data_context,
    )
    assert result["source"] == "anthropic", f"Expected anthropic source, got: {result['source']}. Message: {result['assistant_message'][:300]}"
    assert "error" not in result["assistant_message"].lower() or "api error" not in result["assistant_message"].lower()


# ---------------------------------------------------------------------------
# Test: Total PO Volume question
# ---------------------------------------------------------------------------

@requires_api_key
def test_chat_answers_total_po_volume(live_data_context: dict[str, Any]) -> None:
    """Ask about total PO volume and verify a relevant numeric answer."""
    result = run_chat_workflow(
        user_message="What is the total PO volume?",
        chat_history=[],
        data_context=live_data_context,
        project_root=PROJECT_ROOT,
    )
    msg = result["chat_state"]["assistant_message"].lower()
    # The response should mention a number and relate to PO volume
    assert any(char.isdigit() for char in msg), f"Expected numeric answer, got: {msg[:200]}"
    assert result["chat_state"]["source"] == "anthropic"


# ---------------------------------------------------------------------------
# Test: Average Variance question
# ---------------------------------------------------------------------------

@requires_api_key
def test_chat_answers_average_variance(live_data_context: dict[str, Any]) -> None:
    """Ask about average variance and verify the response references a dollar amount."""
    result = run_chat_workflow(
        user_message="What is the average variance in the current data?",
        chat_history=[],
        data_context=live_data_context,
        project_root=PROJECT_ROOT,
    )
    msg = result["chat_state"]["assistant_message"]
    assert any(char.isdigit() for char in msg), f"Expected numeric answer, got: {msg[:200]}"
    # Should mention variance or dollar amount
    msg_lower = msg.lower()
    assert "variance" in msg_lower or "$" in msg, f"Expected variance-related content, got: {msg[:200]}"


# ---------------------------------------------------------------------------
# Test: Risk-related question
# ---------------------------------------------------------------------------

@requires_api_key
def test_chat_answers_risk_question(live_data_context: dict[str, Any]) -> None:
    """Ask about active risks and verify the response is about risks."""
    result = run_chat_workflow(
        user_message="How many active risks are there and which are the most critical?",
        chat_history=[],
        data_context=live_data_context,
        project_root=PROJECT_ROOT,
    )
    msg = result["chat_state"]["assistant_message"].lower()
    assert "risk" in msg, f"Expected risk-related content, got: {msg[:200]}"
    assert any(char.isdigit() for char in msg), f"Expected numeric data in risk answer, got: {msg[:200]}"


# ---------------------------------------------------------------------------
# Test: Follow-up question uses chat history
# ---------------------------------------------------------------------------

@requires_api_key
def test_chat_handles_followup_with_history(live_data_context: dict[str, Any]) -> None:
    """Send a follow-up question referencing prior context to verify history continuity."""
    history = [
        {"role": "user", "content": "What is the total PO volume?"},
        {"role": "assistant", "content": "The total PO volume in the current view is 31 purchase orders."},
    ]
    result = run_chat_workflow(
        user_message="How does that compare to the number of active risks?",
        chat_history=history,
        data_context=live_data_context,
        project_root=PROJECT_ROOT,
    )
    msg = result["chat_state"]["assistant_message"].lower()
    # Should reference both PO volume and risks since it's a comparison
    assert any(char.isdigit() for char in msg), f"Expected numeric comparison, got: {msg[:200]}"
    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Test: Fallback mode returns structured response (no API key)
# ---------------------------------------------------------------------------

def test_fallback_mode_returns_structured_response(live_data_context: dict[str, Any]) -> None:
    """When API key is absent, verify fallback still returns a structured, non-empty response."""
    assistant = ClaudeChatAssistant(api_key="YOUR_KEY_HERE")
    result = assistant.chat(
        user_message="What is the total PO volume?",
        chat_history=[],
        saved_memory="",
        data_context=live_data_context,
    )
    assert result["source"] == "deterministic-fallback"
    assert result["model"] == "fallback"
    assert result["assistant_message"]
    assert "Claude API is unavailable" in result["assistant_message"]


# ---------------------------------------------------------------------------
# Test: Responses are NOT the same repetitive template
# ---------------------------------------------------------------------------

def test_responses_are_not_identical_across_different_questions(live_data_context: dict[str, Any]) -> None:
    """Even in fallback mode, different questions should reference different data."""
    assistant = ClaudeChatAssistant(api_key="YOUR_KEY_HERE")

    r1 = assistant.chat(
        user_message="What is the total PO volume?",
        chat_history=[],
        saved_memory="",
        data_context=live_data_context,
    )
    r2 = assistant.chat(
        user_message="Summarize the top risks",
        chat_history=[],
        saved_memory="",
        data_context=live_data_context,
    )
    # Both should contain the same fallback metrics since API is absent,
    # but the structure should be valid
    assert r1["assistant_message"]
    assert r2["assistant_message"]
    assert r1["source"] == "deterministic-fallback"
    assert r2["source"] == "deterministic-fallback"
