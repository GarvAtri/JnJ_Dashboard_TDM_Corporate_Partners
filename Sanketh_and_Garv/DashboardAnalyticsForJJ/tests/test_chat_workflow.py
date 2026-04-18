from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.graph import run_chat_workflow, run_save_response_workflow


@pytest.fixture(autouse=True)
def fallback_mode(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("ANTHROPIC_API_KEY", "YOUR_KEY_HERE")


def test_chat_workflow_returns_structured_chat_state(tmp_path: Path) -> None:
	project_root = tmp_path / "project"
	project_root.mkdir(parents=True, exist_ok=True)

	result = run_chat_workflow(
		user_message="What does current spend look like?",
		chat_history=[{"role": "user", "content": "Give me a quick summary."}],
		data_context={"metrics": {"total_po_volume": "31", "average_variance": "$497.13"}},
		project_root=project_root,
	)

	assert result["status"] == "success"
	assert "chat_state" in result
	assert "assistant_message" in result["chat_state"]
	assert result["chat_state"]["assistant_message"]


def test_saved_memory_is_injected_into_future_chat_calls(tmp_path: Path) -> None:
	project_root = tmp_path / "project"
	project_root.mkdir(parents=True, exist_ok=True)

	save = run_save_response_workflow("Remember this prior response.", project_root=project_root)
	assert save["status"] == "success"

	result = run_chat_workflow(
		user_message="Use saved context in this answer.",
		chat_history=[],
		data_context={"metrics": {}},
		project_root=project_root,
	)

	assert "Remember this prior response." in result["chat_state"]["saved_memory"]
	assert "Claude API is unavailable" in result["chat_state"]["assistant_message"]
