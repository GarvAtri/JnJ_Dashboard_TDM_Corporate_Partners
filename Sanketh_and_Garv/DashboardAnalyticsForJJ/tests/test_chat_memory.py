from __future__ import annotations

from pathlib import Path

from src.agent.memory import append_saved_response, load_chat_memory, resolve_chat_memory_path


def test_chat_memory_persists_saved_response(tmp_path: Path) -> None:
	project_root = tmp_path / "project"
	memory_path = resolve_chat_memory_path(project_root)

	initial = load_chat_memory(memory_path)
	assert "# Saved Chat Responses" in initial

	append_saved_response(memory_path, "Assistant insight to persist.")
	reloaded = load_chat_memory(memory_path)

	assert "Assistant insight to persist." in reloaded
	assert "Saved Response" in reloaded


def test_memory_file_location_is_under_claude_memory(tmp_path: Path) -> None:
	project_root = tmp_path / "workspace"
	memory_path = resolve_chat_memory_path(project_root)
	assert memory_path == project_root / ".claude" / "memory" / "chat_context.md"
