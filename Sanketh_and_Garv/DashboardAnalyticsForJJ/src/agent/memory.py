"""Persistent memory helpers for chatbot saved responses."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def resolve_chat_memory_path(project_root: Path) -> Path:
	"""Resolve the persistent chat memory file path."""
	return project_root / ".claude" / "memory" / "chat_context.md"


def load_chat_memory(path: Path) -> str:
	"""Load persistent memory text, creating file if needed."""
	path.parent.mkdir(parents=True, exist_ok=True)
	if not path.exists():
		path.write_text("# Saved Chat Responses\n\n", encoding="utf-8")
	return path.read_text(encoding="utf-8")


def append_saved_response(path: Path, response_text: str, source: str = "assistant") -> dict[str, str]:
	"""Append a saved assistant response to persistent memory."""
	path.parent.mkdir(parents=True, exist_ok=True)
	if not path.exists():
		path.write_text("# Saved Chat Responses\n\n", encoding="utf-8")
	timestamp = datetime.now().isoformat(timespec="seconds")
	entry = (
		f"## Saved Response - {timestamp}\n"
		f"Source: {source}\n\n"
		f"{response_text.strip()}\n\n"
	)
	with path.open("a", encoding="utf-8") as file_handle:
		file_handle.write(entry)
	return {
		"status": "success",
		"message": "Response saved to persistent chat memory.",
		"path": str(path),
	}
