"""Chatbot workflow nodes for Claude panel orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from src.agent.llm import ClaudeChatAssistant
from src.agent.memory import append_saved_response, load_chat_memory, resolve_chat_memory_path


class ChatWorkflowState(TypedDict, total=False):
	"""State contract for chat-specific workflows."""

	config: dict[str, Any]
	user_message: str
	chat_history: list[dict[str, str]]
	saved_memory: str
	chat_result: dict[str, str]
	assistant_message: str
	save_target_message: str
	save_result: dict[str, str]
	data_context: dict[str, Any]
	chat_state: dict[str, Any]
	status: str
	errors: list[str]


def load_memory_node(state: ChatWorkflowState) -> ChatWorkflowState:
	"""Load persistent saved response memory into workflow state."""
	project_root = Path(state.get("config", {}).get("project_root", Path(__file__).resolve().parents[2]))
	memory_path = resolve_chat_memory_path(project_root)
	saved_memory = load_chat_memory(memory_path)
	return {
		"saved_memory": saved_memory,
		"status": "loaded_memory",
	}


def chat_query_node(state: ChatWorkflowState) -> ChatWorkflowState:
	"""Send a user query plus chat context to Claude."""
	assistant = ClaudeChatAssistant(api_key=state.get("config", {}).get("anthropic_api_key"))
	result = assistant.chat(
		user_message=state.get("user_message", ""),
		chat_history=state.get("chat_history", []),
		saved_memory=state.get("saved_memory", ""),
		data_context=state.get("data_context", {}),
	)
	return {
		"chat_result": result,
		"assistant_message": result.get("assistant_message", ""),
		"status": "answered",
	}


def update_chat_state_node(state: ChatWorkflowState) -> ChatWorkflowState:
	"""Return a structured chat payload for Reflex state updates."""
	chat_state = {
		"assistant_message": state.get("assistant_message", ""),
		"source": state.get("chat_result", {}).get("source", "unknown"),
		"model": state.get("chat_result", {}).get("model", "unknown"),
		"saved_memory": state.get("saved_memory", ""),
	}
	return {
		"chat_state": chat_state,
		"status": "success",
	}


def save_response_node(state: ChatWorkflowState) -> ChatWorkflowState:
	"""Persist a selected assistant response in chat memory."""
	project_root = Path(state.get("config", {}).get("project_root", Path(__file__).resolve().parents[2]))
	memory_path = resolve_chat_memory_path(project_root)
	save_result = append_saved_response(
		path=memory_path,
		response_text=state.get("save_target_message", ""),
		source="assistant",
	)
	return {
		"save_result": save_result,
		"status": save_result.get("status", "success"),
	}
