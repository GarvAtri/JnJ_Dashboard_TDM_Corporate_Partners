"""Anthropic client integration for deterministic variance explanations."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv


@dataclass(frozen=True)
class ClaudeExplanationResult:
	"""Structured explanation payload returned to workflow callers."""

	message: str
	source: str
	confidence: str

	def as_dict(self) -> dict[str, str]:
		return {
			"message": self.message,
			"source": self.source,
			"confidence": self.confidence,
		}


class ClaudeVarianceExplainer:
	"""Generate variance explanations using Anthropic when configured."""

	def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6") -> None:
		load_dotenv()
		resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
		self._model = model
		self._client = Anthropic(api_key=resolved_key) if resolved_key and resolved_key != "YOUR_KEY_HERE" else None

	def explain_variance(self, context: dict[str, Any]) -> dict[str, str]:
		"""Explain a selected variance driver with a safe fallback path."""
		if self._client is None:
			return self._fallback_explanation(context).as_dict()

		prompt = self._build_prompt(context)
		try:
			response = self._client.messages.create(
				model=self._model,
				max_tokens=220,
				temperature=0,
				messages=[{"role": "user", "content": prompt}],
			)
			message = " ".join(
				block.text.strip()
				for block in response.content
				if hasattr(block, "text") and block.text.strip()
			)
			if not message:
				return self._fallback_explanation(context).as_dict()
			message = self._two_sentence_text(message)
			return ClaudeExplanationResult(
				message=message,
				source="anthropic",
				confidence="high",
			).as_dict()
		except Exception:
			return self._fallback_explanation(context).as_dict()

	def _build_prompt(self, context: dict[str, Any]) -> str:
		return (
			"You are explaining a spend variance to an executive audience. "
			"Use only the structured context provided. Keep the explanation concise, actionable, and specific. "
			"Mention the root cause, its materiality, recent trend behavior, and one recommended next step.\n\n"
			f"Context:\n{json.dumps(context, indent=2, default=str)}"
		)

	def _fallback_explanation(self, context: dict[str, Any]) -> ClaudeExplanationResult:
		root_cause = str(context.get("root_cause") or "the selected driver")
		variance_total = float(context.get("variance_total", 0.0))
		drift_direction = str(context.get("variance_drift", {}).get("direction", "stable"))
		concentration_share = float(context.get("concentration_risk", {}).get("top_share", 0.0)) * 100
		summary = (
			f"{root_cause} is contributing ${variance_total:,.0f} of variance in the current view. "
			f"Variance drift is {drift_direction} and the leading spend segment accounts for {concentration_share:.1f}% of filtered spend."
		)
		return ClaudeExplanationResult(
			message=self._two_sentence_text(summary),
			source="deterministic-fallback",
			confidence="medium",
		)

	def _two_sentence_text(self, text: str) -> str:
		sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
		if len(sentences) >= 2:
			return f"{sentences[0]} {sentences[1]}"
		if len(sentences) == 1:
			base = sentences[0].rstrip(".!?")
			return f"{base}. Review recent transactions linked to this root cause."
		return "Variance signal is available but no explanatory sentence was produced. Review recent transactions linked to this root cause."


class ClaudeChatAssistant:
	"""Chat-oriented Anthropic client with memory and history context."""

	def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6") -> None:
		load_dotenv()
		resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
		self._model = model
		self._client = Anthropic(api_key=resolved_key) if resolved_key and resolved_key != "YOUR_KEY_HERE" else None

	def chat(
		self,
		user_message: str,
		chat_history: list[dict[str, str]],
		saved_memory: str,
		data_context: dict[str, Any],
	) -> dict[str, str]:
		"""Generate a chat answer with explicit memory and context injection."""
		system_prompt = self._build_system_prompt(saved_memory, data_context)
		messages = self._build_messages(user_message, chat_history)
		if self._client is None:
			return {
				"assistant_message": "[API key not configured] " + self._fallback_response(user_message, saved_memory, data_context),
				"source": "deterministic-fallback",
				"model": "fallback",
			}
		try:
			response = self._client.messages.create(
				model=self._model,
				max_tokens=1024,
				temperature=0.2,
				system=system_prompt,
				messages=messages,
			)
			assistant_message = " ".join(
				block.text.strip()
				for block in response.content
				if hasattr(block, "text") and block.text.strip()
			)
			if not assistant_message:
				assistant_message = self._fallback_response(user_message, saved_memory, data_context)
			return {
				"assistant_message": assistant_message,
				"source": "anthropic",
				"model": self._model,
			}
		except Exception as exc:
			error_hint = str(exc)[:200]
			return {
				"assistant_message": f"[Claude API error: {error_hint}] " + self._fallback_response(user_message, saved_memory, data_context),
				"source": "deterministic-fallback",
				"model": "fallback",
			}

	def _build_system_prompt(self, saved_memory: str, data_context: dict[str, Any]) -> str:
		metrics = data_context.get("metrics", {})
		filters = data_context.get("filters", {})
		open_risks = data_context.get("open_risks", [])
		sector_breakdown = data_context.get("sector_breakdown", [])
		variance_breakdown = data_context.get("variance_breakdown", [])
		insights = data_context.get("insights", [])

		sections = [
			"You are Claude, the AI analytics assistant embedded in DashboardAnalyticsForJJ — an executive spend, variance, and risk analytics dashboard.",
			"Answer the user's questions directly and specifically using the live dashboard data provided below.",
			"Be concise, data-driven, and actionable. Use specific numbers from the data. Do not repeat the question back.",
			"",
			"=== CURRENT DASHBOARD METRICS ===",
			f"Total PO Volume: {metrics.get('total_po_volume', 'n/a')}",
			f"Average Variance: {metrics.get('average_variance', 'n/a')}",
			f"Active Risk Count: {metrics.get('active_risk_count', 'n/a')}",
			f"Addressable Spend %: {metrics.get('addressable_spend_pct', 'n/a')}",
			f"Total Spend: {metrics.get('total_spend', 'n/a')}",
		]

		if filters:
			sections.append("")
			sections.append("=== ACTIVE FILTERS ===")
			for key, val in filters.items():
				sections.append(f"{key}: {val}")

		if sector_breakdown:
			sections.append("")
			sections.append("=== SECTOR SPEND BREAKDOWN (top entries) ===")
			for entry in sector_breakdown[:10]:
				sections.append(f"- {entry.get('sector', '?')} ({entry.get('po_status', '?')}): ${entry.get('amount', 0):,.2f}")

		if variance_breakdown:
			sections.append("")
			sections.append("=== ROOT CAUSE VARIANCE BREAKDOWN (top entries) ===")
			for entry in variance_breakdown[:10]:
				sections.append(f"- {entry.get('root_cause', '?')} / {entry.get('sector', '?')}: ${entry.get('variance', 0):,.2f}")

		if open_risks:
			sections.append("")
			sections.append("=== TOP OPEN RISKS ===")
			for risk in open_risks[:8]:
				sections.append(
					f"- [{risk.get('risk_id', '?')}] {risk.get('risk_description', '?')[:80]} "
					f"(owner: {risk.get('risk_owner', '?')}, status: {risk.get('risk_status', '?')}, "
					f"days open: {risk.get('days_open', '?')})"
				)

		if insights:
			sections.append("")
			sections.append("=== DASHBOARD INSIGHTS ===")
			for line in insights:
				sections.append(f"- {line}")

		if data_context.get("variance_explanation"):
			sections.append("")
			sections.append(f"=== LAST VARIANCE EXPLANATION ===\n{data_context['variance_explanation']}")

		if saved_memory and saved_memory.strip():
			sections.append("")
			sections.append("=== SAVED CHAT MEMORY ===")
			sections.append(saved_memory[-2000:])

		return "\n".join(sections)

	def _build_messages(self, user_message: str, chat_history: list[dict[str, str]]) -> list[dict[str, str]]:
		"""Build a proper multi-turn message list for the Claude API."""
		messages: list[dict[str, str]] = []
		recent = chat_history[-16:]
		for entry in recent:
			role = entry.get("role", "user")
			if role in ("user", "assistant") and entry.get("content", "").strip():
				messages.append({"role": role, "content": entry["content"]})
		# Ensure messages alternate properly and start with user
		deduped: list[dict[str, str]] = []
		for msg in messages:
			if deduped and deduped[-1]["role"] == msg["role"]:
				deduped[-1]["content"] += "\n" + msg["content"]
			else:
				deduped.append(msg)
		if deduped and deduped[0]["role"] != "user":
			deduped = deduped[1:]
		deduped.append({"role": "user", "content": user_message})
		return deduped

	def _fallback_response(self, user_message: str, saved_memory: str, data_context: dict[str, Any]) -> str:
		metrics = data_context.get("metrics", {})
		parts = []
		if metrics.get("total_po_volume"):
			parts.append(f"Total PO Volume is {metrics['total_po_volume']}")
		if metrics.get("average_variance"):
			parts.append(f"Average Variance is {metrics['average_variance']}")
		if metrics.get("active_risk_count"):
			parts.append(f"Active Risk Count is {metrics['active_risk_count']}")
		if metrics.get("addressable_spend_pct"):
			parts.append(f"Addressable Spend is {metrics['addressable_spend_pct']}")
		summary = ". ".join(parts) + "." if parts else "No dashboard data available."
		return f"Based on current dashboard data: {summary} (Note: Claude API is unavailable — this is a limited fallback response.)"
