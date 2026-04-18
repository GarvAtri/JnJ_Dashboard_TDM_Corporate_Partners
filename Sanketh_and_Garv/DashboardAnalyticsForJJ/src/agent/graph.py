"""Compiled LangGraph workflows for dashboard operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agent.chat import (
	ChatWorkflowState,
	chat_query_node,
	load_memory_node,
	save_response_node,
	update_chat_state_node,
)
from src.agent.workflow import (
	WorkflowState,
	compute_metrics_node,
	email_risk_owner_node,
	explain_variance_node,
	export_report_node,
	generate_charts_node,
	load_data_node,
	send_email_report_node,
	update_state_node,
)


def build_dashboard_graph():
	"""Compile the main dashboard workflow graph."""
	graph = StateGraph(WorkflowState)
	graph.add_node("load_data", load_data_node)
	graph.add_node("compute_metrics", compute_metrics_node)
	graph.add_node("generate_charts", generate_charts_node)
	graph.add_node("update_state", update_state_node)
	graph.add_edge(START, "load_data")
	graph.add_edge("load_data", "compute_metrics")
	graph.add_edge("compute_metrics", "generate_charts")
	graph.add_edge("generate_charts", "update_state")
	graph.add_edge("update_state", END)
	return graph.compile()


def build_variance_graph():
	"""Compile the variance explanation workflow."""
	graph = StateGraph(WorkflowState)
	graph.add_node("explain_variance", explain_variance_node)
	graph.add_edge(START, "explain_variance")
	graph.add_edge("explain_variance", END)
	return graph.compile()


def build_email_graph():
	"""Compile the risk owner email workflow."""
	graph = StateGraph(WorkflowState)
	graph.add_node("email_risk_owner", email_risk_owner_node)
	graph.add_edge(START, "email_risk_owner")
	graph.add_edge("email_risk_owner", END)
	return graph.compile()


def build_export_graph():
	"""Compile the report export workflow."""
	graph = StateGraph(WorkflowState)
	graph.add_node("export_report", export_report_node)
	graph.add_edge(START, "export_report")
	graph.add_edge("export_report", END)
	return graph.compile()


def build_send_email_report_graph():
	"""Compile the daily email-report workflow."""
	graph = StateGraph(WorkflowState)
	graph.add_node("send_email_report", send_email_report_node)
	graph.add_edge(START, "send_email_report")
	graph.add_edge("send_email_report", END)
	return graph.compile()


def build_chat_graph():
	"""Compile the chatbot query workflow."""
	graph = StateGraph(ChatWorkflowState)
	graph.add_node("load_memory", load_memory_node)
	graph.add_node("chat_query", chat_query_node)
	graph.add_node("update_state", update_chat_state_node)
	graph.add_edge(START, "load_memory")
	graph.add_edge("load_memory", "chat_query")
	graph.add_edge("chat_query", "update_state")
	graph.add_edge("update_state", END)
	return graph.compile()


def build_save_response_graph():
	"""Compile the save-response workflow."""
	graph = StateGraph(ChatWorkflowState)
	graph.add_node("save_response", save_response_node)
	graph.add_edge(START, "save_response")
	graph.add_edge("save_response", END)
	return graph.compile()


def run_dashboard_workflow(filters: dict[str, str] | None = None, project_root: str | Path | None = None) -> dict[str, Any]:
	"""Execute the main dashboard workflow and return a structured payload."""
	try:
		result = build_dashboard_graph().invoke(
			{
				"filters": filters or {},
				"config": {"project_root": str(project_root) if project_root else str(Path(__file__).resolve().parents[2])},
			}
		)
		return {"status": "success", **result}
	except Exception as exc:
		return {
			"status": "error",
			"errors": [str(exc)],
			"dashboard_state": {
				"metrics": {},
				"charts": {},
				"tables": {},
				"filter_options": {},
				"filters": filters or {},
				"insights": [],
				"status": "error",
			},
		}


def run_variance_explanation_workflow(
	filters: dict[str, str],
	root_cause: str,
	project_root: str | Path | None = None,
) -> dict[str, Any]:
	"""Execute the variance explanation workflow on the current filter context."""
	base_result = run_dashboard_workflow(filters=filters, project_root=project_root)
	if base_result["status"] != "success":
		return base_result
	state = {
		**base_result,
		"filters": {**filters, "selected_root_cause": root_cause},
	}
	explained = build_variance_graph().invoke(state)
	return {"status": "success", **explained}


def run_email_workflow(
	filters: dict[str, str],
	risk_id: str,
	project_root: str | Path | None = None,
) -> dict[str, Any]:
	"""Execute the risk-owner email workflow."""
	base_result = run_dashboard_workflow(filters=filters, project_root=project_root)
	if base_result["status"] != "success":
		return base_result
	state = {
		**base_result,
		"filters": {**filters, "selected_risk_id": risk_id},
	}
	emailed = build_email_graph().invoke(state)
	return {"status": emailed.get("status", "success"), **emailed}


def run_export_workflow(filters: dict[str, str], project_root: str | Path | None = None) -> dict[str, Any]:
	"""Execute the PDF export workflow."""
	base_result = run_dashboard_workflow(filters=filters, project_root=project_root)
	if base_result["status"] != "success":
		return base_result
	exported = build_export_graph().invoke(base_result)
	return {"status": exported.get("status", "success"), **exported}


def run_send_email_report_workflow(
	filters: dict[str, str],
	project_root: str | Path | None = None,
) -> dict[str, Any]:
	"""Execute send_email_report using today's analytics workflow payload."""
	base_result = run_dashboard_workflow(filters=filters, project_root=project_root)
	if base_result["status"] != "success":
		return base_result
	report_result = build_send_email_report_graph().invoke(base_result)
	return {"status": report_result.get("status", "success"), **report_result}


def run_chat_workflow(
	user_message: str,
	chat_history: list[dict[str, str]],
	data_context: dict[str, Any],
	project_root: str | Path | None = None,
	anthropic_api_key: str | None = None,
) -> dict[str, Any]:
	"""Execute load_memory -> chat_query -> update_state workflow."""
	resolved_project_root = str(project_root) if project_root else str(Path(__file__).resolve().parents[2])
	result = build_chat_graph().invoke(
		{
			"config": {
				"project_root": resolved_project_root,
				"anthropic_api_key": anthropic_api_key,
			},
			"user_message": user_message,
			"chat_history": chat_history,
			"data_context": data_context,
		}
	)
	return {"status": result.get("status", "success"), **result}


def run_save_response_workflow(
	message_text: str,
	project_root: str | Path | None = None,
) -> dict[str, Any]:
	"""Persist selected assistant response through the save workflow."""
	resolved_project_root = str(project_root) if project_root else str(Path(__file__).resolve().parents[2])
	result = build_save_response_graph().invoke(
		{
			"config": {"project_root": resolved_project_root},
			"save_target_message": message_text,
		}
	)
	return {"status": result.get("status", "success"), **result}
