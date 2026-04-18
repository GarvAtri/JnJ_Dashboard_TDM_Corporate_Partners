"""Claude-powered column relevance analyzer.

Takes scraped file metadata (column names, types, samples) and uses
Claude to classify each column into relevance categories for the
enterprise spend/risk analytics dashboard.
"""

from __future__ import annotations

import json
import os
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv


# Known dashboard schema — the columns the analytics engine actually uses.
DASHBOARD_SCHEMA = {
    "spend_header": {
        "PO_Number", "PO_Status", "Business_Sector",
        "Addressable_Flag", "PO_Total_Amount", "Last_Updated_Timestamp",
    },
    "spend_detail": {
        "PO_Number", "Sector", "Spend_Amount", "Variance_vs_Budget",
        "Root_Cause_Code", "Last_Updated_Timestamp",
    },
    "risk_register": {
        "Risk #", "Risk Description", "Risk Owner", "Risk Status",
        "Risk Category", "Risk Level", "Risk ERM Type",
        "Open Date", "Closed Date",
    },
}

_SYSTEM_PROMPT_TEMPLATE = """\
You are a data analyst for an enterprise procurement / spend analytics platform \
inspired by SAP ERP systems used at companies like Johnson & Johnson.

The dashboard tracks:
- Purchase Order spend across business sectors
- Budget variance and root-cause analysis
- Risk register governance and aging

Known schema groups:
1. Spend Header: {spend_header}
2. Spend Detail: {spend_detail}
3. Risk Register: {risk_register}

Your task: Given a list of column names (with optional data types and sample \
values) from an uploaded file, classify EVERY column into one of these categories:

- "spend"      — relevant to spend/PO analysis
- "variance"   — relevant to budget variance tracking
- "risk"       — relevant to risk register / governance
- "time"       — temporal / date fields useful for trend analysis
- "identifier" — ID or key fields that link records
- "metadata"   — descriptive but not directly analytical
- "irrelevant" — not useful for the dashboard analytics

For each column also provide:
- "confidence": "high" | "medium" | "low"
- "reason": one-sentence explanation
- "maps_to": the closest known dashboard column it maps to, or null

Respond with ONLY valid JSON in this exact structure:
"""

_JSON_SCHEMA_EXAMPLE = """{
  "file_summary": "one paragraph describing what this data is about",
  "columns": [
    {
      "name": "column_name",
      "category": "spend",
      "confidence": "high",
      "reason": "...",
      "maps_to": "PO_Total_Amount"
    }
  ],
  "relevant_count": 8,
  "irrelevant_count": 2,
  "recommendation": "one paragraph — what the user should do with this file"
}"""


def _build_system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(
        spend_header=", ".join(sorted(DASHBOARD_SCHEMA["spend_header"])),
        spend_detail=", ".join(sorted(DASHBOARD_SCHEMA["spend_detail"])),
        risk_register=", ".join(sorted(DASHBOARD_SCHEMA["risk_register"])),
    ) + _JSON_SCHEMA_EXAMPLE


def _build_user_prompt(file_insights: list[dict[str, Any]]) -> str:
    """Build the user message from scraped file metadata."""
    parts: list[str] = []
    for insight in file_insights:
        name = insight.get("name", "unknown")
        ftype = insight.get("type", "unknown")
        summary = insight.get("summary", "")
        columns = insight.get("column_names", [])

        part = f"### File: {name} ({ftype})\n{summary}\n"

        # Add column details if available (Excel sheet details)
        sheet_details = insight.get("sheet_details", [])
        if sheet_details:
            for sheet in sheet_details:
                part += f"\n**Sheet: {sheet['sheet_name']}** ({sheet['row_count']} rows)\n"
                for col in sheet.get("columns", []):
                    samples = ", ".join(col.get("sample_values", [])[:3])
                    part += f"- `{col['name']}` ({col['dtype']}) — samples: [{samples}]\n"
        elif columns:
            # JSON or PDF columns
            col_types = insight.get("column_types", {})
            for col in columns:
                dtype = col_types.get(col, "unknown")
                part += f"- `{col}` ({dtype})\n"

        # PDF headings
        headings = insight.get("headings", [])
        if headings:
            part += "\nDetected headings:\n"
            for h in headings[:15]:
                part += f"- {h}\n"

        parts.append(part)

    return (
        "Analyze the following uploaded file(s) and classify every column.\n\n"
        + "\n---\n".join(parts)
    )


class ClaudeColumnAnalyzer:
    """Use Claude to classify column relevance for dashboard analytics."""

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6") -> None:
        load_dotenv()
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._model = model
        self._client: Anthropic | None = (
            Anthropic(api_key=resolved_key)
            if resolved_key and resolved_key != "YOUR_KEY_HERE"
            else None
        )

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def analyze(self, file_insights: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze scraped file insights and return relevance report."""
        if not file_insights:
            return {"error": "No file insights provided."}

        if self._client is None:
            return self._fallback_analysis(file_insights)

        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(file_insights)

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = " ".join(
                block.text.strip()
                for block in response.content
                if hasattr(block, "text") and block.text.strip()
            )
            # Extract JSON from response (handle markdown code blocks)
            json_str = raw
            if "```" in json_str:
                # Strip markdown code fences
                import re
                match = re.search(r"```(?:json)?\s*([\s\S]*?)```", json_str)
                if match:
                    json_str = match.group(1)

            result = json.loads(json_str)
            result["source"] = "claude"
            return result

        except json.JSONDecodeError:
            return {
                "error": "Claude response was not valid JSON.",
                "raw_response": raw[:2000] if 'raw' in dir() else "",
                "source": "claude-error",
            }
        except Exception as exc:
            return self._fallback_analysis(file_insights, error=str(exc))

    def _fallback_analysis(
        self, file_insights: list[dict[str, Any]], error: str | None = None,
    ) -> dict[str, Any]:
        """Deterministic fallback when Claude is unavailable."""
        all_known = set()
        for group in DASHBOARD_SCHEMA.values():
            all_known.update(col.lower() for col in group)

        columns_report: list[dict[str, Any]] = []
        relevant = 0
        irrelevant = 0

        for insight in file_insights:
            for col_name in insight.get("column_names", []):
                col_lower = col_name.lower().strip()
                # Check exact match
                matched_group = None
                maps_to = None
                for group_name, group_cols in DASHBOARD_SCHEMA.items():
                    for known_col in group_cols:
                        if col_lower == known_col.lower():
                            matched_group = group_name
                            maps_to = known_col
                            break
                    if matched_group:
                        break

                if matched_group:
                    category_map = {
                        "spend_header": "spend",
                        "spend_detail": "variance",
                        "risk_register": "risk",
                    }
                    columns_report.append({
                        "name": col_name,
                        "category": category_map.get(matched_group, "metadata"),
                        "confidence": "high",
                        "reason": f"Exact match to known {matched_group} schema.",
                        "maps_to": maps_to,
                    })
                    relevant += 1
                elif any(kw in col_lower for kw in ("date", "time", "timestamp", "period", "year", "month")):
                    columns_report.append({
                        "name": col_name,
                        "category": "time",
                        "confidence": "medium",
                        "reason": "Contains temporal keyword.",
                        "maps_to": None,
                    })
                    relevant += 1
                elif any(kw in col_lower for kw in ("id", "number", "code", "key", "#")):
                    columns_report.append({
                        "name": col_name,
                        "category": "identifier",
                        "confidence": "medium",
                        "reason": "Appears to be an identifier or key field.",
                        "maps_to": None,
                    })
                    relevant += 1
                elif any(kw in col_lower for kw in ("spend", "amount", "cost", "price", "budget", "total", "variance")):
                    columns_report.append({
                        "name": col_name,
                        "category": "spend",
                        "confidence": "medium",
                        "reason": "Contains financial keyword.",
                        "maps_to": None,
                    })
                    relevant += 1
                elif any(kw in col_lower for kw in ("risk", "severity", "impact", "likelihood", "owner")):
                    columns_report.append({
                        "name": col_name,
                        "category": "risk",
                        "confidence": "medium",
                        "reason": "Contains risk-related keyword.",
                        "maps_to": None,
                    })
                    relevant += 1
                else:
                    columns_report.append({
                        "name": col_name,
                        "category": "irrelevant",
                        "confidence": "low",
                        "reason": "No match to known schema or analytics keywords.",
                        "maps_to": None,
                    })
                    irrelevant += 1

        file_names = ", ".join(i.get("name", "?") for i in file_insights)
        return {
            "file_summary": f"Deterministic analysis of {file_names}. Claude API was not available.",
            "columns": columns_report,
            "relevant_count": relevant,
            "irrelevant_count": irrelevant,
            "recommendation": (
                "Review the irrelevant columns manually. "
                "Connect your Anthropic API key for deeper AI-powered analysis."
                + (f" (Error: {error})" if error else "")
            ),
            "source": "deterministic-fallback",
        }
