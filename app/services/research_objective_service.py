from __future__ import annotations

import json
from typing import Any


SELECTION_CONTEXT_EVENT_TYPE = "selection_context_created"


class ResearchObjectiveService:
    def build_selection_context(
        self,
        *,
        topic: str,
        selected_suggestions: list[object],
        supporting_suggestions: list[object],
    ) -> dict[str, Any]:
        return {
            "topic": topic,
            "selected_directions": [
                self._suggestion_to_context_item(suggestion)
                for suggestion in selected_suggestions
            ],
            "supporting_directions": [
                self._suggestion_to_context_item(suggestion)
                for suggestion in supporting_suggestions
            ],
            "source_discovery_mix": {
                "selected": 0.6,
                "supporting": 0.3,
                "counter_verification": 0.1,
            },
        }

    def context_to_message(self, context: dict[str, Any]) -> str:
        return json.dumps(context, ensure_ascii=True, separators=(",", ":"))

    def context_from_job(self, job: object) -> dict[str, Any] | None:
        events = list(getattr(job, "events", []) or [])
        for event in reversed(events):
            if getattr(event, "type", "") != SELECTION_CONTEXT_EVENT_TYPE:
                continue
            message = getattr(event, "message", None)
            if not message:
                continue
            try:
                context = json.loads(message)
            except json.JSONDecodeError:
                return None
            return context if isinstance(context, dict) else None
        return None

    def objective_from_job(self, job: object) -> str:
        context = self.context_from_job(job)
        if context:
            return self.objective_from_context(context)

        suggestion = getattr(job, "suggestion", None)
        if suggestion is not None:
            return str(getattr(suggestion, "title", "")).strip()
        return f"Research job {getattr(job, 'id', '')}"

    def objective_from_context(self, context: dict[str, Any]) -> str:
        topic = str(context.get("topic") or "Selected research").strip()
        selected = self._items(context.get("selected_directions"))
        supporting = self._items(context.get("supporting_directions"))

        lines = [f"Research topic: {topic}", "Selected research directions:"]
        lines.extend(f"- {item['title']}: {item['summary']}" for item in selected)

        if supporting:
            lines.append("Supporting related directions:")
            lines.extend(f"- {item['title']}: {item['summary']}" for item in supporting)

        lines.extend(
            [
                "Source discovery mix:",
                "- Selected directions carry 60% of source discovery weight.",
                "- Supporting directions carry 30% of source discovery weight.",
                "- Counter-evidence and verification queries carry 10% of source discovery weight.",
            ]
        )
        return "\n".join(lines)

    def display_title_from_job(self, job: object) -> str:
        context = self.context_from_job(job)
        if not context:
            suggestion = getattr(job, "suggestion", None)
            return str(getattr(suggestion, "title", "Selected research")).strip()

        topic = str(context.get("topic") or "Selected research").strip()
        selected = self._items(context.get("selected_directions"))
        if not selected:
            return topic
        if len(selected) == 1:
            return selected[0]["title"]
        return f"{topic}: {len(selected)} selected directions"

    def _suggestion_to_context_item(self, suggestion: object) -> dict[str, Any]:
        return {
            "id": str(getattr(suggestion, "id", "")),
            "rank": int(getattr(suggestion, "rank", 0) or 0),
            "title": str(getattr(suggestion, "title", "")).strip(),
            "summary": str(getattr(suggestion, "summary", "")).strip(),
            "reason": str(getattr(suggestion, "reason", "")).strip(),
        }

    def _items(self, value: object) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []

        items: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            items.append(
                {
                    "title": title,
                    "summary": str(item.get("summary") or "").strip(),
                }
            )
        return items
