from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class _SafeFormat(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


class PromptLibrary:
    def __init__(self, prompt_dir: Path) -> None:
        self.prompt_dir = prompt_dir

    def render(self, name: str, **context: Any) -> str:
        template_path = self.prompt_dir / f"{name}.md"
        template = template_path.read_text(encoding="utf-8")
        rendered_context = _SafeFormat({key: _render_value(value) for key, value in context.items()})
        return template.format_map(rendered_context)

