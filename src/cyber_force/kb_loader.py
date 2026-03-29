from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schemas import KnowledgeBaseBundle


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_yaml_list(path: Path) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def load_knowledge_base(kb_dir: Path) -> KnowledgeBaseBundle:
    return KnowledgeBaseBundle(
        constitution_text=read_text(kb_dir / "identity" / "constitution.md"),
        principles=read_yaml(kb_dir / "identity" / "principles.yaml"),
        self_facts=read_yaml(kb_dir / "identity" / "self_facts.yaml"),
        style=read_yaml(kb_dir / "identity" / "style.yaml"),
        modes=read_yaml(kb_dir / "identity" / "modes.yaml"),
        decision_policy=read_yaml(kb_dir / "governance" / "decision_policy.yaml"),
        write_guardrails=read_yaml(kb_dir / "governance" / "write_guardrails.yaml"),
        action_policy=read_yaml(kb_dir / "governance" / "action_policy.yaml"),
        memory_classification=read_yaml(kb_dir / "memory" / "classification.yaml"),
        promotion_policy=read_yaml(kb_dir / "memory" / "promotion_policy.yaml"),
        bootstrap_memories=read_yaml_list(kb_dir / "memory" / "bootstrap.yaml"),
    )
