from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .provider import OpenAICompatibleProvider, ProviderUnavailable
from .schemas import (
    AuditRecord,
    ConflictRecord,
    InputEvent,
    MemoryRecord,
    MemoryStatus,
    MemoryZone,
    PromotionRecord,
    utc_now,
)


ENTRY_RE = re.compile(r"^- \[(?P<memory_id>[^\]]+)\]\s+(?P<fact>.+)$")
DATE_STEM_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MEMORY_SECTIONS = (
    "工作与职业",
    "思维方式与偏好",
    "当前关注的事",
    "重要决策记录",
    "关于AI和产品的认知",
)
MEMORY_SECTION_COMMENTS = {
    "工作与职业": "<!-- 求职方向、工作经历、职业判断等 -->",
    "思维方式与偏好": "<!-- 做决策的方式、审美偏好、沟通风格等 -->",
    "当前关注的事": "<!-- 正在做的项目、近期优先级等 -->",
    "重要决策记录": "<!-- 做过的重要决定和背后的理由 -->",
    "关于AI和产品的认知": "<!-- 对AI产品、Agent设计的理解和判断 -->",
}
LOG_SECTIONS = (
    "原始输入",
    "提取的关键事实",
    "待升级到长期记忆",
)
LOG_SECTION_COMMENTS = {
    "原始输入": "<!-- 当天收到的消息原文摘要 -->",
    "提取的关键事实": "<!-- 模型提取后的结构化条目 -->",
    "待升级到长期记忆": "<!-- 标记需要在下次整理时写入 MEMORY.md 的条目 -->",
}
ZONE_TO_LOG_SECTION = {
    MemoryZone.short_term.value: "提取的关键事实",
    MemoryZone.observation.value: "提取的关键事实",
    MemoryZone.candidate.value: "待升级到长期记忆",
}
LEGACY_LAYER_TO_ZONE = {
    "principles": MemoryZone.long_term.value,
    "identity": MemoryZone.long_term.value,
    "style": MemoryZone.long_term.value,
    "modes": MemoryZone.long_term.value,
    "ephemeral": MemoryZone.short_term.value,
    "long_term": MemoryZone.long_term.value,
    "candidate": MemoryZone.candidate.value,
    "observation": MemoryZone.observation.value,
    "short_term": MemoryZone.short_term.value,
}


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class ExtractedMemoryItem(BaseModel):
    memory_id: str
    fact: str
    target_zone: str
    section: str | None = None
    status: str = "added"
    previous_fact: str | None = None


class AddMemoryResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    extracted_items: list[ExtractedMemoryItem] = Field(default_factory=list)
    updated_long_term: bool = False
    memory_path: Path
    log_path: Path | None = None


class SearchMemoryHit(BaseModel):
    memory_id: str
    fact: str
    target_zone: str
    source_path: str
    section: str | None = None
    reason: str


class MarkdownMemoryStore:
    def __init__(
        self,
        settings: Settings | None = None,
        provider: object | None = None,
        now_fn: Callable[[], datetime] = utc_now,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.provider = provider or OpenAICompatibleProvider(self.settings.memory_model)
        self.now_fn = now_fn
        self.memory_root = self.settings.paths.kb_dir / "memory"
        self.memory_path = self.memory_root / "MEMORY.md"
        self.logs_dir = self.memory_root / "logs"
        self.scanned_path = self.memory_root / "scanned.json"
        self._ensure_scaffold()

    def add_memory(self, content: str, layer: str) -> AddMemoryResult:
        target_zone = self._normalize_target_zone(layer)
        extracted_items = self._extract_memories(content, target_zone)
        result = AddMemoryResult(
            extracted_items=extracted_items,
            memory_path=self.memory_path,
        )

        if not extracted_items:
            self._update_scanned_metadata()
            return result

        if target_zone == MemoryZone.long_term.value:
            sections = self._read_memory_sections()
            changed = False
            for item in extracted_items:
                section = item.section or "思维方式与偏好"
                values = sections.setdefault(section, {})
                existing_fact = values.get(item.memory_id)
                if existing_fact == item.fact:
                    item.status = "duplicate"
                    continue
                if existing_fact is not None:
                    item.status = "updated"
                    item.previous_fact = existing_fact
                values[item.memory_id] = item.fact
                changed = True
            if changed:
                self._write_memory_sections(sections)
            result.updated_long_term = changed
        else:
            log_path = self.logs_dir / f"{self.now_fn().date().isoformat()}.md"
            log_doc = self._read_log_document(log_path)
            summary = self._summarize_input(content)
            if summary and summary not in log_doc["原始输入"]:
                log_doc["原始输入"].append(summary)
            section_name = ZONE_TO_LOG_SECTION[target_zone]
            changed = False
            section_entries = log_doc[section_name]
            for item in extracted_items:
                existing_fact = section_entries.get(item.memory_id)
                if existing_fact == item.fact:
                    item.status = "duplicate"
                    continue
                if existing_fact is not None:
                    item.status = "updated"
                    item.previous_fact = existing_fact
                section_entries[item.memory_id] = item.fact
                changed = True
            if changed:
                self._write_log_document(log_path, log_doc)
            result.log_path = log_path

        self._update_scanned_metadata()
        return result

    def mirror_record(self, record: MemoryRecord) -> None:
        fact = self._normalize_text(record.statement)
        if not fact:
            return

        if record.zone == MemoryZone.long_term:
            item = ExtractedMemoryItem(
                memory_id=record.memory_key,
                fact=fact,
                target_zone=record.zone.value,
                section=self._section_for_record(record),
            )
            self._apply_long_term_items([item])
            self._update_scanned_metadata()
            return

        target_zone = record.zone.value
        log_path = self.logs_dir / f"{record.last_seen_at.date().isoformat()}.md"
        log_doc = self._read_log_document(log_path)
        summary = self._normalize_text(record.reason or record.source)
        if summary and summary not in log_doc["原始输入"]:
            log_doc["原始输入"].append(summary)
        section_name = ZONE_TO_LOG_SECTION.get(target_zone)
        if section_name:
            log_doc[section_name][record.memory_key] = fact
            self._write_log_document(log_path, log_doc)
            self._update_scanned_metadata()

    def search_memory(self, query: str, layer: str | None = None) -> list[SearchMemoryHit]:
        target_zone = self._normalize_target_zone(layer) if layer else None
        candidates = self._collect_search_candidates(target_zone)
        if not candidates:
            return []

        if self._provider_enabled():
            try:
                return self._search_with_model(query, candidates)
            except Exception:
                pass

        return self._search_with_keywords(query, candidates)

    def _ensure_scaffold(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists():
            self._write_memory_sections({section: {} for section in MEMORY_SECTIONS})
        if not self.scanned_path.exists():
            self._write_scanned_json(
                {
                    "updated_at": self.now_fn().isoformat(),
                    "long_term_count": 0,
                    "log_files": [],
                    "codex": {},
                    "claude_code": {},
                }
            )

    def _provider_enabled(self) -> bool:
        if not getattr(self.provider, "configured", False):
            return False
        config = getattr(self.provider, "config", None)
        if config is None:
            return True
        return bool(config.base_url and config.model)

    def _extract_memories(self, content: str, target_zone: str) -> list[ExtractedMemoryItem]:
        if self._provider_enabled():
            try:
                items = self._extract_memories_with_model(content, target_zone)
                if items:
                    return items
            except Exception:
                pass
        return self._extract_memories_fallback(content, target_zone)

    def _extract_memories_with_model(
        self,
        content: str,
        target_zone: str,
    ) -> list[ExtractedMemoryItem]:
        payload = self.provider.complete_json(
            system_prompt=(
                "你负责把输入内容提炼成结构化记忆。"
                "不要保留原文，改写成简洁陈述句。只返回 JSON。"
            ),
            user_prompt=json.dumps(
                {
                    "target_zone": target_zone,
                    "instruction": (
                        "从以下内容中提取值得记住的关键事实，"
                        "每条不超过30字；如果写入长期记忆，还要选择一个最合适的 section。"
                    ),
                    "content": content,
                    "memory_sections": list(MEMORY_SECTIONS),
                    "schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "memory_id": {"type": "string"},
                                "fact": {"type": "string"},
                                "section": {"type": "string"},
                            },
                            "required": ["memory_id", "fact"],
                        },
                    },
                },
                ensure_ascii=False,
            ),
        )
        if isinstance(payload, dict):
            payload = payload.get("items", payload.get("results", []))
        if not isinstance(payload, list):
            raise ProviderUnavailable("Memory extraction payload is not a list.")

        items: list[ExtractedMemoryItem] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            memory_id = str(row.get("memory_id", "")).strip()
            fact = self._normalize_text(str(row.get("fact", "")))
            section = str(row.get("section", "")).strip() or None
            if not memory_id or not fact:
                continue
            items.append(
                ExtractedMemoryItem(
                    memory_id=memory_id,
                    fact=fact,
                    target_zone=target_zone,
                    section=section if section in MEMORY_SECTIONS else None,
                )
            )
        return items

    def _extract_memories_fallback(
        self,
        content: str,
        target_zone: str,
    ) -> list[ExtractedMemoryItem]:
        text = self._normalize_text(content)
        lowered = text.lower()

        birthday_match = re.search(r"生日(?:是|为)?\s*(\d{4}-\d{2}-\d{2})", text)
        if birthday_match:
            return [
                ExtractedMemoryItem(
                    memory_id="identity.birthday",
                    fact=f"生日：{birthday_match.group(1)}",
                    target_zone=target_zone,
                    section="工作与职业" if target_zone == MemoryZone.long_term.value else None,
                )
            ]

        city_match = re.search(r"常驻城市(?:是|为)?\s*([A-Za-z\u4e00-\u9fff]+)", text)
        if city_match:
            return [
                ExtractedMemoryItem(
                    memory_id="identity.city",
                    fact=f"常驻城市：{city_match.group(1)}",
                    target_zone=target_zone,
                    section="工作与职业" if target_zone == MemoryZone.long_term.value else None,
                )
            ]

        if "求职" in text or "agent pm" in lowered or "创业公司" in text:
            return [
                ExtractedMemoryItem(
                    memory_id="identity.job_target",
                    fact="主理人的求职目标是 AI-native 创业公司的 Agent PM",
                    target_zone=target_zone,
                    section="工作与职业" if target_zone == MemoryZone.long_term.value else None,
                )
            ]

        if "心情很差" in text or "心情差" in text:
            suffix = "，暂缓重大决定" if "重大决定" in text else ""
            return [
                ExtractedMemoryItem(
                    memory_id=f"{target_zone}.mood",
                    fact=f"临时状态：心情差{suffix}",
                    target_zone=target_zone,
                )
            ]

        if "重大决定" in text and ("不要做" in text or "先不要做" in text):
            return [
                ExtractedMemoryItem(
                    memory_id=f"{target_zone}.major_decision_guard",
                    fact="临时状态：暂缓重大决定",
                    target_zone=target_zone,
                )
            ]

        if not text:
            return []

        memory_id = self._slugify_memory_id(text, target_zone)
        section = self._infer_section(text) if target_zone == MemoryZone.long_term.value else None
        return [
            ExtractedMemoryItem(
                memory_id=memory_id,
                fact=self._limit_fact_length(text),
                target_zone=target_zone,
                section=section,
            )
        ]

    def _apply_long_term_items(self, items: list[ExtractedMemoryItem]) -> bool:
        sections = self._read_memory_sections()
        changed = False
        for item in items:
            section = item.section or "思维方式与偏好"
            values = sections.setdefault(section, {})
            existing_fact = values.get(item.memory_id)
            if existing_fact == item.fact:
                item.status = "duplicate"
                continue
            if existing_fact is not None:
                item.status = "updated"
                item.previous_fact = existing_fact
            values[item.memory_id] = item.fact
            changed = True
        if changed:
            self._write_memory_sections(sections)
        return changed

    def _collect_search_candidates(self, target_zone: str | None) -> list[SearchMemoryHit]:
        hits: list[SearchMemoryHit] = []
        if target_zone in (None, MemoryZone.long_term.value):
            hits.extend(self._memory_hits())

        if target_zone in (None, MemoryZone.short_term.value, MemoryZone.observation.value, MemoryZone.candidate.value):
            cutoff = self.now_fn().date() - timedelta(days=6)
            for log_path in sorted(self.logs_dir.glob("*.md")):
                if not DATE_STEM_RE.match(log_path.stem):
                    continue
                try:
                    log_date = datetime.fromisoformat(log_path.stem).date()
                except ValueError:
                    continue
                if log_date < cutoff:
                    continue
                hits.extend(self._log_hits(log_path))

        if target_zone is None:
            return hits
        return [hit for hit in hits if hit.target_zone == target_zone]

    def _search_with_model(
        self,
        query: str,
        candidates: list[SearchMemoryHit],
    ) -> list[SearchMemoryHit]:
        payload = self.provider.complete_json(
            system_prompt=(
                "你负责从记忆候选中选出与 query 最相关的条目。"
                "只返回 JSON 数组，每项至少包含 memory_id。"
            ),
            user_prompt=json.dumps(
                {
                    "query": query,
                    "candidates": [item.model_dump(mode="json") for item in candidates],
                },
                ensure_ascii=False,
            ),
        )
        if isinstance(payload, dict):
            payload = payload.get("items", payload.get("results", []))
        if not isinstance(payload, list):
            raise ProviderUnavailable("Memory search payload is not a list.")

        indexed = {item.memory_id: item for item in candidates}
        selected: list[SearchMemoryHit] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            memory_id = str(row.get("memory_id", "")).strip()
            hit = indexed.get(memory_id)
            if hit is None:
                continue
            reason = self._normalize_text(str(row.get("reason", ""))) or "model match"
            selected.append(hit.model_copy(update={"reason": reason}))
        return selected or self._search_with_keywords(query, candidates)

    def _search_with_keywords(
        self,
        query: str,
        candidates: list[SearchMemoryHit],
    ) -> list[SearchMemoryHit]:
        terms = self._search_terms(query)
        scored: list[tuple[int, SearchMemoryHit]] = []
        for hit in candidates:
            haystack = f"{hit.memory_id} {hit.fact}".lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, hit))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in scored]

    def _read_memory_sections(self) -> dict[str, dict[str, str]]:
        sections = {section: {} for section in MEMORY_SECTIONS}
        if not self.memory_path.exists():
            return sections

        current_section: str | None = None
        for raw_line in self.memory_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("## "):
                title = line[3:].strip()
                current_section = title if title in sections else None
                continue
            if current_section is None:
                continue
            match = ENTRY_RE.match(line)
            if match:
                sections[current_section][match.group("memory_id")] = match.group("fact").strip()
        return sections

    def _write_memory_sections(self, sections: dict[str, dict[str, str]]) -> None:
        lines = ["# 主理人的长期记忆", ""]
        for section in MEMORY_SECTIONS:
            lines.append(f"## {section}")
            lines.append(MEMORY_SECTION_COMMENTS[section])
            values = sections.get(section, {})
            for memory_id, fact in values.items():
                lines.append(f"- [{memory_id}] {fact}")
            lines.append("")
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _read_log_document(self, path: Path) -> dict[str, Any]:
        document: dict[str, Any] = {
            "原始输入": [],
            "提取的关键事实": {},
            "待升级到长期记忆": {},
        }
        if not path.exists():
            return document

        current_section: str | None = None
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("## "):
                title = line[3:].strip()
                current_section = title if title in document else None
                continue
            if current_section is None:
                continue
            if current_section == "原始输入":
                if line.startswith("- "):
                    document[current_section].append(line[2:].strip())
                continue
            match = ENTRY_RE.match(line)
            if match:
                document[current_section][match.group("memory_id")] = match.group("fact").strip()
        return document

    def _write_log_document(self, path: Path, document: dict[str, Any]) -> None:
        lines = [f"# {path.stem} 日志", ""]
        for section in LOG_SECTIONS:
            lines.append(f"## {section}")
            lines.append(LOG_SECTION_COMMENTS[section])
            if section == "原始输入":
                for summary in document[section]:
                    lines.append(f"- {summary}")
            else:
                for memory_id, fact in document[section].items():
                    lines.append(f"- [{memory_id}] {fact}")
            lines.append("")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _memory_hits(self) -> list[SearchMemoryHit]:
        sections = self._read_memory_sections()
        hits: list[SearchMemoryHit] = []
        for section, values in sections.items():
            for memory_id, fact in values.items():
                hits.append(
                    SearchMemoryHit(
                        memory_id=memory_id,
                        fact=fact,
                        target_zone=MemoryZone.long_term.value,
                        source_path=str(self.memory_path),
                        section=section,
                        reason="fallback match",
                    )
                )
        return hits

    def _log_hits(self, path: Path) -> list[SearchMemoryHit]:
        document = self._read_log_document(path)
        hits: list[SearchMemoryHit] = []
        for section in ("提取的关键事实", "待升级到长期记忆"):
            target_zone = MemoryZone.candidate.value if section == "待升级到长期记忆" else MemoryZone.observation.value
            for memory_id, fact in document[section].items():
                prefix = memory_id.split(".", 1)[0]
                if prefix in {MemoryZone.short_term.value, MemoryZone.observation.value, MemoryZone.candidate.value}:
                    target_zone = prefix
                hits.append(
                    SearchMemoryHit(
                        memory_id=memory_id,
                        fact=fact,
                        target_zone=target_zone,
                        source_path=str(path),
                        section=section,
                        reason="fallback match",
                    )
                )
        return hits

    def _section_for_record(self, record: MemoryRecord) -> str:
        text = f"{record.memory_key} {record.statement} {' '.join(record.tags)}"
        return self._infer_section(text)

    def _infer_section(self, text: str) -> str:
        lowered = text.lower()
        if any(token in text for token in ("求职", "职业", "工作", "岗位", "公司", "创业")):
            return "工作与职业"
        if any(token in text for token in ("当前", "近期", "优先级", "正在做", "项目")):
            return "当前关注的事"
        if any(token in text for token in ("决定", "决策", "为什么这样做", "取舍")):
            return "重要决策记录"
        if "ai" in lowered or any(token in text for token in ("Agent", "产品", "模型")):
            return "关于AI和产品的认知"
        return "思维方式与偏好"

    def _summarize_input(self, content: str) -> str:
        summary = self._limit_fact_length(self._normalize_text(content), limit=60)
        if not summary:
            return ""
        return f"摘要：{summary}"

    def _update_scanned_metadata(self) -> None:
        payload = self._read_scanned_json()
        payload["updated_at"] = self.now_fn().isoformat()
        payload["long_term_count"] = sum(
            len(values) for values in self._read_memory_sections().values()
        )
        payload["log_files"] = sorted(path.name for path in self.logs_dir.glob("*.md"))
        payload.setdefault("codex", {})
        payload.setdefault("claude_code", {})
        self._write_scanned_json(payload)

    def _read_scanned_json(self) -> dict[str, Any]:
        default = {
            "version": 1,
            "updated_at": self.now_fn().isoformat(),
            "long_term_count": 0,
            "log_files": [],
            "codex": {},
            "claude_code": {},
        }
        if not self.scanned_path.exists():
            return default

        payload = json.loads(self.scanned_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return default

        merged = default | payload
        merged["codex"] = payload.get("codex", {}) if isinstance(payload.get("codex"), dict) else {}
        merged["claude_code"] = (
            payload.get("claude_code", {})
            if isinstance(payload.get("claude_code"), dict)
            else {}
        )
        log_files = payload.get("log_files", [])
        merged["log_files"] = log_files if isinstance(log_files, list) else []
        return merged

    def _write_scanned_json(self, payload: dict[str, Any]) -> None:
        self.scanned_path.parent.mkdir(parents=True, exist_ok=True)
        self.scanned_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _normalize_target_zone(self, layer: str | None) -> str:
        if layer is None:
            return MemoryZone.long_term.value
        normalized = str(layer).strip().lower()
        if normalized in LEGACY_LAYER_TO_ZONE:
            return LEGACY_LAYER_TO_ZONE[normalized]
        if normalized in {zone.value for zone in MemoryZone}:
            return normalized
        raise ValueError(f"Unsupported memory layer: {layer}")

    def _slugify_memory_id(self, text: str, target_zone: str) -> str:
        slug = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", text).strip("_").lower()
        return f"{target_zone}.{slug[:40] or 'item'}"

    def _limit_fact_length(self, text: str, limit: int = 30) -> str:
        compact = self._normalize_text(text)
        if len(compact) <= limit:
            return compact
        return compact[: limit - 1].rstrip() + "…"

    def _normalize_text(self, text: str) -> str:
        compact = text.replace("\r", " ").replace("\n", " ")
        compact = re.sub(r"\s+", " ", compact).strip()
        return compact.replace("。", "").strip()

    def _search_terms(self, query: str) -> set[str]:
        normalized = self._normalize_text(query).lower()
        if not normalized:
            return set()
        parts = re.split(r"[\s,，。:：;；!?！？]+", normalized)
        tokens = {part for part in parts if len(part) >= 2}
        collapsed = re.sub(r"\s+", "", normalized)
        if collapsed:
            tokens.add(collapsed)

        for segment in re.findall(r"[\u4e00-\u9fff]{2,}", collapsed):
            tokens.add(segment)
            for size in (2, 3):
                if len(segment) <= size:
                    tokens.add(segment)
                    continue
                tokens.update(segment[index : index + size] for index in range(len(segment) - size + 1))

        return tokens | {normalized}


class FileBackedStore:
    def __init__(
        self,
        state_dir: Path,
        settings: Settings | None = None,
        markdown_store: MarkdownMemoryStore | None = None,
    ) -> None:
        self.state_dir = state_dir
        self.events_path = state_dir / "events.jsonl"
        self.short_term_path = state_dir / "short_term.jsonl"
        self.observations_path = state_dir / "observations.jsonl"
        self.candidates_path = state_dir / "candidates.jsonl"
        self.memories_path = state_dir / "memories.jsonl"
        self.promotions_path = state_dir / "promotions.jsonl"
        self.conflicts_path = state_dir / "conflicts.jsonl"
        self.audit_path = state_dir / "audit.jsonl"
        self.markdown_store = markdown_store or (
            MarkdownMemoryStore(settings=settings) if settings is not None else None
        )

    def append_event(self, event: InputEvent) -> None:
        _append_jsonl(self.events_path, event.model_dump(mode="json"))

    def append_memory(self, record: MemoryRecord) -> None:
        _append_jsonl(self._path_for_zone(record.zone), record.model_dump(mode="json"))
        if self.markdown_store is not None:
            self.markdown_store.mirror_record(record)

    def append_memories(self, records: Iterable[MemoryRecord]) -> None:
        for record in records:
            self.append_memory(record)

    def append_promotion(self, record: PromotionRecord) -> None:
        _append_jsonl(self.promotions_path, record.model_dump(mode="json"))

    def append_conflict(self, record: ConflictRecord) -> None:
        _append_jsonl(self.conflicts_path, record.model_dump(mode="json"))

    def append_audit(self, record: AuditRecord) -> None:
        _append_jsonl(self.audit_path, record.model_dump(mode="json"))

    def list_zone(self, zone: MemoryZone, limit: int = 200) -> list[MemoryRecord]:
        path = self._path_for_zone(zone)
        records = [MemoryRecord.model_validate(item) for item in _read_jsonl(path)]
        return self._filter_active(records)[-limit:]

    def list_contextual_memories(self, limit: int = 80) -> list[MemoryRecord]:
        now = utc_now()
        records = self.list_latest_records(
            zones=[
                MemoryZone.long_term,
                MemoryZone.candidate,
                MemoryZone.observation,
                MemoryZone.short_term,
            ]
        )
        active = [
            record
            for record in records
            if record.status == MemoryStatus.active
            and (record.expires_at is None or record.expires_at >= now)
        ]
        active.sort(key=lambda item: item.last_seen_at)
        return active[-limit:]

    def list_conflicts(self, limit: int = 80) -> list[ConflictRecord]:
        records = [ConflictRecord.model_validate(item) for item in _read_jsonl(self.conflicts_path)]
        return records[-limit:]

    def find_latest_memory(self, memory_key: str) -> MemoryRecord | None:
        records = self.list_latest_records()
        for record in reversed(records):
            if record.memory_key == memory_key:
                return record
        return None

    def list_latest_records(self, zones: list[MemoryZone] | None = None) -> list[MemoryRecord]:
        zone_list = zones or [
            MemoryZone.short_term,
            MemoryZone.observation,
            MemoryZone.candidate,
            MemoryZone.long_term,
        ]
        combined: list[MemoryRecord] = []
        for zone in zone_list:
            combined.extend(self.list_zone(zone, limit=500))

        latest: dict[str, MemoryRecord] = {}
        for record in combined:
            current = latest.get(record.memory_key)
            if current is None or record.last_seen_at >= current.last_seen_at:
                latest[record.memory_key] = record
        return list(latest.values())

    def _filter_active(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        now = utc_now()
        return [
            record
            for record in records
            if record.status in {MemoryStatus.active, MemoryStatus.frozen}
            and (record.expires_at is None or record.expires_at >= now)
        ]

    def _path_for_zone(self, zone: MemoryZone) -> Path:
        mapping = {
            MemoryZone.short_term: self.short_term_path,
            MemoryZone.observation: self.observations_path,
            MemoryZone.candidate: self.candidates_path,
            MemoryZone.long_term: self.memories_path,
        }
        return mapping[zone]


def add_memory(
    content: str,
    layer: str,
    settings: Settings | None = None,
) -> AddMemoryResult:
    return MarkdownMemoryStore(settings=settings).add_memory(content, layer)


def search_memory(
    query: str,
    layer: str | None = None,
    settings: Settings | None = None,
) -> list[SearchMemoryHit]:
    return MarkdownMemoryStore(settings=settings).search_memory(query, layer)
