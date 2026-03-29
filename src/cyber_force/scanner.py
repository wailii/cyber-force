from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATUS_PATH = PROJECT_ROOT / "kb" / "memory" / "scanned.json"
DEFAULT_CODEX_ROOT = Path.home() / ".codex" / "sessions"
DEFAULT_CLAUDE_ROOT = Path.home() / ".claude"
RECENT_WINDOW = timedelta(hours=24)
TEXT_ITEM_TYPES = {"text", "input_text", "output_text"}
CONTAINER_ITEM_TYPES = {"tool_result"}
ROLE_LABELS = {"user", "assistant"}


@dataclass(slots=True)
class ScannedConversation:
    source: str
    file_id: str
    path: Path
    modified_at: datetime
    text: str


@dataclass(slots=True)
class ScanStats:
    source: str
    discovered: int = 0
    recent: int = 0
    scanned: int = 0
    skipped_known: int = 0
    skipped_stale: int = 0
    skipped_empty: int = 0
    failures: int = 0
    processed_ids: list[str] = field(default_factory=list)


MemoryCallback = Callable[[ScannedConversation], object | None]


class ConversationScanner:
    def __init__(
        self,
        *,
        status_path: Path | None = None,
        memory_callback: MemoryCallback | None = None,
        now: datetime | None = None,
    ) -> None:
        self.status_path = status_path or DEFAULT_STATUS_PATH
        self.memory_callback = memory_callback
        self._now = now

    def scan_codex(self, root: Path | None = None) -> ScanStats:
        return self._scan_source(
            source="codex",
            root=root or DEFAULT_CODEX_ROOT,
            id_resolver=_resolve_codex_id,
            extractor=_extract_codex_segments,
        )

    def scan_claude_code(self, root: Path | None = None) -> ScanStats:
        return self._scan_source(
            source="claude_code",
            root=root or DEFAULT_CLAUDE_ROOT,
            id_resolver=_resolve_claude_id,
            extractor=_extract_claude_segments,
        )

    def scan_all(
        self,
        *,
        codex_root: Path | None = None,
        claude_root: Path | None = None,
    ) -> dict[str, ScanStats]:
        return {
            "codex": self.scan_codex(root=codex_root),
            "claude_code": self.scan_claude_code(root=claude_root),
        }

    def _scan_source(
        self,
        *,
        source: str,
        root: Path,
        id_resolver: Callable[[list[dict[str, Any]], Path], str],
        extractor: Callable[[list[dict[str, Any]], list[str]], list[tuple[str, str]]],
    ) -> ScanStats:
        stats = ScanStats(source=source)
        if not root.exists():
            return stats

        files = sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        stats.discovered = len(files)

        cutoff = self._current_time() - RECENT_WINDOW
        status = self._load_status()
        known = status.setdefault(source, {})
        dirty = False

        for path in files:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            if modified_at < cutoff:
                stats.skipped_stale += 1
                continue

            stats.recent += 1
            records, raw_lines = _read_jsonish_lines(path)
            file_id = id_resolver(records, path)
            if file_id in known:
                stats.skipped_known += 1
                continue

            segments = extractor(records, raw_lines)
            text = _format_segments(segments)
            if not text:
                stats.skipped_empty += 1
                continue

            conversation = ScannedConversation(
                source=source,
                file_id=file_id,
                path=path,
                modified_at=modified_at,
                text=text,
            )

            try:
                if self.memory_callback is not None:
                    self.memory_callback(conversation)
            except Exception:
                stats.failures += 1
                continue

            known[file_id] = {
                "path": str(path),
                "modified_at": modified_at.isoformat(),
                "scanned_at": self._current_time().isoformat(),
            }
            stats.scanned += 1
            stats.processed_ids.append(file_id)
            dirty = True

        if dirty:
            self._write_status(status)
        return stats

    def _current_time(self) -> datetime:
        return self._now or datetime.now(tz=UTC)

    def _load_status(self) -> dict[str, Any]:
        default = {
            "version": 1,
            "updated_at": "",
            "long_term_count": 0,
            "log_files": [],
            "codex": {},
            "claude_code": {},
        }
        if not self.status_path.exists():
            return default

        try:
            data = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, JSONDecodeError):
            return default

        if not isinstance(data, dict):
            return default

        normalized = {
            "version": data.get("version", 1),
            "updated_at": data.get("updated_at", ""),
            "long_term_count": data.get("long_term_count", 0),
            "log_files": data.get("log_files", []),
        }
        for source in ("codex", "claude_code"):
            raw = data.get(source, {})
            normalized[source] = raw if isinstance(raw, dict) else {}
        return normalized

    def _write_status(self, status: dict[str, Any]) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(
            json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def scan_codex(
    *,
    scanner: ConversationScanner | None = None,
    root: Path | None = None,
    status_path: Path | None = None,
    memory_callback: MemoryCallback | None = None,
) -> ScanStats:
    active_scanner = scanner or ConversationScanner(
        status_path=status_path,
        memory_callback=memory_callback,
    )
    return active_scanner.scan_codex(root=root)


def scan_claude_code(
    *,
    scanner: ConversationScanner | None = None,
    root: Path | None = None,
    status_path: Path | None = None,
    memory_callback: MemoryCallback | None = None,
) -> ScanStats:
    active_scanner = scanner or ConversationScanner(
        status_path=status_path,
        memory_callback=memory_callback,
    )
    return active_scanner.scan_claude_code(root=root)


def scan_all(
    *,
    scanner: ConversationScanner | None = None,
    codex_root: Path | None = None,
    claude_root: Path | None = None,
    status_path: Path | None = None,
    memory_callback: MemoryCallback | None = None,
) -> dict[str, ScanStats]:
    active_scanner = scanner or ConversationScanner(
        status_path=status_path,
        memory_callback=memory_callback,
    )
    return active_scanner.scan_all(codex_root=codex_root, claude_root=claude_root)


def scan(query: str = "") -> str:
    del query
    from cyber_force.memory import add_memory

    scanner = ConversationScanner(
        memory_callback=make_add_memory_callback(add_memory, layer="observation")
    )
    report = scanner.scan_all()
    lines: list[str] = []
    for source, stats in report.items():
        lines.append(
            f"{source}: 新增 {stats.scanned}，重复 {stats.skipped_known}，过期 {stats.skipped_stale}"
        )
    return "\n".join(lines) if lines else "扫描已完成。"


def make_add_memory_callback(
    add_memory: Callable[..., object],
    *,
    layer: str = "observation",
) -> MemoryCallback:
    def _callback(item: ScannedConversation) -> object:
        return add_memory(item.text, layer=layer)

    return _callback


def _read_jsonish_lines(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    raw_lines: list[str] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except JSONDecodeError:
                raw_lines.append(stripped)
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
            else:
                raw_lines.append(str(parsed))

    return records, raw_lines


def _resolve_codex_id(records: list[dict[str, Any]], path: Path) -> str:
    for record in records:
        if record.get("type") != "session_meta":
            continue
        payload = record.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("id"), str):
            return payload["id"]
    return path.stem


def _resolve_claude_id(records: list[dict[str, Any]], path: Path) -> str:
    for record in records:
        for key in ("sessionId", "session_id"):
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
    return path.stem


def _extract_codex_segments(
    records: list[dict[str, Any]], raw_lines: list[str]
) -> list[tuple[str, str]]:
    del raw_lines
    segments: list[tuple[str, str]] = []
    for record in records:
        record_type = record.get("type")
        payload = record.get("payload")
        if record_type == "response_item" and isinstance(payload, dict):
            if payload.get("type") != "message":
                continue
            role = payload.get("role")
            if role not in ROLE_LABELS:
                continue
            text = _join_fragments(_extract_content_fragments(payload.get("content")))
            if text:
                segments.append((role, text))
        elif record_type == "event_msg" and isinstance(payload, dict):
            if payload.get("type") != "agent_message":
                continue
            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                segments.append(("assistant", message.strip()))
    return segments


def _extract_claude_segments(
    records: list[dict[str, Any]], raw_lines: list[str]
) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    for record in records:
        message = record.get("message")
        role = None
        content: Any = None

        if isinstance(message, dict):
            if message.get("role") in ROLE_LABELS:
                role = message["role"]
                content = message.get("content")
        elif record.get("type") in ROLE_LABELS:
            role = str(record.get("type"))
            content = record.get("content")

        if role is None:
            continue

        text = _join_fragments(_extract_content_fragments(content))
        if text:
            segments.append((role, text))

    for raw in raw_lines:
        cleaned = raw.strip()
        if cleaned:
            segments.append(("raw", cleaned))

    return segments


def _extract_content_fragments(value: Any) -> list[str]:
    fragments: list[str] = []

    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            fragments.append(cleaned)
        return fragments

    if isinstance(value, list):
        for item in value:
            fragments.extend(_extract_content_fragments(item))
        return fragments

    if not isinstance(value, dict):
        return fragments

    item_type = value.get("type")
    text = value.get("text")
    if item_type in TEXT_ITEM_TYPES and isinstance(text, str):
        cleaned = text.strip()
        if cleaned:
            fragments.append(cleaned)
        return fragments

    if item_type in CONTAINER_ITEM_TYPES:
        fragments.extend(_extract_content_fragments(value.get("content")))
        return fragments

    if item_type in {"tool_use", "thinking", "reasoning", "function_call_output"}:
        return fragments

    if isinstance(text, str) and text.strip():
        fragments.append(text.strip())

    if "content" in value:
        fragments.extend(_extract_content_fragments(value.get("content")))
    return fragments


def _join_fragments(fragments: list[str]) -> str:
    unique: list[str] = []
    for fragment in fragments:
        if fragment and (not unique or unique[-1] != fragment):
            unique.append(fragment)
    return "\n".join(unique).strip()


def _format_segments(segments: list[tuple[str, str]]) -> str:
    formatted = [f"[{role}] {text.strip()}" for role, text in segments if text.strip()]
    return "\n\n".join(formatted)
