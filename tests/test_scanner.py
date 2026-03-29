import json
import os
import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCANNER_PATH = PROJECT_ROOT / "src" / "cyber_force" / "scanner.py"

spec = importlib.util.spec_from_file_location("scanner_under_test", SCANNER_PATH)
scanner_module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = scanner_module
spec.loader.exec_module(scanner_module)

ConversationScanner = scanner_module.ConversationScanner
scan_all = scanner_module.scan_all


def _write_jsonl(path: Path, rows: list[object], modified_at: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row + "\n")
            else:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    timestamp = modified_at.timestamp()
    os.utime(path, (timestamp, timestamp))


def test_scan_codex_processes_recent_sessions_once(tmp_path: Path) -> None:
    now = datetime(2026, 3, 29, 1, 0, tzinfo=UTC)
    codex_root = tmp_path / ".codex" / "sessions"
    recent = codex_root / "2026" / "03" / "29" / "rollout-recent.jsonl"
    stale = codex_root / "2026" / "03" / "27" / "rollout-stale.jsonl"
    status_path = tmp_path / "kb" / "memory" / "scanned.json"
    calls: list[tuple[str, str, Path]] = []

    _write_jsonl(
        recent,
        [
            {
                "type": "session_meta",
                "payload": {"id": "codex-session-1"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "记录我偏好简洁回复"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "已收到，会保持简洁。"}],
                },
            },
        ],
        modified_at=now - timedelta(hours=1),
    )
    _write_jsonl(
        stale,
        [
            {
                "type": "session_meta",
                "payload": {"id": "codex-session-stale"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "这条不该被扫描"}],
                },
            },
        ],
        modified_at=now - timedelta(hours=30),
    )

    scanner = ConversationScanner(
        status_path=status_path,
        memory_callback=lambda item: calls.append((item.source, item.text, item.path)),
        now=now,
    )

    first = scanner.scan_codex(root=codex_root)
    second = scanner.scan_codex(root=codex_root)

    assert first.discovered == 2
    assert first.recent == 1
    assert first.scanned == 1
    assert first.skipped_stale == 1
    assert first.skipped_known == 0
    assert calls == [
        (
            "codex",
            "[user] 记录我偏好简洁回复\n\n[assistant] 已收到，会保持简洁。",
            recent,
        )
    ]

    assert second.scanned == 0
    assert second.skipped_known == 1

    scanned = json.loads(status_path.read_text(encoding="utf-8"))
    assert "codex-session-1" in scanned["codex"]
    assert scanned["codex"]["codex-session-1"]["path"] == str(recent)


def test_scan_claude_code_extracts_readable_text_with_fallback(tmp_path: Path) -> None:
    now = datetime(2026, 3, 29, 1, 0, tzinfo=UTC)
    claude_root = tmp_path / ".claude"
    session = claude_root / "projects" / "demo" / "session-abc.jsonl"
    status_path = tmp_path / "kb" / "memory" / "scanned.json"
    calls: list[str] = []

    _write_jsonl(
        session,
        [
            {
                "type": "user",
                "sessionId": "claude-session-1",
                "message": {"role": "user", "content": "请记住我要先写测试"},
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "好的，我会先补测试。"},
                        {"type": "tool_use", "name": "Read"},
                    ],
                },
            },
            "人工补记: 这是一条非 JSON 的兜底文本",
        ],
        modified_at=now - timedelta(minutes=15),
    )

    scanner = ConversationScanner(
        status_path=status_path,
        memory_callback=lambda item: calls.append(item.text),
        now=now,
    )

    result = scanner.scan_claude_code(root=claude_root)

    assert result.discovered == 1
    assert result.recent == 1
    assert result.scanned == 1
    assert calls == [
        "[user] 请记住我要先写测试\n\n[assistant] 好的，我会先补测试。\n\n[raw] 人工补记: 这是一条非 JSON 的兜底文本"
    ]

    scanned = json.loads(status_path.read_text(encoding="utf-8"))
    assert "claude-session-1" in scanned["claude_code"]
    assert scanned["claude_code"]["claude-session-1"]["path"] == str(session)


def test_scan_all_returns_per_source_summary(tmp_path: Path) -> None:
    now = datetime(2026, 3, 29, 1, 0, tzinfo=UTC)
    codex_root = tmp_path / ".codex" / "sessions"
    claude_root = tmp_path / ".claude"
    status_path = tmp_path / "kb" / "memory" / "scanned.json"
    seen: list[tuple[str, str]] = []

    _write_jsonl(
        codex_root / "2026" / "03" / "29" / "rollout-1.jsonl",
        [
            {"type": "session_meta", "payload": {"id": "codex-all-1"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Codex 对话"}],
                },
            },
        ],
        modified_at=now - timedelta(minutes=5),
    )
    _write_jsonl(
        claude_root / "projects" / "demo" / "session-1.jsonl",
        [
            {
                "type": "user",
                "sessionId": "claude-all-1",
                "message": {"role": "user", "content": "Claude 对话"},
            }
        ],
        modified_at=now - timedelta(minutes=3),
    )

    scanner = ConversationScanner(
        status_path=status_path,
        memory_callback=lambda item: seen.append((item.source, item.file_id)),
        now=now,
    )

    report = scan_all(scanner=scanner, codex_root=codex_root, claude_root=claude_root)

    assert report["codex"].scanned == 1
    assert report["claude_code"].scanned == 1
    assert seen == [("codex", "codex-all-1"), ("claude_code", "claude-all-1")]
