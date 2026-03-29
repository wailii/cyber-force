from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from cyber_force.config import Settings
from cyber_force.provider import OpenAICompatibleProvider, ProviderUnavailable


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

LOGGER = logging.getLogger(__name__)
SCHEDULE_INTERVAL_SECONDS = 6 * 3600
SYNC_RETRY_INTERVAL_SECONDS = 60
KB_ROOT = PROJECT_ROOT / "kb"
MEMORY_MARKDOWN_PATH = PROJECT_ROOT / "kb" / "memory" / "MEMORY.md"
SYNC_STATE_PATH = PROJECT_ROOT / "state" / "feishu_sync.json"
SYNC_QUEUE_PATH = PROJECT_ROOT / "state" / "feishu_sync_queue.json"
SYNC_FAILURE_LOG_PATH = PROJECT_ROOT / "state" / "feishu_sync_failures.jsonl"
SKIP_SYNC_FILE_NAMES = {".DS_Store", ".gitkeep"}
LANGUAGE_BY_SUFFIX = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".txt": "text",
}


class FeishuBotConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    app_id: str
    app_secret: str
    memory_doc_id: str
    kb_doc_id: str = ""
    port: int = 9000
    transport: str = "stream"

    @classmethod
    def from_env(cls) -> "FeishuBotConfig":
        transport = os.getenv("FEISHU_BOT_TRANSPORT", "stream").strip().lower() or "stream"
        if transport not in {"stream", "webhook"}:
            transport = "stream"
        return cls(
            app_id=os.getenv("FEISHU_APP_ID", ""),
            app_secret=os.getenv("FEISHU_APP_SECRET", ""),
            memory_doc_id=os.getenv("FEISHU_MEMORY_DOC_ID", ""),
            kb_doc_id=os.getenv("FEISHU_KB_DOC_ID", ""),
            port=int(os.getenv("FEISHU_PORT", "9000")),
            transport=transport,
        )


class FeishuMessage(BaseModel):
    message_id: str
    text: str
    sender_id: str | None = None


class FeishuReplyClientProtocol(Protocol):
    def reply_text(self, message_id: str, text: str) -> None: ...


class MessageRouteDecision(BaseModel):
    action: str
    normalized_text: str
    memory_layer: str | None = None
    target_doc_kind: str | None = None
    log_date: str | None = None


class SyncAttemptResult(BaseModel):
    status: str
    target_doc_kind: str
    log_date: str | None = None
    detail: str = ""


class StoredMemoryResult(BaseModel):
    facts: str
    target_doc_kind: str
    log_date: str | None = None
    sync_status: str = "synced"
    sync_detail: str = ""


class PendingSyncItem(BaseModel):
    target_doc_kind: str
    log_date: str | None = None
    attempts: int = 0
    first_failed_at: str
    last_failed_at: str
    next_retry_at: str
    last_error: str = ""


class MemoryGatewayProtocol(Protocol):
    def add_memory(self, content: str, layer: str = "long_term") -> object: ...

    def search_memory(self, query: str, layer: str | None = None) -> object: ...


class ScannerGatewayProtocol(Protocol):
    def scan(self, query: str = "") -> str: ...

    def scan_all(self) -> object: ...


class EngineGatewayProtocol(Protocol):
    def respond(self, user_text: str, message_id: str, sender_id: str | None = None) -> str: ...


class MemorySyncerProtocol(Protocol):
    def sync(self) -> None: ...

    def sync_target(self, target_doc_kind: str, log_date: str | None = None) -> None: ...

    def sync_target_with_recovery(
        self, target_doc_kind: str, log_date: str | None = None
    ) -> SyncAttemptResult: ...

    def retry_pending(self) -> list[SyncAttemptResult]: ...


class MessageClassifierProtocol(Protocol):
    def classify(self, text: str) -> MessageRouteDecision: ...


class DefaultMemoryGateway:
    def add_memory(self, content: str, layer: str = "long_term") -> object:
        from cyber_force.memory import add_memory

        return add_memory(content, layer=layer)

    def search_memory(self, query: str, layer: str | None = None) -> object:
        from cyber_force.memory import search_memory

        return search_memory(query, layer=layer)


class DefaultScannerGateway:
    def scan(self, query: str = "") -> str:
        from cyber_force.scanner import scan

        return scan(query=query)

    def scan_all(self) -> object:
        from cyber_force.scanner import scan_all

        return scan_all()


class DefaultEngineGateway:
    def __init__(self) -> None:
        from cyber_force.engine import CyberForceEngine
        from cyber_force.schemas import InputEnvelope

        self._engine = CyberForceEngine()
        self._input_envelope_cls = InputEnvelope

    def respond(self, user_text: str, message_id: str, sender_id: str | None = None) -> str:
        response = self._engine.handle(
            self._input_envelope_cls(
                content=user_text,
                source="feishu",
                carrier_id="feishu_bot",
                session_id=message_id,
                audience="ai",
                metadata={"sender_id": sender_id or ""},
            )
        )
        return response.message


class DefaultMessageClassifier:
    def __init__(self) -> None:
        self.settings = Settings.from_env()
        self.provider = OpenAICompatibleProvider(self.settings.memory_model)

    def classify(self, text: str) -> MessageRouteDecision:
        normalized = text.strip()
        if self.provider.configured:
            try:
                return self._classify_with_model(normalized)
            except Exception:
                pass
        return self._classify_fallback(normalized)

    def _classify_with_model(self, text: str) -> MessageRouteDecision:
        payload = self.provider.complete_json(
            system_prompt=(
                "你是飞书记忆路由器。"
                "你的任务是先判断消息属于记忆写入、检索、扫描还是普通对话，"
                "再给出目标文档类型。只返回 JSON。"
            ),
            user_prompt=json.dumps(
                {
                    "text": text,
                    "rules": {
                        "actions": ["remember", "recall", "scan", "chat"],
                        "memory_layers": ["long_term", "candidate", "observation", "short_term"],
                        "target_doc_kind": ["memory_main", "kb_reference", "daily_log", "none"],
                        "guidance": [
                            "长期稳定事实 -> long_term + memory_main",
                            "临时观察/对话沉淀 -> observation + daily_log",
                            "待升级条目 -> candidate + daily_log",
                            "检索请求 -> recall",
                            "扫描请求 -> scan",
                            "普通问答 -> chat",
                        ],
                    },
                    "schema": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "normalized_text": {"type": "string"},
                            "memory_layer": {"type": "string"},
                            "target_doc_kind": {"type": "string"},
                            "log_date": {"type": "string"},
                        },
                        "required": ["action", "normalized_text"],
                    },
                },
                ensure_ascii=False,
            ),
        )
        if not isinstance(payload, dict):
            raise ProviderUnavailable("Message classifier payload is not an object.")
        decision = MessageRouteDecision.model_validate(payload)
        return self._normalize_decision(decision, original_text=text)

    def _classify_fallback(self, text: str) -> MessageRouteDecision:
        remember_content = _strip_command_prefix(text, ["记住", "记录", "remember"])
        if remember_content is not None:
            return MessageRouteDecision(
                action="remember",
                normalized_text=remember_content,
                memory_layer="long_term",
                target_doc_kind="memory_main",
            )

        recall_query = _strip_command_prefix(text, ["查", "recall", "我说过"])
        if recall_query is not None:
            return MessageRouteDecision(action="recall", normalized_text=recall_query)

        scan_query = _strip_command_prefix(text, ["扫描", "scan"])
        if scan_query is not None:
            return MessageRouteDecision(action="scan", normalized_text=scan_query)

        if any(token in text for token in ("记下来", "记一下", "帮我记住", "别忘了")):
            return MessageRouteDecision(
                action="remember",
                normalized_text=text,
                memory_layer="long_term",
                target_doc_kind="memory_main",
            )

        return MessageRouteDecision(action="chat", normalized_text=text)

    def _normalize_decision(
        self,
        decision: MessageRouteDecision,
        *,
        original_text: str,
    ) -> MessageRouteDecision:
        normalized_text = decision.normalized_text.strip() or original_text
        action = decision.action.strip().lower() or "chat"
        memory_layer = (decision.memory_layer or "").strip().lower() or None
        target_doc_kind = (decision.target_doc_kind or "").strip().lower() or None
        if action not in {"remember", "recall", "scan", "chat"}:
            action = "chat"
        if memory_layer not in {None, "long_term", "candidate", "observation", "short_term"}:
            memory_layer = None
        if target_doc_kind not in {None, "memory_main", "kb_reference", "daily_log", "none"}:
            target_doc_kind = None
        if action == "remember" and memory_layer is None:
            memory_layer = "long_term"
        if action == "remember" and target_doc_kind is None:
            target_doc_kind = "memory_main" if memory_layer == "long_term" else "daily_log"
        return MessageRouteDecision(
            action=action,
            normalized_text=normalized_text,
            memory_layer=memory_layer,
            target_doc_kind=target_doc_kind,
            log_date=decision.log_date,
        )


class CommandRunner:
    def run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, capture_output=True, check=True, text=True)


class LarkCliMessageSubscriber:
    def __init__(
        self,
        service: FeishuBotService | Any,
        *,
        reconnect_delay_seconds: float = 5.0,
    ) -> None:
        self.service = service
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.cli_executable = self._resolve_cli_executable()

    def build_command(self) -> list[str]:
        return [
            self.cli_executable,
            "event",
            "+subscribe",
            "--as",
            "bot",
            "--event-types",
            "im.message.receive_v1",
            "--quiet",
        ]

    def handle_stdout_line(self, line: str) -> bool:
        payload_text = line.strip()
        if not payload_text:
            return False
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            LOGGER.warning("ignored non-json Feishu event line: %s", payload_text[:200])
            return False
        if not isinstance(payload, dict):
            LOGGER.warning("ignored non-object Feishu event payload: %r", payload)
            return False
        try:
            self.service.handle_event(payload)
        except Exception as exc:
            LOGGER.warning("failed to handle Feishu event: %s", exc)
            return False
        return True

    async def run_forever(self) -> None:
        self._ensure_lark_cli()
        while True:
            try:
                await self._run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.warning("Feishu long connection exited: %s", exc)
            await asyncio.sleep(self.reconnect_delay_seconds)

    def _ensure_lark_cli(self) -> None:
        if shutil.which("lark-cli") is not None:
            return
        LOGGER.warning("lark-cli not found; installing @larksuite/cli")
        subprocess.run(["npm", "install", "-g", "@larksuite/cli"], check=True, text=True)
        self.cli_executable = self._resolve_cli_executable()

    def _resolve_cli_executable(self) -> str:
        executable = shutil.which("lark-cli")
        if executable is None:
            return "lark-cli"
        wrapper_path = Path(executable).resolve()
        compiled_path = wrapper_path.parent.parent / "bin" / wrapper_path.name
        if compiled_path.exists():
            return str(compiled_path)
        return executable

    async def _run_once(self) -> None:
        process = await asyncio.create_subprocess_exec(
            *self.build_command(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stderr_task = asyncio.create_task(self._drain_stderr(process.stderr))
        try:
            if process.stdout is None:
                raise RuntimeError("lark-cli event stream stdout is unavailable")
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                self.handle_stdout_line(line.decode("utf-8", errors="replace"))
            returncode = await process.wait()
            if returncode != 0:
                raise RuntimeError(f"lark-cli event stream exited with code {returncode}")
        finally:
            stderr_task.cancel()
            with suppress(asyncio.CancelledError):
                await stderr_task
            if process.returncode is None:
                process.terminate()
                with suppress(ProcessLookupError, TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=5)

    async def _drain_stderr(self, stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            message = line.decode("utf-8", errors="replace").strip()
            if message:
                LOGGER.info("lark-cli: %s", message)


class LarkCliMemorySyncer:
    def __init__(
        self,
        config: FeishuBotConfig,
        runner: CommandRunner | None = None,
        memory_path: Path | None = None,
        kb_root: Path = KB_ROOT,
        state_path: Path = SYNC_STATE_PATH,
        queue_path: Path = SYNC_QUEUE_PATH,
        failure_log_path: Path = SYNC_FAILURE_LOG_PATH,
    ) -> None:
        self.config = config
        self.runner = runner or CommandRunner()
        self.kb_root = kb_root
        self.memory_path = memory_path or (kb_root / "memory" / "MEMORY.md")
        self.state_path = state_path
        self.queue_path = queue_path
        self.failure_log_path = failure_log_path

    def sync(self) -> None:
        if not self.kb_root.exists():
            LOGGER.warning("kb root missing: %s", self.kb_root)
            return
        self._ensure_lark_cli()
        manifest = self._load_state()
        manifest = self._sync_memory_doc(manifest)
        manifest = self._sync_kb_doc(manifest)
        for log_path in self._iter_log_files():
            manifest = self._sync_log_doc(manifest, log_path.stem)
        self._write_state(manifest)

    def sync_target(self, target_doc_kind: str, log_date: str | None = None) -> None:
        if not self.kb_root.exists():
            LOGGER.warning("kb root missing: %s", self.kb_root)
            return
        self._ensure_lark_cli()
        manifest = self._load_state()
        if target_doc_kind == "memory_main":
            manifest = self._sync_memory_doc(manifest)
        elif target_doc_kind == "kb_reference":
            manifest = self._sync_kb_doc(manifest)
        elif target_doc_kind == "daily_log":
            if log_date:
                manifest = self._sync_log_doc(manifest, log_date)
            else:
                for log_path in self._iter_log_files():
                    manifest = self._sync_log_doc(manifest, log_path.stem)
        else:
            manifest = self._sync_memory_doc(manifest)
            manifest = self._sync_kb_doc(manifest)
        self._write_state(manifest)

    def sync_target_with_recovery(
        self, target_doc_kind: str, log_date: str | None = None
    ) -> SyncAttemptResult:
        try:
            self.sync_target(target_doc_kind, log_date)
        except Exception as exc:
            item = self._enqueue_pending_sync(target_doc_kind, log_date, str(exc))
            self._append_failure_log(item, phase="initial_sync")
            return SyncAttemptResult(
                status="queued",
                target_doc_kind=target_doc_kind,
                log_date=log_date,
                detail=str(exc),
            )
        self._remove_pending_sync(target_doc_kind, log_date)
        return SyncAttemptResult(status="synced", target_doc_kind=target_doc_kind, log_date=log_date)

    def retry_pending(self) -> list[SyncAttemptResult]:
        items = self._load_pending_syncs()
        if not items:
            return []

        now = self._now()
        results: list[SyncAttemptResult] = []
        remaining: list[PendingSyncItem] = []

        for item in items:
            due_at = self._parse_timestamp(item.next_retry_at)
            if due_at > now:
                remaining.append(item)
                continue
            try:
                self.sync_target(item.target_doc_kind, item.log_date)
            except Exception as exc:
                updated = self._bump_pending_sync(item, str(exc))
                remaining.append(updated)
                self._append_failure_log(updated, phase="retry")
                results.append(
                    SyncAttemptResult(
                        status="queued",
                        target_doc_kind=item.target_doc_kind,
                        log_date=item.log_date,
                        detail=str(exc),
                    )
                )
                continue

            results.append(
                SyncAttemptResult(
                    status="synced",
                    target_doc_kind=item.target_doc_kind,
                    log_date=item.log_date,
                )
            )

        self._write_pending_syncs(remaining)
        return results

    def _ensure_lark_cli(self) -> None:
        try:
            self.runner.run(["which", "lark-cli"])
        except subprocess.CalledProcessError:
            LOGGER.warning("lark-cli not found; installing @larksuite/cli")
            self.runner.run(["npm", "install", "-g", "@larksuite/cli"])

    def _render_kb_snapshot(self) -> str:
        lines = [
            "# 赛博外力知识与规则",
            "",
            "以下内容同步自本地 `kb/identity`、`kb/governance` 与 `kb/memory` 的规则/索引文件。",
            "",
        ]

        for path in self._iter_kb_reference_files():
            relative_path = Path("kb") / path.relative_to(self.kb_root)
            language = LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "text")
            content = self._read_text(path, fallback="")
            lines.extend(
                [
                    f"## {relative_path.as_posix()}",
                    "",
                    f"```{language}",
                    content.rstrip(),
                    "```",
                    "",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"

    def _iter_kb_reference_files(self) -> list[Path]:
        files = []
        for path in sorted(self.kb_root.rglob("*")):
            if not path.is_file():
                continue
            if path.name in SKIP_SYNC_FILE_NAMES:
                continue
            if path == self.memory_path:
                continue
            if path.parent == self.memory_path.parent / "logs":
                continue
            files.append(path)
        return files

    def _iter_log_files(self) -> list[Path]:
        logs_root = self.memory_path.parent / "logs"
        if not logs_root.exists():
            return []
        files: list[Path] = []
        for path in sorted(logs_root.glob("*.md")):
            if path.name in SKIP_SYNC_FILE_NAMES:
                continue
            files.append(path)
        return files

    def _sync_memory_doc(self, manifest: dict[str, Any]) -> dict[str, Any]:
        memory_markdown = self._read_text(self.memory_path, fallback="# 主理人的长期记忆\n")
        memory_doc_id = self._resolve_doc_id(
            explicit_doc_id=self.config.memory_doc_id,
            manifest_doc_id=str(manifest.get("memory_doc_id", "")).strip(),
            title="赛博外力 MEMORY",
            initial_markdown=memory_markdown,
        )
        self._update_doc(memory_doc_id, memory_markdown)
        manifest["memory_doc_id"] = memory_doc_id
        return manifest

    def _sync_kb_doc(self, manifest: dict[str, Any]) -> dict[str, Any]:
        kb_markdown = self._render_kb_snapshot()
        kb_doc_id = self._resolve_doc_id(
            explicit_doc_id=self.config.kb_doc_id,
            manifest_doc_id=str(manifest.get("kb_doc_id", "")).strip(),
            title="赛博外力 KB",
            initial_markdown=kb_markdown,
        )
        self._update_doc(kb_doc_id, kb_markdown)
        manifest["kb_doc_id"] = kb_doc_id
        return manifest

    def _sync_log_doc(self, manifest: dict[str, Any], log_date: str) -> dict[str, Any]:
        log_path = self.memory_path.parent / "logs" / f"{log_date}.md"
        if not log_path.exists():
            return manifest
        log_docs = manifest.get("log_docs", {})
        if not isinstance(log_docs, dict):
            log_docs = {}
        log_doc_id = str(log_docs.get(log_path.name, "")).strip()
        log_markdown = self._read_text(log_path, fallback=f"# {log_path.stem} 日志\n")
        if not log_doc_id:
            log_doc_id = self._create_doc(
                title=f"赛博外力 日志 {log_path.stem}",
                markdown=log_markdown,
            )
        self._update_doc(log_doc_id, log_markdown)
        log_docs[log_path.name] = log_doc_id
        manifest["log_docs"] = log_docs
        return manifest

    def _resolve_doc_id(
        self,
        *,
        explicit_doc_id: str,
        manifest_doc_id: str,
        title: str,
        initial_markdown: str,
    ) -> str:
        if explicit_doc_id.strip():
            return explicit_doc_id.strip()
        if manifest_doc_id.strip():
            return manifest_doc_id.strip()
        return self._create_doc(title=title, markdown=initial_markdown)

    def _create_doc(self, *, title: str, markdown: str) -> str:
        result = self.runner.run(
            [
                "lark-cli",
                "docs",
                "+create",
                "--title",
                title,
                "--markdown",
                markdown,
            ]
        )
        payload = self._parse_json_stdout(result)
        doc_id = str(((payload.get("data") or {}).get("doc_id", ""))).strip()
        if not doc_id:
            raise RuntimeError(f"failed to create Feishu doc for {title}")
        return doc_id

    def _update_doc(self, doc_id: str, markdown: str) -> None:
        self.runner.run(
            [
                "lark-cli",
                "docs",
                "+update",
                "--doc",
                doc_id,
                "--mode",
                "overwrite",
                "--markdown",
                markdown,
            ]
        )

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"memory_doc_id": "", "kb_doc_id": "", "log_docs": {}}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"memory_doc_id": "", "kb_doc_id": "", "log_docs": {}}
        if not isinstance(payload, dict):
            return {"memory_doc_id": "", "kb_doc_id": "", "log_docs": {}}
        log_docs = payload.get("log_docs", {})
        payload["log_docs"] = log_docs if isinstance(log_docs, dict) else {}
        payload.setdefault("memory_doc_id", "")
        payload.setdefault("kb_doc_id", "")
        return payload

    def _write_state(self, payload: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _load_pending_syncs(self) -> list[PendingSyncItem]:
        if not self.queue_path.exists():
            return []
        try:
            payload = json.loads(self.queue_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(payload, list):
            return []
        items: list[PendingSyncItem] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                items.append(PendingSyncItem.model_validate(item))
            except Exception:
                continue
        return items

    def _write_pending_syncs(self, items: list[PendingSyncItem]) -> None:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.queue_path.write_text(
            json.dumps([item.model_dump(mode="json") for item in items], ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    def _enqueue_pending_sync(
        self, target_doc_kind: str, log_date: str | None, error: str
    ) -> PendingSyncItem:
        items = self._load_pending_syncs()
        now = self._now()
        existing = None
        for item in items:
            if item.target_doc_kind == target_doc_kind and item.log_date == log_date:
                existing = item
                break
        if existing is None:
            queued = PendingSyncItem(
                target_doc_kind=target_doc_kind,
                log_date=log_date,
                attempts=1,
                first_failed_at=self._format_timestamp(now),
                last_failed_at=self._format_timestamp(now),
                next_retry_at=self._format_timestamp(now + timedelta(seconds=SYNC_RETRY_INTERVAL_SECONDS)),
                last_error=error,
            )
            items.append(queued)
        else:
            queued = self._bump_pending_sync(existing, error)
            items = [
                queued if item.target_doc_kind == target_doc_kind and item.log_date == log_date else item
                for item in items
            ]
        self._write_pending_syncs(items)
        return queued

    def _remove_pending_sync(self, target_doc_kind: str, log_date: str | None) -> None:
        items = self._load_pending_syncs()
        remaining = [
            item
            for item in items
            if not (item.target_doc_kind == target_doc_kind and item.log_date == log_date)
        ]
        if len(remaining) != len(items):
            self._write_pending_syncs(remaining)

    def _bump_pending_sync(self, item: PendingSyncItem, error: str) -> PendingSyncItem:
        now = self._now()
        delay_seconds = min(SYNC_RETRY_INTERVAL_SECONDS * (2 ** max(item.attempts - 1, 0)), 3600)
        return PendingSyncItem(
            target_doc_kind=item.target_doc_kind,
            log_date=item.log_date,
            attempts=item.attempts + 1,
            first_failed_at=item.first_failed_at,
            last_failed_at=self._format_timestamp(now),
            next_retry_at=self._format_timestamp(now + timedelta(seconds=delay_seconds)),
            last_error=error,
        )

    def _append_failure_log(self, item: PendingSyncItem, *, phase: str) -> None:
        self.failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": self._format_timestamp(self._now()),
            "phase": phase,
            "target_doc_kind": item.target_doc_kind,
            "log_date": item.log_date,
            "attempts": item.attempts,
            "next_retry_at": item.next_retry_at,
            "last_error": item.last_error,
        }
        with self.failure_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _now(self) -> datetime:
        return datetime.now().astimezone()

    def _format_timestamp(self, value: datetime) -> str:
        return value.isoformat()

    def _parse_timestamp(self, value: str) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return self._now()

    def _parse_json_stdout(self, result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
        stdout = (result.stdout or "").strip()
        if not stdout:
            raise RuntimeError("lark-cli returned empty stdout")
        payload = json.loads(stdout)
        if not isinstance(payload, dict):
            raise RuntimeError("lark-cli returned non-object payload")
        if payload.get("ok") is False:
            raise RuntimeError(str(payload.get("error") or payload))
        return payload

    def _read_text(self, path: Path, *, fallback: str) -> str:
        if not path.exists():
            return fallback
        return path.read_text(encoding="utf-8")


class FeishuReplyClient:
    def __init__(
        self,
        config: FeishuBotConfig,
        http_client: httpx.Client | None = None,
        base_url: str = "https://open.feishu.cn",
    ) -> None:
        self.config = config
        self.http_client = http_client or httpx.Client(timeout=30.0, trust_env=False)
        self.base_url = base_url.rstrip("/")
        self._cached_token: str | None = None
        self._token_expires_at = 0.0

    def reply_text(self, message_id: str, text: str) -> None:
        token = self._tenant_access_token()
        response = self.http_client.post(
            f"{self.base_url}/open-apis/im/v1/messages/{message_id}/reply",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json={
                "msg_type": "text",
                "content": json.dumps({"text": text}, ensure_ascii=False),
            },
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            details = exc.response.text.strip()
            if details:
                raise RuntimeError(f"Feishu reply failed: {details}") from exc
            raise

    def _tenant_access_token(self) -> str:
        now = time.time()
        if self._cached_token and now < self._token_expires_at:
            return self._cached_token

        response = self.http_client.post(
            f"{self.base_url}/open-apis/auth/v3/tenant_access_token/internal",
            headers={"Content-Type": "application/json; charset=utf-8"},
            json={"app_id": self.config.app_id, "app_secret": self.config.app_secret},
        )
        response.raise_for_status()
        payload = response.json()
        token = str(payload.get("tenant_access_token", "")).strip()
        if not token:
            raise RuntimeError("failed to obtain Feishu tenant access token")
        expire = int(payload.get("expire", 7200))
        self._cached_token = token
        self._token_expires_at = now + max(expire - 60, 60)
        return token


class FeishuBotService:
    def __init__(
        self,
        config: FeishuBotConfig,
        memory_gateway: MemoryGatewayProtocol | None = None,
        reply_client: FeishuReplyClientProtocol | None = None,
        syncer: MemorySyncerProtocol | None = None,
        scanner_gateway: ScannerGatewayProtocol | None = None,
        engine_gateway: EngineGatewayProtocol | None = None,
        classifier: MessageClassifierProtocol | None = None,
    ) -> None:
        self.config = config
        self._memory_gateway = memory_gateway
        self._reply_client = reply_client
        self._syncer = syncer
        self._scanner_gateway = scanner_gateway
        self._engine_gateway = engine_gateway
        self._classifier = classifier

    @property
    def memory_gateway(self) -> MemoryGatewayProtocol:
        if self._memory_gateway is None:
            self._memory_gateway = DefaultMemoryGateway()
        return self._memory_gateway

    @property
    def reply_client(self) -> FeishuReplyClientProtocol:
        if self._reply_client is None:
            self._reply_client = FeishuReplyClient(self.config)
        return self._reply_client

    @property
    def syncer(self) -> MemorySyncerProtocol:
        if self._syncer is None:
            self._syncer = LarkCliMemorySyncer(self.config)
        return self._syncer

    @property
    def scanner_gateway(self) -> ScannerGatewayProtocol:
        if self._scanner_gateway is None:
            self._scanner_gateway = DefaultScannerGateway()
        return self._scanner_gateway

    @property
    def engine_gateway(self) -> EngineGatewayProtocol:
        if self._engine_gateway is None:
            self._engine_gateway = DefaultEngineGateway()
        return self._engine_gateway

    @property
    def classifier(self) -> MessageClassifierProtocol:
        if self._classifier is None:
            self._classifier = DefaultMessageClassifier()
        return self._classifier

    def process_text_message(self, text: str, message_id: str, sender_id: str | None = None) -> str:
        self._retry_pending_syncs()
        reply = self._build_reply(FeishuMessage(message_id=message_id, text=text.strip(), sender_id=sender_id))
        self.reply_client.reply_text(message_id, reply)
        return reply

    def handle_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("type") == "url_verification":
            return {"challenge": payload.get("challenge", "")}

        message = self._parse_message(payload)
        if message is None:
            return {"ok": True}
        self.process_text_message(message.text, message.message_id, message.sender_id)
        return {"ok": True}

    def scheduled_scan_once(self) -> str:
        try:
            result = self.scanner_gateway.scan_all()
            return _format_scan_result(result)
        except Exception as exc:
            LOGGER.warning("scheduled scan failed: %s", exc)
            return f"扫描失败：{exc}"

    def retry_pending_syncs(self) -> list[SyncAttemptResult]:
        return self._retry_pending_syncs()

    def _build_reply(self, message: FeishuMessage) -> str:
        decision = self.classifier.classify(message.text)

        if decision.action == "remember":
            content = decision.normalized_text.strip()
            if not content:
                return "请提供要记住的内容。"
            result = self._store_memory(
                content,
                layer=decision.memory_layer or "long_term",
                target_doc_kind=decision.target_doc_kind or "memory_main",
                log_date=decision.log_date,
            )
            return _format_memory_acknowledgement(result)

        if decision.action == "recall":
            recall_query = decision.normalized_text.strip()
            if not recall_query:
                return "请提供检索关键词。"
            results = self.memory_gateway.search_memory(recall_query)
            return _format_search_results(results)

        if decision.action == "scan":
            return self.scanner_gateway.scan(decision.normalized_text.strip())

        reply = self.engine_gateway.respond(
            user_text=decision.normalized_text or message.text,
            message_id=message.message_id,
            sender_id=message.sender_id,
        )
        stored = self._store_memory(
            f"用户：{message.text}\n助手：{reply}",
            layer=decision.memory_layer or "observation",
            target_doc_kind=decision.target_doc_kind or "daily_log",
            log_date=decision.log_date,
        )
        note = _format_sync_warning(stored)
        if note:
            return f"{reply}\n\n{note}"
        return reply

    def _store_memory(
        self,
        content: str,
        *,
        layer: str,
        target_doc_kind: str,
        log_date: str | None,
    ) -> StoredMemoryResult:
        try:
            result = self.memory_gateway.add_memory(content, layer=layer)
        except Exception as exc:
            LOGGER.warning("add_memory failed: %s", exc)
            return StoredMemoryResult(
                facts="记忆提取失败",
                target_doc_kind=target_doc_kind,
                log_date=log_date,
                sync_status="failed",
                sync_detail=str(exc),
            )

        facts = _format_add_memory_result(result)
        derived_log_date = log_date
        result_log_path = getattr(result, "log_path", None)
        if derived_log_date is None and result_log_path is not None:
            derived_log_date = Path(str(result_log_path)).stem
        sync_result = self.syncer.sync_target_with_recovery(target_doc_kind, derived_log_date)
        if sync_result.status == "queued":
            LOGGER.warning("memory sync queued for retry: %s", sync_result.detail)
        return StoredMemoryResult(
            facts=facts,
            target_doc_kind=target_doc_kind,
            log_date=derived_log_date,
            sync_status=sync_result.status,
            sync_detail=sync_result.detail,
        )

    def _retry_pending_syncs(self) -> list[SyncAttemptResult]:
        try:
            results = self.syncer.retry_pending()
        except Exception as exc:
            LOGGER.warning("pending sync retry failed: %s", exc)
            return []
        queued = [result for result in results if result.status == "queued"]
        if queued:
            LOGGER.warning("pending syncs still queued: %s", len(queued))
        return results

    def _parse_message(self, payload: dict[str, Any]) -> FeishuMessage | None:
        event = payload.get("event") or {}
        message = event.get("message") or {}
        if message.get("message_type") != "text":
            return None

        content_raw = message.get("content")
        if not isinstance(content_raw, str):
            return None
        try:
            content = json.loads(content_raw)
        except json.JSONDecodeError:
            return None

        text = str(content.get("text", "")).strip()
        if not text:
            return None

        sender = event.get("sender") or {}
        sender_id_block = sender.get("sender_id") or {}
        sender_id = sender_id_block.get("open_id") or sender_id_block.get("user_id")
        return FeishuMessage(
            message_id=str(message.get("message_id", "")),
            text=text,
            sender_id=str(sender_id) if sender_id else None,
        )


async def scheduled_scan(service: FeishuBotService) -> None:
    while True:
        await asyncio.sleep(SCHEDULE_INTERVAL_SECONDS)
        service.scheduled_scan_once()


async def scheduled_sync_retry(service: FeishuBotService) -> None:
    while True:
        await asyncio.sleep(SYNC_RETRY_INTERVAL_SECONDS)
        service.retry_pending_syncs()


def _strip_command_prefix(text: str, prefixes: list[str]) -> str | None:
    normalized = text.strip()
    lowered = normalized.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix.lower()):
            return normalized[len(prefix) :].strip(" \t\r\n:：,，")
    return None


def _format_add_memory_result(result: object) -> str:
    items = getattr(result, "extracted_items", None)
    if isinstance(items, list) and items:
        facts = [str(getattr(item, "fact", "")).strip() for item in items]
        cleaned = [fact for fact in facts if fact]
        if cleaned:
            return "；".join(cleaned)
    return "已完成提取"


def _memory_destination_label(target_doc_kind: str, log_date: str | None = None) -> str:
    if target_doc_kind == "memory_main":
        return "长期记忆"
    if target_doc_kind == "kb_reference":
        return "知识文档"
    if target_doc_kind == "daily_log":
        return f"{log_date or '今日日志'}"
    return "记忆"


def _format_memory_acknowledgement(result: StoredMemoryResult) -> str:
    if result.sync_status == "failed":
        return result.facts
    destination = _memory_destination_label(result.target_doc_kind, result.log_date)
    if result.sync_status == "queued":
        return f"已写入{destination}，但飞书同步失败，已进入重试队列：{result.facts}"
    return f"已写入{destination}：{result.facts}"


def _format_sync_warning(result: StoredMemoryResult) -> str:
    if result.sync_status != "queued":
        return ""
    destination = _memory_destination_label(result.target_doc_kind, result.log_date)
    return f"[同步告警] 本次内容已写入{destination}，但飞书同步失败，系统会自动重试。"


def _format_search_results(results: object) -> str:
    if isinstance(results, str):
        return results or "未找到相关记忆。"
    if isinstance(results, list):
        facts = [str(getattr(item, "fact", "")).strip() for item in results]
        cleaned = [fact for fact in facts if fact]
        return "\n".join(cleaned) if cleaned else "未找到相关记忆。"
    return str(results) if results else "未找到相关记忆。"


def _format_scan_result(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        lines: list[str] = []
        for source, stats in result.items():
            scanned = getattr(stats, "scanned", None)
            recent = getattr(stats, "recent", None)
            if scanned is None:
                continue
            lines.append(f"{source}: 扫描 {scanned} 个新会话 / 最近 {recent} 个")
        return "\n".join(lines) if lines else "扫描已完成。"
    return "扫描已完成。"


def create_app(service: FeishuBotService | None = None) -> FastAPI:
    bot_service = service or FeishuBotService(FeishuBotConfig.from_env())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        bot_service.retry_pending_syncs()
        scan_task = asyncio.create_task(scheduled_scan(bot_service))
        sync_task = asyncio.create_task(scheduled_sync_retry(bot_service))
        try:
            yield
        finally:
            scan_task.cancel()
            sync_task.cancel()
            with suppress(asyncio.CancelledError):
                await scan_task
            with suppress(asyncio.CancelledError):
                await sync_task

    app = FastAPI(title="Cyber Force Feishu Bot", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/webhook/feishu")
    def feishu_webhook(payload: dict[str, Any]) -> dict[str, Any]:
        return bot_service.handle_event(payload)

    return app


app = create_app()


async def run_stream_mode(service: FeishuBotService) -> None:
    subscriber = LarkCliMessageSubscriber(service=service)
    service.retry_pending_syncs()
    scan_task = asyncio.create_task(scheduled_scan(service))
    sync_task = asyncio.create_task(scheduled_sync_retry(service))
    try:
        await subscriber.run_forever()
    finally:
        scan_task.cancel()
        sync_task.cancel()
        with suppress(asyncio.CancelledError):
            await scan_task
        with suppress(asyncio.CancelledError):
            await sync_task


def main() -> None:
    config = FeishuBotConfig.from_env()
    service = FeishuBotService(config)
    if config.transport == "webhook":
        uvicorn.run(create_app(service=service), host="0.0.0.0", port=config.port)
        return
    asyncio.run(run_stream_mode(service))


if __name__ == "__main__":
    main()
