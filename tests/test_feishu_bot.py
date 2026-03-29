import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feishu_bot import (
    FeishuBotConfig,
    FeishuBotService,
    FeishuReplyClient,
    LarkCliMemorySyncer,
    LarkCliMessageSubscriber,
    MessageRouteDecision,
    SyncAttemptResult,
    create_app,
)


class FakeExtractedItem:
    def __init__(self, fact: str) -> None:
        self.fact = fact


class FakeAddMemoryResult:
    def __init__(self, *facts: str) -> None:
        self.extracted_items = [FakeExtractedItem(fact) for fact in facts]


class FakeMemoryGateway:
    def __init__(self, add_result: object | None = None, search_result: str = "") -> None:
        self.add_result = add_result
        self.search_result = search_result
        self.add_calls: list[tuple[str, str]] = []
        self.search_calls: list[str] = []

    def add_memory(self, content: str, layer: str = "long_term") -> object:
        self.add_calls.append((content, layer))
        return self.add_result or FakeAddMemoryResult(content)

    def search_memory(self, query: str, layer: str | None = None) -> str:
        del layer
        self.search_calls.append(query)
        return self.search_result


class FakeReplyClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def reply_text(self, message_id: str, text: str) -> None:
        self.calls.append((message_id, text))


class FakeHttpResponse:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class FakeHttpClient:
    def __init__(self, responses: list[FakeHttpResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object], dict[str, str]]] = []

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> FakeHttpResponse:
        self.calls.append((url, json, headers))
        return self.responses.pop(0)


class FakeSyncer:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls = 0
        self.target_calls: list[tuple[str, str | None]] = []
        self.retry_calls = 0

    def sync(self) -> None:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("sync failed")

    def sync_target(self, target_doc_kind: str, log_date: str | None = None) -> None:
        self.target_calls.append((target_doc_kind, log_date))
        if self.should_fail:
            raise RuntimeError("sync failed")

    def sync_target_with_recovery(
        self, target_doc_kind: str, log_date: str | None = None
    ) -> SyncAttemptResult:
        self.target_calls.append((target_doc_kind, log_date))
        if self.should_fail:
            return SyncAttemptResult(
                status="queued",
                target_doc_kind=target_doc_kind,
                log_date=log_date,
                detail="sync failed",
            )
        return SyncAttemptResult(status="synced", target_doc_kind=target_doc_kind, log_date=log_date)

    def retry_pending(self) -> list[SyncAttemptResult]:
        self.retry_calls += 1
        return []


class FakeCommandRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.created_doc_ids = ["doc_kb_created", "doc_log_2026_03_29"]

    def run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        stdout = ""
        if args[:3] == ["lark-cli", "docs", "+create"]:
            doc_id = self.created_doc_ids.pop(0)
            stdout = json.dumps(
                {
                    "ok": True,
                    "data": {"doc_id": doc_id},
                },
                ensure_ascii=False,
            )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")


class FlakyCommandRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail_create_once = True

    def run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        if args == ["which", "lark-cli"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="/usr/bin/lark-cli\n", stderr="")
        if args[:3] == ["lark-cli", "docs", "+create"]:
            if self.fail_create_once:
                self.fail_create_once = False
                raise subprocess.CalledProcessError(returncode=1, cmd=args, stderr="create failed")
            stdout = json.dumps({"ok": True, "data": {"doc_id": "doc_log_retry"}}, ensure_ascii=False)
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


class FakeScannerGateway:
    def __init__(self, result: str = "扫描接口已预留") -> None:
        self.result = result
        self.calls: list[str] = []

    def scan(self, query: str) -> str:
        self.calls.append(query)
        return self.result


class FakeEngineGateway:
    def __init__(self, message: str = "默认回复") -> None:
        self.message = message
        self.calls: list[str] = []

    def respond(self, user_text: str, message_id: str, sender_id: str | None = None) -> str:
        del message_id, sender_id
        self.calls.append(user_text)
        return self.message


class FakeEventService:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []
        self.raise_on_handle = False

    def handle_event(self, payload: dict[str, object]) -> dict[str, bool]:
        if self.raise_on_handle:
            raise RuntimeError("reply failed")
        self.payloads.append(payload)
        return {"ok": True}


class FakeClassifier:
    def __init__(self, decision: MessageRouteDecision) -> None:
        self.decision = decision
        self.calls: list[str] = []

    def classify(self, text: str) -> MessageRouteDecision:
        self.calls.append(text)
        return self.decision


def build_service(
    *,
    memory: FakeMemoryGateway | None = None,
    reply: FakeReplyClient | None = None,
    syncer: FakeSyncer | None = None,
    scanner: FakeScannerGateway | None = None,
    engine: FakeEngineGateway | None = None,
    classifier: FakeClassifier | None = None,
) -> FeishuBotService:
    return FeishuBotService(
        config=FeishuBotConfig(
            app_id="cli_test",
            app_secret="secret",
            memory_doc_id="doc_test",
        ),
        memory_gateway=memory or FakeMemoryGateway(),
        reply_client=reply or FakeReplyClient(),
        syncer=syncer or FakeSyncer(),
        scanner_gateway=scanner or FakeScannerGateway(),
        engine_gateway=engine or FakeEngineGateway(),
        classifier=classifier,
    )


def test_remember_message_adds_long_term_memory_and_replies() -> None:
    memory = FakeMemoryGateway(add_result=FakeAddMemoryResult("午饭后整理扫描清单"))
    reply = FakeReplyClient()
    syncer = FakeSyncer()
    classifier = FakeClassifier(
        MessageRouteDecision(
            action="remember",
            normalized_text="午饭后提醒我整理扫描清单",
            memory_layer="long_term",
            target_doc_kind="memory_main",
        )
    )
    service = build_service(memory=memory, reply=reply, syncer=syncer, classifier=classifier)

    service.process_text_message(
        text="记住 午饭后提醒我整理扫描清单",
        message_id="om_1",
        sender_id="ou_1",
    )

    assert memory.add_calls == [("午饭后提醒我整理扫描清单", "long_term")]
    assert reply.calls == [("om_1", "已写入长期记忆：午饭后整理扫描清单")]
    assert syncer.target_calls == [("memory_main", None)]


def test_recall_message_uses_memory_search_and_replies_with_result() -> None:
    memory = FakeMemoryGateway(search_result="找到 2 条相关记忆")
    reply = FakeReplyClient()
    classifier = FakeClassifier(
        MessageRouteDecision(
            action="recall",
            normalized_text="项目优先级",
        )
    )
    service = build_service(memory=memory, reply=reply, classifier=classifier)

    service.process_text_message(
        text="查 项目优先级",
        message_id="om_2",
        sender_id="ou_2",
    )

    assert memory.search_calls == ["项目优先级"]
    assert reply.calls == [("om_2", "找到 2 条相关记忆")]


def test_scan_message_goes_through_injected_scanner_gateway() -> None:
    scanner = FakeScannerGateway(result="扫描接口已接受：内网端口")
    reply = FakeReplyClient()
    classifier = FakeClassifier(
        MessageRouteDecision(
            action="scan",
            normalized_text="内网端口",
        )
    )
    service = build_service(scanner=scanner, reply=reply, classifier=classifier)

    service.process_text_message(
        text="扫描 内网端口",
        message_id="om_3",
        sender_id="ou_3",
    )

    assert scanner.calls == ["内网端口"]
    assert reply.calls == [("om_3", "扫描接口已接受：内网端口")]


def test_default_conversation_uses_engine_and_persists_dialogue_memory() -> None:
    memory = FakeMemoryGateway(add_result=FakeAddMemoryResult("用户想先明确目标系统和边界"))
    reply = FakeReplyClient()
    syncer = FakeSyncer()
    engine = FakeEngineGateway(message="先补充目标系统和边界。")
    classifier = FakeClassifier(
        MessageRouteDecision(
            action="chat",
            normalized_text="帮我分析一下这个需求",
        )
    )
    service = build_service(
        memory=memory,
        reply=reply,
        syncer=syncer,
        engine=engine,
        classifier=classifier,
    )

    service.process_text_message(
        text="帮我分析一下这个需求",
        message_id="om_4",
        sender_id="ou_4",
    )

    assert engine.calls == ["帮我分析一下这个需求"]
    assert reply.calls == [("om_4", "先补充目标系统和边界。")]
    assert memory.add_calls == [("用户：帮我分析一下这个需求\n助手：先补充目标系统和边界。", "observation")]
    assert syncer.target_calls == [("daily_log", None)]


def test_sync_failure_only_warns_without_breaking_main_flow() -> None:
    memory = FakeMemoryGateway(add_result=FakeAddMemoryResult("incident checklist"))
    reply = FakeReplyClient()
    syncer = FakeSyncer(should_fail=True)
    classifier = FakeClassifier(
        MessageRouteDecision(
            action="remember",
            normalized_text="draft the incident checklist",
            memory_layer="long_term",
            target_doc_kind="memory_main",
        )
    )
    service = build_service(memory=memory, reply=reply, syncer=syncer, classifier=classifier)

    service.process_text_message(
        text="remember draft the incident checklist",
        message_id="om_5",
        sender_id="ou_5",
    )

    assert reply.calls == [("om_5", "已写入长期记忆，但飞书同步失败，已进入重试队列：incident checklist")]
    assert memory.add_calls == [("draft the incident checklist", "long_term")]
    assert syncer.target_calls == [("memory_main", None)]


def test_webhook_route_handles_url_verification_and_message_event() -> None:
    memory = FakeMemoryGateway(add_result=FakeAddMemoryResult("每天晨会前看告警面板"))
    reply = FakeReplyClient()
    classifier = FakeClassifier(
        MessageRouteDecision(
            action="remember",
            normalized_text="每天晨会前看告警面板",
            memory_layer="long_term",
            target_doc_kind="memory_main",
        )
    )
    service = build_service(memory=memory, reply=reply, classifier=classifier)
    client = TestClient(create_app(service=service))

    verification = client.post(
        "/webhook/feishu",
        json={"type": "url_verification", "challenge": "challenge-token"},
    )
    assert verification.status_code == 200
    assert verification.json() == {"challenge": "challenge-token"}

    event_payload = {
        "schema": "2.0",
        "header": {"event_type": "im.message.receive_v1"},
        "event": {
            "sender": {"sender_id": {"open_id": "ou_6"}},
            "message": {
                "message_id": "om_6",
                "message_type": "text",
                "content": json.dumps({"text": "记录 每天晨会前看告警面板"}, ensure_ascii=False),
            },
        },
    }
    response = client.post("/webhook/feishu", json=event_payload)

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert memory.add_calls == [("每天晨会前看告警面板", "long_term")]
    assert reply.calls == [("om_6", "已写入长期记忆：每天晨会前看告警面板")]


def test_classifier_can_route_plain_natural_language_memory_without_prefix() -> None:
    memory = FakeMemoryGateway(add_result=FakeAddMemoryResult("主理人想去 AI-native 创业公司做 Agent PM"))
    reply = FakeReplyClient()
    syncer = FakeSyncer()
    classifier = FakeClassifier(
        MessageRouteDecision(
            action="remember",
            normalized_text="我想去 AI-native 创业公司做 Agent PM，这件事记下来",
            memory_layer="long_term",
            target_doc_kind="memory_main",
        )
    )
    service = build_service(memory=memory, reply=reply, syncer=syncer, classifier=classifier)

    service.process_text_message(
        text="我想去 AI-native 创业公司做 Agent PM，这件事记下来",
        message_id="om_7",
        sender_id="ou_7",
    )

    assert classifier.calls == ["我想去 AI-native 创业公司做 Agent PM，这件事记下来"]
    assert memory.add_calls == [("我想去 AI-native 创业公司做 Agent PM，这件事记下来", "long_term")]
    assert syncer.target_calls == [("memory_main", None)]


def test_chat_reply_includes_sync_warning_when_daily_log_sync_is_queued() -> None:
    memory = FakeMemoryGateway(add_result=FakeAddMemoryResult("用户对 AI 时代竞争力感到困惑"))
    reply = FakeReplyClient()
    syncer = FakeSyncer(should_fail=True)
    engine = FakeEngineGateway(message="先把你真正的差异化来源拆开。")
    classifier = FakeClassifier(
        MessageRouteDecision(
            action="chat",
            normalized_text="我最近在想我的竞争力到底在哪",
        )
    )
    service = build_service(
        memory=memory,
        reply=reply,
        syncer=syncer,
        engine=engine,
        classifier=classifier,
    )

    service.process_text_message(
        text="我最近在想我的竞争力到底在哪",
        message_id="om_7b",
        sender_id="ou_7b",
    )

    assert reply.calls == [
        (
            "om_7b",
            "先把你真正的差异化来源拆开。\n\n[同步告警] 本次内容已写入今日日志，但飞书同步失败，系统会自动重试。",
        )
    ]


def test_memory_syncer_splits_memory_kb_and_logs_into_multiple_docs(tmp_path: Path) -> None:
    kb_root = tmp_path / "kb"
    (kb_root / "identity").mkdir(parents=True, exist_ok=True)
    (kb_root / "governance").mkdir(parents=True, exist_ok=True)
    (kb_root / "memory" / "logs").mkdir(parents=True, exist_ok=True)
    (kb_root / "identity" / "constitution.md").write_text("# 身份\n我是主理人\n", encoding="utf-8")
    (kb_root / "governance" / "decision_policy.yaml").write_text("mode: plan_only\n", encoding="utf-8")
    (kb_root / "memory" / "MEMORY.md").write_text("# 长期记忆\n- 项目: cyber-force\n", encoding="utf-8")
    (kb_root / "memory" / "logs" / "2026-03-29.md").write_text("# 日志\n- 今天新增候选记忆\n", encoding="utf-8")
    (kb_root / "memory" / "logs" / ".gitkeep").write_text("", encoding="utf-8")
    (kb_root / ".DS_Store").write_text("ignore", encoding="utf-8")

    runner = FakeCommandRunner()
    syncer = LarkCliMemorySyncer(
        config=FeishuBotConfig(
            app_id="cli_test",
            app_secret="secret",
            memory_doc_id="doc_memory",
            kb_doc_id="",
        ),
        runner=runner,
        kb_root=kb_root,
        state_path=tmp_path / "state" / "feishu_sync.json",
    )

    syncer.sync()

    assert runner.calls[0] == ["which", "lark-cli"]
    memory_update = runner.calls[1]
    assert memory_update[:7] == [
        "lark-cli",
        "docs",
        "+update",
        "--doc",
        "doc_memory",
        "--mode",
        "overwrite",
    ]
    assert memory_update[8] == "# 长期记忆\n- 项目: cyber-force\n"

    kb_create = runner.calls[2]
    assert kb_create[:3] == ["lark-cli", "docs", "+create"]
    assert kb_create[3:5] == ["--title", "赛博外力 KB"]

    kb_update = runner.calls[3]
    assert kb_update[:7] == [
        "lark-cli",
        "docs",
        "+update",
        "--doc",
        "doc_kb_created",
        "--mode",
        "overwrite",
    ]
    kb_markdown = kb_update[8]
    assert "# 赛博外力知识与规则" in kb_markdown
    assert "## kb/identity/constitution.md" in kb_markdown
    assert "## kb/governance/decision_policy.yaml" in kb_markdown
    assert "我是主理人" in kb_markdown
    assert "mode: plan_only" in kb_markdown
    assert "kb/memory/MEMORY.md" not in kb_markdown
    assert "2026-03-29.md" not in kb_markdown

    log_create = runner.calls[4]
    assert log_create[:3] == ["lark-cli", "docs", "+create"]
    assert log_create[3:5] == ["--title", "赛博外力 日志 2026-03-29"]

    log_update = runner.calls[5]
    assert log_update[:7] == [
        "lark-cli",
        "docs",
        "+update",
        "--doc",
        "doc_log_2026_03_29",
        "--mode",
        "overwrite",
    ]
    assert log_update[8] == "# 日志\n- 今天新增候选记忆\n"

    manifest = json.loads((tmp_path / "state" / "feishu_sync.json").read_text(encoding="utf-8"))
    assert manifest["memory_doc_id"] == "doc_memory"
    assert manifest["kb_doc_id"] == "doc_kb_created"
    assert manifest["log_docs"] == {"2026-03-29.md": "doc_log_2026_03_29"}


def test_memory_syncer_queues_failed_log_sync_and_retries_later(tmp_path: Path) -> None:
    kb_root = tmp_path / "kb"
    (kb_root / "memory" / "logs").mkdir(parents=True, exist_ok=True)
    (kb_root / "memory" / "MEMORY.md").write_text("# 长期记忆\n", encoding="utf-8")
    (kb_root / "memory" / "logs" / "2026-03-29.md").write_text("# 2026-03-29 日志\n", encoding="utf-8")

    runner = FlakyCommandRunner()
    queue_path = tmp_path / "state" / "feishu_sync_queue.json"
    failure_log_path = tmp_path / "state" / "feishu_sync_failures.jsonl"
    syncer = LarkCliMemorySyncer(
        config=FeishuBotConfig(
            app_id="cli_test",
            app_secret="secret",
            memory_doc_id="doc_memory",
            kb_doc_id="doc_kb",
        ),
        runner=runner,
        kb_root=kb_root,
        state_path=tmp_path / "state" / "feishu_sync.json",
        queue_path=queue_path,
        failure_log_path=failure_log_path,
    )

    initial = syncer.sync_target_with_recovery("daily_log", "2026-03-29")

    assert initial.status == "queued"
    queued_items = json.loads(queue_path.read_text(encoding="utf-8"))
    assert len(queued_items) == 1
    assert queued_items[0]["target_doc_kind"] == "daily_log"
    assert queued_items[0]["log_date"] == "2026-03-29"
    queued_items[0]["next_retry_at"] = "2000-01-01T00:00:00+08:00"
    queue_path.write_text(json.dumps(queued_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failure_lines = failure_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(failure_lines) == 1
    assert json.loads(failure_lines[0])["phase"] == "initial_sync"

    retried = syncer.retry_pending()

    assert retried == [SyncAttemptResult(status="synced", target_doc_kind="daily_log", log_date="2026-03-29", detail="")]
    assert json.loads(queue_path.read_text(encoding="utf-8")) == []
    manifest = json.loads((tmp_path / "state" / "feishu_sync.json").read_text(encoding="utf-8"))
    assert manifest["log_docs"] == {"2026-03-29.md": "doc_log_retry"}


def test_lark_cli_message_subscriber_uses_bot_long_connection_command() -> None:
    subscriber = LarkCliMessageSubscriber(service=FakeEventService())

    command = subscriber.build_command()

    assert command[1:] == [
        "event",
        "+subscribe",
        "--as",
        "bot",
        "--event-types",
        "im.message.receive_v1",
        "--quiet",
    ]
    assert command[0].endswith("lark-cli")


def test_lark_cli_message_subscriber_routes_ndjson_event_to_service() -> None:
    service = FakeEventService()
    subscriber = LarkCliMessageSubscriber(service=service)
    payload = {
        "schema": "2.0",
        "header": {"event_type": "im.message.receive_v1"},
        "event": {
            "sender": {"sender_id": {"open_id": "ou_8"}},
            "message": {
                "message_id": "om_8",
                "message_type": "text",
                "content": json.dumps({"text": "记住 今天补长连接"}, ensure_ascii=False),
            },
        },
    }

    handled = subscriber.handle_stdout_line(json.dumps(payload, ensure_ascii=False))

    assert handled is True
    assert service.payloads == [payload]


def test_lark_cli_message_subscriber_keeps_stream_alive_when_service_raises() -> None:
    service = FakeEventService()
    service.raise_on_handle = True
    subscriber = LarkCliMessageSubscriber(service=service)
    payload = {
        "schema": "2.0",
        "header": {"event_type": "im.message.receive_v1"},
        "event": {
            "message": {
                "message_id": "om_9",
                "message_type": "text",
                "content": json.dumps({"text": "hihi"}, ensure_ascii=False),
            },
        },
    }

    handled = subscriber.handle_stdout_line(json.dumps(payload, ensure_ascii=False))

    assert handled is False
    assert service.payloads == []


def test_feishu_reply_client_sends_text_content_as_json_string() -> None:
    http_client = FakeHttpClient(
        responses=[
            FakeHttpResponse({"tenant_access_token": "tenant-token", "expire": 7200}),
            FakeHttpResponse({}),
        ]
    )
    client = FeishuReplyClient(
        config=FeishuBotConfig(
            app_id="cli_test",
            app_secret="secret",
            memory_doc_id="doc_test",
        ),
        http_client=http_client,
        base_url="https://open.feishu.cn",
    )

    client.reply_text("om_reply", "你好")

    token_call = http_client.calls[0]
    assert token_call[0] == "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    reply_call = http_client.calls[1]
    assert reply_call[0] == "https://open.feishu.cn/open-apis/im/v1/messages/om_reply/reply"
    assert reply_call[1] == {
        "msg_type": "text",
        "content": json.dumps({"text": "你好"}, ensure_ascii=False),
    }
