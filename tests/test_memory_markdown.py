from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from cyber_force.config import ModelConfig, PathsConfig, Settings, ThresholdConfig
from cyber_force.memory import MarkdownMemoryStore


def _settings(tmp_path: Path, memory_model: ModelConfig | None = None) -> Settings:
    project_root = tmp_path / "project"
    return Settings(
        paths=PathsConfig(
            project_root=project_root,
            kb_dir=project_root / "kb",
            prompts_dir=project_root / "prompts",
            protocol_dir=project_root / "protocol",
            state_dir=project_root / "state",
        ),
        model=ModelConfig(),
        memory_model=memory_model or ModelConfig(),
        thresholds=ThresholdConfig(),
    )


def test_settings_from_env_loads_memory_model_defaults(monkeypatch) -> None:
    monkeypatch.delenv("MEMORY_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("MEMORY_MODEL_NAME", raising=False)
    monkeypatch.delenv("MEMORY_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MEMORY_MODEL_BASE_URL", raising=False)

    settings = Settings.from_env(project_root=Path("/tmp/cyber-force"))

    assert settings.memory_model.provider == "deepseek"
    assert settings.memory_model.model == "deepseek-chat"
    assert settings.memory_model.base_url == "https://api.deepseek.com/v1"


def test_add_memory_writes_long_term_fact_into_memory_markdown(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(_settings(tmp_path))

    result = store.add_memory("我的生日是 1990-01-01。", layer="long_term")

    memory_md = (tmp_path / "project" / "kb" / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    scanned = json.loads(
        (tmp_path / "project" / "kb" / "memory" / "scanned.json").read_text(encoding="utf-8")
    )

    assert result.extracted_items[0].fact == "生日：1990-01-01"
    assert result.updated_long_term is True
    assert "# 外力的长期记忆" in memory_md
    assert "## 工作与职业" in memory_md
    assert "生日：1990-01-01" in memory_md
    assert "我的生日是 1990-01-01" not in memory_md
    assert scanned["long_term_count"] == 1
    assert scanned["codex"] == {}
    assert scanned["claude_code"] == {}


def test_add_memory_writes_observation_entry_to_daily_log_only(tmp_path: Path) -> None:
    frozen_now = datetime(2026, 3, 29, 8, 0, tzinfo=timezone.utc)
    store = MarkdownMemoryStore(_settings(tmp_path), now_fn=lambda: frozen_now)

    result = store.add_memory("我今天心情很差，先不要做重大决定。", layer="observation")

    memory_root = tmp_path / "project" / "kb" / "memory"
    log_path = memory_root / "logs" / "2026-03-29.md"
    memory_md = (memory_root / "MEMORY.md").read_text(encoding="utf-8")
    log_text = log_path.read_text(encoding="utf-8")

    assert result.updated_long_term is False
    assert result.log_path == log_path
    assert "## 原始输入" in log_text
    assert "摘要：我今天心情很差，先不要做重大决定" in log_text
    assert "## 提取的关键事实" in log_text
    assert "临时状态：心情差，暂缓重大决定" in log_text
    assert "临时状态：心情差，暂缓重大决定" not in memory_md


def test_add_memory_updates_existing_long_term_fact_without_raw_text(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(_settings(tmp_path))

    first = store.add_memory("我的常驻城市是上海。", layer="long_term")
    second = store.add_memory("我的常驻城市是杭州。", layer="long_term")

    memory_md = (tmp_path / "project" / "kb" / "memory" / "MEMORY.md").read_text(encoding="utf-8")

    assert first.updated_long_term is True
    assert second.updated_long_term is True
    assert second.extracted_items[0].status == "updated"
    assert "常驻城市：杭州" in memory_md
    assert "常驻城市：上海" not in memory_md
    assert "我的常驻城市是杭州" not in memory_md


def test_search_memory_fallback_reads_memory_and_recent_logs(tmp_path: Path) -> None:
    frozen_now = datetime(2026, 3, 29, 8, 0, tzinfo=timezone.utc)
    store = MarkdownMemoryStore(_settings(tmp_path), now_fn=lambda: frozen_now)
    memory_root = tmp_path / "project" / "kb" / "memory"
    logs_dir = memory_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (memory_root / "MEMORY.md").write_text(
        (
            "# 外力的长期记忆\n\n"
            "## 工作与职业\n"
            "<!-- 求职方向、工作经历、职业判断等 -->\n"
            "- [identity.city] 常驻城市：杭州\n\n"
            "## 思维方式与偏好\n"
            "<!-- 做决策的方式、审美偏好、沟通风格等 -->\n\n"
            "## 当前关注的事\n"
            "<!-- 正在做的项目、近期优先级等 -->\n\n"
            "## 重要决策记录\n"
            "<!-- 做过的重要决定和背后的理由 -->\n\n"
            "## 关于AI和产品的认知\n"
            "<!-- 对AI产品、Agent设计的理解和判断 -->\n"
        ),
        encoding="utf-8",
    )
    (logs_dir / "2026-03-29.md").write_text(
        (
            "# 2026-03-29 日志\n\n"
            "## 原始输入\n"
            "<!-- 当天收到的消息原文摘要 -->\n"
            "- 摘要：我今天心情很差，先不要做重大决定\n\n"
            "## 提取的关键事实\n"
            "<!-- 模型提取后的结构化条目 -->\n"
            "- [observation.mood] 临时状态：心情差，暂缓重大决定\n\n"
            "## 待升级到长期记忆\n"
            "<!-- 标记需要在下次整理时写入 MEMORY.md 的条目 -->\n"
        ),
        encoding="utf-8",
    )
    (logs_dir / "2026-03-10.md").write_text(
        (
            "# 2026-03-10 日志\n\n"
            "## 原始输入\n"
            "<!-- 当天收到的消息原文摘要 -->\n\n"
            "## 提取的关键事实\n"
            "<!-- 模型提取后的结构化条目 -->\n"
            "- [observation.old] 临时状态：这条太旧，不该被检索到\n\n"
            "## 待升级到长期记忆\n"
            "<!-- 标记需要在下次整理时写入 MEMORY.md 的条目 -->\n"
        ),
        encoding="utf-8",
    )

    results = store.search_memory("重大决定", layer="observation")

    assert len(results) == 1
    assert results[0].fact == "临时状态：心情差，暂缓重大决定"
    assert results[0].source_path.endswith("2026-03-29.md")


def test_search_memory_uses_model_selection_when_available(tmp_path: Path) -> None:
    class FakeProvider:
        configured = True

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def complete_json(self, system_prompt: str, user_prompt: str) -> list[dict]:
            self.calls.append((system_prompt, user_prompt))
            return [{"memory_id": "identity.city", "reason": "query match"}]

    provider = FakeProvider()
    settings = _settings(
        tmp_path,
        memory_model=ModelConfig(
            provider="deepseek",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        ),
    )
    store = MarkdownMemoryStore(settings, provider=provider)
    memory_root = tmp_path / "project" / "kb" / "memory"
    memory_root.mkdir(parents=True, exist_ok=True)
    (memory_root / "MEMORY.md").write_text(
        (
            "# 外力的长期记忆\n\n"
            "## 工作与职业\n"
            "<!-- 求职方向、工作经历、职业判断等 -->\n"
            "- [identity.city] 常驻城市：杭州\n\n"
            "## 思维方式与偏好\n"
            "<!-- 做决策的方式、审美偏好、沟通风格等 -->\n\n"
            "## 当前关注的事\n"
            "<!-- 正在做的项目、近期优先级等 -->\n\n"
            "## 重要决策记录\n"
            "<!-- 做过的重要决定和背后的理由 -->\n\n"
            "## 关于AI和产品的认知\n"
            "<!-- 对AI产品、Agent设计的理解和判断 -->\n"
        ),
        encoding="utf-8",
    )

    results = store.search_memory("我住在哪", layer="long_term")

    assert len(results) == 1
    assert results[0].memory_id == "identity.city"
    assert provider.calls


def test_memory_smoke_long_term_round_trip(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    from cyber_force.memory import add_memory, search_memory

    add_memory("外力求职目标是 AI-native 创业公司的 Agent PM", "long_term", settings=settings)

    memory_md = (tmp_path / "project" / "kb" / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    results = search_memory("外力的求职方向", settings=settings)

    assert "外力求职目标是 AI-native 创业公司的 Agent PM" in memory_md
    assert any("Agent PM" in item.fact for item in results)
