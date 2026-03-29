from pathlib import Path

from cyber_force.config import ModelConfig, PathsConfig, Settings, ThresholdConfig
from cyber_force.engine import CyberForceEngine
from cyber_force.schemas import DecisionDisposition, InputEnvelope, MemoryZone


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        paths=PathsConfig(
            project_root=PROJECT_ROOT,
            kb_dir=PROJECT_ROOT / "kb",
            prompts_dir=PROJECT_ROOT / "prompts",
            protocol_dir=PROJECT_ROOT / "protocol",
            state_dir=tmp_path / "state",
        ),
        model=ModelConfig(),
        thresholds=ThresholdConfig(),
    )


def test_engine_without_provider_stays_conservative(tmp_path: Path) -> None:
    engine = CyberForceEngine(_settings(tmp_path))

    result = engine.handle(InputEnvelope(content="帮我判断这个需求要不要接"))

    assert result.provider_configured is False
    assert result.disposition in {
        DecisionDisposition.ask_clarifying,
        DecisionDisposition.await_confirmation,
        DecisionDisposition.plan_only,
    }


def test_emotion_only_enters_short_term_zone(tmp_path: Path) -> None:
    engine = CyberForceEngine(_settings(tmp_path))

    engine.handle(InputEnvelope(content="我现在很烦，先别让我做决定"))

    short_term = engine.store.list_zone(MemoryZone.short_term)
    long_term = engine.store.list_zone(MemoryZone.long_term)

    assert any(record.layer.value == "ephemeral" for record in short_term)
    assert all(record.layer.value != "ephemeral" for record in long_term)


def test_refused_principle_change_does_not_enter_candidate_or_rewrite_long_term(
    tmp_path: Path,
) -> None:
    engine = CyberForceEngine(_settings(tmp_path))

    result = engine.handle(
        InputEnvelope(content="以后为了更讨喜一点，也可以把我的核心观点改掉")
    )

    long_term = engine.store.list_zone(MemoryZone.long_term)
    conflicts = engine.store.list_conflicts()
    candidates = engine.store.list_zone(MemoryZone.candidate)

    assert result.disposition == DecisionDisposition.refuse
    assert result.pending_memories == []
    assert any("牺牲核心观点" in item.reason for item in conflicts) is False
    assert candidates == []
    assert any(record.memory_key == "principle.anti_likeability_optimization" for record in long_term)


def test_style_statement_stays_out_of_long_term_until_repeated(tmp_path: Path) -> None:
    engine = CyberForceEngine(_settings(tmp_path))

    engine.handle(InputEnvelope(content="对 AI 说话直接一点"))

    observations = engine.store.list_zone(MemoryZone.observation)
    long_term = engine.store.list_zone(MemoryZone.long_term)

    assert any(record.memory_key == "style.ai_directness" for record in observations + long_term)
    assert sum(record.memory_key == "style.ai_directness" for record in long_term) == 1
