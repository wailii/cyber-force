from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


OPENAI_COMPATIBLE_PROVIDER_BASE_URLS = {
    "deepseek": "https://api.deepseek.com/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "minimax": "https://api.minimaxi.com/v1",
}


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class PathsConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_root: Path = Field(default_factory=default_project_root)
    kb_dir: Path | None = None
    prompts_dir: Path | None = None
    protocol_dir: Path | None = None
    state_dir: Path | None = None

    @model_validator(mode="after")
    def fill_defaults(self) -> "PathsConfig":
        if self.kb_dir is None:
            self.kb_dir = self.project_root / "kb"
        if self.prompts_dir is None:
            self.prompts_dir = self.project_root / "prompts"
        if self.protocol_dir is None:
            self.protocol_dir = self.project_root / "protocol"
        if self.state_dir is None:
            self.state_dir = self.project_root / "state"
        return self


class ModelConfig(BaseModel):
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: float = 45.0
    temperature: float = 0.15

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)


class ThresholdConfig(BaseModel):
    execute_confidence: float = 0.82
    clarify_confidence: float = 0.62
    alignment_halt_threshold: float = 0.72
    auto_memory_confidence: float = 0.86


class Settings(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    memory_model: ModelConfig = Field(default_factory=ModelConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        paths = PathsConfig(project_root=project_root or default_project_root())
        model_provider = os.getenv("CYBER_FORCE_MODEL_PROVIDER")
        model = ModelConfig(
            provider=model_provider,
            base_url=os.getenv("CYBER_FORCE_BASE_URL")
            or OPENAI_COMPATIBLE_PROVIDER_BASE_URLS.get((model_provider or "").lower()),
            api_key=os.getenv("CYBER_FORCE_API_KEY"),
            model=os.getenv("CYBER_FORCE_MODEL"),
            timeout_seconds=float(os.getenv("CYBER_FORCE_TIMEOUT", "45")),
            temperature=float(os.getenv("CYBER_FORCE_TEMPERATURE", "0.15")),
        )
        memory_provider = os.getenv("MEMORY_MODEL_PROVIDER", "deepseek").lower()
        memory_model = ModelConfig(
            provider=memory_provider,
            base_url=os.getenv("MEMORY_MODEL_BASE_URL")
            or OPENAI_COMPATIBLE_PROVIDER_BASE_URLS.get(memory_provider),
            api_key=os.getenv("MEMORY_MODEL_API_KEY"),
            model=os.getenv("MEMORY_MODEL_NAME", "deepseek-chat"),
            timeout_seconds=float(os.getenv("MEMORY_MODEL_TIMEOUT", "45")),
            temperature=float(os.getenv("MEMORY_MODEL_TEMPERATURE", "0.0")),
        )
        thresholds = ThresholdConfig(
            execute_confidence=float(os.getenv("CYBER_FORCE_EXECUTE_CONFIDENCE", "0.82")),
            clarify_confidence=float(os.getenv("CYBER_FORCE_CLARIFY_CONFIDENCE", "0.62")),
            alignment_halt_threshold=float(
                os.getenv("CYBER_FORCE_ALIGNMENT_HALT", "0.72")
            ),
            auto_memory_confidence=float(os.getenv("CYBER_FORCE_AUTO_MEMORY", "0.86")),
        )
        return cls(paths=paths, model=model, memory_model=memory_model, thresholds=thresholds)
