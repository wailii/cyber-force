from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def future_utc(days: int) -> datetime:
    return utc_now() + timedelta(days=days)


class ConfidenceBand(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class EventType(str, Enum):
    principle = "principle"
    fact_self = "fact_self"
    fact_world = "fact_world"
    preference = "preference"
    style = "style"
    context_mode = "context_mode"
    emotion = "emotion"
    task_decision = "task_decision"
    conflict_signal = "conflict_signal"
    meta_instruction = "meta_instruction"
    unknown = "unknown"


class IdentityLayer(str, Enum):
    principles = "principles"
    identity = "identity"
    style = "style"
    modes = "modes"
    ephemeral = "ephemeral"


class MemoryZone(str, Enum):
    short_term = "short_term"
    observation = "observation"
    candidate = "candidate"
    long_term = "long_term"


class MemoryStatus(str, Enum):
    active = "active"
    frozen = "frozen"
    revoked = "revoked"
    superseded = "superseded"


class ClaimSourceKind(str, Enum):
    direct = "direct"
    inferred = "inferred"
    external = "external"


class DecisionDisposition(str, Enum):
    act = "act"
    plan_only = "plan_only"
    ask_clarifying = "ask_clarifying"
    await_confirmation = "await_confirmation"
    challenge = "challenge"
    refuse = "refuse"
    defer = "defer"
    log_only = "log_only"


class CarrierCapability(str, Enum):
    text_input = "text_input"
    text_output = "text_output"
    voice_input = "voice_input"
    push_notify = "push_notify"
    local_files = "local_files"
    browser_action = "browser_action"
    calendar_read = "calendar_read"
    calendar_write = "calendar_write"
    email_read = "email_read"
    email_write = "email_write"
    webhooks = "webhooks"


class PermissionLevel(str, Enum):
    none = "none"
    read = "read"
    propose = "propose"
    act = "act"


class InputEnvelope(BaseModel):
    content: str
    source: str = "cli"
    carrier_id: str | None = None
    session_id: str | None = None
    audience: str | None = None
    intent_hint: str | None = None
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    received_at: datetime = Field(default_factory=utc_now)


class InputEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    carrier_id: str | None = None
    raw_text: str
    normalized_request: str
    audience_mode: str | None = None
    event_type: EventType = EventType.unknown
    source_kind: ClaimSourceKind = ClaimSourceKind.direct
    urgency: Literal["low", "normal", "high"] = "normal"
    emotional_volatility: bool = False
    ambiguity_score: float = 0.0
    missing_context: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    received_at: datetime = Field(default_factory=utc_now)


class IntentFrame(BaseModel):
    intent: Literal["question", "task", "note", "reflection", "update", "unknown"] = "unknown"
    user_goal: str | None = None
    requested_action: str | None = None
    external_effect: bool = False
    requires_decision: bool = True
    ambiguity_score: float = 0.0


class IntakePacket(BaseModel):
    event: InputEvent
    intent: IntentFrame


class MemoryCandidate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    memory_key: str
    layer: IdentityLayer
    proposed_zone: MemoryZone = MemoryZone.observation
    event_type: EventType
    statement: str
    reason: str
    confidence: float = 0.0
    source_kind: ClaimSourceKind = ClaimSourceKind.direct
    evidence: list[str] = Field(default_factory=list)
    requires_confirmation: bool = True
    expires_in_days: int | None = None
    tags: list[str] = Field(default_factory=list)


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    memory_key: str
    layer: IdentityLayer
    zone: MemoryZone
    event_type: EventType
    statement: str
    reason: str
    source: str
    source_kind: ClaimSourceKind = ClaimSourceKind.direct
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    observation_count: int = 1
    version: int = 1
    status: MemoryStatus = MemoryStatus.active
    requires_confirmation: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    first_seen_at: datetime = Field(default_factory=utc_now)
    last_seen_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None


class PromotionRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    memory_key: str
    from_zone: MemoryZone
    to_zone: MemoryZone
    reason: str
    event_id: str
    recorded_at: datetime = Field(default_factory=utc_now)


class ConflictRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    memory_key: str
    layer: IdentityLayer
    existing_statement: str
    proposed_statement: str
    reason: str
    severity: Literal["medium", "high"] = "high"
    event_id: str | None = None
    recorded_at: datetime = Field(default_factory=utc_now)


class ContextFrame(BaseModel):
    identity_snapshot: dict[str, Any]
    active_modes: list[dict[str, Any]] = Field(default_factory=list)
    relevant_memories: list[MemoryRecord] = Field(default_factory=list)
    pending_candidates: list[MemoryRecord] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    capability_scope: list[str] = Field(default_factory=list)


class ActionProposal(BaseModel):
    kind: str
    description: str
    capability: CarrierCapability | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class CritiqueReport(BaseModel):
    mode: Literal["pass", "revise", "halt"] = "halt"
    aligned: bool = False
    alignment_score: float = 0.0
    conflicts: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    requires_confirmation: bool = False
    recommended_changes: list[str] = Field(default_factory=list)


class DecisionAssessment(BaseModel):
    summary: str
    response_message: str
    disposition: DecisionDisposition
    confidence: float
    confidence_band: ConfidenceBand = ConfidenceBand.low
    reasoning: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    requires_user_confirmation: bool = False
    clarification_questions: list[str] = Field(default_factory=list)
    proposed_actions: list[ActionProposal] = Field(default_factory=list)
    voice_alignment_checks: list[str] = Field(default_factory=list)
    refusal_reason: str | None = None
    next_step: str | None = None


class AuditRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    input: InputEnvelope
    event: InputEvent
    intent: IntentFrame
    context: ContextFrame
    critique: CritiqueReport
    decision: DecisionAssessment
    written_memories: list[MemoryRecord] = Field(default_factory=list)
    pending_memories: list[MemoryCandidate] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    recorded_at: datetime = Field(default_factory=utc_now)


class EngineResponse(BaseModel):
    disposition: DecisionDisposition
    confidence: float
    confidence_band: ConfidenceBand
    message: str
    clarification_questions: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    proposed_actions: list[ActionProposal] = Field(default_factory=list)
    pending_memories: list[MemoryCandidate] = Field(default_factory=list)
    provider_configured: bool = False
    next_step: str | None = None
    audit_id: str


class KnowledgeBaseBundle(BaseModel):
    constitution_text: str
    principles: dict[str, Any]
    self_facts: dict[str, Any]
    style: dict[str, Any]
    modes: dict[str, Any]
    decision_policy: dict[str, Any]
    write_guardrails: dict[str, Any]
    action_policy: dict[str, Any]
    memory_classification: dict[str, Any]
    promotion_policy: dict[str, Any]
    bootstrap_memories: list[dict[str, Any]] = Field(default_factory=list)


class CarrierManifest(BaseModel):
    protocol_version: str = "0.2"
    carrier_id: str
    name: str
    version: str
    transport: Literal["local_cli", "local_http", "desktop_app", "wearable", "daemon", "webhook"]
    adapter: str | None = None
    capabilities: list[CarrierCapability] = Field(default_factory=list)
    permissions: dict[str, PermissionLevel] = Field(default_factory=dict)
    identity_claim: dict[str, Any] = Field(default_factory=dict)
    session_defaults: dict[str, Any] = Field(default_factory=dict)
    bundle_mount: dict[str, Any] = Field(default_factory=dict)
    integrity: dict[str, Any] = Field(default_factory=dict)
    device: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionContract(BaseModel):
    carrier_id: str
    accepted_capabilities: list[CarrierCapability] = Field(default_factory=list)
    denied_capabilities: list[CarrierCapability] = Field(default_factory=list)
    permissions: dict[str, PermissionLevel] = Field(default_factory=dict)
    required_confirmation_for: list[str] = Field(default_factory=list)
    execution_allowed: bool = False
    identity_fingerprint: str | None = None
    policy_version: str | None = None
