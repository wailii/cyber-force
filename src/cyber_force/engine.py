from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

from .config import Settings
from .kb_loader import load_knowledge_base
from .memory import FileBackedStore
from .prompts import PromptLibrary
from .provider import OpenAICompatibleProvider
from .schemas import (
    ActionProposal,
    AuditRecord,
    ClaimSourceKind,
    ConfidenceBand,
    ConflictRecord,
    ContextFrame,
    CritiqueReport,
    DecisionAssessment,
    DecisionDisposition,
    EngineResponse,
    EventType,
    IdentityLayer,
    InputEnvelope,
    InputEvent,
    IntakePacket,
    IntentFrame,
    MemoryCandidate,
    MemoryRecord,
    MemoryStatus,
    MemoryZone,
    PromotionRecord,
    future_utc,
    utc_now,
)


class CyberForceEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.kb = load_knowledge_base(self.settings.paths.kb_dir)
        self.store = FileBackedStore(self.settings.paths.state_dir, settings=self.settings)
        self.prompts = PromptLibrary(self.settings.paths.prompts_dir)
        self.provider = OpenAICompatibleProvider(self.settings.model)
        self._seed_bootstrap_memories()

    def handle(self, envelope: InputEnvelope) -> EngineResponse:
        intake = self._run_intake(envelope)
        self.store.append_event(intake.event)

        recent_memories = self.store.list_contextual_memories()
        context = self._build_context(envelope, intake.event, intake.intent, recent_memories)
        written_memories, pending_candidates, conflicts = self._ingest_memory_candidates(
            envelope=envelope,
            event=intake.event,
            intent=intake.intent,
            context=context,
            recent_memories=recent_memories,
        )
        context.pending_candidates = self.store.list_zone(MemoryZone.candidate, limit=24)
        critique = self._run_critique(
            envelope=envelope,
            event=intake.event,
            intent=intake.intent,
            context=context,
            conflicts=conflicts,
            pending_candidates=pending_candidates,
        )
        decision = self._run_decision(
            envelope=envelope,
            event=intake.event,
            intent=intake.intent,
            context=context,
            critique=critique,
            conflicts=conflicts,
            pending_candidates=pending_candidates,
        )
        response = self._finalize(decision, pending_candidates)

        audit = AuditRecord(
            input=envelope,
            event=intake.event,
            intent=intake.intent,
            context=context,
            critique=critique,
            decision=decision,
            written_memories=written_memories,
            pending_memories=pending_candidates,
            conflicts=conflicts,
        )
        self.store.append_audit(audit)
        response.audit_id = audit.id
        return response

    def _run_intake(self, envelope: InputEnvelope) -> IntakePacket:
        if self.provider.configured:
            return self._call_model(
                prompt_name="intake",
                output_model=IntakePacket,
                input=envelope.model_dump(mode="json"),
                constitution=self.kb.constitution_text,
                principles=self.kb.principles,
                self_facts=self.kb.self_facts,
                style=self.kb.style,
                schema=IntakePacket.model_json_schema(),
            )

        content = envelope.content.strip()
        event_type = self._infer_event_type(content)
        missing_context = []
        if len(content) < 12:
            missing_context.append("输入过短，缺少对象、目标或边界。")

        emotional_volatility = event_type == EventType.emotion or bool(
            re.search(r"(烦|怒|气死|焦虑|崩溃|委屈|冲动|失眠|急死)", content)
        )
        ambiguity_score = 0.2
        if missing_context:
            ambiguity_score += 0.35
        if envelope.intent_hint is None and not re.search(r"(帮我|请你|记录|我觉得|我想)", content):
            ambiguity_score += 0.15

        event = InputEvent(
            source=envelope.source,
            carrier_id=envelope.carrier_id,
            raw_text=content,
            normalized_request=content,
            audience_mode=envelope.audience,
            event_type=event_type,
            source_kind=ClaimSourceKind.direct,
            urgency=envelope.metadata.get("urgency", "normal"),
            emotional_volatility=emotional_volatility,
            ambiguity_score=min(ambiguity_score, 1.0),
            missing_context=missing_context,
            tags=self._event_tags(content, event_type),
            received_at=envelope.received_at,
        )
        intent = IntentFrame(
            intent=self._infer_intent(content, envelope.intent_hint),
            user_goal=content,
            requested_action=self._infer_requested_action(content),
            external_effect=self._has_external_effect(content),
            requires_decision=True,
            ambiguity_score=event.ambiguity_score,
        )
        return IntakePacket(event=event, intent=intent)

    def _build_context(
        self,
        envelope: InputEnvelope,
        event: InputEvent,
        intent: IntentFrame,
        recent_memories: list[MemoryRecord],
    ) -> ContextFrame:
        del intent
        audience = envelope.audience or event.audience_mode
        active_modes = []
        for mode in self.kb.modes.get("modes", []):
            audiences = mode.get("audiences", [])
            if audience and audience in audiences:
                active_modes.append(mode)

        relevant_memories = recent_memories[-16:]
        candidate_records = self.store.list_zone(MemoryZone.candidate, limit=16)
        style_by_audience = self.kb.style.get("audiences", {}).get(audience or "", {})
        identity_snapshot = {
            "principles": self.kb.principles,
            "self_facts": self.kb.self_facts,
            "style": self.kb.style,
            "audience_style": style_by_audience,
            "decision_policy": self.kb.decision_policy,
            "write_guardrails": self.kb.write_guardrails,
        }
        return ContextFrame(
            identity_snapshot=identity_snapshot,
            active_modes=active_modes,
            relevant_memories=relevant_memories,
            pending_candidates=candidate_records,
            missing_facts=list(event.missing_context),
            capability_scope=envelope.metadata.get(
                "capabilities", ["text_input", "text_output"]
            ),
        )

    def _ingest_memory_candidates(
        self,
        envelope: InputEnvelope,
        event: InputEvent,
        intent: IntentFrame,
        context: ContextFrame,
        recent_memories: list[MemoryRecord],
    ) -> tuple[list[MemoryRecord], list[MemoryCandidate], list[ConflictRecord]]:
        if self._hard_refusal_reason(envelope.content):
            return [], [], []

        candidates = self._extract_candidate_memories(
            envelope=envelope,
            event=event,
            intent=intent,
            context=context,
            recent_memories=recent_memories,
        )

        written: list[MemoryRecord] = []
        pending: list[MemoryCandidate] = []
        conflicts: list[ConflictRecord] = []
        promotions: list[PromotionRecord] = []

        for candidate in candidates:
            record, candidate_conflict, promotion = self._materialize_candidate(candidate, event)
            if candidate_conflict is not None:
                conflicts.append(candidate_conflict)
                pending.append(candidate)
                self.store.append_conflict(candidate_conflict)
                continue

            if record is None:
                continue

            written.append(record)
            self.store.append_memory(record)
            if record.zone == MemoryZone.candidate:
                pending.append(candidate)
            if promotion is not None:
                promotions.append(promotion)
                self.store.append_promotion(promotion)

        return written, pending, conflicts

    def _extract_candidate_memories(
        self,
        envelope: InputEnvelope,
        event: InputEvent,
        intent: IntentFrame,
        context: ContextFrame,
        recent_memories: list[MemoryRecord],
    ) -> list[MemoryCandidate]:
        if self.provider.configured:
            return self._call_model(
                prompt_name="memory_extract",
                output_model=list[MemoryCandidate],
                input=envelope.model_dump(mode="json"),
                event=event.model_dump(mode="json"),
                intent=intent.model_dump(mode="json"),
                context=context.model_dump(mode="json"),
                recent_memories=[item.model_dump(mode="json") for item in recent_memories[-12:]],
                classification_policy=self.kb.memory_classification,
                promotion_policy=self.kb.promotion_policy,
                schema={"type": "array", "items": MemoryCandidate.model_json_schema()},
            )

        content = envelope.content.strip()
        candidates: list[MemoryCandidate] = []
        event_type = event.event_type

        if event_type in {EventType.unknown, EventType.fact_world, EventType.task_decision}:
            return candidates

        layer = self._layer_for_event_type(event_type)
        if layer is None:
            return candidates

        candidate = MemoryCandidate(
            memory_key=self._memory_key_for(content, event_type),
            layer=layer,
            proposed_zone=self._default_zone_for_layer(layer),
            event_type=event_type,
            statement=content,
            reason=self._candidate_reason(event_type),
            confidence=self._candidate_confidence(event_type, event),
            source_kind=ClaimSourceKind.direct,
            evidence=[content],
            requires_confirmation=layer not in {IdentityLayer.ephemeral, IdentityLayer.modes},
            expires_in_days=self._default_expiry_for(layer),
            tags=event.tags,
        )
        candidates.append(candidate)
        return candidates

    def _materialize_candidate(
        self, candidate: MemoryCandidate, event: InputEvent
    ) -> tuple[MemoryRecord | None, ConflictRecord | None, PromotionRecord | None]:
        existing = self.store.find_latest_memory(candidate.memory_key)
        if (
            existing is not None
            and existing.zone == MemoryZone.long_term
            and self._existing_conflict_with_candidate(existing, candidate)
            and existing.layer == candidate.layer
        ):
            return (
                None,
                ConflictRecord(
                    memory_key=candidate.memory_key,
                    layer=candidate.layer,
                    existing_statement=existing.statement,
                    proposed_statement=candidate.statement,
                    reason="新输入与已确认长期条目冲突，暂停升级。",
                    event_id=event.id,
                ),
                None,
            )

        zone = candidate.proposed_zone
        promotion: PromotionRecord | None = None
        observation_count = 1
        version = 1

        if existing is not None:
            observation_count = existing.observation_count + 1
            version = existing.version + 1
            if existing.zone == MemoryZone.observation and self._eligible_for_candidate(
                candidate, existing, observation_count
            ):
                zone = MemoryZone.candidate
                promotion = PromotionRecord(
                    memory_key=candidate.memory_key,
                    from_zone=existing.zone,
                    to_zone=zone,
                    reason="达到候选晋升条件。",
                    event_id=event.id,
                )
            elif existing.zone == MemoryZone.candidate:
                zone = MemoryZone.candidate

        record = MemoryRecord(
            memory_key=candidate.memory_key,
            layer=candidate.layer,
            zone=zone,
            event_type=candidate.event_type,
            statement=candidate.statement,
            reason=candidate.reason,
            source=event.source,
            source_kind=candidate.source_kind,
            confidence=candidate.confidence,
            evidence=candidate.evidence,
            tags=candidate.tags,
            observation_count=observation_count,
            version=version,
            status=MemoryStatus.active,
            requires_confirmation=zone == MemoryZone.candidate or candidate.requires_confirmation,
            first_seen_at=existing.first_seen_at if existing is not None else event.received_at,
            last_seen_at=event.received_at,
            expires_at=future_utc(candidate.expires_in_days)
            if candidate.expires_in_days
            else None,
        )
        return record, None, promotion

    def _run_critique(
        self,
        envelope: InputEnvelope,
        event: InputEvent,
        intent: IntentFrame,
        context: ContextFrame,
        conflicts: list[ConflictRecord],
        pending_candidates: list[MemoryCandidate],
    ) -> CritiqueReport:
        if self.provider.configured:
            return self._call_model(
                prompt_name="critic",
                output_model=CritiqueReport,
                input=envelope.model_dump(mode="json"),
                event=event.model_dump(mode="json"),
                intent=intent.model_dump(mode="json"),
                context=context.model_dump(mode="json"),
                constitution=self.kb.constitution_text,
                principles=self.kb.principles,
                write_guardrails=self.kb.write_guardrails,
                conflicts=[item.model_dump(mode="json") for item in conflicts],
                pending_candidates=[item.model_dump(mode="json") for item in pending_candidates],
                schema=CritiqueReport.model_json_schema(),
            )

        hard_refusal = self._hard_refusal_reason(envelope.content)
        critique = CritiqueReport(
            mode="pass",
            aligned=True,
            alignment_score=0.78 if self.provider.configured else 0.52,
            conflicts=[item.reason for item in conflicts],
            risk_flags=[],
            assumptions=[],
            requires_confirmation=False,
            recommended_changes=[],
        )

        if hard_refusal:
            critique.mode = "halt"
            critique.aligned = False
            critique.alignment_score = 0.1
            critique.conflicts.append(hard_refusal)
            critique.risk_flags.append("hard_refusal")
            critique.recommended_changes.append("拒绝该请求，不让载体继续执行。")
            return critique

        if event.missing_context:
            critique.mode = "revise"
            critique.requires_confirmation = True
            critique.risk_flags.append("missing_context")
            critique.recommended_changes.append("先补齐对象、目标和边界，再判断。")

        if conflicts:
            critique.mode = "revise"
            critique.requires_confirmation = True
            critique.risk_flags.append("identity_conflict")
            critique.recommended_changes.append("先显性指出冲突，再决定是否接受新的长期规则。")

        if intent.external_effect:
            critique.requires_confirmation = True
            critique.risk_flags.append("external_effect")
            critique.recommended_changes.append("先给方案或草稿，不直接替你对外发出。")

        if pending_candidates:
            critique.risk_flags.append("memory_pending")

        if not self.provider.configured:
            critique.risk_flags.append("no_model_provider")
            critique.recommended_changes.append("没有模型时维持保守，不假装高置信。")

        return critique

    def _run_decision(
        self,
        envelope: InputEnvelope,
        event: InputEvent,
        intent: IntentFrame,
        context: ContextFrame,
        critique: CritiqueReport,
        conflicts: list[ConflictRecord],
        pending_candidates: list[MemoryCandidate],
    ) -> DecisionAssessment:
        if self.provider.configured:
            decision = self._call_model(
                prompt_name="decision",
                output_model=DecisionAssessment,
                input=envelope.model_dump(mode="json"),
                event=event.model_dump(mode="json"),
                intent=intent.model_dump(mode="json"),
                context=context.model_dump(mode="json"),
                critique=critique.model_dump(mode="json"),
                constitution=self.kb.constitution_text,
                action_policy=self.kb.action_policy,
                pending_candidates=[item.model_dump(mode="json") for item in pending_candidates],
                schema=DecisionAssessment.model_json_schema(),
            )
            decision.confidence_band = self._confidence_band(decision.confidence)
            return decision

        hard_refusal = self._hard_refusal_reason(envelope.content)
        if hard_refusal:
            return DecisionAssessment(
                summary="请求触碰了主权边界。",
                response_message=f"我不会这样代表你行动：{hard_refusal}",
                disposition=DecisionDisposition.refuse,
                confidence=0.98,
                confidence_band=ConfidenceBand.high,
                reasoning=[
                    "该请求要求系统背离已确认原则。",
                    "主权层规则优先于当前任务意图。",
                ],
                contradictions=[hard_refusal],
                refusal_reason=hard_refusal,
                voice_alignment_checks=["拒绝比顺着说更符合你的治理规则。"],
                next_step="换一个不违背核心原则的做法。",
            )

        if conflicts:
            return DecisionAssessment(
                summary="当前输入和已确认身份条目冲突。",
                response_message="这次我先不顺着执行。你的新说法和已确认长期原则有冲突，我需要先把矛盾摊开。",
                disposition=DecisionDisposition.challenge,
                confidence=0.8,
                confidence_band=ConfidenceBand.medium,
                reasoning=[
                    "代表你行动之前，系统必须先指出冲突。",
                    "冲突未解时不能静默改写人格层。",
                ],
                contradictions=[item.reason for item in conflicts],
                clarification_questions=[
                    "这是你要长期修订的原则，还是这次情境下的策略性说法？"
                ],
                voice_alignment_checks=["系统显性指出矛盾，而不是自动圆过去。"],
                next_step="确认是否修订长期原则。",
            )

        if event.missing_context:
            return DecisionAssessment(
                summary="上下文不足，不能安全代你判断。",
                response_message="信息不够，我先不替你拍板。请补对象、目标和绝对不能越过的边界。",
                disposition=DecisionDisposition.ask_clarifying,
                confidence=0.32,
                confidence_band=ConfidenceBand.low,
                reasoning=[
                    "低置信度默认停下来。",
                    "缺少关键事实时继续判断只是在猜。",
                ],
                clarification_questions=self._clarification_questions(intent),
                voice_alignment_checks=["系统选择停下来，而不是瞎猜。"],
                next_step="补充最小必要上下文。",
            )

        if critique.requires_confirmation or intent.external_effect:
            return DecisionAssessment(
                summary="方向可以继续，但不应越权执行。",
                response_message="我先给保守方案，不直接替你拍板或对外执行。你确认后再继续。",
                disposition=DecisionDisposition.await_confirmation,
                confidence=0.62,
                confidence_band=ConfidenceBand.medium,
                reasoning=[
                    "涉及外部后果或长期记忆写入时，需要确认。",
                    "当前版本默认先给方案，不默认执行。",
                ],
                proposed_actions=self._proposed_actions(intent),
                voice_alignment_checks=["先给保守方案，符合默认保守原则。"],
                next_step="确认是否继续，以及是否允许写入候选记忆。",
            )

        if intent.intent == "note":
            return DecisionAssessment(
                summary="内容已收下并进入审计链。",
                response_message="我先把这条记下来，不把它自动升级成长期人格。",
                disposition=DecisionDisposition.plan_only,
                confidence=0.58,
                confidence_band=ConfidenceBand.medium,
                reasoning=["当前更像记录，而不是立即行动。"],
                voice_alignment_checks=["记录不等于学习进长期层。"],
                next_step="后续如果重复出现，再进入候选层。",
            )

        return DecisionAssessment(
            summary="当前没有足够安全条件直接替你行动。",
            response_message="我能先给你一个保守判断或草稿，但这版不会直接代你执行高后果动作。",
            disposition=DecisionDisposition.plan_only,
            confidence=0.56,
            confidence_band=ConfidenceBand.medium,
            reasoning=[
                "当前运行时优先证明治理资格，而不是追求执行面。",
                "没有更高置信度或授权前，先停在方案层。",
            ],
            proposed_actions=self._proposed_actions(intent),
            voice_alignment_checks=["先做 plan，不装作已经能完全代你执行。"],
            next_step="如需执行，再明确授权载体能力。",
        )

    def _finalize(
        self, decision: DecisionAssessment, pending_memories: list[MemoryCandidate]
    ) -> EngineResponse:
        return EngineResponse(
            disposition=decision.disposition,
            confidence=decision.confidence,
            confidence_band=self._confidence_band(decision.confidence),
            message=decision.response_message,
            clarification_questions=decision.clarification_questions,
            contradictions=decision.contradictions,
            proposed_actions=decision.proposed_actions,
            pending_memories=pending_memories,
            provider_configured=self.provider.configured,
            next_step=decision.next_step,
            audit_id="pending",
        )

    def _call_model(self, prompt_name: str, output_model: Any, **context: Any) -> Any:
        system_prompt = (
            "You are a component inside a sovereign personal agent runtime. "
            "Return valid JSON only. Do not explain outside the JSON structure."
        )
        prompt = self.prompts.render(prompt_name, **context)
        payload = self.provider.complete_json(system_prompt=system_prompt, user_prompt=prompt)
        try:
            return output_model.model_validate(payload)
        except AttributeError:
            if output_model == list[MemoryCandidate]:
                return [MemoryCandidate.model_validate(item) for item in payload]
            raise

    def _confidence_band(self, confidence: float) -> ConfidenceBand:
        if confidence >= 0.82:
            return ConfidenceBand.high
        if confidence >= 0.55:
            return ConfidenceBand.medium
        return ConfidenceBand.low

    def _seed_bootstrap_memories(self) -> None:
        if self.store.list_zone(MemoryZone.long_term, limit=1):
            return

        records: list[MemoryRecord] = []
        for item in self.kb.bootstrap_memories:
            records.append(
                MemoryRecord(
                    memory_key=item["memory_key"],
                    layer=IdentityLayer(item["layer"]),
                    zone=MemoryZone(item.get("zone", MemoryZone.long_term.value)),
                    event_type=EventType(item["event_type"]),
                    statement=item["statement"],
                    reason=item.get("reason", "bootstrap memory"),
                    source="bootstrap",
                    source_kind=ClaimSourceKind(item.get("source_kind", ClaimSourceKind.direct.value)),
                    confidence=float(item.get("confidence", 1.0)),
                    tags=item.get("tags", ["bootstrap"]),
                    requires_confirmation=False,
                    expires_at=future_utc(int(item["expires_in_days"]))
                    if item.get("expires_in_days")
                    else None,
                )
            )

        if records:
            self.store.append_memories(records)

    def _infer_event_type(self, content: str) -> EventType:
        if re.search(r"(烦|怒|焦虑|崩溃|委屈|冲动|失眠|很累|很急)", content):
            return EventType.emotion
        if re.search(r"(以后|原则|必须|不能只|宁可|绝对不能)", content):
            return EventType.principle
        if re.search(r"(不要从一次聊天学|别从一次聊天学|临时情绪绝对不能)", content):
            return EventType.meta_instruction
        if re.search(r"(我是|我做|我负责|我有计算机背景|我是 B2B AI 产品经理)", content):
            return EventType.fact_self
        if re.search(r"(讨厌|不喜欢|偏好|更喜欢|不想被)", content):
            return EventType.preference
        if re.search(r"(对 AI|对领导|对公众|表达方式|说话很直接|包装)", content):
            return EventType.style
        if re.search(r"(这次|本轮|当前场景|这轮)", content):
            return EventType.context_mode
        if re.search(r"(矛盾|前后不一致|冲突)", content):
            return EventType.conflict_signal
        if re.search(r"(客户|会议|下周|今天|明天|这个需求)", content):
            return EventType.fact_world
        return EventType.unknown

    def _infer_intent(self, content: str, hint: str | None) -> str:
        if hint:
            return hint
        if "?" in content or "？" in content or "吗" in content:
            return "question"
        if re.search(r"(记录|备注|记住|存一下)", content):
            return "note"
        if re.search(r"(我觉得|我发现|我在想)", content):
            return "reflection"
        if re.search(r"(帮我|请你|判断|回复|写|安排|拒绝)", content):
            return "task"
        return "unknown"

    def _infer_requested_action(self, content: str) -> str | None:
        if re.search(r"(回复|回信|回消息)", content):
            return "draft_reply"
        if re.search(r"(判断|拍板|决策)", content):
            return "judge"
        if re.search(r"(记录|存一下|记住)", content):
            return "record"
        if re.search(r"(写|整理|起草)", content):
            return "draft"
        return None

    def _event_tags(self, content: str, event_type: EventType) -> list[str]:
        tags = [event_type.value]
        if "领导" in content:
            tags.append("leadership")
        if "公众" in content:
            tags.append("public")
        if "AI" in content or "ai" in content:
            tags.append("ai")
        return tags

    def _has_external_effect(self, content: str) -> bool:
        return bool(re.search(r"(发给|发送|回复客户|对外|承诺|公开说|发邮件|回消息)", content))

    def _layer_for_event_type(self, event_type: EventType) -> IdentityLayer | None:
        mapping = {
            EventType.principle: IdentityLayer.principles,
            EventType.meta_instruction: IdentityLayer.principles,
            EventType.fact_self: IdentityLayer.identity,
            EventType.preference: IdentityLayer.style,
            EventType.style: IdentityLayer.style,
            EventType.context_mode: IdentityLayer.modes,
            EventType.emotion: IdentityLayer.ephemeral,
        }
        return mapping.get(event_type)

    def _default_zone_for_layer(self, layer: IdentityLayer) -> MemoryZone:
        if layer in {IdentityLayer.ephemeral, IdentityLayer.modes}:
            return MemoryZone.short_term
        if layer in {IdentityLayer.principles, IdentityLayer.identity}:
            return MemoryZone.candidate
        return MemoryZone.observation

    def _default_expiry_for(self, layer: IdentityLayer) -> int | None:
        rules = self.kb.promotion_policy.get("zones", {})
        if layer == IdentityLayer.ephemeral:
            return rules.get("short_term", {}).get("ephemeral_expiry_days", 2)
        if layer == IdentityLayer.modes:
            return rules.get("short_term", {}).get("mode_expiry_days", 7)
        return None

    def _candidate_reason(self, event_type: EventType) -> str:
        reasons = {
            EventType.principle: "疑似长期原则，需要确认后才能稳固进人格层。",
            EventType.meta_instruction: "疑似系统治理规则，需要写保护。",
            EventType.fact_self: "疑似长期身份事实，需要确认后才上升。",
            EventType.preference: "疑似稳定偏好，需要多次观察。",
            EventType.style: "疑似表达风格，需要跨场景观察。",
            EventType.context_mode: "疑似本轮情境模式，只应短期生效。",
            EventType.emotion: "疑似临时情绪，只能短期缓存。",
        }
        return reasons.get(event_type, "可学习性不足，默认不升级。")

    def _candidate_confidence(self, event_type: EventType, event: InputEvent) -> float:
        base = {
            EventType.principle: 0.78,
            EventType.meta_instruction: 0.82,
            EventType.fact_self: 0.76,
            EventType.preference: 0.68,
            EventType.style: 0.68,
            EventType.context_mode: 0.72,
            EventType.emotion: 0.7,
        }.get(event_type, 0.45)
        if event.emotional_volatility and event_type != EventType.emotion:
            base -= 0.15
        return max(0.0, min(base, 0.99))

    def _eligible_for_candidate(
        self,
        candidate: MemoryCandidate,
        existing: MemoryRecord,
        observation_count: int,
    ) -> bool:
        layer_rules = self.kb.promotion_policy.get("layers", {}).get(candidate.layer.value, {})
        min_observations = int(layer_rules.get("min_observations", 2))
        min_gap_days = int(layer_rules.get("min_separation_days", 0))
        enough_gap = True
        if min_gap_days > 0:
            enough_gap = utc_now() - existing.first_seen_at >= timedelta(days=min_gap_days)
        return (
            observation_count >= min_observations
            and candidate.confidence >= float(layer_rules.get("candidate_confidence", 0.68))
            and enough_gap
        )

    def _memory_key_for(self, content: str, event_type: EventType) -> str:
        mapping = {
            (EventType.principle, "置信度"): "principle.uncertainty_pause",
            (EventType.principle, "讨喜"): "principle.anti_likeability_optimization",
            (EventType.principle, "矛盾"): "principle.expose_conflicts",
            (EventType.meta_instruction, "临时情绪"): "principle.no_emotion_promotion",
            (EventType.fact_self, "B2B AI 产品经理"): "identity.primary_role",
            (EventType.fact_self, "计算机背景"): "identity.technical_background",
            (EventType.fact_self, "Agent 工作流"): "identity.agent_workflow_familiar",
            (EventType.preference, "讨厌被优化成更讨喜"): "style.anti_likeability",
            (EventType.style, "对 AI"): "style.ai_directness",
            (EventType.style, "对领导"): "style.leadership_steady",
            (EventType.style, "对公众"): "style.public_packaging",
            (EventType.context_mode, "领导"): "mode.leadership_safe_mode",
        }
        for (mapped_type, token), key in mapping.items():
            if event_type == mapped_type and token in content:
                return key
        slug = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", content).strip("_").lower()
        return f"{event_type.value}.{slug[:40] or 'statement'}"

    def _hard_refusal_reason(self, content: str) -> str | None:
        if re.search(r"(更讨喜|更圆滑|更好听).*(核心观点|真实观点|本来观点)", content):
            return "请求为了讨喜牺牲核心观点。"
        if re.search(r"(假装确定|装作确定|瞎猜|编一个确定答案)", content):
            return "请求伪造置信度。"
        if re.search(r"(临时情绪|一时情绪).*(写进|升级成).*(人格|长期)", content):
            return "请求把临时情绪写成长期人格。"
        return None

    def _existing_conflict_with_candidate(
        self, existing: MemoryRecord, candidate: MemoryCandidate
    ) -> bool:
        proposed = candidate.statement
        if candidate.memory_key == "principle.anti_likeability_optimization":
            return bool(re.search(r"(可以|允许).*(更讨喜|更圆滑|改掉核心观点)", proposed))
        if candidate.memory_key == "principle.uncertainty_pause":
            return bool(re.search(r"(不用确认|不需要确认|直接拍板|瞎猜也行)", proposed))
        if candidate.memory_key == "principle.expose_conflicts":
            return bool(re.search(r"(不要指出矛盾|顺着说就好)", proposed))
        if candidate.memory_key == "principle.preserve_core_view":
            return bool(re.search(r"(可以|允许).*(丢掉|改掉).*(核心观点)", proposed))
        if candidate.layer in {IdentityLayer.style, IdentityLayer.identity}:
            return False
        return existing.statement != proposed

    def _clarification_questions(self, intent: IntentFrame) -> list[str]:
        questions = [
            "对象是谁？",
            "你想达成什么结果？",
            "绝对不能越过的边界是什么？",
        ]
        if intent.external_effect:
            questions.append("这是只要草稿，还是要真的替你发出去？")
        return questions[:4]

    def _proposed_actions(self, intent: IntentFrame) -> list[ActionProposal]:
        if intent.requested_action == "draft_reply":
            return [
                ActionProposal(
                    kind="draft_reply",
                    description="先起草回复，不直接发送。",
                    capability=None,
                )
            ]
        if intent.requested_action == "judge":
            return [
                ActionProposal(
                    kind="decision_note",
                    description="先输出保守判断和冲突点，不直接执行。",
                    capability=None,
                )
            ]
        return []
