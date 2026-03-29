from __future__ import annotations

from fastapi import FastAPI

from .config import Settings
from .engine import CyberForceEngine
from .handshake import build_session_contract
from .schemas import CarrierManifest, EngineResponse, InputEnvelope, MemoryZone, SessionContract


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Cyber Force", version="0.2.0")
    engine = CyberForceEngine(settings=settings)

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/kb/summary")
    def kb_summary() -> dict[str, object]:
        return {
            "principles": engine.kb.principles,
            "self_facts": engine.kb.self_facts,
            "style": engine.kb.style,
            "modes": engine.kb.modes,
            "decision_policy": engine.kb.decision_policy,
        }

    @app.get("/state/summary")
    def state_summary() -> dict[str, object]:
        return {
            "long_term": [item.model_dump(mode="json") for item in engine.store.list_zone(MemoryZone.long_term)],
            "candidates": [item.model_dump(mode="json") for item in engine.store.list_zone(MemoryZone.candidate)],
            "observations": [
                item.model_dump(mode="json") for item in engine.store.list_zone(MemoryZone.observation)
            ],
            "short_term": [item.model_dump(mode="json") for item in engine.store.list_zone(MemoryZone.short_term)],
            "conflicts": [item.model_dump(mode="json") for item in engine.store.list_conflicts()],
        }

    @app.post("/attach/handshake", response_model=SessionContract)
    def attach_handshake(manifest: CarrierManifest) -> SessionContract:
        return build_session_contract(manifest, engine.kb)

    @app.post("/ingest", response_model=EngineResponse)
    def ingest(envelope: InputEnvelope) -> EngineResponse:
        return engine.handle(envelope)

    return app


app = create_app()
