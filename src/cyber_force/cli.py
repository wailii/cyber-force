from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn
import yaml

from .config import Settings
from .engine import CyberForceEngine
from .handshake import build_session_contract
from .kb_loader import load_knowledge_base
from .schemas import CarrierManifest, InputEnvelope

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _settings(project_root: Path | None = None) -> Settings:
    return Settings.from_env(project_root=project_root)


def _print_json(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command()
def say(
    content: str,
    source: str = "cli",
    audience: str | None = None,
    intent: str | None = None,
    carrier_id: str | None = None,
    metadata_json: str | None = None,
) -> None:
    metadata = json.loads(metadata_json) if metadata_json else {}
    engine = CyberForceEngine(_settings())
    result = engine.handle(
        InputEnvelope(
            content=content,
            source=source,
            carrier_id=carrier_id,
            audience=audience,
            intent_hint=intent,
            metadata=metadata,
        )
    )
    _print_json(result.model_dump(mode="json"))


@app.command()
def attach(manifest_path: Path, project_root: Path | None = None) -> None:
    settings = _settings(project_root)
    manifest = CarrierManifest.model_validate(
        yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    )
    kb = load_knowledge_base(settings.paths.kb_dir)
    contract = build_session_contract(manifest, kb)
    _print_json(contract.model_dump(mode="json"))


@app.command()
def show_kb(project_root: Path | None = None) -> None:
    settings = _settings(project_root)
    kb = load_knowledge_base(settings.paths.kb_dir)
    _print_json(kb.model_dump(mode="json"))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8787) -> None:
    uvicorn.run("cyber_force.server:app", host=host, port=port, reload=False)
