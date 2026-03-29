from pathlib import Path

import yaml

from cyber_force.handshake import build_session_contract
from cyber_force.kb_loader import load_knowledge_base
from cyber_force.schemas import CarrierCapability, CarrierManifest, PermissionLevel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_carrier_manifest_example_is_valid() -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "protocol" / "carrier-manifest.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest = CarrierManifest.model_validate(payload)
    assert CarrierCapability.text_input in manifest.capabilities


def test_handshake_respects_declared_permissions() -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "protocol" / "carrier-manifest.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest = CarrierManifest.model_validate(payload)
    kb = load_knowledge_base(PROJECT_ROOT / "kb")
    contract = build_session_contract(manifest, kb)

    assert contract.permissions["text_input"] == PermissionLevel.act
    assert CarrierCapability.local_files in contract.denied_capabilities
