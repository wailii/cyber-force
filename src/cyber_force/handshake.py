from __future__ import annotations

from .schemas import (
    CarrierCapability,
    CarrierManifest,
    KnowledgeBaseBundle,
    PermissionLevel,
    SessionContract,
)


def build_session_contract(
    manifest: CarrierManifest, kb: KnowledgeBaseBundle
) -> SessionContract:
    accepted: list[CarrierCapability] = []
    denied: list[CarrierCapability] = []
    normalized_permissions: dict[str, PermissionLevel] = {}

    for capability in manifest.capabilities:
        declared = manifest.permissions.get(capability.value, PermissionLevel.propose)
        permission = declared if isinstance(declared, PermissionLevel) else PermissionLevel(declared)
        normalized_permissions[capability.value] = permission
        if permission == PermissionLevel.none:
            denied.append(capability)
        else:
            accepted.append(capability)

    required_confirmation = list(
        dict.fromkeys(
            [
                *kb.action_policy.get("requires_confirmation", []),
                *manifest.session_defaults.get("require_confirmation_for", []),
            ]
        )
    )

    return SessionContract(
        carrier_id=manifest.carrier_id,
        accepted_capabilities=accepted,
        denied_capabilities=denied,
        permissions=normalized_permissions,
        required_confirmation_for=required_confirmation,
        execution_allowed=any(
            permission == PermissionLevel.act for permission in normalized_permissions.values()
        ),
        identity_fingerprint=manifest.identity_claim.get("owner_id"),
        policy_version=str(kb.decision_policy.get("version", "unknown")),
    )
