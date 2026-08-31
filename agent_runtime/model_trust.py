"""User-owned trust for immutable model definitions and per-Turn bindings."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from agent_runtime.core.messages import canonical_json_text
from agent_runtime.model_config_io import (
    CommitOutcomeUnknown,
    UntrustedConfigPathError,
    atomic_install_bytes,
    confirm_parent_directory_durability,
    ensure_durable_directory,
    exclusive_config_lock,
    reconcile_atomic_install_link,
    validate_user_config_path,
)
from agent_runtime.model_definition import (
    ModelExecutionDefinition,
    canonical_definition_json,
)
from agent_runtime.tools.tool import JsonValue

_TRUST_VERSION = 1
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_TRUST_FILE_LIMIT = 4096
_DEFINITION_FILE_LIMIT = 4 * 1024 * 1024
_REVISION_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SIGNATURE_PATTERN = re.compile(r"^hmac-sha256:[0-9a-f]{64}$")
_BINDING_SCHEMA_VERSION = 2
_AUTHENTICATION_SCHEMA_VERSION = 1


class TrustDomainNotInitializedError(RuntimeError):
    """The operator has not explicitly initialized model-binding trust."""


class TrustDomainValidationError(ValueError):
    """The persisted trust domain is malformed or permission-unsafe."""


class BindingAuthenticationError(ValueError):
    """A model binding association failed authentication."""


class TrustedDefinitionValidationError(ValueError):
    """An archived model definition is malformed, mutable, or mismatched."""


class TrustedDefinitionNotFoundError(FileNotFoundError):
    """A requested immutable model definition is absent from the archive."""


@dataclass(frozen=True, slots=True)
class TrustDomainStatus:
    trust_domain_id: str
    signing_key_id: str


@dataclass(frozen=True, slots=True)
class _TrustMaterial:
    status: TrustDomainStatus
    key: bytes


class ModelBindingTrustDomain:
    """Explicitly initialized HMAC trust domain stored outside the workspace."""

    def __init__(self, path: Path, *, workspace: Path, worktree: Path) -> None:
        _reject_managed_path_symlink(path, subject="trust domain")
        self.path = validate_user_config_path(
            path,
            workspace=workspace,
            worktree=worktree,
        )

    def initialize(self) -> TrustDomainStatus:
        _ensure_private_directory(
            self.path.parent,
            subject="trust domain parent",
            error_type=TrustDomainValidationError,
        )
        with exclusive_config_lock(self.path):
            _reconcile_install_if_present(self.path)
            try:
                existing = self._load()
            except TrustDomainNotInitializedError:
                existing = None
            if existing is not None:
                _confirm_visible_install(self.path, subject="trust domain")
                return existing.status

            material, payload = _new_trust_material()
            outcome = atomic_install_bytes(self.path, payload)
            if outcome == "exists":
                winner = self._load()
                _confirm_visible_install(self.path, subject="trust domain")
                return winner.status
            return material.status

    def status(self) -> TrustDomainStatus:
        return self._load().status

    def sign(self, association: Mapping[str, JsonValue]) -> str:
        material = self._load()
        digest = hmac.new(
            material.key,
            _validated_association_bytes(association, status=material.status),
            hashlib.sha256,
        ).hexdigest()
        return f"hmac-sha256:{digest}"

    def verify(self, association: Mapping[str, JsonValue], signature: str) -> None:
        material = self._load()
        expected = hmac.new(
            material.key,
            _validated_association_bytes(association, status=material.status),
            hashlib.sha256,
        ).hexdigest()
        provided = signature.removeprefix("hmac-sha256:") if isinstance(signature, str) else ""
        if type(signature) is not str or _SIGNATURE_PATTERN.fullmatch(
            signature
        ) is None or not hmac.compare_digest(
            expected,
            provided,
        ):
            raise BindingAuthenticationError("model binding authentication failed")

    def _load(self) -> _TrustMaterial:
        try:
            _reconcile_install_if_needed(self.path)
            _validate_private_directory(
                self.path.parent,
                subject="trust domain parent",
                error_type=TrustDomainValidationError,
            )
            payload = _read_private_file(
                self.path,
                size_limit=_TRUST_FILE_LIMIT,
                error_type=TrustDomainValidationError,
                subject="trust domain",
            )
        except FileNotFoundError as error:
            raise TrustDomainNotInitializedError(
                "Model binding trust is not initialized; run `agent model trust init`"
            ) from error
        return _parse_trust_material(payload)


class TrustedModelDefinitionArchive:
    """Immutable, content-addressed definitions trusted for Turn resume."""

    def __init__(self, path: Path, *, workspace: Path, worktree: Path) -> None:
        _reject_managed_path_symlink(path, subject="model definition archive")
        self.path = validate_user_config_path(
            path,
            workspace=workspace,
            worktree=worktree,
        )

    def ensure(self, definition: ModelExecutionDefinition) -> str:
        normalized = _normalize_definition(definition)
        revision = normalized.definition_revision
        payload = canonical_definition_json(normalized)
        _ensure_private_directory(
            self.path,
            subject="model definition archive",
            error_type=TrustedDefinitionValidationError,
        )
        target = self._target(revision)
        with exclusive_config_lock(target):
            _reconcile_install_if_present(target)
            try:
                existing = self._load_target(target, revision=revision)
            except TrustedDefinitionNotFoundError:
                existing = None
            if existing is not None:
                if canonical_definition_json(existing) != payload:
                    raise TrustedDefinitionValidationError(
                        "archived model definition conflicts with canonical contents"
                    )
                _confirm_visible_install(target, subject="archived model definition")
                return revision

            outcome = atomic_install_bytes(target, payload)
            if outcome == "exists":
                winner = self._load_target(target, revision=revision)
                if canonical_definition_json(winner) != payload:
                    raise TrustedDefinitionValidationError(
                        "archived model definition conflicts with canonical contents"
                    )
                _confirm_visible_install(target, subject="archived model definition")
            return revision

    def load(self, definition_revision: str) -> ModelExecutionDefinition:
        target = self._target(definition_revision)
        try:
            _validate_private_directory(
                self.path,
                subject="model definition archive",
                error_type=TrustedDefinitionValidationError,
            )
        except FileNotFoundError as error:
            raise TrustedDefinitionNotFoundError(
                f"trusted model definition {definition_revision!r} is not installed"
            ) from error
        _reconcile_install_if_needed(target)
        return self._load_target(target, revision=definition_revision)

    def _target(self, revision: str) -> Path:
        if not isinstance(revision, str) or _REVISION_PATTERN.fullmatch(revision) is None:
            raise TrustedDefinitionValidationError("model definition revision is malformed")
        return self.path / f"{revision}.json"

    def _load_target(self, target: Path, *, revision: str) -> ModelExecutionDefinition:
        try:
            payload = _read_private_file(
                target,
                size_limit=_DEFINITION_FILE_LIMIT,
                error_type=TrustedDefinitionValidationError,
                subject="archived model definition",
            )
        except FileNotFoundError as error:
            raise TrustedDefinitionNotFoundError(
                f"trusted model definition {revision!r} is not installed"
            ) from error
        definition = _parse_archived_definition(payload)
        if definition.definition_revision != revision:
            raise TrustedDefinitionValidationError(
                "archived model definition digest does not match its filename"
            )
        return definition


def _new_trust_material() -> tuple[_TrustMaterial, bytes]:
    key = secrets.token_bytes(32)
    status = TrustDomainStatus(
        trust_domain_id=str(uuid.uuid4()),
        signing_key_id=_key_id(key),
    )
    document = {
        "version": _TRUST_VERSION,
        "trust_domain_id": status.trust_domain_id,
        "signing_key_id": status.signing_key_id,
        "hmac_key_base64": base64.b64encode(key).decode("ascii"),
    }
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _TrustMaterial(status=status, key=key), payload


def _parse_trust_material(payload: bytes) -> _TrustMaterial:
    document = _strict_json_object(
        payload,
        error_type=TrustDomainValidationError,
        subject="trust domain",
    )
    expected_fields = {
        "version",
        "trust_domain_id",
        "signing_key_id",
        "hmac_key_base64",
    }
    if set(document) != expected_fields:
        raise TrustDomainValidationError("trust domain has unexpected or missing fields")
    if type(document["version"]) is not int or document["version"] != _TRUST_VERSION:
        raise TrustDomainValidationError("trust domain version must be 1")
    trust_domain_id = document["trust_domain_id"]
    signing_key_id = document["signing_key_id"]
    encoded_key = document["hmac_key_base64"]
    if not isinstance(trust_domain_id, str) or not isinstance(signing_key_id, str):
        raise TrustDomainValidationError("trust domain identifiers must be strings")
    try:
        parsed_id = uuid.UUID(trust_domain_id)
    except (ValueError, AttributeError) as error:
        raise TrustDomainValidationError("trust domain ID must be a canonical UUID") from error
    if str(parsed_id) != trust_domain_id:
        raise TrustDomainValidationError("trust domain ID must be a canonical UUID")
    if not isinstance(encoded_key, str):
        raise TrustDomainValidationError("trust domain key must be base64 text")
    try:
        key = base64.b64decode(encoded_key, validate=True)
    except (ValueError, binascii.Error) as error:
        raise TrustDomainValidationError("trust domain key is not valid base64") from error
    if len(key) != 32 or base64.b64encode(key).decode("ascii") != encoded_key:
        raise TrustDomainValidationError("trust domain key must be exactly 256 bits")
    if signing_key_id != _key_id(key):
        raise TrustDomainValidationError("trust domain signing key ID does not match key material")
    return _TrustMaterial(
        status=TrustDomainStatus(
            trust_domain_id=trust_domain_id,
            signing_key_id=signing_key_id,
        ),
        key=key,
    )


def _parse_archived_definition(payload: bytes) -> ModelExecutionDefinition:
    document = _strict_json_object(
        payload,
        error_type=TrustedDefinitionValidationError,
        subject="archived model definition",
    )
    try:
        definition = ModelExecutionDefinition.model_validate(document)
    except ValidationError as error:
        raise TrustedDefinitionValidationError(
            "archived model definition does not match the strict schema"
        ) from error
    if canonical_definition_json(definition) != payload:
        raise TrustedDefinitionValidationError(
            "archived model definition bytes are not canonical"
        )
    return definition


def _normalize_definition(definition: ModelExecutionDefinition) -> ModelExecutionDefinition:
    if not isinstance(definition, ModelExecutionDefinition):
        raise TypeError("definition must be a ModelExecutionDefinition")
    try:
        return ModelExecutionDefinition.model_validate(
            definition.model_dump(mode="python", exclude_none=False)
        )
    except ValidationError as error:
        raise TrustedDefinitionValidationError("model definition is invalid") from error


def build_model_binding_envelope(
    *,
    alias: str,
    origin: str,
    definition: ModelExecutionDefinition,
    policy_revision: str,
) -> dict[str, JsonValue]:
    """Build the exact digest-bearing envelope authenticated for one Turn."""

    normalized = _normalize_definition(definition)
    envelope: dict[str, JsonValue] = {
        "schema_version": _BINDING_SCHEMA_VERSION,
        "alias": alias,
        "origin": origin,
        "definition_revision": normalized.definition_revision,
        "definition": cast(
            JsonValue,
            normalized.model_dump(mode="json", by_alias=True, exclude_none=False),
        ),
        "policy_revision": policy_revision,
    }
    envelope["binding_digest"] = _json_digest(envelope)
    _validate_binding_envelope(envelope)
    return envelope


def build_model_binding_association(
    *,
    status: TrustDomainStatus,
    thread_id: str,
    turn_id: str,
    selection_requester: str,
    binding: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Build the exact identity association accepted by the trust domain."""

    association: dict[str, JsonValue] = {
        "authentication_schema_version": _AUTHENTICATION_SCHEMA_VERSION,
        "trust_domain_id": status.trust_domain_id,
        "signing_key_id": status.signing_key_id,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "selection_requester": selection_requester,
        "binding": dict(binding),
    }
    _validated_association_bytes(association, status=status)
    return association


def _strict_json_object(
    payload: bytes,
    *,
    error_type: type[ValueError],
    subject: str,
) -> dict[str, object]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        document: dict[str, object] = {}
        for key, value in pairs:
            if key in document:
                raise error_type(f"{subject} contains duplicate key {key!r}")
            document[key] = value
        return document

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise error_type(f"{subject} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise error_type(f"{subject} must be a JSON object")
    return cast(dict[str, object], value)


def _validated_association_bytes(
    association: Mapping[str, JsonValue],
    *,
    status: TrustDomainStatus,
) -> bytes:
    if not isinstance(association, Mapping):
        raise BindingAuthenticationError("model binding association must be a mapping")
    document = dict(association)
    required = {
        "authentication_schema_version",
        "trust_domain_id",
        "signing_key_id",
        "thread_id",
        "turn_id",
        "selection_requester",
        "binding",
    }
    if set(document) != required:
        raise BindingAuthenticationError(
            "model binding association has unexpected or missing fields"
        )
    if (
        type(document["authentication_schema_version"]) is not int
        or document["authentication_schema_version"] != _AUTHENTICATION_SCHEMA_VERSION
    ):
        raise BindingAuthenticationError("model binding authentication schema is unsupported")
    trust_domain_id = document["trust_domain_id"]
    signing_key_id = document["signing_key_id"]
    if (
        type(trust_domain_id) is not str
        or type(signing_key_id) is not str
        or trust_domain_id != status.trust_domain_id
        or signing_key_id != status.signing_key_id
    ):
        raise BindingAuthenticationError("model binding trust identity does not match")
    for field_name in ("thread_id", "turn_id"):
        value = document[field_name]
        if type(value) is not str or not value or value != value.strip():
            raise BindingAuthenticationError(
                f"model binding {field_name} must be a non-empty trimmed string"
            )
    requester = document["selection_requester"]
    if type(requester) is not str or requester not in ("user", "agent", "system"):
        raise BindingAuthenticationError("model binding selection requester is invalid")
    binding = document["binding"]
    if not isinstance(binding, Mapping):
        raise BindingAuthenticationError("model binding envelope must be a mapping")
    _validate_binding_envelope(binding)
    try:
        return canonical_json_text(cast(JsonValue, document)).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BindingAuthenticationError(
            "model binding association must contain canonical JSON values"
        ) from error


def _validate_binding_envelope(binding: Mapping[str, object]) -> None:
    document = dict(binding)
    required = {
        "schema_version",
        "alias",
        "origin",
        "definition_revision",
        "definition",
        "policy_revision",
        "binding_digest",
    }
    if set(document) != required:
        raise BindingAuthenticationError("model binding envelope has unexpected or missing fields")
    if type(document["schema_version"]) is not int or document[
        "schema_version"
    ] != _BINDING_SCHEMA_VERSION:
        raise BindingAuthenticationError("model binding schema is unsupported")
    alias = document["alias"]
    if type(alias) is not str or not alias or alias != alias.strip():
        raise BindingAuthenticationError("model binding alias is invalid")
    origin = document["origin"]
    if type(origin) is not str or origin not in ("builtin", "user", "override"):
        raise BindingAuthenticationError("model binding origin is invalid")
    revision = document["definition_revision"]
    if type(revision) is not str or _REVISION_PATTERN.fullmatch(revision) is None:
        raise BindingAuthenticationError("model binding definition revision is malformed")
    policy_revision = document["policy_revision"]
    if (
        type(policy_revision) is not str
        or not policy_revision
        or policy_revision != policy_revision.strip()
    ):
        raise BindingAuthenticationError("model binding policy revision is invalid")
    raw_definition = document["definition"]
    if not isinstance(raw_definition, Mapping):
        raise BindingAuthenticationError("model binding definition must be a mapping")
    try:
        definition = ModelExecutionDefinition.model_validate(dict(raw_definition))
    except ValidationError as error:
        raise BindingAuthenticationError("model binding definition is invalid") from error
    if definition.definition_revision != revision:
        raise BindingAuthenticationError("model binding definition digest does not match")
    binding_digest = document["binding_digest"]
    if type(binding_digest) is not str or _REVISION_PATTERN.fullmatch(binding_digest) is None:
        raise BindingAuthenticationError("model binding envelope digest is malformed")
    unsigned = {key: value for key, value in document.items() if key != "binding_digest"}
    if not hmac.compare_digest(binding_digest, _json_digest(unsigned)):
        raise BindingAuthenticationError("model binding envelope digest does not match")


def _json_digest(value: Mapping[str, object]) -> str:
    try:
        payload = canonical_json_text(cast(JsonValue, dict(value))).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BindingAuthenticationError("model binding contains non-JSON values") from error
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _key_id(key: bytes) -> str:
    return f"sha256:{hashlib.sha256(key).hexdigest()}"


def _reject_managed_path_symlink(path: Path, *, subject: str) -> None:
    if not path.is_absolute():
        return
    try:
        target_stat = path.lstat()
    except FileNotFoundError:
        target_stat = None
    if target_stat is not None and stat.S_ISLNK(target_stat.st_mode):
        raise UntrustedConfigPathError(f"{subject} path must not be a symlink")
    try:
        parent_stat = path.parent.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(parent_stat.st_mode):
        raise UntrustedConfigPathError(f"{subject} parent must not be a symlink")


def _ensure_private_directory(
    path: Path,
    *,
    subject: str,
    error_type: type[ValueError],
) -> None:
    try:
        ensure_durable_directory(path, mode=_PRIVATE_DIRECTORY_MODE)
    except CommitOutcomeUnknown:
        raise
    except OSError as error:
        raise error_type(f"{subject} is not a private directory") from error
    _validate_private_directory(path, subject=subject, error_type=error_type)


def _validate_private_directory(
    path: Path,
    *,
    subject: str,
    error_type: type[ValueError],
) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
        raise error_type(f"{subject} must be a non-symlink directory")
    if path_stat.st_uid != os.geteuid():
        raise error_type(f"{subject} must be owned by the current user")
    if stat.S_IMODE(path_stat.st_mode) != _PRIVATE_DIRECTORY_MODE:
        raise error_type(f"{subject} mode must be 0700")


def _read_private_file(
    path: Path,
    *,
    size_limit: int,
    error_type: type[ValueError],
    subject: str,
) -> bytes:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise error_type(f"{subject} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise error_type(f"{subject} could not be opened safely") from error
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise error_type(f"{subject} changed while it was opened")
        if not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.geteuid():
            raise error_type(f"{subject} must be a regular file owned by the current user")
        if stat.S_IMODE(opened.st_mode) != _PRIVATE_FILE_MODE:
            raise error_type(f"{subject} mode must be 0600")
        if opened.st_nlink != 1:
            raise error_type(f"{subject} must not be hard-linked")
        if opened.st_size > size_limit:
            raise error_type(f"{subject} exceeds its size limit")
        chunks: list[bytes] = []
        remaining = size_limit + 1
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > size_limit:
            raise error_type(f"{subject} exceeds its size limit")
        return payload
    finally:
        os.close(fd)


def _confirm_visible_install(path: Path, *, subject: str) -> None:
    try:
        confirm_parent_directory_durability(path)
    except OSError as error:
        raise CommitOutcomeUnknown(f"{subject} durability is still unconfirmed") from error


def _reconcile_install_if_present(path: Path) -> None:
    try:
        reconcile_atomic_install_link(path)
    except FileNotFoundError:
        return


def _reconcile_install_if_needed(path: Path) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    if observed.st_nlink == 1:
        return
    with exclusive_config_lock(path):
        reconcile_atomic_install_link(path)


__all__ = [
    "BindingAuthenticationError",
    "ModelBindingTrustDomain",
    "TrustDomainNotInitializedError",
    "TrustDomainStatus",
    "TrustDomainValidationError",
    "TrustedDefinitionNotFoundError",
    "TrustedDefinitionValidationError",
    "TrustedModelDefinitionArchive",
    "build_model_binding_association",
    "build_model_binding_envelope",
]
