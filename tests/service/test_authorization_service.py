from __future__ import annotations

from dataclasses import dataclass

from rag.retrieval.authorization_service import AuthorizationService
from rag.schema.core import Document, Source, SourceType
from rag.schema.runtime import AccessPolicy


@dataclass
class _Resolver:
    def allowed_doc_ids_for_user(self, user_id: str):
        return {"42", "99"} if user_id == "alice" else set()

    def access_policy_for_user(self, user_id: str):
        if user_id != "alice":
            return None
        return AccessPolicy.default()


def test_authorization_service_resolves_user_scoped_doc_ids_and_policy() -> None:
    service = AuthorizationService(resolver=_Resolver())

    resolved = service.resolve_query(
        user_id="alice",
        access_policy=AccessPolicy.default(),
        source_scope=(),
    )

    assert resolved.user_id == "alice"
    assert set(resolved.allowed_doc_ids) == {"42", "99"}
    assert set(resolved.source_scope) == {"42", "99"}
    # Web search permission will be managed by AgentToolPolicy, not AccessPolicy


@dataclass
class _MetadataFallbackResolver:
    def list_sources(self):
        return [
            Source(
                source_id=7, source_type=SourceType.MARKDOWN, location="docs/a.md", content_hash="", owner_id="alice"
            ),
            Source(source_id=8, source_type=SourceType.MARKDOWN, location="docs/b.md", content_hash="", owner_id="bob"),
        ]

    def list_documents(self, source_id=None, *, active_only: bool = False):
        del source_id, active_only
        return [
            Document(source_id=7, doc_id=42, file_hash="hash-42", version_group_id=42),
            Document(source_id=8, doc_id=99, file_hash="hash-99", version_group_id=99),
        ]


def test_authorization_service_falls_back_to_source_owner_scope_when_no_explicit_view_exists() -> None:
    service = AuthorizationService(resolver=_MetadataFallbackResolver())

    resolved = service.resolve_query(
        user_id="alice",
        access_policy=AccessPolicy.default(),
        source_scope=(),
    )

    assert resolved.allowed_doc_ids == ("42",)
    assert resolved.source_scope == ("42",)
