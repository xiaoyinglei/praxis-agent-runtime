from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rag.schema.runtime import AccessPolicy


@dataclass(frozen=True, slots=True)
class AuthorizationContext:
    user_id: str | None
    access_policy: AccessPolicy
    source_scope: tuple[str, ...]
    allowed_doc_ids: tuple[str, ...] = ()
    resolved_user_view: bool = False


@dataclass(slots=True)
class AuthorizationService:
    resolver: object | None = None

    def resolve_query(
        self,
        *,
        user_id: str | None,
        access_policy: AccessPolicy,
        source_scope: Sequence[str],
    ) -> AuthorizationContext:
        if not user_id:
            return AuthorizationContext(
                user_id=None,
                access_policy=access_policy,
                source_scope=tuple(source_scope),
                allowed_doc_ids=(),
                resolved_user_view=False,
            )
        effective_access_policy = access_policy
        allowed_doc_ids, resolved_user_view = self._allowed_doc_ids_for_user(user_id)
        if allowed_doc_ids:
            if source_scope:
                narrowed_scope = tuple(item for item in source_scope if str(item) in allowed_doc_ids)
            else:
                narrowed_scope = tuple(sorted(allowed_doc_ids))
            return AuthorizationContext(
                user_id=user_id,
                access_policy=effective_access_policy,
                source_scope=narrowed_scope,
                allowed_doc_ids=tuple(sorted(allowed_doc_ids)),
                resolved_user_view=resolved_user_view,
            )
        return AuthorizationContext(
            user_id=user_id,
            access_policy=effective_access_policy,
            source_scope=tuple(source_scope),
            allowed_doc_ids=(),
            resolved_user_view=resolved_user_view,
        )

    def is_document_allowed(
        self,
        *,
        user_id: str | None,
        doc_id: str,
        access_policy: AccessPolicy,
    ) -> bool:
        context = self.resolve_query(user_id=user_id, access_policy=access_policy, source_scope=())
        if not context.allowed_doc_ids:
            return True
        return str(doc_id) in set(context.allowed_doc_ids)

    def _allowed_doc_ids_for_user(self, user_id: str) -> tuple[set[str], bool]:
        resolver = self.resolver
        if resolver is None:
            return set(), False
        resolve_user_view = getattr(resolver, "resolve_user_view", None)
        if callable(resolve_user_view):
            view = resolve_user_view(user_id)
            doc_ids = getattr(view, "allowed_doc_ids", None)
            if isinstance(doc_ids, Iterable):
                return {str(item) for item in doc_ids if str(item).strip()}, True
        allowed_doc_ids_for_user = getattr(resolver, "allowed_doc_ids_for_user", None)
        if callable(allowed_doc_ids_for_user):
            doc_ids = allowed_doc_ids_for_user(user_id)
            if isinstance(doc_ids, Iterable):
                return {str(item) for item in doc_ids if str(item).strip()}, True
        fallback = self._fallback_owner_doc_ids(user_id)
        return fallback, bool(fallback)

    def _fallback_owner_doc_ids(self, user_id: str) -> set[str]:
        resolver = self.resolver
        list_sources = getattr(resolver, "list_sources", None)
        list_documents = getattr(resolver, "list_documents", None)
        if not callable(list_sources) or not callable(list_documents):
            return set()
        owned_source_ids = {
            str(source.source_id)
            for source in list_sources()
            if str(getattr(source, "owner_id", "") or "").strip() == user_id
        }
        if not owned_source_ids:
            return set()
        return {
            str(document.doc_id)
            for document in list_documents(active_only=False)
            if str(getattr(document, "source_id", "")) in owned_source_ids
        }


__all__ = ["AuthorizationContext", "AuthorizationService"]
