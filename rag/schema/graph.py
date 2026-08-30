from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ==========================================
# 知识图谱定义 
# ==========================================
class GraphNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str
    node_type: str
    label: str
    metadata: dict[str, str] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    edge_id: str
    from_node_id: str
    to_node_id: str
    relation_type: str
    confidence: float
    evidence_ids: list[str]
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("evidence_ids must not be empty")
        return value

__all__ = ["GraphNode", "GraphEdge"]