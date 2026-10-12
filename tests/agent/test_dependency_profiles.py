from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]


def test_default_dependency_profile_preserves_agent_data_capabilities_without_rag_backends() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    default = {Requirement(requirement).name for requirement in project["dependencies"]}
    extras = {
        extra: {Requirement(requirement).name for requirement in requirements}
        for extra, requirements in project["optional-dependencies"].items()
    }
    assert {"pandas", "openpyxl", "pymupdf", "python-docx", "scipy", "matplotlib", "tokenizers"} <= default
    rag_backends = {"docling", "flagembedding", "duckdb", "pymilvus", "neo4j", "redis", "psycopg", "boto3"}
    assert default.isdisjoint(rag_backends | {"mlx-lm", "transformers"})
    assert rag_backends | {"transformers"} <= extras["rag"]
    assert {"mlx-lm", "transformers"} <= extras["local-mlx"]


def test_reduced_server_profile_is_covered_by_default_locked_export() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    packages = {package["name"]: package for package in lock["package"]}
    pending = ["praxis-agent-runtime"]
    default_names: set[str] = set()
    while pending:
        name = pending.pop()
        if name in default_names:
            continue
        default_names.add(name)
        pending.extend(dependency["name"] for dependency in packages[name].get("dependencies", ()))
    reduced = {
        Requirement(line).name
        for line in (ROOT / "scripts/ubuntu/requirements-core.in").read_text().splitlines()
        if line and not line.startswith("#")
    }
    assert reduced <= default_names
