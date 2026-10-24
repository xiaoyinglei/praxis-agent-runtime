#!/usr/bin/env python3
"""Project the reviewed cloud snapshot into the existing runtime YAML files.

No provider requests, credential loading, runtime discovery, or budget changes.
Local/legacy research definitions and operational defaults remain in models.yaml.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from agent_runtime.core.llm_registry import ModelRegistry
from agent_runtime.model_config_io import load_model_yaml

ROOT = Path(__file__).resolve().parents[1]


def rendered_configs(root: Path) -> dict[Path, str]:
    snapshot = load_model_yaml((root / "configs/model-catalog.yaml").read_bytes())
    research_path = root / "configs/models.yaml"
    research = load_model_yaml(research_path.read_bytes())
    public_path = root / "configs/agent-models.yaml"
    previous_public = load_model_yaml(public_path.read_bytes())
    public_models = {}
    for model_id, entry in snapshot["models"].items():
        if not entry.get("sources"):
            raise ValueError(f"{model_id}: missing official sources")
        integration = entry["integration"]
        if integration == "catalog_only":
            if not entry.get("blocker"):
                raise ValueError(f"{model_id}: missing integration limitation")
            continue
        if integration != "chat_completions":
            raise ValueError(f"{model_id}: unknown integration status {integration!r}")
        config = deepcopy(entry["config"])
        if config["provider"] != entry["provider"] or config["capability"] != "chat":
            raise ValueError(f"{model_id}: inconsistent runtime declaration")
        public_models[model_id] = config
    providers = {entry["provider"] for entry in public_models.values()}
    public_providers = {name: deepcopy(snapshot["providers"][name]) for name in sorted(providers)}
    default = snapshot["policy"]["default_model"]
    if default not in public_models:
        raise ValueError("snapshot default is not integrated")
    research["providers"].update(deepcopy(public_providers))
    # The previous generated view identifies deleted entries too. Remove only
    # managed IDs, including newly catalog-only entries; leave local/legacy IDs.
    for model_id in previous_public["models"].keys() | snapshot["models"].keys():
        research["models"].pop(model_id, None)
    research["models"].update(deepcopy(public_models))
    public = {key: deepcopy(research[key]) for key in (
        "generation", "auto_compact_token_limit", "llm_budgets", "tokenizer",
    ) if key in research}
    public = {"providers": public_providers, "models": public_models,
              "defaults": {"primary_model": default}, **public}
    header = (
        f"# Cloud snapshot: {snapshot['as_of']}; maintained in configs/model-catalog.yaml.\n"
        "# Regenerate with: uv run python scripts/sync_model_catalog.py\n"
    )
    return {
        research_path: header + "# Local and older research entries are preserved; they are outside this snapshot.\n"
        + yaml.safe_dump(research, allow_unicode=True, sort_keys=False),
        public_path: header
        + "# Generated public/wheel catalog. Do not edit cloud definitions here.\n"
        + yaml.safe_dump(public, allow_unicode=True, sort_keys=False),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated views differ; do not write")
    args = parser.parse_args()
    outputs = rendered_configs(ROOT)
    # Validate the complete candidate using the real production parser before writing.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="praxis-catalog-check-") as directory:
        for index, content in enumerate(outputs.values()):
            path = Path(directory) / f"{index}.yaml"
            path.write_text(content, encoding="utf-8")
            ModelRegistry(ModelRegistry._load_yaml_file(path))
    stale = [path for path, content in outputs.items() if path.read_text(encoding="utf-8") != content]
    if args.check:
        for path in stale:
            print(f"Out of date: {path.relative_to(ROOT)}")
        return int(bool(stale))
    for path in stale:
        path.write_text(outputs[path], encoding="utf-8")
        print(f"Updated {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
