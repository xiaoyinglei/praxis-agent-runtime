#!/usr/bin/env bash
# Verify the built distributions from fresh environments outside the checkout.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && $2 != --chromium ) ]]; then
    echo "Usage: bash scripts/agent_install_smoke.sh NEW_EVIDENCE_DIR [--chromium]" >&2
    exit 2
fi

chromium=${2:-}
if [[ -n "$chromium" && "${PLAYWRIGHT_BROWSERS_PATH:-}" != /* ]]; then
    echo "Set PLAYWRIGHT_BROWSERS_PATH to the absolute path to an installed Chromium cache." >&2
    exit 2
fi

repo="$(cd "$(dirname "$0")/.." && pwd -P)"
mkdir -- "$1"
out="$(cd "$1" && pwd -P)"
work="$(mktemp -d "${TMPDIR:-/tmp}/praxis-install.XXXXXX")"
trap 'rm -rf -- "$work"' EXIT

unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export UV_LINK_MODE=copy UV_NO_PROGRESS=1
set -x
cd "$repo"
uname -smr
uv --version
git rev-parse HEAD
git status --short
uv build --python 3.12 --out-dir "$out/dist"
shasum -a 256 "$out"/dist/*.tar.gz "$out"/dist/*.whl > "$out/SHA256SUMS"
uv export --locked --all-extras --no-dev --no-hashes --no-emit-project \
    --output-file "$out/constraints.txt" >/dev/null

verify_base() {
    "$venv/bin/agent" --help
    "$venv/bin/agent" run --help
    "$venv/bin/agent" background status --help
    "$venv/bin/agent" connections --help
    "$venv/bin/agent" model list --source
    "$python" -I - "$artifact" <<'PY'
import json
import sys
from importlib.metadata import distribution, distributions
from importlib.resources import files
from pathlib import Path
from urllib.parse import unquote, urlsplit

import agent_runtime
from agent_runtime.modeling.tokenization import TokenAccountingService, TokenizerContract
from rag.models.runtime import resolve_runtime_config

package = distribution("praxis-agent")
direct_url = json.loads(package.read_text("direct_url.json") or "{}")
assert Path(unquote(urlsplit(direct_url["url"]).path)) == Path(sys.argv[1]), direct_url
assert "archive_info" in direct_url, direct_url
assert Path(agent_runtime.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
print("Installed package", package.version, agent_runtime.__file__, direct_url)
print(package.read_text("WHEEL"), end="")

installed = {dist.metadata["Name"].lower() for dist in distributions()}
assert {"pandas", "openpyxl", "pymupdf", "python-docx", "scipy", "matplotlib"} <= installed
assert {
    "docling", "flagembedding", "torch", "transformers", "mlx-lm", "pymilvus", "playwright", "pytest"
}.isdisjoint(installed)
for module in ("pandas", "openpyxl", "fitz", "docx", "scipy", "matplotlib"):
    __import__(module)

agent_data = files("agent_runtime")
assert agent_data.joinpath("_data", "models.yaml").is_file()
assert agent_data.joinpath("_data", "model-catalog.yaml").is_file()
assert "连接服务" in agent_data.joinpath("connection_assets", "index.html").read_text(encoding="utf-8")
for asset in ("style.css", "app.js"):
    assert agent_data.joinpath("connection_assets", asset).is_file()
tokenizer = TokenAccountingService(TokenizerContract("", "deepseek-v4-official", "deepseek-v4-official"))
assert tokenizer.budget_count_source() == "tokenizers:deepseek-v4-official"
assert tokenizer.count_for_budget("安装验证") > 0

catalog = resolve_runtime_config()
print("Bundled RAG defaults", catalog.primary_model.id, catalog.embedding_model.id)
try:
    resolve_runtime_config(catalog_path="missing.yaml")
except FileNotFoundError:
    pass
else:
    raise AssertionError("An explicit missing catalog must report the missing path")
PY
    "$python" -I "$repo/scripts/agent_delivery_smoke.py" --fake-model
}

verify_extra() {
    case "$profile" in
        browser)
            "$python" -I -m playwright --version
            if [[ -n "$chromium" ]]; then
                "$python" -I - <<'PY'
from playwright.sync_api import sync_playwright

with sync_playwright() as playwright:
    with playwright.chromium.launch(headless=True) as browser:
        page = browser.new_page()
        page.set_content("<title>Praxis installed browser</title>")
        assert page.title() == "Praxis installed browser"
        print("Headless Chromium", browser.version)
PY
            fi
            ;;
        rag)
            "$venv/bin/rag" --help
            "$python" -I - <<'PY'
from importlib.metadata import version

import duckdb
from docling.document_converter import DocumentConverter
from rag import RAGRuntime

with duckdb.connect(":memory:") as connection:
    assert connection.sql("select 42").fetchone() == (42,)
print("RAG dependencies", version("duckdb"), version("docling"), RAGRuntime, DocumentConverter)
PY
            ;;
        local-mlx)
            "$python" -I -m mlx_lm.server --help
            "$python" -I - <<'PY'
from importlib.metadata import version

import mlx.core as mx

assert mx.sum(mx.arange(4, stream=mx.cpu), stream=mx.cpu).item() == 6
metal_available = mx.metal.is_available()
if metal_available:
    assert mx.sum(mx.arange(4, stream=mx.gpu), stream=mx.gpu).item() == 6
print("MLX", version("mlx"), "MLX-LM", version("mlx-lm"), "Metal available", metal_available)
PY
            ;;
    esac
}

verify() (
    profile="$1"
    artifact="$2"
    requirement="$3"
    venv="$work/$profile/venv"
    python="$venv/bin/python"
    mkdir -p "$work/$profile/workspace" "$work/$profile/home"
    cd "$work/$profile/workspace"
    uv venv --no-project --python 3.12 "$venv"
    uv pip install --python "$python" --constraint "$out/constraints.txt" "$requirement"
    uv pip check --python "$python"
    uv pip freeze --python "$python" > "$out/$profile-requirements.txt"
    "$python" -VV

    export HOME="$work/$profile/home" XDG_CONFIG_HOME="$work/$profile/home/.config"
    export PRAXIS_CONFIG_DIR="$XDG_CONFIG_HOME/praxis"
    export PRAXIS_MODEL_REGISTRY_PATH="$PRAXIS_CONFIG_DIR/models.yaml" PRAXIS_DISABLE_DOTENV=1
    export HF_HUB_OFFLINE=1

    if [[ "$profile" == wheel || "$profile" == sdist ]]; then
        verify_base
    else
        verify_extra
    fi
    echo "PASS $profile"
)

wheel=("$out"/dist/*.whl)
sdist=("$out"/dist/*.tar.gz)
[[ ${#wheel[@]} -eq 1 && ${#sdist[@]} -eq 1 ]]
verify wheel "${wheel[0]}" "${wheel[0]}"
verify sdist "${sdist[0]}" "${sdist[0]}"
verify browser "${wheel[0]}" "${wheel[0]}[browser]"
verify rag "${wheel[0]}" "${wheel[0]}[rag]"
if [[ "$(uname -sm)" == 'Darwin arm64' ]]; then
    verify local-mlx "${wheel[0]}" "${wheel[0]}[local-mlx]"
else
    echo "SKIP local-mlx (requires Apple Silicon)"
fi
