from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

PLACEHOLDER_EXPEDITIONS = "__EXPEDITIONS_JSON__"
PLACEHOLDER_TRANSLATIONS = "__TRANSLATIONS_JSON__"
PLACEHOLDER_GENERATED_AT = "__GENERATED_AT__"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def json_literal(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    serialized = serialized.replace("<", "\\u003c").replace(">", "\\u003e")
    return serialized


def read_yaml(path: Path) -> Any:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    metadata = {}
    for line in content.splitlines():
        if line.startswith("#"):
            key_value = line[1:].strip().split(":", 1)
            if len(key_value) == 2:
                key, value = key_value
                metadata[key.strip()] = value.strip()
            else:
                metadata[key_value[0].strip()] = ""
        else:
            break
    return yaml.safe_load(content), metadata


def main() -> None:
    root = repo_root()
    template_path = root / "assets" / "template.html"
    results_path = root / "results.yaml"
    translations_path = root / "translations.yaml"
    web_root = root / "web"
    output_path = web_root / "index.html"

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    expeditions, expeditions_metadata = read_yaml(results_path)
    translations, _ = read_yaml(translations_path)
    expeditions = expeditions if isinstance(expeditions, list) else []
    translations = translations if isinstance(translations, dict) else {}

    expeditions = [e for e in expeditions if not e.get("excluded", False) and not e.get("outdated", False)]

    if "hide_date" in expeditions_metadata:
        for expedition in expeditions:
            expedition.pop("date", None)

    template = template_path.read_text(encoding="utf-8")
    html = (
        template.replace(PLACEHOLDER_EXPEDITIONS, json_literal(expeditions))
        .replace(PLACEHOLDER_TRANSLATIONS, json_literal(translations))
        .replace(PLACEHOLDER_GENERATED_AT, str(int(time.time() * 1000)))
    )

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
