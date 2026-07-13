from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import yaml


PLACEHOLDER_EXPEDITIONS = "__EXPEDITIONS_JSON__"
PLACEHOLDER_TRANSLATIONS = "__TRANSLATIONS_JSON__"
PLACEHOLDER_GENERATED_AT = "__GENERATED_AT__"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_yaml(path: Path) -> Any:
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    root = repo_root()
    template_path = root / "assets" / "template.html"
    results_path = root / "results.yaml"
    translations_path = root / "translations.yaml"
    output_path = root / "index.html"

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    expeditions = read_yaml(results_path)
    translations = read_yaml(translations_path)
    expeditions = expeditions if isinstance(expeditions, list) else []
    translations = translations if isinstance(translations, dict) else {}

    template = template_path.read_text(encoding="utf-8")
    html = (
        template.replace(PLACEHOLDER_EXPEDITIONS, json.dumps(expeditions, ensure_ascii=False))
        .replace(PLACEHOLDER_TRANSLATIONS, json.dumps(translations, ensure_ascii=False))
        .replace(PLACEHOLDER_GENERATED_AT, dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )

    output_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
