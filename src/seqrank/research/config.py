"""Configuration inheritance, explicit overrides, artifact fingerprints."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


def merge(base: dict, update: dict) -> dict:
    result = dict(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: str | Path, overrides: list[str] | None = None, _seen=None) -> dict:
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f'Configuration inheritance cycle: {path}')
    seen.add(path)
    config = yaml.safe_load(path.read_text()) or {}
    parents = config.pop('extends', [])
    if isinstance(parents, str):
        parents = [parents]
    result = {}
    for parent in parents:
        result = merge(result, load_config(path.parent / parent, _seen=seen))
    result = merge(result, config)
    for override in overrides or []:
        key, sep, value = override.partition('=')
        if not sep:
            raise ValueError('Override syntax: section.key=value')
        target = result
        parts = key.split('.')
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(value)
    def expand(value):
        if isinstance(value, str):
            def env(match):
                name, _, default = match.group(1).partition(':-')
                if name not in os.environ and not default:
                    raise ValueError(f'Missing environment variable: {name}')
                return os.environ.get(name, default)
            return re.sub(r'\$\{([^}]+)\}', env, value)
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return value
    return expand(result)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def empty_output(path: str | Path) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path
