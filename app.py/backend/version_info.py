"""Central definitions for application version and third-party notices."""
from __future__ import annotations

from typing import Dict, List

APP_NAME = "DeepFunGen"
APP_VERSION = "1.2.0"

THIRD_PARTY_LIBRARIES: List[Dict[str, str]] = [
    {
        "name": "uv",
        "url": "https://github.com/astral-sh/uv",
        "license": "Apache License 2.0",
        "notes": "Used for packaging and virtual environment management.",
    },
]


def get_version_info() -> Dict[str, object]:
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "third_party": [dict(item) for item in THIRD_PARTY_LIBRARIES],
    }


__all__ = ["APP_NAME", "APP_VERSION", "THIRD_PARTY_LIBRARIES", "get_version_info"]
