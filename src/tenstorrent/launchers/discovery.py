# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Ask a running server what it is serving.

Models served by tt are unauthenticated, so one GET /v1/models answers both "is
anything there" and "what model id does it expect" — for tt-inference-server and
tt-model containers alike. Which port to ask is the caller's job: `commands.launch`
reads it off the running container (docker) when neither --port nor --url pins one.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..errors import ExitCode, TTError
from .base import RunningModel

DEFAULT_PORT = 20000
DEFAULT_BASE_URL = f"http://127.0.0.1:{DEFAULT_PORT}/v1"
_TIMEOUT_S = 5.0


def base_url_for(url: str | None, port: int | None) -> str:
    """The endpoint to probe, from --url, --port, or the server's own default."""
    if url and port:
        raise TTError(
            "--url and --port cannot both be given.",
            why="They set the same thing.",
            next_step="Pass --port for a local server, --url for anything else.",
            exit_code=ExitCode.USAGE,
        )
    if url:
        return url.rstrip("/")
    if port:
        return f"http://127.0.0.1:{port}/v1"
    return DEFAULT_BASE_URL


def discover(base_url: str, *, timeout_s: float = _TIMEOUT_S) -> list[RunningModel]:
    """Every model the server at `base_url` reports, in its own order."""
    url = f"{base_url}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            doc = json.load(response)
        if not isinstance(doc, dict):
            # A foreign server returning a JSON array or scalar is "not a model
            # server", not a crash.
            raise ValueError(f"expected an object, got {type(doc).__name__}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise TTError(
            f"No OpenAI-compatible server answered at {base_url}.",
            why=f"GET {url} failed ({exc}).",
            # `tt serve` returns once the container is listed, which is minutes
            # before a large model finishes loading — so "not up yet" is the more
            # likely cause than "not started".
            next_step="If you just started it, the model is probably still loading — "
            "watch `docker logs -f <container>` and retry. Otherwise serve one "
            "(`tt serve <model>`), or pass --port/--url if it is listening elsewhere.",
            exit_code=ExitCode.ERROR,
            details={"base_url": base_url},
        ) from exc
    served = [
        RunningModel(
            served_id=str(item["id"]),
            base_url=base_url,
            max_context=item.get("max_model_len"),
        )
        for item in (doc.get("data") or [])
        if isinstance(item, dict) and item.get("id")
    ]
    if not served:
        raise TTError(
            f"The server at {base_url} reports no models.",
            why="GET /v1/models returned an empty list.",
            next_step="Check the server's own log; it may still be starting up.",
            exit_code=ExitCode.ERROR,
            details={"base_url": base_url},
        )
    return served


def probe(base_url: str, *, timeout_s: float = _TIMEOUT_S) -> list[RunningModel] | None:
    """discover() for callers that only want to know whether a model server answers.

    None when nothing OpenAI-compatible replied — refused, timed out, a foreign
    server (TT-Studio's own backend answers 404 on 8000), or an empty list — so
    `tt model ps` can mark a row "starting" instead of failing the listing."""
    try:
        return discover(base_url, timeout_s=timeout_s)
    except TTError:
        return None


# Checked in order: vLLM lists audio and embedding routes beside chat, so chat
# has to win; the media server registers only its own service's routes.
_MODALITY_ROUTES = (
    ("/v1/chat/completions", "llm"),
    ("/v1/images/generations", "image"),
    ("/v1/videos/generations", "video"),
    ("/v1/audio/speech", "text_to_speech"),
    ("/v1/audio/transcriptions", "audio"),
    ("/v1/audio/translations", "audio"),
    ("/v1/embeddings", "embedding"),
    ("/v1/cnn/search-image", "cnn"),
)


def detect_modality(base_url: str, *, timeout_s: float = _TIMEOUT_S) -> str | None:
    """The catalog model_type a server's routes imply, read from its FastAPI
    /openapi.json. None if it publishes no schema or no known route."""
    root = base_url.removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/openapi.json", timeout=timeout_s) as response:
            paths = json.load(response).get("paths") or {}
    except (urllib.error.URLError, OSError, ValueError, AttributeError):
        return None
    return next((modality for route, modality in _MODALITY_ROUTES if route in paths), None)
