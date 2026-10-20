# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""Gradio adapter: tt's own Gradio app, run on a Python the user installed Gradio in.

tt ships the app (apps/gradio_ui/tt_gradio.py) but never Gradio itself: the
user installs it anywhere, and tt runs the app with the Python behind `gradio`.
One app covers every model being served, with a tab per task: chat,
embeddings, speech, images, video and image classification. Configured by
environment, so nothing persists past the run.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

from ...errors import ExitCode, TTError
from ..base import LaunchEnv, LaunchOptions, Preparation, RunningModel, free_web_port
from ..discovery import detect_modality

# In a folder of its own: run as a script, its folder joins sys.path, and this
# module would then shadow the real `gradio` package.
APP = Path(__file__).resolve().parent / "gradio_ui" / "tt_gradio.py"
# Catalog model_types the app has a UI for.
MODALITIES = frozenset(
    {"llm", "vlm", "embedding", "audio", "text_to_speech", "image", "video", "cnn"}
)
MIN_MAJOR_VERSION = 6
DEFAULT_PORT = 7860  # Gradio's own default
_VERSION_PROBE = "from importlib.metadata import version; print(version('gradio'))"


def interpreter(executable: str) -> str:
    """The Python behind a `gradio` console script, or `executable` if it is one."""
    if Path(executable).name.startswith("python"):
        return executable
    try:
        with open(executable, "rb") as script:
            shebang = script.readline().decode(errors="replace")
    except OSError:
        shebang = ""
    argv = shebang[2:].split() if shebang.startswith("#!") else []
    if len(argv) > 1 and Path(argv[0]).name == "env":
        argv = argv[1:]
    python = shutil.which(argv[0]) if argv and Path(argv[0]).name.startswith("python") else None
    if python:
        return python
    raise TTError(
        f"Cannot tell which Python runs {executable}.",
        why="Its first line does not name a Python interpreter that exists.",
        next_step="Point TT_TOOL_BIN_GRADIO at the Python that has Gradio installed, "
        "e.g. TT_TOOL_BIN_GRADIO=/path/to/venv/bin/python.",
        exit_code=ExitCode.TOOL_MISSING,
        details={"tool": "gradio", "executable": executable},
    )


def _labels(models: list[RunningModel]) -> list[str]:
    """Short names for the UI: the catalog name, else the id's last path segment
    (a server may report a local path such as /weight-cache/hf_view/s2-pro). Two
    servers of one model are told apart by port."""
    names = [
        m.entry.name if m.entry else m.served_id.rstrip("/").rsplit("/", 1)[-1] for m in models
    ]
    counts = Counter(names)
    return [
        f"{name} :{urlsplit(m.base_url).port}" if counts[name] > 1 else name
        for name, m in zip(names, models, strict=True)
    ]


class Gradio:
    id = "gradio"
    binaries = ("gradio",)
    install_hint = (
        f"Install Gradio {MIN_MAJOR_VERSION} or later so `gradio` is on PATH "
        "(`uv tool install gradio`, `pipx install gradio`, or `pip install gradio` in "
        "an active venv), or point TT_TOOL_BIN_GRADIO at a Python that has it. "
        "Then re-run."
    )
    requires_tool_calling = False
    hands_over_terminal = True  # Gradio serves in the foreground; Ctrl+C stops it
    kind = "web UI"
    modalities = MODALITIES
    multi_model = True  # one UI for every served model it handles, a tab per task

    def target(self) -> str:
        return "environment (this run only)"

    def plan(
        self,
        model: RunningModel,
        options: LaunchOptions,
        *,
        executable: str | None,
        runner,
    ) -> Preparation:
        return self.plan_all([model], options, executable=executable, runner=runner)

    def plan_all(
        self,
        models: list[RunningModel],
        options: LaunchOptions,
        *,
        executable: str | None,
        runner,
    ) -> Preparation:
        python = interpreter(executable) if executable else "python3"
        if executable:
            self._check_version(python, runner)
        probed: dict[str, str | None] = {}  # one /openapi.json read per server
        served = []
        for model, label in zip(models, _labels(models), strict=True):
            detected = model.entry.model_type if model.entry else None
            if detected is None:
                if model.base_url not in probed:
                    probed[model.base_url] = detect_modality(model.base_url)
                detected = probed[model.base_url]
            served.append(
                {
                    "id": model.served_id,
                    "label": label,
                    "base_url": model.base_url,
                    "modality": detected or "llm",
                    "detected": detected is not None,
                }
            )
        if not options.web_port_pinned:
            options = replace(options, web_port=DEFAULT_PORT)
        web_port = free_web_port(options)
        app_models = [{k: m[k] for k in ("id", "label", "base_url", "modality")} for m in served]
        env = {
            "TT_GRADIO_MODELS": json.dumps(app_models, separators=(",", ":")),
            "GRADIO_SERVER_NAME": "127.0.0.1",
            "GRADIO_SERVER_PORT": str(web_port),
            "GRADIO_ANALYTICS_ENABLED": "False",
        }
        return Preparation(
            rows={
                "models": [
                    {
                        "model": m["id"],
                        "base_url": m["base_url"],
                        "modality": m["modality"]
                        if m["detected"]
                        else f"{m['modality']} (not detected, assumed)",
                    }
                    for m in served
                ],
                "python": python,
            },
            env=env,
            steps=[[python, str(APP)]],
            url=f"http://127.0.0.1:{web_port}",
        )

    def apply(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        """Nothing on disk: the environment is applied at hand-off."""

    def handoff(self, model: RunningModel, prep: Preparation, env: LaunchEnv) -> None:
        env.runner.exec_tty(prep.steps[0], env=prep.env)

    def disconnect_plan(self, executable: str | None, runner) -> str | None:
        """Nothing persists, so there is never anything to undo."""
        return None

    def disconnect(self, env: LaunchEnv) -> None:
        pass

    def _check_version(self, python: str, runner) -> None:
        result = runner.capture([python, "-c", _VERSION_PROBE], check=False, tool=self.id)
        version = result.stdout.strip()
        major = version.split(".", 1)[0]
        if result.returncode == 0 and major.isdigit() and int(major) >= MIN_MAJOR_VERSION:
            return
        found = f"Gradio {version}" if result.returncode == 0 else "no Gradio installed"
        raise TTError(
            f"{python} has {found}.",
            why=f"tt launch gradio needs Gradio {MIN_MAJOR_VERSION} or later.",
            next_step=self.install_hint,
            exit_code=ExitCode.TOOL_MISSING,
            details={"tool": self.id, "python": python, "version": version or None},
        )
