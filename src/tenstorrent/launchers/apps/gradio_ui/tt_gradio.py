# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""A Gradio UI for every model being served, with a tab per task.

Run by `tt launch gradio` on the user's own Python, never imported by tt, so it
may only depend on Gradio and what Gradio itself installs (httpx, pandas).
Configured entirely by environment: TT_GRADIO_MODELS, a JSON list of
{id, label, base_url, modality}, and API_KEY when the servers were started with one.
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

import gradio as gr
import httpx
import pandas as pd


@dataclass(frozen=True)
class Model:
    id: str  # the id the API expects
    label: str  # what the UI calls it; unique
    base_url: str  # OpenAI-compatible base, ending in /v1
    modality: str  # a catalog model_type


HEADERS = {"Authorization": f"Bearer {os.environ['API_KEY']}"} if os.environ.get("API_KEY") else {}
# Generation can take minutes on a cold model; connecting should not.
TIMEOUT = httpx.Timeout(600.0, connect=10.0)
VIDEO_POLL_S = 3.0
_VIDEO_DONE = {"completed", "failed", "cancelled"}
HERE = Path(__file__).resolve().parent
# Waveforms in the theme's sand and teal rather than Gradio's orange.
WAVEFORM = gr.WaveformOptions(waveform_color="#9c9686", waveform_progress_color="#23a0c7")


# -- HTTP -------------------------------------------------------------------------
def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except (ValueError, AttributeError):
        return response.text[:500]


def _check(path: str, response: httpx.Response) -> httpx.Response:
    if response.is_error:
        raise gr.Error(f"{path} answered {response.status_code}: {_detail(response)}")
    return response


def _request(
    model: Model, method: str, path: str, *, check: bool = True, **kwargs
) -> httpx.Response:
    """One call to a model's server, with failures shown in the UI."""
    try:
        response = httpx.request(
            method, f"{model.base_url}{path}", headers=HEADERS, timeout=TIMEOUT, **kwargs
        )
    except httpx.HTTPError as exc:
        raise gr.Error(f"Could not reach {model.base_url}: {exc}") from exc
    return _check(path, response) if check else response


def _save(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as out:
        out.write(data)
    return out.name


def _took(started: float, model: Model, note: str = "") -> str:
    """The badge under a result: how long the model took, and which model it was."""
    seconds = time.monotonic() - started
    extra = f"<span>{escape(note)}</span>" if note else ""
    return (
        f'<div class="tt-took"><b>{seconds:.2f} s</b>'
        f"<span>{escape(model.label)}</span>{extra}</div>"
    )


# -- chat (llm, vlm) --------------------------------------------------------------
def _image_part(path: str) -> dict:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as image:
        data = base64.b64encode(image.read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def _parts(content) -> list[dict]:
    """OpenAI content parts from any shape Gradio gives a message or its files."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, (list, tuple)):
        return [part for item in content for part in _parts(item)]
    if not isinstance(content, dict):
        return []
    if "files" in content:  # a MultimodalTextbox submission
        return _parts(content.get("text") or "") + [_image_part(f) for f in content["files"]]
    if content.get("type") == "text":
        return _parts(content.get("text") or "")
    path = content.get("path") or (content.get("file") or {}).get("path")
    return [_image_part(path)] if path else []


def _message(role: str, content) -> dict:
    parts = _parts(content)
    if all(part["type"] == "text" for part in parts):
        return {"role": role, "content": "".join(part["text"] for part in parts)}
    return {"role": role, "content": parts}


def chat(model: Model, message, history: list[dict]) -> Iterator[str]:
    messages = [_message(turn["role"], turn["content"]) for turn in history]
    messages.append(_message("user", message))
    body = {"model": model.id, "messages": messages, "stream": True}
    try:
        with httpx.stream(
            "POST", f"{model.base_url}/chat/completions",
            json=body, headers=HEADERS, timeout=TIMEOUT,
        ) as response:
            if response.is_error:
                response.read()
                _check("/chat/completions", response)
            reply = ""
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                choices = json.loads(line[len("data: "):]).get("choices") or [{}]
                reply += (choices[0].get("delta") or {}).get("content") or ""
                yield reply
    except httpx.HTTPError as exc:
        raise gr.Error(f"Could not reach {model.base_url}: {exc}") from exc


def build_chat(models: list[Model], which: gr.components.Component, multimodal: bool) -> None:
    examples = [
        "Summarize the plot of Hamlet in three sentences.",
        "Write a Python function that checks whether a number is prime.",
        "Explain the difference between latency and throughput.",
    ]

    def respond(message, history: list[dict], label: str) -> Iterator[str]:
        yield from chat(_by_label(models, label), message, history)

    gr.ChatInterface(
        respond,
        multimodal=multimodal,
        additional_inputs=[which],
        # None leaves the picker alone: an example runs on the model chosen now.
        examples=None if multimodal else [[text, None] for text in examples],
        chatbot=gr.Chatbot(height=520, show_label=False, elem_classes="tt-chatbot"),
    )


# -- embedding --------------------------------------------------------------------
def _embed(model: Model, text: str) -> list[float]:
    response = _request(model, "POST", "/embeddings", json={"model": model.id, "input": text})
    return response.json()["data"][0]["embedding"]


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


def similarity(model: Model, texts: str) -> tuple[pd.DataFrame, str]:
    lines = [line.strip() for line in texts.splitlines() if line.strip()]
    if not lines:
        raise gr.Error("Enter at least one line of text.")
    started = time.monotonic()
    vectors = [_embed(model, line) for line in lines]
    columns = [str(n) for n in range(1, len(lines) + 1)]
    rows = [f"{n}. {line if len(line) <= 40 else line[:39] + '…'}" for n, line in zip(columns, lines, strict=True)]
    matrix = [[round(_cosine(a, b), 3) for b in vectors] for a in vectors]
    table = pd.DataFrame(matrix, columns=columns)
    table.insert(0, "text", rows)
    return table, _took(started, model, f"{len(vectors)} × {len(vectors[0])} dims")


def build_embedding(models: list[Model], which: gr.components.Component) -> None:
    with gr.Row(equal_height=False):
        with gr.Column(variant="panel"):
            texts = gr.Textbox(label="Texts, one per line", lines=8)
            run = gr.Button("Embed and compare", variant="primary")
            gr.Examples(
                [["The cat sat on the mat.\nA kitten rested on the rug.\nStock markets fell sharply today."]],
                texts,
            )
        with gr.Column(variant="panel"):
            table = gr.Dataframe(label="Cosine similarity", interactive=False)
            took = gr.HTML()
    run.click(lambda label, t: similarity(_by_label(models, label), t), [which, texts], [table, took])


# -- speech to text (audio) -------------------------------------------------------
def transcribe(model: Model, path: str | None) -> tuple[str, str]:
    if not path:
        raise gr.Error("Record or upload audio first.")
    started = time.monotonic()
    # One task per deployment: a translation server has no /transcriptions.
    for route in ("/audio/transcriptions", "/audio/translations"):
        with open(path, "rb") as audio:
            response = _request(
                model, "POST", route, check=False,
                files={"file": (os.path.basename(path), audio)},
                data={"model": model.id, "response_format": "json"},
            )
        if response.status_code != 404:
            break
    text = _check(route, response).json().get("text", "")
    return text.strip(), _took(started, model)


def build_audio(models: list[Model], which: gr.components.Component) -> None:
    with gr.Row(equal_height=False):
        with gr.Column(variant="panel"):
            audio = gr.Audio(
                sources=["microphone", "upload"], type="filepath", label="Audio",
                waveform_options=WAVEFORM,
            )
            run = gr.Button("Transcribe", variant="primary")
        with gr.Column(variant="panel"):
            text = gr.Textbox(label="Transcript", lines=10, buttons=["copy"])
            took = gr.HTML()
    run.click(lambda label, p: transcribe(_by_label(models, label), p), [which, audio], [text, took])


# -- text to speech ---------------------------------------------------------------
def speak(model: Model, text: str) -> tuple[str, str]:
    if not text.strip():
        raise gr.Error("Enter some text first.")
    # The media server's TTS reads `text`; OpenAI-style servers (Fish Speech) read
    # `input`. Each ignores the other's field, so send both.
    body = {"model": model.id, "input": text, "text": text, "response_format": "wav"}
    started = time.monotonic()
    audio = _request(model, "POST", "/audio/speech", json=body).content
    return _save(audio, ".wav"), _took(started, model)


def build_text_to_speech(models: list[Model], which: gr.components.Component) -> None:
    with gr.Row(equal_height=False):
        with gr.Column(variant="panel"):
            text = gr.Textbox(label="Text", lines=6)
            run = gr.Button("Speak", variant="primary")
            gr.Examples(
                [
                    ["Hello! This voice is being generated on Tenstorrent hardware."],
                    ["The quick brown fox jumps over the lazy dog, then naps in the sun."],
                ],
                text,
            )
        with gr.Column(variant="panel"):
            audio = gr.Audio(
                label="Speech", type="filepath", autoplay=True, waveform_options=WAVEFORM
            )
            took = gr.HTML()
    run.click(lambda label, t: speak(_by_label(models, label), t), [which, text], [audio, took])


# -- image generation -------------------------------------------------------------
def _decode_image(encoded: str) -> str:
    _, _, data = encoded.rpartition(",")  # tolerate a data: URL prefix
    raw = base64.b64decode(data)
    return _save(raw, ".png" if raw.startswith(b"\x89PNG") else ".jpg")


def generate_image(
    model: Model, prompt: str, negative: str, steps: int, seed: int, count: int
) -> tuple[list[str], str]:
    if not prompt.strip():
        raise gr.Error("Enter a prompt first.")
    body = {
        "prompt": prompt,
        "negative_prompt": negative or None,
        "num_inference_steps": int(steps),
        "seed": None if seed < 0 else int(seed),
        "number_of_images": int(count),
    }
    started = time.monotonic()
    images = _request(model, "POST", "/images/generations", json=body).json().get("images") or []
    return [_decode_image(image) for image in images], _took(started, model, f"{int(steps)} steps")


def build_image(models: list[Model], which: gr.components.Component) -> None:
    with gr.Row(equal_height=False):
        with gr.Column(variant="panel", scale=2):
            prompt = gr.Textbox(label="Prompt", lines=3)
            run = gr.Button("Generate", variant="primary")
            with gr.Accordion("Settings", open=False):
                negative = gr.Textbox(label="Negative prompt")
                steps = gr.Slider(4, 50, value=20, step=1, label="Steps")
                count = gr.Slider(1, 4, value=1, step=1, label="Images")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
            gr.Examples(
                [
                    ["A lighthouse on a rocky coast at dusk, oil painting"],
                    ["A tiny robot watering a bonsai tree, studio photo"],
                ],
                prompt,
            )
        with gr.Column(variant="panel", scale=3):
            gallery = gr.Gallery(label="Images", columns=2, height=520, object_fit="contain")
            took = gr.HTML()
    run.click(
        lambda label, *args: generate_image(_by_label(models, label), *args),
        [which, prompt, negative, steps, seed, count],
        [gallery, took],
    )


# -- video generation -------------------------------------------------------------
def generate_video(
    model: Model, prompt: str, negative: str, seed: int
) -> Iterator[tuple[str | None, str]]:
    if not prompt.strip():
        raise gr.Error("Enter a prompt first.")
    body = {"prompt": prompt, "negative_prompt": negative or None, "seed": None if seed < 0 else int(seed)}
    started = time.monotonic()
    response = _request(model, "POST", "/videos/generations", json=body)
    # Servers configured for synchronous video answer with the file itself.
    if response.headers.get("content-type", "").startswith("video/"):
        yield _save(response.content, ".mp4"), _took(started, model)
        return
    job = response.json()
    while job.get("status") not in _VIDEO_DONE:
        yield None, _took(started, model, f"job {job['id']}: {job.get('status')}")
        time.sleep(VIDEO_POLL_S)
        job = _request(model, "GET", f"/videos/generations/{job['id']}").json()
    if job["status"] != "completed":
        raise gr.Error(f"Video job {job['status']}: {job.get('error') or 'no detail'}")
    video = _request(model, "GET", f"/videos/generations/{job['id']}/download").content
    yield _save(video, ".mp4"), _took(started, model)


def build_video(models: list[Model], which: gr.components.Component) -> None:
    with gr.Row(equal_height=False):
        with gr.Column(variant="panel", scale=2):
            prompt = gr.Textbox(label="Prompt", lines=3)
            run = gr.Button("Generate", variant="primary")
            with gr.Accordion("Settings", open=False):
                negative = gr.Textbox(label="Negative prompt")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
            gr.Examples([["A paper boat drifting down a rainy street, cinematic"]], prompt)
        with gr.Column(variant="panel", scale=3):
            video = gr.Video(label="Video")
            took = gr.HTML()

    def run_video(label: str, *args) -> Iterator[tuple[str | None, str]]:
        yield from generate_video(_by_label(models, label), *args)

    run.click(run_video, [which, prompt, negative, seed], [video, took])


# -- image classification (cnn) ---------------------------------------------------
def _confidences(result: dict) -> dict[str, float]:
    """Label → confidence (0–1), from either prediction shape runners return."""
    first = (result.get("image_data") or [None])[0]
    if isinstance(first, list):
        return {
            p["object"]: p["confidence_level"] / 100
            for p in first
            if isinstance(p, dict) and "object" in p and "confidence_level" in p
        }
    if isinstance(first, dict) and "top1_class_label" in first:
        found = re.search(r"\d+(?:\.\d+)?", str(first.get("top1_class_probability", "")))
        value = float(found.group()) if found else 0.0
        return {first["top1_class_label"]: value / 100 if value > 1 else value}
    return {}


def classify(model: Model, path: str | None) -> tuple[dict[str, float], dict, str]:
    if not path:
        raise gr.Error("Upload an image first.")
    started = time.monotonic()
    with open(path, "rb") as image:
        response = _request(
            model, "POST", "/cnn/search-image",
            files={"file": (os.path.basename(path), image)},
            data={"top_k": "5", "min_confidence": "0"},
        )
    result = response.json()
    return _confidences(result), result, _took(started, model)


def build_cnn(models: list[Model], which: gr.components.Component) -> None:
    with gr.Row(equal_height=False):
        with gr.Column(variant="panel"):
            image = gr.Image(type="filepath", label="Image", height=360)
            run = gr.Button("Classify", variant="primary")
        with gr.Column(variant="panel"):
            labels = gr.Label(label="Predictions", num_top_classes=5)
            took = gr.HTML()
            with gr.Accordion("Raw response", open=False):
                raw = gr.JSON(show_label=False)
    run.click(lambda label, p: classify(_by_label(models, label), p), [which, image], [labels, raw, took])


# -- page -------------------------------------------------------------------------
Builder = Callable[[list[Model], gr.components.Component], None]
# Tabs appear in this order.
TASKS: dict[str, tuple[str, str, Builder]] = {
    "llm": (
        "Chat", "Chat with the model. Replies stream in as they are generated.",
        lambda models, which: build_chat(models, which, multimodal=False),
    ),
    "vlm": (
        "Vision chat", "Chat with the model, and attach images for it to look at.",
        lambda models, which: build_chat(models, which, multimodal=True),
    ),
    "audio": ("Speech to text", "Record or upload audio and get a transcript.", build_audio),
    "text_to_speech": ("Text to speech", "Type something and hear it spoken.", build_text_to_speech),
    "image": ("Image generation", "Describe an image and generate it.", build_image),
    "video": (
        "Video generation",
        "Describe a clip. It renders as a job, which can take a few minutes.",
        build_video,
    ),
    "embedding": (
        "Embeddings", "Embed a few texts and compare how similar the model finds them.",
        build_embedding,
    ),
    "cnn": ("Image classification", "Upload an image and see what the model recognizes.", build_cnn),
}


def _load_models() -> list[Model]:
    models = [
        Model(
            id=m["id"],
            label=m["label"],
            base_url=m["base_url"].rstrip("/"),
            # tt only sends types it has a UI for; anything else is most likely chat.
            modality=m["modality"] if m["modality"] in TASKS else "llm",
        )
        for m in json.loads(os.environ["TT_GRADIO_MODELS"])
    ]
    if not models:
        raise SystemExit("TT_GRADIO_MODELS lists no models.")
    return models


MODELS = _load_models()
LOGO = (HERE / "tenstorrent_logo.svg").read_text()


def _by_label(models: list[Model], label: str | None) -> Model:
    # An example run before the picker was touched may arrive without a label.
    return next((m for m in models if m.label == label), models[0])


def _where(model: Model) -> str:
    """The server, shortened to its port when it is on this machine."""
    url = urlsplit(model.base_url)
    local = url.hostname in {"127.0.0.1", "localhost", "::1"}
    return f":{url.port}" if local and url.port else url.netloc


def _online(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/models", headers=HEADERS, timeout=3.0).is_success
    except httpx.HTTPError:
        return False


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# The short code on a model's tag, as tenstorrent.com tags its cards; the colour
# comes from _CSS (.tt-tag-<modality>).
CODES = {
    "llm": "LLM", "vlm": "VLM", "audio": "STT", "text_to_speech": "TTS",
    "image": "IMG", "video": "VID", "embedding": "EMB", "cnn": "CNN",
}
# Lucide icons (ISC), as the site's theme switcher shows them.
_ICONS = {
    "system": '<rect width="20" height="14" x="2" y="3" rx="2"/><path d="M8 21h8M12 17v4"/>',
    "light": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2m-7.07-17.07 1.41 1.41m11.32 11.32 1.41 1.41M2 12h2m16 0h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>',
    "dark": '<circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 0 0 20z" fill="currentColor"/>',
}


def _icon(name: str) -> str:
    return (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_ICONS[name]}</svg>'
    )


def _tag(model: Model) -> str:
    return f'<span class="tt-tag tt-tag-{model.modality}">{CODES[model.modality]}</span>'


def header() -> str:
    """The nav bar, and a card per model; re-rendered on every page load so the
    status dots are live."""
    servers = sorted({m.base_url for m in MODELS})
    with ThreadPoolExecutor(max_workers=len(servers)) as pool:
        online = dict(zip(servers, pool.map(_online, servers), strict=True))
    cards = "".join(
        f"""
    <div class="tt-card" title="{escape(m.id)} at {escape(m.base_url)}">
      <div class="tt-card-text">
        <div class="tt-card-name">{_tag(m)}<span>{escape(m.label)}</span></div>
        <div class="tt-card-sub">{escape(TASKS[m.modality][0])} · {escape(_where(m))}{"" if online[m.base_url] else " · not answering"}</div>
      </div>
      <span class="tt-dot{"" if online[m.base_url] else " tt-down"}" title="{"online" if online[m.base_url] else "not answering"}"></span>
    </div>"""
        for m in MODELS
    )
    up = sum(online.values())
    status = "all online" if up == len(servers) else f"{up} of {len(servers)} online"
    themes = "".join(
        f'<button type="button" data-tt-theme="{mode}" aria-label="Switch color theme to {label}" '
        f'title="{label.capitalize()}">{_icon(mode)}</button>'
        for mode, label in (("system", "system preference"), ("light", "light mode"), ("dark", "dark mode"))
    )
    return f"""
<nav class="tt-nav">
  <div class="tt-nav-logo">{LOGO}</div>
  <div class="tt-nav-app">Model Playground</div>
  <div class="tt-nav-end">
    <span class="tt-nav-status"><span class="tt-dot{"" if up == len(servers) else " tt-down"}"></span>{_plural(len(MODELS), "model")} · {status}</span>
    <div class="tt-theme" role="group" aria-label="Color theme">{themes}</div>
  </div>
</nav>
<section class="tt-hero">
  <div class="tt-hero-text">
    <h1>Playground</h1>
    <p>Every model tt is serving,
ready to try on Tenstorrent hardware.
Pick a task below.</p>
  </div>
  <div class="tt-cards">{cards}
  </div>
</section>
"""


def _picker(models: list[Model]) -> gr.components.Component:
    """Which model a tab talks to: a choice when it has several, else fixed."""
    if len(models) == 1:
        return gr.State(models[0].label)
    return gr.Radio(
        [m.label for m in models], value=models[0].label, show_label=False,
        container=False, scale=0, min_width=0, elem_classes="tt-picker",
    )


def _task(modality: str, models: list[Model]) -> None:
    _, blurb, builder = TASKS[modality]
    with gr.Row(equal_height=True, elem_classes="tt-toolbar"):
        gr.HTML(f'<p class="tt-blurb">{escape(blurb)}</p>', padding=False)
        which = _picker(models)
        if len(models) == 1:
            gr.HTML(
                f'<span class="tt-chip" title="{escape(models[0].id)}">'
                f"{_tag(models[0])}{escape(models[0].label)}</span>",
                padding=False, scale=0, min_width=0,
            )
    builder(models, which)


# -- theme ------------------------------------------------------------------------
# tenstorrent.com's developer design language ("SDL"): the stone and teal scales,
# which the site redefines for dark mode, its yellow buttons, mono labels, and a
# dot grid. Its typefaces (Berkeley Mono, Unica Pro) are licensed, so they are
# used only when installed, ahead of system fonts; a web font would be fetched
# from a third party on every page load.
_MONO = ("Berkeley Mono", "JetBrains Mono", "IBM Plex Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace")
_SANS = ("Unica Pro", "ui-sans-serif", "system-ui", "sans-serif")
_YELLOW = gr.themes.Color(
    c50="#fff5e0", c100="#ffebc2", c200="#ffd88a", c300="#ffc757", c400="#ffb71b",
    c500="#d99a12", c600="#986f16", c700="#6d5217", c800="#473715", c900="#2e2614",
    c950="#211908", name="tt-yellow",
)
_TEAL = gr.themes.Color(
    c50="#e2f3f9", c100="#caecf7", c200="#a1e2f7", c300="#8ad2ea", c400="#5cbcdb",
    c500="#32abd2", c600="#1b8eb1", c700="#176782", c800="#144a5d", c900="#0c2f3b",
    c950="#071d25", name="tt-teal",
)
# Light stone up to 600; past it the dark-mode stone, which is what Gradio's
# defaults reach for (neutral 700–950) in dark mode.
_STONE = gr.themes.Color(
    c50="#f1f8f8", c100="#e5f2f1", c200="#d0e2df", c300="#c7d9d8", c400="#9eb2b0",
    c500="#678583", c600="#415e5b", c700="#344d49", c800="#172624", c900="#0f1d1b",
    c950="#071614", name="tt-stone",
)
_RADIUS = gr.themes.Size(xxs="1px", xs="2px", sm="4px", md="4px", lg="8px", xl="8px", xxl="8px", name="tt-radius")
# The dot grid: a 2px square every 16px.
_DOTS = (
    "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16'%3E"
    "%3Crect x='7' y='7' width='2' height='2' fill='{}'/%3E%3C/svg%3E\")"
)
_CSS = """
/* Gradio paints the body from --bg and --col, which it sets once for the theme
   it loaded in; the theme switcher flips .dark without a reload, so read the
   theme's own variables, which follow .dark. */
body {
  background-color: var(--body-background-fill) !important; color: var(--body-text-color) !important;
  background-image: DOTS_LIGHT;
}
body.dark { background-image: DOTS_DARK; }
gradio-app, .gradio-container, .gradio-container .main { background: transparent !important; }
.gradio-container { width: 100% !important; max-width: 1240px !important; margin: 0 auto !important; }

/* Nav: the site's near-black bar, in both modes. */
.tt-nav {
  display: flex; align-items: stretch; min-height: 60px; margin-top: 4px;
  background: #071614; border: 1px solid #263634; color: #c2d6d3;
  font-family: var(--font-mono); font-size: 12px; font-weight: 600;
  letter-spacing: 1px; text-transform: uppercase;
}
/* Gradio's .prose * hands every child the page's text colour, which in light
   mode is dark on this dark bar. */
.tt-nav * { color: inherit !important; }
.tt-nav-logo { display: flex; align-items: center; padding: 0 20px; border-right: 1px solid #263634; }
.tt-nav .tt-nav-logo { color: #f1f8f8 !important; }
.tt-nav-logo svg { height: 22px; width: auto; display: block; }
.tt-nav-logo svg polygon { fill: #5cbcdb; }
.tt-nav-app { display: flex; align-items: center; gap: 8px; padding: 0 20px; }
.tt-nav-app::before { content: ""; width: 9px; height: 9px; background: #ffc757; }
.tt-nav-end { display: flex; align-items: center; gap: 20px; margin-left: auto; padding: 0 12px 0 20px; }
.tt-nav-status { display: flex; align-items: center; gap: 10px; white-space: nowrap; }
.tt-theme { display: flex; border: 1px solid #344d49; border-radius: 6px; padding: 3px; gap: 2px; }
.tt-theme button {
  display: grid; place-items: center; width: 32px; height: 26px; border-radius: 4px;
  border: none; background: none; cursor: pointer;
}
.tt-nav .tt-theme button { color: #7a9994 !important; }
.tt-nav .tt-theme button:not([aria-pressed="true"]):hover { color: #e7f0ee !important; }
.tt-theme button svg { width: 16px; height: 16px; }
.tt-nav .tt-theme button[aria-pressed="true"] { color: #071614 !important; background: #c2d6d3; }

/* Hero: a mono display line and teal mono copy, model cards beside it. */
.tt-hero {
  display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 380px); gap: 24px 40px;
  align-items: center; padding: 36px 20px 32px;
}
.tt-hero h1 {
  margin: 0 0 22px; font-family: var(--font-mono); font-size: clamp(44px, 7vw, 72px);
  font-weight: 400; letter-spacing: -1px; line-height: 1; color: var(--body-text-color);
}
.tt-hero p {
  margin: 0; white-space: pre-line; font-family: var(--font-mono); font-size: 17px;
  line-height: 1.35; color: #176782;
}
.dark .tt-hero p { color: #6cc7e5; }
.tt-cards { display: grid; gap: 8px; }
.tt-card {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding: 12px 14px 12px 16px; border-radius: 4px; min-width: 0;
  background: #e5f2f1; border: 1px solid #9eb2b0;
}
.dark .tt-card { background: #0f1d1b; border-color: #486762; }
.tt-card-text { min-width: 0; }
.tt-card-name {
  display: flex; align-items: center; gap: 8px; min-width: 0;
  font-family: var(--font-mono); font-weight: 600; font-size: 15px; letter-spacing: 0.25px;
  color: var(--body-text-color);
}
.tt-card-name > span:last-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tt-card-sub { margin-top: 5px; font-size: 12px; letter-spacing: 0.25px; color: #293d3b; }
.dark .tt-card-sub { color: #c2d6d3; }
.tt-dot { flex: none; width: 8px; height: 8px; border-radius: 50%; background: #29a38d;
  box-shadow: 0 0 0 3px rgb(41 163 141 / 0.25); }
.tt-dot.tt-down { background: #d93c1c; box-shadow: 0 0 0 3px rgb(217 60 28 / 0.25); }

/* Task tags, after the site's lettered card tags. */
.tt-tag {
  flex: none; display: inline-grid; place-items: center; height: 20px; padding: 0 5px;
  font-family: var(--font-mono); font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
  color: #092221;
}
.dark .tt-tag { color: #e7f0ee; }
.tt-tag-llm { background: #a1e2f7; }            .dark .tt-tag-llm { background: #073a4a; }
.tt-tag-vlm { background: #9acef4; }            .dark .tt-tag-vlm { background: #1a3046; }
.tt-tag-audio { background: #a5e9dc; }          .dark .tt-tag-audio { background: #0f3d35; }
.tt-tag-text_to_speech { background: #ffd88a; } .dark .tt-tag-text_to_speech { background: #473715; }
.tt-tag-image { background: #ffade0; }          .dark .tt-tag-image { background: #441d35; }
.tt-tag-video { background: #c0b7f9; }          .dark .tt-tag-video { background: #272150; }
.tt-tag-embedding { background: #ffc88a; }      .dark .tt-tag-embedding { background: #473015; }
.tt-tag-cnn { background: #faad9e; }            .dark .tt-tag-cnn { background: #523028; }

/* Tabs and labels in the site's mono caps. */
.tt-tabs { margin-top: 4px; }
.tt-tabs button[role="tab"] {
  font-family: var(--font-mono); font-size: 12px; font-weight: 600; letter-spacing: 1px;
  text-transform: uppercase; padding: 12px 18px;
}
.tt-tabs button[role="tab"].selected { color: var(--body-text-color); }
.tt-tabs button[role="tab"].selected::after { background: #ffc757; height: 3px; }
[data-testid="block-info"], [data-testid="block-label"] {
  font-family: var(--font-mono) !important; font-size: 11px !important; font-weight: 600 !important;
  letter-spacing: 1px; text-transform: uppercase;
}

/* The strip above each task: what it does, and which model it talks to, on one
   line that wraps on narrow screens. Gradio sizes every Row child equally. */
.tt-toolbar { align-items: center !important; justify-content: space-between; margin: 6px 0 4px; }
.tt-toolbar > * { flex: 0 0 auto !important; width: auto !important; }
.tt-toolbar > :first-child { flex: 1 1 260px !important; }
.tt-blurb { margin: 0; color: var(--body-text-color-subdued); }
.tt-chip {
  display: inline-flex; align-items: center; gap: 8px; padding: 3px 10px 3px 3px; white-space: nowrap;
  border: 1px solid var(--border-color-primary); border-radius: 4px;
  font-family: var(--font-mono); font-size: 13px; color: var(--body-text-color);
}
.tt-picker .wrap { gap: 6px; }
.tt-picker label {
  border-radius: 4px !important; padding: 5px 12px !important; cursor: pointer;
  font-family: var(--font-mono); font-size: 13px; background: transparent !important;
  border: 1px solid #9eb2b0 !important;
}
.dark .tt-picker label { border-color: #486762 !important; }
.tt-picker label.selected { background: #092221 !important; border-color: #092221 !important; color: #f1f8f8 !important; }
.dark .tt-picker label.selected { background: #e7f0ee !important; border-color: #e7f0ee !important; color: #071614 !important; }
/* The pill itself shows the choice; keep the radio for keyboard and screen readers. */
.tt-picker input[type="radio"] { position: absolute; opacity: 0; width: 0; height: 0; }

/* Buttons and examples, as on the site: mono, outlined. */
button.primary, button.secondary { font-family: var(--font-mono); font-weight: 700; letter-spacing: 0.25px; }
.gallery-item { font-size: 13px; }

/* The timing badge under a result. */
.tt-took {
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px; padding: 2px 0 0;
  font-family: var(--font-mono); font-size: 11px; letter-spacing: 1px; text-transform: uppercase;
  color: var(--body-text-color-subdued);
}
.tt-took > * { padding: 2px 6px; background: #d8eae8; }
.dark .tt-took > * { background: #172624; }
.tt-took b { color: #176782; font-weight: 600; }
.dark .tt-took b { color: #6cc7e5; }

/* Gradio paints a slider's empty track with --neutral-200 in both modes. */
.dark { --neutral-200: #344d49; }

@media (max-width: 760px) {
  .tt-hero { grid-template-columns: 1fr; padding: 28px 4px 24px; }
  .tt-nav-app, .tt-nav-status { display: none; }
  .tt-nav-logo { border-right: none; padding: 0 14px; }
  .tt-nav-logo svg { height: 18px; }
  .tt-nav-end { padding: 0 8px; }
  .tt-theme button { width: 30px; }
}
""".replace("DOTS_LIGHT", _DOTS.format("%23d0e2df")).replace("DOTS_DARK", _DOTS.format("%231f2d2b"))
# The choice is kept in the URL's __theme, which Gradio itself reads on load
# (and its Settings panel writes), and in localStorage so a fresh visit starts
# where the last one left off. Switching flips Gradio's body.dark in place
# rather than reloading, so nothing on the page is lost. Gradio injects `head`
# after it has set its own theme, so this applies the choice itself too. The
# chosen button is marked with aria-pressed rather than by a selector on <html>:
# Gradio scopes custom CSS under .gradio-container, which an html rule can't be.
_THEME_JS = """
<script>
(() => {
  const KEY = "tt-playground-theme";
  const media = matchMedia("(prefers-color-scheme: dark)");
  const url = new URL(location.href);
  let mode = url.searchParams.get("__theme");
  try { mode = mode || localStorage.getItem(KEY); } catch (e) {}
  mode = ["light", "dark"].includes(mode) ? mode : "system";
  const save = () => {
    const url = new URL(location.href);
    if (mode === "system") url.searchParams.delete("__theme");
    else url.searchParams.set("__theme", mode);
    history.replaceState(history.state, "", url);
    try { localStorage.setItem(KEY, mode); } catch (e) {}
    mark();
  };
  const mark = () => document.querySelectorAll("button[data-tt-theme]").forEach(
    (button) => button.setAttribute("aria-pressed", button.dataset.ttTheme === mode)
  );
  const apply = () => document.body?.classList.toggle(
    "dark", mode === "dark" || (mode === "system" && media.matches)
  );
  save();
  apply();
  setTimeout(apply, 0);
  document.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-tt-theme]");
    if (!button) return;
    mode = button.dataset.ttTheme;
    save();
    apply();
  });
  // Gradio follows the OS while it thinks the mode is "system"; win after it.
  media.addEventListener("change", () => setTimeout(apply, 0));
  // The nav is rendered after this runs, and again on every page load.
  new MutationObserver(() => {
    if (document.querySelector("button[data-tt-theme]:not([aria-pressed])")) mark();
  }).observe(document.documentElement, { childList: true, subtree: true });
})();
</script>
"""


def tenstorrent_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=_YELLOW,
        secondary_hue=_TEAL,
        neutral_hue=_STONE,
        radius_size=_RADIUS,
        font=_SANS,
        font_mono=_MONO,
    ).set(
        body_background_fill="#f1f8f8",
        body_background_fill_dark="#071614",
        body_text_color="#092221",
        body_text_color_dark="#e7f0ee",
        body_text_color_subdued="#415e5b",
        body_text_color_subdued_dark="#7a9994",
        background_fill_primary="#f1f8f8",
        background_fill_primary_dark="#0f1d1b",
        background_fill_secondary="#e5f2f1",
        background_fill_secondary_dark="#0f1d1b",
        block_background_fill="#f1f8f8",
        block_background_fill_dark="#0f1d1b",
        block_border_color="#c7d9d8",
        block_border_color_dark="#263634",
        block_label_background_fill="#e5f2f1",
        block_label_background_fill_dark="#172624",
        block_label_text_color="#415e5b",
        block_label_text_color_dark="#7a9994",
        block_info_text_color="#415e5b",
        block_info_text_color_dark="#7a9994",
        block_title_text_color="#415e5b",
        block_title_text_color_dark="#7a9994",
        block_shadow="none",
        block_shadow_dark="none",
        panel_border_color="#c7d9d8",
        panel_border_color_dark="#263634",
        border_color_primary="#c7d9d8",
        border_color_primary_dark="#263634",
        border_color_accent="#9eb2b0",
        border_color_accent_dark="#486762",
        input_background_fill="#fbfdfd",
        input_background_fill_dark="#071614",
        input_border_width="1px",  # buttons inherit it: the site outlines both
        input_border_color="#c7d9d8",
        input_border_color_dark="#344d49",
        input_border_color_focus="#1b8eb1",
        input_border_color_focus_dark="#6cc7e5",
        input_shadow_focus="0 0 0 3px rgb(27 142 177 / 0.18)",
        input_shadow_focus_dark="0 0 0 3px rgb(108 199 229 / 0.18)",
        button_large_radius="8px",
        button_medium_radius="8px",
        button_small_radius="8px",
        button_primary_background_fill="#ffc757",
        button_primary_background_fill_dark="#ffc757",
        button_primary_background_fill_hover="#ffb71b",
        button_primary_background_fill_hover_dark="#ffb71b",
        button_primary_text_color="#092221",
        button_primary_text_color_dark="#071614",
        button_primary_border_color="#092221",
        button_primary_border_color_dark="#ffc757",
        button_secondary_background_fill="transparent",
        button_secondary_background_fill_dark="transparent",
        button_secondary_background_fill_hover="#d8eae8",
        button_secondary_background_fill_hover_dark="#172624",
        button_secondary_text_color="#092221",
        button_secondary_text_color_dark="#e7f0ee",
        button_secondary_border_color="#9eb2b0",
        button_secondary_border_color_dark="#486762",
        color_accent="#1b8eb1",
        color_accent_soft="#caecf7",
        color_accent_soft_dark="#0c2f3b",
        link_text_color="#176782",
        link_text_color_dark="#6cc7e5",
        link_text_color_hover="#144a5d",
        link_text_color_hover_dark="#8ad2ea",
        slider_color="#1b8eb1",
        slider_color_dark="#6cc7e5",
        loader_color="#1b8eb1",
        loader_color_dark="#6cc7e5",
        table_border_color="#c7d9d8",
        table_border_color_dark="#263634",
    )


def build() -> gr.Blocks:
    by_task = {
        modality: [m for m in MODELS if m.modality == modality]
        for modality in TASKS
        if any(m.modality == modality for m in MODELS)
    }
    title = TASKS[next(iter(by_task))][0] if len(by_task) == 1 else "Model Playground"
    # fill_width hands the width to _CSS, which caps it the same for every task.
    with gr.Blocks(title=f"{title} · Tenstorrent", fill_width=True) as demo:
        gr.HTML(header, padding=False)
        if len(by_task) == 1:  # a lone tab bar would only repeat the task's name
            _task(*next(iter(by_task.items())))
        else:
            with gr.Tabs(elem_classes="tt-tabs"):
                for modality, models in by_task.items():
                    label = TASKS[modality][0]
                    with gr.Tab(f"{label} · {len(models)}" if len(models) > 1 else label):
                        _task(modality, models)
    return demo


if __name__ == "__main__":
    build().launch(
        inbrowser=True,
        theme=tenstorrent_theme(),
        css=_CSS,
        head=_THEME_JS,
        favicon_path=str(HERE / "tenstorrent_mark.svg"),
    )
