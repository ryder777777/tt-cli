# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025-2026 Tenstorrent USA, Inc.

"""A Gradio UI for one served model, shaped by its modality.

Run by `tt launch gradio` on the user's own Python, never imported by tt, so it
may only depend on Gradio and what Gradio itself installs (httpx, pandas).
Configured entirely by environment: TT_GRADIO_BASE_URL, TT_GRADIO_MODEL,
TT_GRADIO_MODALITY, and API_KEY when the server was started with one.
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import tempfile
import time
from collections.abc import Callable, Iterator

import gradio as gr
import httpx
import pandas as pd

BASE_URL = os.environ["TT_GRADIO_BASE_URL"].rstrip("/")
MODEL = os.environ["TT_GRADIO_MODEL"]
MODALITY = os.environ.get("TT_GRADIO_MODALITY", "llm")
HEADERS = {"Authorization": f"Bearer {os.environ['API_KEY']}"} if os.environ.get("API_KEY") else {}
# Generation can take minutes on a cold model; connecting should not.
TIMEOUT = httpx.Timeout(600.0, connect=10.0)
VIDEO_POLL_S = 3.0
_VIDEO_DONE = {"completed", "failed", "cancelled"}


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


def _request(method: str, path: str, *, check: bool = True, **kwargs) -> httpx.Response:
    """One call to the model server, with failures shown in the UI."""
    try:
        response = httpx.request(
            method, f"{BASE_URL}{path}", headers=HEADERS, timeout=TIMEOUT, **kwargs
        )
    except httpx.HTTPError as exc:
        raise gr.Error(f"Could not reach {BASE_URL}: {exc}") from exc
    return _check(path, response) if check else response


def _save(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as out:
        out.write(data)
    return out.name


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


def chat(message, history: list[dict]) -> Iterator[str]:
    messages = [_message(turn["role"], turn["content"]) for turn in history]
    messages.append(_message("user", message))
    body = {"model": MODEL, "messages": messages, "stream": True}
    try:
        with httpx.stream(
            "POST", f"{BASE_URL}/chat/completions",
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
        raise gr.Error(f"Could not reach {BASE_URL}: {exc}") from exc


def build_chat(multimodal: bool) -> None:
    gr.ChatInterface(chat, multimodal=multimodal)


# -- embedding --------------------------------------------------------------------
def _embed(text: str) -> list[float]:
    response = _request("POST", "/embeddings", json={"model": MODEL, "input": text})
    return response.json()["data"][0]["embedding"]


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


def similarity(texts: str) -> tuple[pd.DataFrame, str]:
    lines = [line.strip() for line in texts.splitlines() if line.strip()]
    if not lines:
        raise gr.Error("Enter at least one line of text.")
    vectors = [_embed(line) for line in lines]
    labels = [line if len(line) <= 30 else f"{line[:29]}…" for line in lines]
    matrix = [[round(_cosine(a, b), 4) for b in vectors] for a in vectors]
    table = pd.DataFrame(matrix, columns=labels)
    table.insert(0, "text", labels, allow_duplicates=True)
    return table, f"{len(vectors)} embeddings × {len(vectors[0])} dimensions"


def build_embedding() -> None:
    texts = gr.Textbox(label="Texts, one per line", lines=6)
    run = gr.Button("Embed and compare", variant="primary")
    summary = gr.Markdown()
    table = gr.Dataframe(label="Cosine similarity")
    run.click(similarity, texts, [table, summary])


# -- speech to text (audio) -------------------------------------------------------
def transcribe(path: str | None) -> str:
    if not path:
        raise gr.Error("Record or upload audio first.")
    # One task per deployment: a translation server has no /transcriptions.
    for route in ("/audio/transcriptions", "/audio/translations"):
        with open(path, "rb") as audio:
            response = _request(
                "POST", route, check=False,
                files={"file": (os.path.basename(path), audio)},
                data={"model": MODEL, "response_format": "json"},
            )
        if response.status_code != 404:
            break
    return _check(route, response).json().get("text", "")


def build_audio() -> None:
    audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
    run = gr.Button("Transcribe", variant="primary")
    text = gr.Textbox(label="Text", lines=8)
    run.click(transcribe, audio, text)


# -- text to speech ---------------------------------------------------------------
def speak(text: str) -> str:
    if not text.strip():
        raise gr.Error("Enter some text first.")
    body = {"model": MODEL, "text": text, "response_format": "wav"}
    return _save(_request("POST", "/audio/speech", json=body).content, ".wav")


def build_text_to_speech() -> None:
    text = gr.Textbox(label="Text", lines=6)
    run = gr.Button("Speak", variant="primary")
    audio = gr.Audio(label="Speech", type="filepath")
    run.click(speak, text, audio)


# -- image generation -------------------------------------------------------------
def _decode_image(encoded: str) -> str:
    _, _, data = encoded.rpartition(",")  # tolerate a data: URL prefix
    raw = base64.b64decode(data)
    return _save(raw, ".png" if raw.startswith(b"\x89PNG") else ".jpg")


def generate_image(prompt: str, negative: str, steps: int, seed: int, count: int) -> list[str]:
    if not prompt.strip():
        raise gr.Error("Enter a prompt first.")
    body = {
        "prompt": prompt,
        "negative_prompt": negative or None,
        "num_inference_steps": int(steps),
        "seed": None if seed < 0 else int(seed),
        "number_of_images": int(count),
    }
    images = _request("POST", "/images/generations", json=body).json().get("images") or []
    return [_decode_image(image) for image in images]


def build_image() -> None:
    prompt = gr.Textbox(label="Prompt", lines=3)
    negative = gr.Textbox(label="Negative prompt")
    with gr.Row():
        steps = gr.Slider(4, 50, value=20, step=1, label="Steps")
        seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
        count = gr.Slider(1, 4, value=1, step=1, label="Images")
    run = gr.Button("Generate", variant="primary")
    gallery = gr.Gallery(label="Images")
    run.click(generate_image, [prompt, negative, steps, seed, count], gallery)


# -- video generation -------------------------------------------------------------
def generate_video(prompt: str, negative: str, seed: int) -> Iterator[tuple[str | None, str]]:
    if not prompt.strip():
        raise gr.Error("Enter a prompt first.")
    body = {"prompt": prompt, "negative_prompt": negative or None, "seed": None if seed < 0 else int(seed)}
    response = _request("POST", "/videos/generations", json=body)
    # Servers configured for synchronous video answer with the file itself.
    if response.headers.get("content-type", "").startswith("video/"):
        yield _save(response.content, ".mp4"), "Done."
        return
    job = response.json()
    started = time.monotonic()
    while job.get("status") not in _VIDEO_DONE:
        yield None, f"Job {job['id']}: {job.get('status')} ({int(time.monotonic() - started)}s)"
        time.sleep(VIDEO_POLL_S)
        job = _request("GET", f"/videos/generations/{job['id']}").json()
    if job["status"] != "completed":
        raise gr.Error(f"Video job {job['status']}: {job.get('error') or 'no detail'}")
    video = _request("GET", f"/videos/generations/{job['id']}/download").content
    yield _save(video, ".mp4"), f"Done in {int(time.monotonic() - started)}s."


def build_video() -> None:
    prompt = gr.Textbox(label="Prompt", lines=3)
    negative = gr.Textbox(label="Negative prompt")
    seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
    run = gr.Button("Generate", variant="primary")
    status = gr.Markdown()
    video = gr.Video(label="Video")
    run.click(generate_video, [prompt, negative, seed], [video, status])


# -- image classification (cnn) ---------------------------------------------------
def classify(path: str | None) -> dict:
    if not path:
        raise gr.Error("Upload an image first.")
    with open(path, "rb") as image:
        response = _request(
            "POST", "/cnn/search-image",
            files={"file": (os.path.basename(path), image)},
            data={"top_k": "5", "min_confidence": "0"},
        )
    return response.json()


def build_cnn() -> None:
    image = gr.Image(type="filepath", label="Image")
    run = gr.Button("Classify", variant="primary")
    result = gr.JSON(label="Predictions")
    run.click(classify, image, result)


BUILDERS: dict[str, Callable[[], None]] = {
    "llm": lambda: build_chat(multimodal=False),
    "vlm": lambda: build_chat(multimodal=True),
    "embedding": build_embedding,
    "audio": build_audio,
    "text_to_speech": build_text_to_speech,
    "image": build_image,
    "video": build_video,
    "cnn": build_cnn,
}


# -- theme ------------------------------------------------------------------------
# Palettes from tenstorrent.com: "sand" in light mode and the dark teal "stone"
# scale in dark mode, with the site's yellow buttons and teal links in both.
# System fonts only: the site's typefaces are licensed, and a Google Font would
# be fetched from Google on every page load.
_YELLOW = gr.themes.Color(
    c50="#fff5e0", c100="#ffebc2", c200="#ffd88a", c300="#ffc757", c400="#ffb71b",
    c500="#d99a12", c600="#986f16", c700="#6d5217", c800="#473715", c900="#2e2614",
    c950="#211908", name="tt-yellow",
)
_TEAL = gr.themes.Color(
    c50="#e3f6fd", c100="#c8edf9", c200="#a3e0f5", c300="#70c4e1", c400="#4eb3d4",
    c500="#23a0c7", c600="#157493", c700="#165469", c800="#114050", c900="#122930",
    c950="#0a161a", name="tt-teal",
)
_SAND = gr.themes.Color(
    c50="#f5f4ef", c100="#edeae3", c200="#dcd7cb", c300="#bdb7a8", c400="#9c9686",
    c500="#797263", c600="#575247", c700="#3f3c37", c800="#2e2d28", c900="#201f1e",
    c950="#171614", name="tt-sand",
)
# Gradio paints a slider's empty track with --neutral-200 in both modes.
_CSS = ".dark { --neutral-200: #344d49; }"


def tenstorrent_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=_YELLOW,
        secondary_hue=_TEAL,
        neutral_hue=_SAND,
        radius_size=gr.themes.sizes.radius_lg,
        font=("ui-sans-serif", "system-ui", "sans-serif"),
        font_mono=("ui-monospace", "SFMono-Regular", "Menlo", "monospace"),
    ).set(
        body_background_fill="#f5f4ef",
        body_background_fill_dark="#071614",
        body_text_color="#201f1e",
        body_text_color_dark="#e7f0ee",
        body_text_color_subdued="#575247",
        body_text_color_subdued_dark="#7a9994",
        background_fill_primary="#fbfbf9",
        background_fill_primary_dark="#0f1d1b",
        background_fill_secondary="#edeae3",
        background_fill_secondary_dark="#172624",
        block_background_fill="#fbfbf9",
        block_background_fill_dark="#0f1d1b",
        block_border_color="#dcd7cb",
        block_border_color_dark="#263634",
        block_shadow="none",
        block_shadow_dark="none",
        border_color_primary="#dcd7cb",
        border_color_primary_dark="#263634",
        input_background_fill="#ffffff",
        input_background_fill_dark="#172624",
        input_border_width="1px",  # buttons inherit it: the site outlines both
        input_border_color="#dcd7cb",
        input_border_color_dark="#344d49",
        input_border_color_focus="#157493",
        input_border_color_focus_dark="#4eb3d4",
        button_primary_background_fill="#ffc757",
        button_primary_background_fill_dark="#ffc757",
        button_primary_background_fill_hover="#ffb71b",
        button_primary_background_fill_hover_dark="#ffb71b",
        button_primary_text_color="#201f1e",
        button_primary_text_color_dark="#071614",
        button_primary_border_color="#201f1e",
        button_primary_border_color_dark="#ffc757",
        button_secondary_background_fill="#edeae3",
        button_secondary_background_fill_dark="#172624",
        button_secondary_background_fill_hover="#dad5fb",
        button_secondary_background_fill_hover_dark="#263634",
        button_secondary_text_color="#201f1e",
        button_secondary_text_color_dark="#e7f0ee",
        button_secondary_border_color="#201f1e",
        button_secondary_border_color_dark="#486762",
        color_accent="#157493",
        color_accent_soft="#dad5fb",
        color_accent_soft_dark="#0d2c35",
        link_text_color="#157493",
        link_text_color_dark="#4eb3d4",
        link_text_color_hover="#165469",
        link_text_color_hover_dark="#6cc7e5",
        slider_color="#157493",
        slider_color_dark="#4eb3d4",
        loader_color="#157493",
        loader_color_dark="#4eb3d4",
    )


def build() -> gr.Blocks:
    with gr.Blocks(title=f"{MODEL} · Tenstorrent") as demo:
        gr.Markdown(f"### {MODEL}\n`{MODALITY}` · {BASE_URL}")
        BUILDERS.get(MODALITY, BUILDERS["llm"])()
    return demo


if __name__ == "__main__":
    build().launch(inbrowser=True, theme=tenstorrent_theme(), css=_CSS)
