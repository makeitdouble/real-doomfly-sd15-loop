import asyncio
import json
import os
import random
import textwrap
import threading
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
GENERATED_STATIC_DIR = STATIC_DIR / "generated"
GENERATED_STATIC_DIR.mkdir(parents=True, exist_ok=True)

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CONFIG = json.load(f)

_backend_override = os.environ.get("FLY_GENERATION_BACKEND", "").strip().lower()
if _backend_override in {"mock", "comfyui"}:
    CONFIG["generation_backend"] = _backend_override


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def now_ts() -> str:
    return time.strftime("%H:%M:%S")


ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN", "PRESS"]


SEMANTIC_LAYOUT = [
    ["red", "blue", "green", "black", "white", "gold"],
    ["face", "eye", "hand", "house", "city", "tree"],
    ["bird", "insect", "woman", "machine", "fire", "water"],
    ["dream", "war", "peace", "glitch", "ancient", "future"],
    ["dark", "bright", "huge", "tiny", "broken", "alive"],
    ["and", "with", "in", "space", "back", "ENTER"],
]

QWERTY_LAYOUT = [
    ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
    ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";"],
    ["z", "x", "c", "v", "b", "n", "m", ",", ".", "?"],
    ["space", "space", "-", "'", "[", "]", "ENTER", "ENTER", "delete", "delete"],
    ["space", "space", ",", ".", "?", "!", "ENTER", "ENTER", "delete", "delete"],
]


def retinal_samples_rgb(rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    x = uv[:, 0] * (w - 1)
    y = uv[:, 1] * (h - 1)
    x0 = x.astype(int)
    y0 = y.astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0

    def linear_luma(pixels: np.ndarray) -> np.ndarray:
        px = pixels.astype(np.float32) / 255.0
        px = np.where(px <= 0.04045, px / 12.92, ((px + 0.055) / 1.055) ** 2.4)
        return px @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)

    return (
        (1 - dx) * (1 - dy) * linear_luma(rgb[y0, x0])
        + dx * (1 - dy) * linear_luma(rgb[y0, x1])
        + (1 - dx) * dy * linear_luma(rgb[y1, x0])
        + dx * dy * linear_luma(rgb[y1, x1])
    ).astype(np.float32)


class AdaptiveKeyboardBCI:
    DIRECTION_GROUPS = {
        "LEFT": (("DNp20", "L", 1.0), ("DNa02", "L", 0.35)),
        "RIGHT": (("DNp20", "R", 1.0), ("DNa02", "R", 0.35)),
        "UP": (("DNpe017", "L", 1.0), ("MDN", None, 0.25)),
        "DOWN": (("DNpe017", "R", 1.0), ("DNp09", None, 0.25)),
    }

    def __init__(self, readouts: List[Dict[str, Any]], config: Dict[str, Any]):
        self.readouts = [dict(r) for r in readouts]
        self.rate_tau = float(config.get("bci_rate_tau_seconds", 0.12))
        self.baseline_tau = float(config.get("bci_baseline_tau_seconds", 1.5))
        self.nav_z_threshold = float(config.get("bci_nav_z_threshold", 0.30))
        self.nav_idle_fallback_steps = int(config.get("bci_nav_idle_fallback_steps", 4))
        self.press_cooldown_steps = int(config.get("bci_press_cooldown_steps", 4))
        self.press_sync_history = int(config.get("bci_press_sync_history", 8))
        self.rates = np.zeros(len(self.readouts), dtype=np.float64)
        self.baseline = {a: None for a in self.DIRECTION_GROUPS}
        self.variance = {a: 1.0 for a in self.DIRECTION_GROUPS}
        self.step_index = 0
        self.last_press_step = -10_000
        self.idle_steps = 0
        self.press_sync = deque(maxlen=max(3, self.press_sync_history))
        self.prev_sync = 0

    def _match(self, typ: str, side: Optional[str] = None):
        for i, r in enumerate(self.readouts):
            if r.get("type") == typ and (side is None or r.get("side") == side):
                yield i, r

    def _sum_rate(self, typ: str, side: Optional[str] = None) -> float:
        return float(sum(self.rates[i] for i, _ in self._match(typ, side)))

    def _sum_spikes(self, counts: np.ndarray, typ: str, side: Optional[str] = None) -> int:
        return int(sum(int(counts[r["index"]]) for _, r in self._match(typ, side)))

    def _group_rate(self, action: str) -> float:
        total = 0.0
        for typ, side, weight in self.DIRECTION_GROUPS[action]:
            total += weight * self._sum_rate(typ, side)
        return float(total)

    def _adaptive_z(self, action: str, value: float, seconds: float) -> float:
        mean = self.baseline[action]
        if mean is None:
            self.baseline[action] = value
            self.variance[action] = max(1.0, (0.15 * abs(value) + 1.0) ** 2)
            return 0.0
        std = max(1.0, self.variance[action] ** 0.5)
        z = (value - mean) / std
        alpha = 1.0 - np.exp(-seconds / max(1e-6, self.baseline_tau))
        delta = value - mean
        self.baseline[action] = mean + alpha * delta
        self.variance[action] = max(0.25, (1.0 - alpha) * self.variance[action] + alpha * delta * delta)
        return float(z)

    def decode(self, counts: np.ndarray, seconds: float) -> Dict[str, Any]:
        if seconds <= 0:
            raise ValueError("Positive neural interval required")
        self.step_index += 1
        alpha = 1.0 - np.exp(-seconds / max(1e-6, self.rate_tau))
        raw_rates = np.asarray([counts[r["index"]] / seconds for r in self.readouts], dtype=np.float64)
        if self.step_index == 1:
            self.rates[:] = raw_rates
        else:
            self.rates += alpha * (raw_rates - self.rates)

        group_rates = {a: self._group_rate(a) for a in self.DIRECTION_GROUPS}
        zscores = {a: self._adaptive_z(a, v, seconds) for a, v in group_rates.items()}

        mn9_spikes = self._sum_spikes(counts, "MN9")
        dnpe_l_spikes = self._sum_spikes(counts, "DNpe017", "L")
        dnpe_r_spikes = self._sum_spikes(counts, "DNpe017", "R")
        sync = min(dnpe_l_spikes, dnpe_r_spikes)
        hist = list(self.press_sync)
        median_sync = float(np.median(hist)) if hist else 0.0
        rising_sync = len(hist) >= 3 and sync > self.prev_sync and sync >= max(1.0, median_sync)
        press_ready = (self.step_index - self.last_press_step) >= self.press_cooldown_steps
        press_event = press_ready and (mn9_spikes > 0 or rising_sync)
        press_source = "MN9 spike" if (press_event and mn9_spikes > 0) else ("DNpe017 bilateral burst" if press_event else "—")
        self.press_sync.append(sync)
        self.prev_sync = sync

        action = "IDLE"
        reason = "no event above adaptive threshold"
        if press_event:
            action = "PRESS"
            self.last_press_step = self.step_index
            self.idle_steps = 0
            reason = press_source
        else:
            winner = max(zscores, key=lambda a: zscores[a])
            winner_z = zscores[winner]
            if winner_z >= self.nav_z_threshold:
                action = winner
                self.idle_steps = 0
                reason = f"adaptive population winner {winner} z={winner_z:.2f}"
            else:
                self.idle_steps += 1
                if self.idle_steps >= self.nav_idle_fallback_steps:
                    left = group_rates["LEFT"]
                    right = group_rates["RIGHT"]
                    up = group_rates["UP"]
                    down = group_rates["DOWN"]
                    h = (right - left) / (right + left + 1.0)
                    v = (down - up) / (down + up + 1.0)
                    if abs(h) >= abs(v) and abs(h) > 1e-4:
                        action = "RIGHT" if h > 0 else "LEFT"
                        reason = f"stable-activity horizontal bias {h:+.3f}"
                    elif abs(v) > 1e-4:
                        action = "DOWN" if v > 0 else "UP"
                        reason = f"stable-activity vertical bias {v:+.3f}"
                    if action != "IDLE":
                        self.idle_steps = 0

        neuron_rows = []
        for i, r in enumerate(self.readouts):
            neuron_rows.append({
                **r,
                "spikes": int(counts[r["index"]]),
                "rate_hz": round(float(self.rates[i]), 3),
            })

        return {
            "action": action,
            "reason": reason,
            "press_source": press_source,
            "mn9_spikes": mn9_spikes,
            "dnpe_sync": int(sync),
            "group_rates": {k: round(v, 3) for k, v in group_rates.items()},
            "zscores": {k: round(v, 3) for k, v in zscores.items()},
            "readouts": neuron_rows,
        }


class DoomFlyBrainAdapter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.root = Path(config["doomfly_root"])
        self.graph_path = Path(config["doomfly_graph"])
        self.manifest_path = Path(config["doomfly_manifest"])
        self.step_ms = float(config.get("brain_step_ms", 28.6))
        self.lamina_bias = float(config.get("brain_lamina_bias", 12.0))
        self._lock = threading.RLock()

        if not self.root.exists():
            raise RuntimeError(f"DOOMFLY root not found: {self.root}")
        if not self.graph_path.exists():
            raise RuntimeError(f"DOOMFLY graph not found: {self.graph_path}")
        if not self.manifest_path.exists():
            raise RuntimeError(f"DOOMFLY manifest not found: {self.manifest_path}")

        root_text = str(self.root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        from doom.engine import Brain
        self.Brain = Brain
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.current_rgb = np.full((512, 512, 3), 127, dtype=np.uint8)
        self.current_image_label = "neutral-gray bootstrap"
        self.last_metrics: Dict[str, Any] = {}
        self._create_runtime()

    def _create_runtime(self) -> None:
        self.brain = self.Brain(self.graph_path)
        self.keyboard_bci = AdaptiveKeyboardBCI(self.manifest["readouts"], self.config)
        self.last_metrics = {
            "dataset": self.manifest.get("dataset", "malecns_v1"),
            "neurons": int(self.manifest.get("neurons", self.brain.n)),
            "edges": int(self.manifest.get("edges", len(self.brain.post))),
            "retina": int(len(self.brain.retina)),
            "image": self.current_image_label,
            "action": "IDLE",
            "reason": "bootstrap",
            "press_source": "—",
            "spikes": 0,
            "active_neurons": int(self.brain.nactive[0]),
            "sim_ms": 0.0,
            "wall_ms": 0.0,
            "group_rates": {a: 0.0 for a in ("LEFT", "RIGHT", "UP", "DOWN")},
            "zscores": {a: 0.0 for a in ("LEFT", "RIGHT", "UP", "DOWN")},
            "readouts": [{**r, "spikes": 0, "rate_hz": 0.0} for r in self.manifest["readouts"]],
        }

    def reset(self) -> None:
        with self._lock:
            self.current_rgb = np.full((512, 512, 3), 127, dtype=np.uint8)
            self.current_image_label = "neutral-gray bootstrap"
            self._create_runtime()

    def observe_image(self, path: Path) -> None:
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        with self._lock:
            self.current_rgb = np.ascontiguousarray(rgb)
            self.current_image_label = Path(path).name
            self.last_metrics["image"] = self.current_image_label

    def choose_action(self, cursor_row: int, cursor_col: int, layout: List[List[str]], current_prompt_len: int) -> str:
        del cursor_row, cursor_col, layout, current_prompt_len
        with self._lock:
            light = retinal_samples_rgb(self.current_rgb, self.brain.uv)
            counts, wall_seconds = self.brain.step(
                light,
                self.step_ms,
                sugar=False,
                lamina_bias=self.lamina_bias,
            )
            seconds = self.step_ms / 1000.0
            decoded = self.keyboard_bci.decode(counts, seconds)
            action = decoded["action"]
            self.last_metrics = {
                "dataset": self.manifest.get("dataset", "malecns_v1"),
                "neurons": int(self.manifest.get("neurons", self.brain.n)),
                "edges": int(self.manifest.get("edges", len(self.brain.post))),
                "retina": int(len(self.brain.retina)),
                "image": self.current_image_label,
                "action": action,
                "reason": decoded["reason"],
                "press_source": decoded["press_source"],
                "mn9_spikes": decoded["mn9_spikes"],
                "dnpe_sync": decoded["dnpe_sync"],
                "group_rates": decoded["group_rates"],
                "zscores": decoded["zscores"],
                "readouts": decoded["readouts"],
                "spikes": int(counts.sum(dtype=np.int64)),
                "active_neurons": int(self.brain.nactive[0]),
                "sim_ms": round(float(self.brain.sim_ms), 3),
                "wall_ms": round(float(wall_seconds) * 1000.0, 3),
            }
            return action

    def summary(self) -> str:
        return (
            f"REAL DOOMFLY {self.manifest.get('dataset', 'malecns_v1')} | "
            f"{int(self.manifest.get('neurons', self.brain.n)):,} neurons | "
            f"{int(self.manifest.get('edges', len(self.brain.post))):,} edges | "
            f"{len(self.brain.retina):,} retinal inputs"
        )


class ComfyClient:
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config["comfyui_url"].rstrip("/")
        self.output_dir = Path(config["comfyui_output_dir"])
        self.checkpoint = config["checkpoint_name"]
        self.vae_name = str(config.get("vae_name", "")).strip()
        self.width = int(config["width"])
        self.height = int(config["height"])
        self.steps = int(config["steps"])
        self.cfg = float(config["cfg"])
        self.sampler_name = config["sampler_name"]
        self.scheduler = config["scheduler"]
        self.positive_prefix = config["positive_prefix"].strip()
        self.negative_prompt = config["negative_prompt"]
        self.seed_mode = config.get("seed_mode", "random")
        self.seed = int(config.get("seed", 42))

    def _build_workflow(self, typed_prompt: str) -> Dict[str, Any]:
        prompt_text = f"{self.positive_prefix} {typed_prompt}".strip()
        seed = self.seed if self.seed_mode == "fixed" else random.randint(1, 2**31 - 1)
        return {
            "3": {
                "inputs": {
                    "seed": seed,
                    "steps": self.steps,
                    "cfg": self.cfg,
                    "sampler_name": self.sampler_name,
                    "scheduler": self.scheduler,
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
                "class_type": "KSampler",
            },
            "4": {"inputs": {"ckpt_name": self.checkpoint}, "class_type": "CheckpointLoaderSimple"},
            "5": {"inputs": {"width": self.width, "height": self.height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
            "6": {"inputs": {"text": prompt_text, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
            "7": {"inputs": {"text": self.negative_prompt, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
            "8": {"inputs": {"samples": ["3", 0], "vae": ["10", 0] if self.vae_name else ["4", 2]}, "class_type": "VAEDecode"},
            "9": {"inputs": {"filename_prefix": "fly_sd15", "images": ["8", 0]}, "class_type": "SaveImage"},
            **({"10": {"inputs": {"vae_name": self.vae_name}, "class_type": "VAELoader"}} if self.vae_name else {}),
        }

    def validate(self) -> str:
        sres = requests.get(f"{self.base_url}/system_stats", timeout=5)
        sres.raise_for_status()
        info = requests.get(f"{self.base_url}/object_info/CheckpointLoaderSimple", timeout=10)
        info.raise_for_status()
        obj = info.json().get("CheckpointLoaderSimple", {})
        choices = (((obj.get("input") or {}).get("required") or {}).get("ckpt_name") or [[]])[0]
        if isinstance(choices, list) and choices and self.checkpoint not in choices:
            raise RuntimeError(f"Checkpoint '{self.checkpoint}' not found in ComfyUI. Available examples: {', '.join(map(str, choices[:8]))}")
        if self.vae_name:
            vinfo = requests.get(f"{self.base_url}/object_info/VAELoader", timeout=10)
            vinfo.raise_for_status()
            vobj = vinfo.json().get("VAELoader", {})
            vchoices = (((vobj.get("input") or {}).get("required") or {}).get("vae_name") or [[]])[0]
            if isinstance(vchoices, list) and vchoices and self.vae_name not in vchoices:
                raise RuntimeError(f"VAE '{self.vae_name}' not found in ComfyUI. Available examples: {', '.join(map(str, vchoices[:8]))}")
        return "ComfyUI API OK"

    def generate(self, typed_prompt: str) -> Path:
        workflow = self._build_workflow(typed_prompt)
        client_id = str(uuid.uuid4())
        payload = {"prompt": workflow, "client_id": client_id}
        res = requests.post(f"{self.base_url}/prompt", json=payload, timeout=60)
        res.raise_for_status()
        prompt_id = res.json()["prompt_id"]

        deadline = time.time() + 600
        image_meta = None
        history_json = None
        while time.time() < deadline:
            hres = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
            hres.raise_for_status()
            history_json = hres.json()
            if prompt_id in history_json:
                outputs = history_json[prompt_id].get("outputs", {})
                for _, node_output in outputs.items():
                    images = node_output.get("images")
                    if images:
                        image_meta = images[0]
                        break
            if image_meta is not None:
                break
            time.sleep(1.2)

        if image_meta is None:
            raise RuntimeError(f"ComfyUI timed out. history={history_json}")

        params = {
            "filename": image_meta["filename"],
            "subfolder": image_meta.get("subfolder", ""),
            "type": image_meta.get("type", "output"),
        }
        ires = requests.get(f"{self.base_url}/view", params=params, timeout=120)
        ires.raise_for_status()
        local_name = f"{int(time.time())}_{image_meta['filename']}"
        local_path = GENERATED_STATIC_DIR / local_name
        with local_path.open("wb") as f:
            f.write(ires.content)
        return local_path


class MockGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.width = int(config["width"])
        self.height = int(config["height"])

    def generate(self, typed_prompt: str) -> Path:
        bg = (15, 15, 20)
        img = Image.new("RGB", (self.width, self.height), bg)
        draw = ImageDraw.Draw(img)
        rng = random.Random(hash(typed_prompt) & 0xffffffff)
        for _ in range(100):
            x1 = rng.randint(0, self.width)
            y1 = rng.randint(0, self.height)
            x2 = rng.randint(0, self.width)
            y2 = rng.randint(0, self.height)
            color = (rng.randint(40, 255), rng.randint(40, 255), rng.randint(40, 255))
            draw.ellipse([x1, y1, x2, y2], outline=color, width=rng.randint(1, 6))
        font = ImageFont.load_default()
        wrapped = typed_prompt[:200]
        draw.rectangle([20, 20, self.width - 20, 110], fill=(0, 0, 0))
        draw.multiline_text((30, 35), wrapped, fill=(255, 255, 255), font=font, spacing=4)
        local_name = f"mock_{int(time.time())}.png"
        local_path = GENERATED_STATIC_DIR / local_name
        img.save(local_path)
        return local_path


@dataclass
class OutputRecord:
    prompt: str
    image_url: str
    local_path: str
    raw_local_path: str
    timestamp: str


@dataclass
class LoopState:
    config: Dict[str, Any]
    layout: List[List[str]]
    brain: DoomFlyBrainAdapter
    running: bool = False
    generating: bool = False
    current_tokens: List[str] = field(default_factory=list)
    cursor_row: int = 0
    cursor_col: int = 0
    last_prompt: str = ""
    last_image_url: str = ""
    last_image_local_path: str = ""
    last_raw_image_local_path: str = ""
    records: List[OutputRecord] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    last_error: str = ""
    loop_task: Optional[asyncio.Task] = None
    brain_busy: bool = False
    display_revision: int = 0

    def current_prompt_text(self) -> str:
        if self.config.get("keyboard_mode", "qwerty") == "semantic":
            out = []
            for t in self.current_tokens:
                if t == "space":
                    out.append("")
                else:
                    out.append(t)
            return " ".join([x for x in out if x != ""]).strip()
        return "".join(" " if t == "space" else t for t in self.current_tokens)

    def prompt_for_retina(self) -> str:
        cur = self.current_prompt_text()
        if cur.strip():
            return cur
        if self.last_prompt.strip():
            return self.last_prompt
        return "fly retina bootstrap"

    def push_log(self, msg: str) -> None:
        self.logs.append(f"[{now_ts()}] {msg}")
        self.logs = self.logs[-120:]


app = FastAPI(title=CONFIG.get("ui_title", "Fly SD15 Loop"))
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

layout_mode = str(CONFIG.get("keyboard_mode", "qwerty")).lower().strip()
layout = SEMANTIC_LAYOUT if layout_mode == "semantic" else QWERTY_LAYOUT
brain = DoomFlyBrainAdapter(CONFIG)
state = LoopState(config=CONFIG, layout=layout, brain=brain)
generator = ComfyClient(CONFIG) if CONFIG.get("generation_backend") == "comfyui" else MockGenerator(CONFIG)


def wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int, max_lines: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        w = draw.textbbox((0, 0), candidate, font=font)[2]
        if w <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
            if len(lines) >= max_lines - 1:
                break
    if len(lines) < max_lines:
        lines.append(current)
    remaining_words = words[len(" ".join(lines).split()):]
    if remaining_words:
        tail = lines[-1]
        while draw.textbbox((0, 0), tail + "…", font=font)[2] > max_width and len(tail) > 6:
            tail = tail[:-1]
        lines[-1] = tail.rstrip() + "…"
    return lines[:max_lines]


def load_ui_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates += [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    candidates += [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def fit_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    scale = max(target_w / max(1, src_w), target_h / max(1, src_h))
    new_size = (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale))))
    resized = img.resize(new_size, Image.Resampling.LANCZOS)
    left = max(0, (resized.width - target_w) // 2)
    top = max(0, (resized.height - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h))


def render_feedback_image(prompt_text: str, source_image_path: Optional[str], output_path: Path, width: int, height: int) -> None:
    header_h = int(height * float(CONFIG.get("retina_header_ratio", 0.23)))
    header_h = clamp(header_h, 110, 240)
    body_h = height - header_h

    canvas = Image.new("RGB", (width, height), (10, 12, 18))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, header_h), fill=(6, 8, 12))
    draw.line((0, header_h - 1, width, header_h - 1), fill=(65, 72, 94), width=1)

    title_font = load_ui_font(18, bold=True)
    text_font = load_ui_font(24)
    meta_font = load_ui_font(16)

    draw.text((20, 14), "FLY PROMPT MEMORY", fill=(105, 255, 212), font=title_font)
    draw.text((20, 38), "The fly sees its typed prompt above the generated image.", fill=(160, 168, 188), font=meta_font)

    wrapped = wrap_text_to_width(draw, prompt_text or " ", text_font, width - 40, max_lines=4)
    y = 68
    for line in wrapped:
        draw.text((20, y), line, fill=(255, 255, 255), font=text_font)
        y += 28

    if source_image_path and Path(source_image_path).exists():
        base = Image.open(source_image_path).convert("RGB")
        fitted = fit_cover(base, width, body_h)
    else:
        fitted = Image.new("RGB", (width, body_h), (115, 120, 128))
        ph = ImageDraw.Draw(fitted)
        ph.text((width // 2 - 120, body_h // 2 - 10), "No generated image yet", fill=(240, 240, 245), font=load_ui_font(20, bold=True))

    canvas.paste(fitted, (0, header_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def refresh_retina_feedback(log_reason: Optional[str] = None) -> None:
    display_path = GENERATED_STATIC_DIR / "retina_live.png"
    render_feedback_image(
        state.prompt_for_retina(),
        state.last_raw_image_local_path if state.last_raw_image_local_path else None,
        display_path,
        int(CONFIG.get("width", 768)),
        int(CONFIG.get("height", 768)),
    )
    state.display_revision += 1
    state.last_image_local_path = str(display_path)
    state.last_image_url = f"/static/generated/{display_path.name}?v={state.display_revision}"
    state.brain.observe_image(display_path)
    if log_reason:
        state.push_log(log_reason)


def snapshot_feedback_image(prompt_text: str, raw_image_path: str) -> Path:
    snapshot_path = GENERATED_STATIC_DIR / f"retina_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
    render_feedback_image(prompt_text, raw_image_path, snapshot_path, int(CONFIG.get("width", 768)), int(CONFIG.get("height", 768)))
    return snapshot_path


def apply_press_token(state: LoopState, token: str) -> Optional[str]:
    if token == "delete":
        if state.current_tokens:
            removed = state.current_tokens.pop()
            state.push_log(f"DELETE -> removed '{removed}'")
            refresh_retina_feedback()
        return None
    if token in {"ENTER", "OK"}:
        prompt_text = state.current_prompt_text().strip()
        if prompt_text:
            state.push_log(f"PROMPT SUBMIT -> {prompt_text}")
            state.current_tokens.clear()
            return prompt_text
        state.push_log("ENTER ignored: prompt empty")
        return None
    if token == "space":
        max_chars = int(CONFIG.get("max_prompt_chars", 220))
        if len(state.current_prompt_text()) >= max_chars:
            state.push_log("TYPE ignored: max prompt chars reached")
            return None
        state.current_tokens.append("space")
        state.push_log("TYPE -> [space]")
        refresh_retina_feedback()
        return None

    if state.config.get("keyboard_mode", "qwerty") == "semantic":
        max_tokens = int(CONFIG.get("max_prompt_tokens", 64))
        if len(state.current_tokens) >= max_tokens:
            state.push_log("TYPE ignored: max prompt tokens reached")
            return None
        state.current_tokens.append(token)
        state.push_log(f"TYPE -> {token}")
    else:
        max_chars = int(CONFIG.get("max_prompt_chars", 220))
        if len(state.current_prompt_text()) >= max_chars:
            state.push_log("TYPE ignored: max prompt chars reached")
            return None
        state.current_tokens.append(token)
        state.push_log(f"TYPE -> {token}")
    refresh_retina_feedback()
    return None


def generation_worker(prompt_text: str) -> None:
    try:
        raw_path = generator.generate(prompt_text)
        state.last_prompt = prompt_text
        state.last_raw_image_local_path = str(raw_path)
        snapshot_path = snapshot_feedback_image(prompt_text, str(raw_path))
        state.records.insert(0, OutputRecord(
            prompt=prompt_text,
            image_url=f"/static/generated/{snapshot_path.name}",
            local_path=str(snapshot_path),
            raw_local_path=str(raw_path),
            timestamp=now_ts(),
        ))
        state.records = state.records[: int(CONFIG.get("max_history", 24))]
        refresh_retina_feedback(f"IMAGE READY + RETINA UPDATED -> raw={Path(raw_path).name} | feedback={snapshot_path.name}")
        state.last_error = ""
    except Exception as e:
        state.last_error = str(e)
        state.push_log(f"ERROR -> {e}")
    finally:
        state.generating = False


def move_cursor(action: str) -> None:
    rows = len(state.layout)
    cols = len(state.layout[0])
    wrap = bool(CONFIG.get("wrap_keyboard", True))
    if action == "LEFT":
        state.cursor_col = (state.cursor_col - 1) % cols if wrap else clamp(state.cursor_col - 1, 0, cols - 1)
    elif action == "RIGHT":
        state.cursor_col = (state.cursor_col + 1) % cols if wrap else clamp(state.cursor_col + 1, 0, cols - 1)
    elif action == "UP":
        state.cursor_row = (state.cursor_row - 1) % rows if wrap else clamp(state.cursor_row - 1, 0, rows - 1)
    elif action == "DOWN":
        state.cursor_row = (state.cursor_row + 1) % rows if wrap else clamp(state.cursor_row + 1, 0, rows - 1)


def apply_brain_action(action: str) -> Optional[str]:
    if action in {"LEFT", "RIGHT", "UP", "DOWN"}:
        move_cursor(action)
        return None
    if action == "IDLE":
        return None
    if action != "PRESS":
        return None
    token = state.layout[state.cursor_row][state.cursor_col]
    state.push_log(f"NEURAL PRESS -> KEY [{token}]")
    return apply_press_token(state, token)


async def one_brain_action() -> str:
    state.brain_busy = True
    try:
        return await asyncio.to_thread(
            state.brain.choose_action,
            state.cursor_row,
            state.cursor_col,
            state.layout,
            len(state.current_prompt_text()),
        )
    finally:
        state.brain_busy = False


async def fly_loop() -> None:
    state.push_log("REAL connectome loop started")
    while state.running:
        try:
            if not state.generating:
                action = await one_brain_action()
                prompt_to_generate = apply_brain_action(action)
                m = state.brain.last_metrics
                state.push_log(
                    f"NEURAL ACTION -> {action} @ ({state.cursor_row},{state.cursor_col}) | "
                    f"reason={m.get('reason', '—')} | spikes={m.get('spikes', 0)} brain={m.get('wall_ms', 0)}ms"
                )
                if prompt_to_generate:
                    state.generating = True
                    state.last_prompt = prompt_to_generate
                    refresh_retina_feedback("RETINA PREFEEDBACK -> submitted prompt held in view while image is generating")
                    threading.Thread(target=generation_worker, args=(prompt_to_generate,), daemon=True).start()
            await asyncio.sleep(float(CONFIG.get("tick_seconds", 0.45)))
        except asyncio.CancelledError:
            state.push_log("Loop cancelled")
            raise
        except Exception as e:
            state.last_error = str(e)
            state.push_log(f"LOOP ERROR -> {e}")
            await asyncio.sleep(1.0)
    state.push_log("Loop stopped")


@app.on_event("startup")
async def startup_event():
    state.push_log("Server ready")
    state.push_log(state.brain.summary())
    state.push_log(f"Keyboard mode: {CONFIG.get('keyboard_mode', 'qwerty')}")
    state.push_log("Retina feedback mode: prompt header + generated image")
    if CONFIG.get("generation_backend") == "comfyui":
        state.push_log(f"ComfyUI backend: {CONFIG.get('comfyui_url')} | checkpoint={CONFIG.get('checkpoint_name')}")
        try:
            msg = generator.validate()
            state.push_log(msg)
        except Exception as e:
            state.last_error = str(e)
            state.push_log(f"COMFY CHECK FAILED -> {e}")
    else:
        state.push_log("Mock backend enabled")
    await asyncio.to_thread(refresh_retina_feedback)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": CONFIG.get("ui_title", "Fly SD15 Loop"),
        "keyboard_mode": CONFIG.get("keyboard_mode", "qwerty"),
    })


@app.get("/api/state")
async def api_state():
    return JSONResponse({
        "running": state.running,
        "generating": state.generating,
        "brain_busy": state.brain_busy,
        "cursor_row": state.cursor_row,
        "cursor_col": state.cursor_col,
        "layout": state.layout,
        "current_prompt": state.current_prompt_text(),
        "last_prompt": state.last_prompt,
        "last_image_url": state.last_image_url,
        "last_image_local_path": state.last_image_local_path,
        "last_raw_image_local_path": state.last_raw_image_local_path,
        "records": [record.__dict__ for record in state.records],
        "logs": state.logs[-30:],
        "last_error": state.last_error,
        "backend": CONFIG.get("generation_backend"),
        "checkpoint": CONFIG.get("checkpoint_name"),
        "tick_seconds": CONFIG.get("tick_seconds"),
        "brain_summary": state.brain.summary(),
        "brain_metrics": state.brain.last_metrics,
        "keyboard_mode": CONFIG.get("keyboard_mode", "qwerty"),
    })


@app.post("/api/start")
async def api_start():
    if state.running:
        return {"ok": True, "message": "already running"}
    state.running = True
    state.loop_task = asyncio.create_task(fly_loop())
    return {"ok": True}


@app.post("/api/stop")
async def api_stop():
    state.running = False
    if state.loop_task:
        state.loop_task.cancel()
        try:
            await state.loop_task
        except asyncio.CancelledError:
            pass
        state.loop_task = None
    return {"ok": True}


@app.post("/api/step")
async def api_step():
    if state.running:
        raise HTTPException(status_code=400, detail="Stop loop before stepping manually")
    if state.generating:
        raise HTTPException(status_code=400, detail="Generation in progress")
    action = await one_brain_action()
    prompt_to_generate = apply_brain_action(action)
    m = state.brain.last_metrics
    state.push_log(
        f"MANUAL NEURAL STEP -> {action} | reason={m.get('reason', '—')} | spikes={m.get('spikes', 0)} brain={m.get('wall_ms', 0)}ms"
    )
    if prompt_to_generate:
        state.generating = True
        state.last_prompt = prompt_to_generate
        refresh_retina_feedback("RETINA PREFEEDBACK -> submitted prompt held in view while image is generating")
        threading.Thread(target=generation_worker, args=(prompt_to_generate,), daemon=True).start()
    return {"ok": True, "action": action, "brain": state.brain.last_metrics}


@app.post("/api/test-generation")
async def api_test_generation():
    if state.generating:
        raise HTTPException(status_code=400, detail="Generation already in progress")
    state.push_log("TEST GENERATION -> fly test image")
    state.generating = True
    state.last_prompt = "fly test image"
    refresh_retina_feedback("RETINA PREFEEDBACK -> fly test image")
    threading.Thread(target=generation_worker, args=("fly test image",), daemon=True).start()
    return {"ok": True}


@app.post("/api/reset")
async def api_reset():
    state.running = False
    if state.loop_task:
        state.loop_task.cancel()
        try:
            await state.loop_task
        except asyncio.CancelledError:
            pass
        state.loop_task = None
    state.current_tokens.clear()
    state.cursor_row = 0
    state.cursor_col = 0
    state.last_prompt = ""
    state.last_image_url = ""
    state.last_image_local_path = ""
    state.last_raw_image_local_path = ""
    state.records.clear()
    state.logs.clear()
    state.last_error = ""
    state.display_revision = 0
    await asyncio.to_thread(state.brain.reset)
    state.push_log("State reset: full DOOMFLY neural state recreated")
    await asyncio.to_thread(refresh_retina_feedback)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=CONFIG.get("app_host", "127.0.0.1"),
        port=int(CONFIG.get("app_port", 8951)),
        reload=False,
    )
