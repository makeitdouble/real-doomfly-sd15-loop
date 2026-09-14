import asyncio
import json
import os
import random
import re
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

ENGLISH_LEXICON_TEXT = """
a i aah able about above abstract acid across actor actual add after again air alien alive all almost alone alpha already also am amber ancient angel animal another any apple arc area arm around art artist as ash at atlas atom autumn away azure baby back bad ball band bare base basic beach bear beat beautiful because become bed bee before begin behind being bell below best better between beyond big bird black blade blue blur body bone book born both bottle bottom branch bright broken bronze brother brush build building burn burst but by camera cat cave cell center central century chain chair chance change chaos city clay clean clear cliff clock cloth cloud coast cold color comet coming common copper coral core cosmic could craft crystal dark data dawn day dead deep deer delicate dense depth desert design detail dew did dim distant do dog door dream drift drop dry dusk each early earth east easy echo edge edit electric else ember empty end energy engine enough enter epic even evening ever every eye fabric face fade fair fall far fast feather field fire first fish flower fly fog forest form forward found fox frame free fresh friend frost future galaxy garden gate ghost girl glass glow gold good grace grain grass green grey grid ground grow hand happy hard has have he head heart heat heavy her hero high hill home honey horizon horse house how human ice idea image in insect inside iron island it jade jewel joy just keep key kind king knew lake land language large last late leaf left legend light line little live long look lost low machine magic make man many marble mark meadow memory metal micro mist moon more morning moss motion mountain move much music my narrow near nebula need never night no north not oak ocean of off old olive on once one open orange orchard other our out pale paper path peace pearl people petal phase picture pine pink place plain planet plant pool portrait power pretty prism pulse pure purple quartz queen quick quiet rain red reed river road rock root rose round ruin run safe said sand scale sea search secret seed seem shadow shape she shell shine short silver simple sky sleep slow small smoke snow soft solar song soul sound south space spark sphere spirit spring star steel stone storm story strange stream street string sun surreal swan sweet swift table take temple than that the their them then there these they thin thing this through thunder tiny to tower tree true turn under unknown up us valley velvet view violet vision void warm war water wave we wheat when where white who wild wind wing winter with woman wood word world would write yellow you young"""


def tokenize_prompt_words(text: str) -> List[str]:
    return [m.group(0).lower() for m in re.finditer(r"[a-zA-Z']+", text or "") if m.group(0)]


def levenshtein_distance_limited(a: str, b: str, limit: int = 2) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        row_min = cur[0]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
            row_min = min(row_min, cur[-1])
        if row_min > limit:
            return limit + 1
        prev = cur
    return prev[-1]


class EnglishWordRewardDetector:
    def __init__(self):
        self.words = sorted({w.strip().lower() for w in ENGLISH_LEXICON_TEXT.split() if w.strip()})
        self.word_set = set(self.words)
        self.by_first: Dict[str, List[str]] = {}
        for w in self.words:
            self.by_first.setdefault(w[0], []).append(w)

    def _looks_like_english(self, token: str) -> bool:
        vowels = sum(1 for ch in token if ch in 'aeiouy')
        consonants = sum(1 for ch in token if ch.isalpha() and ch not in 'aeiouy')
        if vowels == 0:
            return False
        if len(token) >= 5 and consonants >= 5 and vowels <= 1:
            return False
        if 'qq' in token or 'zx' in token or 'qj' in token or 'jj' in token:
            return False
        if 'q' in token and 'qu' not in token:
            return False
        return True

    def _prefix_score(self, token: str) -> tuple[float, str]:
        if len(token) < 2 or token in self.word_set:
            return 0.0, ''
        candidates = [w for w in self.by_first.get(token[0], []) if w.startswith(token) and len(w) > len(token)]
        if not candidates:
            return 0.0, ''
        best = min(candidates, key=len)
        score = min(0.55, 0.14 * len(token))
        return round(score, 3), best

    def _near_match(self, token: str) -> tuple[float, str, int]:
        if len(token) < 3:
            return 0.0, '', 99
        candidates = [w for w in self.by_first.get(token[0], []) if abs(len(w) - len(token)) <= 2]
        best_word = ''
        best_dist = 99
        for cand in candidates:
            dist = levenshtein_distance_limited(token, cand, limit=2)
            if dist < best_dist:
                best_dist = dist
                best_word = cand
                if dist == 1:
                    break
        if best_dist == 1:
            return 0.72, best_word, best_dist
        if best_dist == 2 and len(token) >= 5:
            return 0.38, best_word, best_dist
        return 0.0, '', best_dist

    def analyze(self, text: str) -> Dict[str, Any]:
        tokens = tokenize_prompt_words(text)
        exact = []
        near = []
        unknown = []
        prefix_bonus = 0.0
        prefix_target = ''
        for token in tokens:
            if token in self.word_set:
                exact.append(token)
                continue
            near_score, near_word, near_dist = self._near_match(token)
            if near_score > 0:
                near.append({'token': token, 'target': near_word, 'score': near_score, 'distance': near_dist})
                continue
            unknown.append(token)

        if tokens:
            prefix_bonus, prefix_target = self._prefix_score(tokens[-1])

        exact_count = len(exact)
        near_score_total = round(sum(item['score'] for item in near), 3)
        unknown_penalty = min(1.6, 0.08 * max(0, len(unknown) - 1))
        multiword_bonus = max(0, exact_count - 1) * 0.45
        unique_bonus = min(0.6, 0.12 * len(set(exact)))
        total_reward = max(0.0, exact_count * 1.0 + near_score_total + prefix_bonus + multiword_bonus + unique_bonus - unknown_penalty)
        typing_reward = max(0.0, exact_count * 0.9 + near_score_total * 0.75 + prefix_bonus - min(1.0, 0.05 * len(unknown)))
        english_ratio = (exact_count + 0.6 * len(near)) / max(1, len(tokens))
        return {
            'tokens': tokens,
            'exact_words': exact,
            'exact_count': exact_count,
            'near_words': near,
            'near_count': len(near),
            'unknown_words': unknown,
            'unknown_count': len(unknown),
            'prefix_bonus': round(prefix_bonus, 3),
            'prefix_target': prefix_target,
            'typing_reward': round(typing_reward, 3),
            'total_reward': round(total_reward, 3),
            'english_ratio': round(float(english_ratio), 3),
        }


class RewardPolicy:
    def __init__(self, config: Dict[str, Any]):
        self.alpha = float(config.get('reward_alpha', 0.18))
        self.gamma = float(config.get('reward_gamma', 0.94))
        self.epsilon = float(config.get('reward_exploration', 0.16))
        self.base_action_bias = float(config.get('reward_base_action_bias', 1.2))
        self.learned_scale = float(config.get('reward_learned_scale', 0.9))
        self.typing_delta_gain = float(config.get('reward_typing_delta_gain', 1.0))
        self.submit_gain = float(config.get('reward_submit_gain', 1.25))
        self.credit_window = int(config.get('reward_credit_window', 72))
        self.max_abs_q = float(config.get('reward_max_abs_q', 8.0))
        self.q_values: Dict[str, np.ndarray] = {}
        self.trace: List[Dict[str, Any]] = []
        self.current_buffer_score = 0.0
        self.total_reward = 0.0
        self.last_delta = 0.0
        self.last_event_reward = 0.0
        self.last_submit_reward = 0.0
        self.last_buffer_report: Dict[str, Any] = {}
        self.last_submit_report: Dict[str, Any] = {}
        self.last_policy_info: Dict[str, Any] = {}
        self.prompt_counter = 0
        self.rewarded_exact_signatures: set[tuple[int, str]] = set()
        self.rewarded_near_signatures: set[tuple[int, str]] = set()

    def reset(self) -> None:
        self.q_values.clear()
        self.trace.clear()
        self.current_buffer_score = 0.0
        self.total_reward = 0.0
        self.last_delta = 0.0
        self.last_event_reward = 0.0
        self.last_submit_reward = 0.0
        self.last_buffer_report = {}
        self.last_submit_report = {}
        self.last_policy_info = {}
        self.prompt_counter = 0
        self.rewarded_exact_signatures.clear()
        self.rewarded_near_signatures.clear()

    def _state_key(self, cursor_row: int, cursor_col: int, layout: List[List[str]], prompt_text: str) -> str:
        token = layout[cursor_row][cursor_col]
        words = tokenize_prompt_words(prompt_text)
        suffix = words[-1][-2:] if words else ''
        length_bucket = min(12, len(prompt_text) // 3)
        return f'{cursor_row}:{cursor_col}|{token}|{suffix}|{length_bucket}'

    def _ensure_q(self, state_key: str) -> np.ndarray:
        if state_key not in self.q_values:
            self.q_values[state_key] = np.zeros(len(ACTIONS), dtype=np.float64)
        return self.q_values[state_key]

    def select_action(self, base_action: str, cursor_row: int, cursor_col: int, layout: List[List[str]], prompt_text: str) -> str:
        state_key = self._state_key(cursor_row, cursor_col, layout, prompt_text)
        q = self._ensure_q(state_key)
        scores = q * self.learned_scale
        if base_action in ACTIONS:
            scores[ACTIONS.index(base_action)] += self.base_action_bias
        if random.random() < self.epsilon:
            final_action = random.choice(ACTIONS)
            policy_mode = 'explore'
        else:
            final_action = ACTIONS[int(np.argmax(scores))]
            policy_mode = 'greedy'
        self.trace.append({'state_key': state_key, 'action': final_action})
        if len(self.trace) > self.credit_window * 3:
            self.trace = self.trace[-self.credit_window * 3:]
        self.last_policy_info = {
            'base_action': base_action,
            'final_action': final_action,
            'policy_mode': policy_mode,
            'override': final_action != base_action,
            'state_key': state_key,
            'q_values': {a: round(float(q[i]), 3) for i, a in enumerate(ACTIONS)},
        }
        return final_action

    def _apply_credit(self, reward: float) -> None:
        if abs(reward) < 1e-6 or not self.trace:
            return
        recent = self.trace[-self.credit_window:]
        for distance, item in enumerate(reversed(recent)):
            decay = self.gamma ** distance
            q = self._ensure_q(item['state_key'])
            idx = ACTIONS.index(item['action'])
            q[idx] = float(np.clip(q[idx] + self.alpha * reward * decay, -self.max_abs_q, self.max_abs_q))
        self.total_reward += reward

    def _word_signatures(self, report: Dict[str, Any]) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
        exact_sigs: list[tuple[int, str]] = []
        for idx, token in enumerate(report.get('tokens', [])):
            if token in report.get('exact_words', []):
                exact_sigs.append((idx, token))
        near_sigs: list[tuple[int, str]] = []
        for item in report.get('near_words', []):
            token = item.get('token', '')
            for idx, t in enumerate(report.get('tokens', [])):
                if t == token:
                    near_sigs.append((idx, token))
                    break
        return exact_sigs, near_sigs

    def _one_shot_typing_bonus(self, report: Dict[str, Any]) -> tuple[float, list[str], list[str]]:
        exact_sigs, near_sigs = self._word_signatures(report)
        current_exact = set(exact_sigs)
        current_near = set(near_sigs)
        # drop signatures that disappeared after delete/backtracking, so re-discovery is possible later in same prompt
        self.rewarded_exact_signatures.intersection_update(current_exact)
        self.rewarded_near_signatures.intersection_update(current_near)

        new_exact = [sig for sig in exact_sigs if sig not in self.rewarded_exact_signatures]
        new_near = [sig for sig in near_sigs if sig not in self.rewarded_near_signatures]

        for sig in new_exact:
            self.rewarded_exact_signatures.add(sig)
        for sig in new_near:
            self.rewarded_near_signatures.add(sig)

        exact_words = [word for _, word in new_exact]
        near_words = [word for _, word in new_near]
        bonus = 0.22 * len(new_exact) + 0.08 * len(new_near)
        return round(bonus, 3), exact_words, near_words

    def on_buffer_updated(self, prompt_text: str, detector: 'EnglishWordRewardDetector') -> Dict[str, Any]:
        report = detector.analyze(prompt_text)
        delta = report['typing_reward'] - self.current_buffer_score
        one_shot_bonus, new_exact_words, new_near_words = self._one_shot_typing_bonus(report)
        event_reward = delta * self.typing_delta_gain + one_shot_bonus
        self.last_delta = round(delta, 3)
        self.last_event_reward = round(event_reward, 3)
        self.current_buffer_score = report['typing_reward']
        self.last_buffer_report = dict(report)
        self.last_buffer_report['delta'] = round(delta, 3)
        self.last_buffer_report['event_reward'] = round(event_reward, 3)
        self.last_buffer_report['new_exact_words'] = new_exact_words
        self.last_buffer_report['new_near_words'] = new_near_words
        self.last_buffer_report['one_shot_bonus'] = round(one_shot_bonus, 3)
        if abs(event_reward) >= 0.01:
            self._apply_credit(event_reward)
        return self.last_buffer_report

    def on_prompt_submitted(self, prompt_text: str, detector: 'EnglishWordRewardDetector') -> Dict[str, Any]:
        report = detector.analyze(prompt_text)
        exact_count = report.get('exact_count', 0)
        near_count = report.get('near_count', 0)
        length_bonus = min(1.5, 0.15 * max(0, len(report.get('tokens', [])) - 1))
        reward = (report['total_reward'] + exact_count * 0.9 + near_count * 0.25 + length_bonus) * self.submit_gain
        self._apply_credit(reward)
        self.last_submit_reward = round(reward, 3)
        self.last_submit_report = dict(report)
        self.last_submit_report['applied_reward'] = round(reward, 3)
        self.last_submit_report['length_bonus'] = round(length_bonus, 3)
        self.prompt_counter += 1
        self.trace.clear()
        self.current_buffer_score = 0.0
        self.last_delta = 0.0
        self.last_event_reward = 0.0
        self.rewarded_exact_signatures.clear()
        self.rewarded_near_signatures.clear()
        return self.last_submit_report

    def telemetry(self) -> Dict[str, Any]:
        return {
            'prompt_counter': self.prompt_counter,
            'current_buffer_score': round(self.current_buffer_score, 3),
            'last_delta': round(self.last_delta, 3),
            'last_event_reward': round(self.last_event_reward, 3),
            'last_submit_reward': round(self.last_submit_reward, 3),
            'total_reward': round(self.total_reward, 3),
            'buffer_report': self.last_buffer_report,
            'submit_report': self.last_submit_report,
            'policy': self.last_policy_info,
        }



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
    reward_total: float = 0.0
    reward_summary: str = ''


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
    reward_status: Dict[str, Any] = field(default_factory=dict)

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
reward_detector = EnglishWordRewardDetector()
reward_policy = RewardPolicy(CONFIG)
state.reward_status = reward_policy.telemetry()
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


def update_language_reward(log_prefix: str = '') -> Dict[str, Any]:
    report = reward_policy.on_buffer_updated(state.current_prompt_text(), reward_detector)
    state.reward_status = reward_policy.telemetry()
    if log_prefix and abs(report.get('delta', 0.0)) >= 0.05:
        state.push_log(
            f"{log_prefix} reward delta {report.get('delta', 0.0):+.2f} | event={report.get('event_reward', 0.0):+.2f} | "
            f"new_exact={','.join(report.get('new_exact_words', [])) or '—'} | new_near={','.join(report.get('new_near_words', [])) or '—'} | prefix={report.get('prefix_bonus', 0.0):.2f}"
        )
    return report


def apply_prompt_submit_reward(prompt_text: str) -> Dict[str, Any]:
    report = reward_policy.on_prompt_submitted(prompt_text, reward_detector)
    state.reward_status = reward_policy.telemetry()
    near_preview = ', '.join(f"{x['token']}→{x['target']}" for x in report.get('near_words', [])[:3]) or '—'
    exact_preview = ', '.join(report.get('exact_words', [])[:5]) or '—'
    state.push_log(
        f"LANGUAGE REWARD -> total={report.get('total_reward', 0.0):.2f} applied={report.get('applied_reward', 0.0):.2f} | "
        f"exact={report.get('exact_count', 0)} [{exact_preview}] | near={report.get('near_count', 0)} [{near_preview}]"
    )
    return report


def apply_press_token(state: LoopState, token: str) -> Optional[str]:
    if token == "delete":
        if state.current_tokens:
            removed = state.current_tokens.pop()
            state.push_log(f"DELETE -> removed '{removed}'")
            refresh_retina_feedback()
            update_language_reward('DELETE')
        return None
    if token in {"ENTER", "OK"}:
        prompt_text = state.current_prompt_text().strip()
        if prompt_text:
            reward_report = apply_prompt_submit_reward(prompt_text)
            state.push_log(f"PROMPT SUBMIT -> {prompt_text}")
            state.current_tokens.clear()
            state.reward_status = reward_policy.telemetry()
            state.reward_status['last_submit_report'] = reward_report
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
        update_language_reward('TYPE')
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
    update_language_reward('TYPE')
    return None


def generation_worker(prompt_text: str) -> None:
    try:
        raw_path = generator.generate(prompt_text)
        state.last_prompt = prompt_text
        state.last_raw_image_local_path = str(raw_path)
        snapshot_path = snapshot_feedback_image(prompt_text, str(raw_path))
        submit_report = reward_policy.last_submit_report or reward_detector.analyze(prompt_text)
        reward_summary = f"exact {submit_report.get('exact_count', 0)} | near {submit_report.get('near_count', 0)} | reward {submit_report.get('total_reward', 0.0):.2f}"
        state.records.insert(0, OutputRecord(
            prompt=prompt_text,
            image_url=f"/static/generated/{snapshot_path.name}",
            local_path=str(snapshot_path),
            raw_local_path=str(raw_path),
            timestamp=now_ts(),
            reward_total=float(submit_report.get('total_reward', 0.0)),
            reward_summary=reward_summary,
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
        base_action = await asyncio.to_thread(
            state.brain.choose_action,
            state.cursor_row,
            state.cursor_col,
            state.layout,
            len(state.current_prompt_text()),
        )
        final_action = reward_policy.select_action(
            base_action,
            state.cursor_row,
            state.cursor_col,
            state.layout,
            state.current_prompt_text(),
        )
        state.reward_status = reward_policy.telemetry()
        state.brain.last_metrics['base_action'] = base_action
        state.brain.last_metrics['policy_action'] = final_action
        state.brain.last_metrics['policy_mode'] = reward_policy.last_policy_info.get('policy_mode', '—')
        state.brain.last_metrics['policy_override'] = reward_policy.last_policy_info.get('override', False)
        state.brain.last_metrics['reward_total'] = state.reward_status.get('total_reward', 0.0)
        state.brain.last_metrics['buffer_reward'] = state.reward_status.get('current_buffer_score', 0.0)
        return final_action
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
                    f"NEURAL ACTION -> {action} (base {m.get('base_action', action)}) @ ({state.cursor_row},{state.cursor_col}) | "
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
    state.push_log("Reward mode: English-word detector + lightweight policy reinforcement")
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
        "reward_status": state.reward_status,
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
        "reward_status": state.reward_status,
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
        f"MANUAL NEURAL STEP -> {action} (base {m.get('base_action', action)}) | reason={m.get('reason', '—')} | spikes={m.get('spikes', 0)} brain={m.get('wall_ms', 0)}ms"
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
    reward_policy.reset()
    state.reward_status = reward_policy.telemetry()
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
