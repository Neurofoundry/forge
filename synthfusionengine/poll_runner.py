#!/usr/bin/env python3
"""
SynthFusion no-instruction poll runner.

Runs style-schema polls against Cloudflare AI (default) or OpenAI-compatible Responses API.

Usage:
  python poll_runner.py --image path/to/image.png
  python poll_runner.py --image path/to/image.png --mode schema_plus_image_minimal
  python poll_runner.py --image path/to/image.png --model gpt-5.2-codex

Cloudflare mode (default) env vars:
  CLOUDFLARE_API_TOKEN
  CLOUDFLARE_ACCOUNT_ID

OpenAI mode env vars:
  OPENAI_API_KEY
Optional:
  OPENAI_BASE_URL (default: https://api.openai.com/v1)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent
SCHEMA_INDEX_PATH = ROOT / "style_schema_index.json"
POLL_PACK_PATH = ROOT / "poll_pack_no_instruction.json"
RESULTS_DIR = ROOT / "poll_results"


@dataclass
class PollConfig:
    model: str
    mode: str
    image_path: Path
    image_id: str
    provider: str


def load_local_env(path: Path) -> None:
    """Load .env values into process env if missing.

    Supports:
    - KEY=value
    - export KEY=value
    - $env:KEY="value" (PowerShell style)
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        m_ps = re.match(r'^\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*["\']?(.*?)["\']?\s*$', line)
        if m_ps:
            key, value = m_ps.group(1), m_ps.group(2)
            if key and value and not os.environ.get(key):
                os.environ[key] = value
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_to_b64(image_path: Path) -> str:
    data = image_path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def image_to_u8(image_path: Path) -> list[int]:
    return list(image_path.read_bytes())


def extract_text_from_responses(resp: dict[str, Any]) -> str:
    # Common Responses API result extraction across variants.
    out = resp.get("output", [])
    texts: list[str] = []
    for item in out:
        for content in item.get("content", []) if isinstance(item, dict) else []:
            if isinstance(content, dict):
                if content.get("type") in ("output_text", "text"):
                    t = content.get("text")
                    if isinstance(t, str):
                        texts.append(t)
                elif isinstance(content.get("text"), str):
                    texts.append(content["text"])
    if texts:
        return "\n".join(texts).strip()

    # Fallbacks
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"].strip()
    return json.dumps(resp, ensure_ascii=False)[:4000]


def extract_text_from_cloudflare(resp: dict[str, Any]) -> str:
    result = resp.get("result")
    if isinstance(result, dict):
        for key in ("response", "generated_text", "output", "caption", "description"):
            v = result.get(key)
            if isinstance(v, str):
                return v.strip()
    if isinstance(result, str):
        return result.strip()
    return json.dumps(resp, ensure_ascii=False)[:4000]


def estimate_compactness_score(text: str) -> int:
    wc = len(re.findall(r"\S+", text))
    if wc <= 40:
        return 1
    if wc <= 65:
        return 3
    if wc <= 90:
        return 5
    return 2


def extract_bracket_prompt(text: str) -> str:
    parts = re.findall(r"\[[^\]]+\]", text)
    return " ".join(parts).strip()


def heuristic_scores(text: str) -> dict[str, int]:
    lower = text.lower()
    def has_any(words: list[str]) -> int:
        return 4 if any(w in lower for w in words) else 1

    return {
        "character_coverage": has_any(["character", "hero", "portrait", "rider", "subject", "mask"]),
        "objects": has_any(["hammer", "sword", "armor", "gloves", "boots", "horse", "object"]),
        "scene": has_any(["scene", "battlefield", "street", "outdoor", "indoor", "environment"]),
        "atmosphere": has_any(["smoke", "haze", "fog", "embers", "mood", "atmosphere", "storm"]),
        "camera": has_any(["angle", "shot", "framing", "close-up", "wide", "low angle"]),
        "style": has_any(["style", "cinematic", "ink", "collage", "painterly", "render"]),
        "compactness": estimate_compactness_score(text),
    }


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_'-]+", text)


def count_words(text: str) -> int:
    return len(word_tokens(text))


def extract_fragments(text: str) -> list[str]:
    groups = re.findall(r"\[([^\]]+)\]", text)
    source_parts: list[str] = groups if groups else [text]
    out: list[str] = []
    seen: set[str] = set()
    for part in source_parts:
        for frag in re.split(r"[,\n;]+", part):
            cleaned = re.sub(r"\s+", " ", frag).strip(" -\t\r\n")
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
    return out


def schema_expansion_pool(schema_json: dict[str, Any]) -> list[str]:
    pool: list[str] = []
    seen: set[str] = set()

    def push(item: str) -> None:
        cleaned = re.sub(r"\s+", " ", item).strip(" -\t\r\n")
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        pool.append(cleaned)

    intent = schema_json.get("intent")
    if isinstance(intent, str):
        for frag in re.split(r"[,\n;]+", intent):
            push(frag)

    layout = schema_json.get("layout", {})
    if isinstance(layout, dict):
        order = layout.get("order", [])
        if isinstance(order, list):
            for lane in order:
                if isinstance(lane, str):
                    push(lane.replace("_", " "))

    bias = schema_json.get("lexicon_bias", {})
    if isinstance(bias, dict):
        prefer = bias.get("prefer", [])
        if isinstance(prefer, list):
            for p in prefer:
                if isinstance(p, str):
                    push(p)

    defaults = schema_json.get("negative_defaults", [])
    if isinstance(defaults, list):
        for n in defaults:
            if isinstance(n, str):
                push(n)

    example = schema_json.get("example_compact")
    if isinstance(example, str):
        for frag in extract_fragments(example):
            push(frag)

    # Generic fillers for "75 wasn't enough, spread it" behavior.
    for extra in [
        "atmospheric depth",
        "camera depth separation",
        "volumetric haze",
        "rim light accents",
        "foreground midground background layering",
        "material texture variation",
        "controlled color contrast",
        "cinematic composition",
        "dynamic perspective",
        "environment context",
    ]:
        push(extra)
    return pool


def chunks(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        return [items]
    return [items[i:i + size] for i in range(0, len(items), size)]


def fragments_to_brackets(fragments: list[str], group_size: int = 3) -> str:
    groups = chunks(fragments, group_size)
    return " ".join(f"[{', '.join(g)}]" for g in groups if g).strip()


def normalize_compact_to_target(
    compact_text: str,
    raw_text: str,
    schema_json: dict[str, Any],
    target_words: int = 75,
) -> tuple[str, int, bool, list[str]]:
    fragments = extract_fragments(compact_text)
    if not fragments:
        fragments = extract_fragments(raw_text)
    if not fragments:
        example = schema_json.get("example_compact")
        if isinstance(example, str):
            fragments = extract_fragments(example)

    pool = schema_expansion_pool(schema_json)
    spread_added: list[str] = []

    def total_words(fs: list[str]) -> int:
        return sum(count_words(f) for f in fs)

    current = total_words(fragments)
    if current < target_words:
        for extra in pool:
            if total_words(fragments) >= target_words:
                break
            if extra.lower() in {f.lower() for f in fragments}:
                continue
            fragments.append(extra)
            spread_added.append(extra)

    # Trim from tail to satisfy exact target.
    while fragments and total_words(fragments) > target_words:
        if count_words(fragments[-1]) <= (total_words(fragments) - target_words):
            fragments.pop()
            continue
        # Partial trim on final fragment when needed.
        words = word_tokens(fragments[-1])
        keep = max(1, len(words) - (total_words(fragments) - target_words))
        fragments[-1] = " ".join(words[:keep])
        break

    # Final pad with stable filler words if still short.
    filler_words = ["atmosphere", "camera", "detail", "texture", "lighting", "depth", "contrast", "motion"]
    while total_words(fragments) < target_words:
        needed = target_words - total_words(fragments)
        take = filler_words[:needed]
        fragments.append(" ".join(take))
        spread_added.append(" ".join(take))

    compact_75 = fragments_to_brackets(fragments, group_size=3)
    count_75 = count_words(compact_75)
    return compact_75, count_75, len(spread_added) > 0, spread_added


def build_input(mode: str, schema_json: dict[str, Any], image_b64: str) -> list[dict[str, Any]]:
    if mode == "image_only":
        return [
            {"type": "input_image", "image_b64": image_b64}
        ]
    if mode == "schema_plus_image_minimal":
        return [
            {
                "type": "input_text",
                "text": "Use this schema format only. Return compact bracket output and negative list."
            },
            {
                "type": "input_text",
                "text": json.dumps(schema_json, ensure_ascii=False)
            },
            {"type": "input_image", "image_b64": image_b64}
        ]
    raise ValueError(f"Unsupported mode: {mode}")


def call_responses_api(model: str, input_items: list[dict[str, Any]]) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/responses"
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": input_items
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1200]}")
    return r.json()


def call_cloudflare_ai(
    model: str,
    mode: str,
    schema_json: dict[str, Any],
    image_b64: str,
    image_u8: list[int],
) -> dict[str, Any]:
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    if not token or not account_id:
        raise RuntimeError("CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID are required for provider=cloudflare")

    # Vision-first payload for image polling.
    if mode == "image_only":
        prompt = "Describe the image with compact bracketed fragments only."
    else:
        prompt = (
            "Use this schema format only. Return compact bracket output and negative list.\n"
            + json.dumps(schema_json, ensure_ascii=False)
        )

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data_uri = f"data:image/png;base64,{image_b64}"
    payloads = [
        {"image": image_u8, "prompt": prompt, "max_tokens": 700},
        {"image": data_uri, "prompt": prompt, "max_tokens": 700},
        {"image": image_b64, "prompt": prompt, "max_tokens": 700},
        {"input": {"image": data_uri, "prompt": prompt, "max_tokens": 700}},
    ]
    last_error = ""
    for payload in payloads:
        r = requests.post(url, headers=headers, json=payload, timeout=180)
        if r.status_code < 400:
            return r.json()
        last_error = f"HTTP {r.status_code}: {r.text[:1200]}"
        # Retry alternate payload only for input-shape errors.
        if r.status_code != 400 or "Tensor error" not in r.text:
            break
    raise RuntimeError(last_error)


def run_poll(cfg: PollConfig) -> dict[str, Any]:
    schema_index = read_json(SCHEMA_INDEX_PATH)
    schemas = schema_index.get("schemas", [])
    image_b64 = image_to_b64(cfg.image_path)
    image_u8 = image_to_u8(cfg.image_path)

    rows: list[dict[str, Any]] = []
    start_all = time.time()
    for s in schemas:
        schema_file = ROOT / s["file"]
        schema_json = read_json(schema_file)

        t0 = time.time()
        if cfg.provider == "cloudflare":
            raw_resp = call_cloudflare_ai(cfg.model, cfg.mode, schema_json, image_b64, image_u8)
            raw_text = extract_text_from_cloudflare(raw_resp)
        else:
            input_items = build_input(cfg.mode, schema_json, image_b64)
            raw_resp = call_responses_api(cfg.model, input_items)
            raw_text = extract_text_from_responses(raw_resp)
        elapsed_ms = int((time.time() - t0) * 1000)
        compact_raw = extract_bracket_prompt(raw_text)
        raw_count = count_words(compact_raw)
        compact_75, count_75, spread_applied, spread_added = normalize_compact_to_target(
            compact_raw, raw_text, schema_json, target_words=75
        )

        row = {
            "image_id": cfg.image_id,
            "schema_id": s["id"],
            "schema_file": s["file"],
            "mode": cfg.mode,
            "model": cfg.model,
            "latency_ms": elapsed_ms,
            "raw_response": raw_text,
            "compact_prompt": compact_75,
            "compact_prompt_raw": compact_raw,
            "token_count_raw": raw_count,
            "token_count": count_75,
            "strict_target_tokens": 75,
            "strict_75_pass": count_75 == 75,
            "spread_applied": spread_applied,
            "spread_added_fragments": spread_added,
            "negative_prompt": "",  # manual extraction step if needed
            "scores": heuristic_scores(raw_text),
        }
        rows.append(row)
        print(f"- {s['id']}: {elapsed_ms}ms")

    total_ms = int((time.time() - start_all) * 1000)
    summary = {
        "image_id": cfg.image_id,
        "mode": cfg.mode,
        "model": cfg.model,
        "total_ms": total_ms,
        "count": len(rows)
    }
    return {"summary": summary, "rows": rows}


def main() -> int:
    load_local_env(ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--image-id", default="", help="Logical image id for report")
    parser.add_argument("--mode", default="schema_plus_image_minimal", choices=["image_only", "schema_plus_image_minimal"])
    parser.add_argument("--provider", default="cloudflare", choices=["cloudflare", "openai"])
    parser.add_argument("--model", default="@cf/llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 2

    image_id = args.image_id or image_path.stem
    cfg = PollConfig(model=args.model, mode=args.mode, image_path=image_path, image_id=image_id, provider=args.provider)

    result = run_poll(cfg)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out).resolve() if args.out else RESULTS_DIR / f"{image_id}_{cfg.mode}_{int(time.time())}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved:", out_path)
    print("Summary:", json.dumps(result["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
