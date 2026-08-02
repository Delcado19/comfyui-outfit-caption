"""NVIDIA NIM outfit captioning node for the comfyui-outfit-caption package."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


DEFAULT_ENV = "NVIDIA_API_KEY_COMFYUI"
MODEL_LIST_URL = "https://integrate.api.nvidia.com/v1/models"
IMAGE_MODEL_CATALOG_URL = (
    "https://build.nvidia.com/models?filters=usecase%3Ausecase_image_to_text"
)
GENERATE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
CATALOG_PATH = Path(__file__).with_name("models_nvidia.json")
_CACHE_ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
CACHE_PATH = _CACHE_ROOT / "ComfyUI" / "outfit-caption-models" / "nvidia-nim.json"
CUSTOM_MODEL = "⚪ — | use custom_model"
PROBE_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
RATING_ORDER = {"green": 0, "yellow": 1, "red": 2, "untested": 3}
RATING_ICON = {"green": "🟢", "yellow": "🟡", "red": "🔴", "untested": "⚪"}

LABELS = (
    "Garment pieces",
    "Coverage and silhouette",
    "Color",
    "Material family",
    "Material evidence",
    "Construction details",
    "Folds and tension",
    "Footwear",
    "Accessories",
    "Preservation",
    "Uncertainty",
)
PROMPT = """Describe only the visible clothing, footwear and accessories for virtual try-on caption QA.

Ignore the person, face, hair, expression, pose, camera, background, setting and lighting setup. Do not describe body shape except where needed to explain garment fit. Return English text only. Do not hardcode any garment category, color, material, or style; infer only from the current outfit reference image.

Use evidence-first material analysis. First describe visible evidence: highlight shape, highlight sharpness, reflection clarity, surface grain if visible, weave if visible, fold behavior, compression creases, seams, panel edges, thickness, stiffness, drape, and how the material changes at joints or hems.

Material family is mandatory. Choose the closest conservative family from visible evidence: woven textile, knit/jersey, denim, satin/silk, suede, smooth coated textile, leather-like synthetic, smooth leather, stretch leather, coated leather, patent leather, latex/PVC/rubber, metal, mesh, lace, fur, or unknown. If uncertain, write a cautious label such as unknown smooth coated material. Use latex/PVC/rubber/patent only with hard mirror reflections or clearly reflected shapes.

Write exactly these labelled sections:
Garment pieces:
Coverage and silhouette:
Color:
Material family:
Material evidence:
Construction details:
Folds and tension:
Footwear:
Accessories:
Preservation:
Uncertainty:

Footwear must state whether toes are open or closed if visible, plus toe shape, heel type, shaft/strap coverage, and material contrast. Zippers, belts, buckles, eyelets, seams, panels, cuffs, hems, cutouts, lacing, buttons, rings and trim must be listed only when visible. In Preservation, list the visual features a diffusion model should preserve. Do not mention non-clothing elements."""
PROMPT_PREFIX = (
    "Transfer only the visible garments, footwear, and outfit accessories described below. "
    "Preserve the target person's identity, pose, face, hair, hands, and background. Do not "
    "invent extra openings, exposure, jewelry, logos, colors, materials, closures, straps, or "
    "footwear details beyond the reference evidence. Respect the described clothing layer order: "
    "outer garments must remain outermost and cover underlayers wherever they overlap, including "
    "the sleeves. Show an underlayer only where the reference exposes it at necklines, openings, "
    "hems, or cuffs. Do not remove described outer-garment sleeves."
)
NO_PERSON_ACCESSORY_PREFIX = (
    "Do not transfer personal or carried accessories from the reference, including bags, "
    "purses, backpacks, eyewear, jewelry, watches, hats, or their loose straps and handles. "
    "Keep construction that is structurally part of a garment, such as garment belts, sewn "
    "straps, buckles, buttons, and zippers. Do not copy reference body parts or pose."
)
SOFTENERS = (
    (re.compile(r"\bdeep,\s*plunging\s+v[- ]neckline\b", re.I), "visible front neckline"),
    (re.compile(r"\bdeep\s+plunging\s+v[- ]neckline\b", re.I), "visible front neckline"),
    (re.compile(r"\bplunging\s+v[- ]neckline\b", re.I), "front neckline"),
    (re.compile(r"\bdeep\s+v[- ]neckline\b", re.I), "front neckline"),
)
PERSON_ACCESSORY_RE = re.compile(
    r"\b(sunglasses?|eyewear|glasses|goggles|visors?|lenses?|earrings?|necklaces?|chokers?|"
    r"bracelets?|watches|piercings?|finger\s+rings?|rings?\s+on\s+(?:the\s+)?"
    r"(?:(?:index|middle|ring|little|thumb)\s+)?fingers?|handbags?|bags?|purses?|clutches?|"
    r"wallets?|backpacks?|totes?|satchels?|briefcases?|bag\s+(?:straps?|handles?)|"
    r"(?:straps?|handles?)\s+(?:of|for|attached\s+to)\s+(?:the\s+)?(?:hand)?bag|"
    r"hats?|headbands?|hair\s+accessories)\b",
    re.I,
)

# ponytail: secrets live only for this process; re-entry after restart/clone is the security boundary.
_SESSION_KEYS: dict[str, str] = {}


def load_catalog(path: Path = CATALOG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_error(error: object, api_key: str = "") -> str:
    text = str(error).replace(api_key, "***") if api_key else str(error)
    return text[:500]


def _request_json(request: urllib.request.Request, timeout: float, api_key: str) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NVIDIA API error {exc.code}: {_sanitize_error(body, api_key)}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"NVIDIA request failed: {_sanitize_error(exc, api_key)}") from exc


def fetch_model_list(api_key: str, timeout: float = 15.0) -> list[dict[str, str]]:
    request = urllib.request.Request(
        MODEL_LIST_URL, headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    )
    payload = _request_json(request, timeout, api_key)
    return [
        {"id": str(item["id"]), "version": str(item.get("created") or "")}
        for item in payload.get("data", [])
        if item.get("id")
    ]


def _parse_image_model_names(page: str) -> list[str]:
    # NVIDIA embeds the filtered catalog as escaped JSON in the server-rendered page.
    pattern = r'\\"resourceType\\":\\"ENDPOINT\\".*?\\"name\\":\\"([^\\"]+)\\"'
    return list(dict.fromkeys(re.findall(pattern, page, re.S)))


def fetch_image_model_names(timeout: float = 15.0) -> list[str]:
    # ponytail: scraping build.nvidia.com is fragile, but /v1/models carries no
    # modality field (verified: only id/object/created/owned_by), so this page is
    # the sole Image-to-Text source. Switch to a JSON API if NVIDIA ever ships one.
    request = urllib.request.Request(
        IMAGE_MODEL_CATALOG_URL,
        headers={"Accept": "text/html", "User-Agent": "ComfyUI-NVIDIA-NIM-Outfit-Caption/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            page = response.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"NVIDIA catalog request failed: {_sanitize_error(exc)}") from exc
    names = _parse_image_model_names(page)
    if not names:
        raise RuntimeError("NVIDIA catalog response contained no Image-to-Text endpoints.")
    return names


def _effective_rating(entry: dict, live_version: str) -> tuple[str, int | None]:
    rating = entry.get("rating", "untested")
    tested_version = entry.get("tested_model_version")
    if rating != "untested" and tested_version and live_version and live_version != tested_version:
        return "untested", None
    score = entry.get("score")
    return rating, score if isinstance(score, int) else None


def _is_official_image_model(model_id: str, image_model_names: set[str]) -> bool:
    slug = model_id.split("/", 1)[-1]
    return slug in image_model_names or model_id.replace("/", "-") in image_model_names


def rank_models(
    catalog: dict, live_models: list[dict], image_model_names: list[str] | None = None
) -> list[dict]:
    catalog_by_id = {entry["id"]: entry for entry in catalog.get("models", [])}
    excluded = set(catalog.get("excluded_models", {}))
    official = set(image_model_names or [])
    ranked = []
    for live_entry in live_models:
        model_id = str(live_entry.get("id") or "")
        if not model_id or model_id in excluded:
            continue
        entry = catalog_by_id.get(model_id)
        if entry is None and not _is_official_image_model(model_id, official):
            continue
        entry = entry or {"id": model_id, "rating": "untested", "score": None}
        rating, score = _effective_rating(entry, str(live_entry.get("version") or ""))
        score_text = str(score) if score is not None else "—"
        # ComfyUI validates saved combo values exactly, so cache state must not alter the label.
        label = f"{RATING_ICON[rating]} {score_text} | {model_id}"
        ranked.append({"id": model_id, "label": label, "rating": rating, "score": score})
    # Stable provider order is retained for equal scores; no secondary criterion is invented.
    ranked.sort(
        key=lambda item: (
            RATING_ORDER[item["rating"]],
            -(item["score"] if item["score"] is not None else -1),
        )
    )
    return ranked


def _read_cache() -> dict:
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        verified = payload.get("verified") is True
        return {
            "models": payload.get("models", []) if verified else [],
            "image_model_names": payload.get("image_model_names", []) if verified else [],
            "verified": verified,
        }
    except (OSError, json.JSONDecodeError):
        return {"models": [], "image_model_names": [], "verified": False}


def _write_cache(models: list[dict], image_model_names: list[str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(
            {"models": models, "image_model_names": image_model_names, "verified": True},
            indent=2,
        ),
        encoding="utf-8",
    )


def _probe_model(model_id: str, api_key: str, timeout: float) -> None:
    image_uri = "data:image/png;base64," + PROBE_IMAGE_B64
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Return exactly: OK"},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 8,
        "stream": False,
    }
    request = urllib.request.Request(
        GENERATE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    # NVIDIA /v1/models can list endpoints that 404 for the account; the probe is the contract.
    extract_output_text(_request_json(request, min(timeout, 30.0), api_key))


def available_models(api_key: str, timeout: float = 15.0) -> dict:
    try:
        live = fetch_model_list(api_key, timeout)
        image_model_names = fetch_image_model_names(timeout)
        candidates = rank_models(load_catalog(), live, image_model_names)
        probe_errors = {}
        verified_ids = set()
        for item in candidates:
            try:
                _probe_model(item["id"], api_key, timeout)
                verified_ids.add(item["id"])
            except Exception as exc:
                probe_errors[item["id"]] = _sanitize_error(exc, api_key)
        verified_live = [item for item in live if item["id"] in verified_ids]
        _write_cache(verified_live, image_model_names)
        return {
            "models": [item for item in candidates if item["id"] in verified_ids],
            "stale": False,
            "error": "",
            "probe_errors": probe_errors,
        }
    except Exception as exc:
        cached = _read_cache()
        return {
            "models": rank_models(
                load_catalog(), cached["models"], cached["image_model_names"]
            ),
            "stale": True,
            "error": _sanitize_error(exc, api_key),
            "probe_errors": {},
        }


def _initial_model_choices() -> list[str]:
    cached = _read_cache()
    choices = [
        item["label"]
        for item in rank_models(load_catalog(), cached["models"], cached["image_model_names"])
    ]
    return choices or [CUSTOM_MODEL]


def resolve_model(model: str, custom_model: str) -> str:
    if custom_model.strip():
        return custom_model.strip()
    if model == CUSTOM_MODEL or " | " not in model:
        raise ValueError("Choose a listed model or enter custom_model.")
    model_id = model.split(" | ", 1)[1].removesuffix(" [stale]").strip()
    excluded = load_catalog().get("excluded_models", {})
    if model_id in excluded:
        # 2026-07-09: stale workflow dropdowns kept known-bad NVIDIA IDs; fail before a 404/timeout.
        raise ValueError(
            f"Model {model_id} is excluded: {excluded[model_id]} "
            "Refresh models, pick another listed model, or enter it in custom_model to force it."
        )
    cache = _read_cache()
    verified_ids = {item["id"] for item in cache["models"]} if cache.get("verified") else set()
    if model_id not in verified_ids:
        raise ValueError(
            f"Model {model_id} has not passed the live NVIDIA probe. "
            "Refresh models, pick a verified model, or enter it in custom_model to force it."
        )
    return model_id


def split_sections(text: str) -> dict[str, str]:
    # Vision models often wrap requested headings in Markdown despite the exact-label instruction.
    pattern = re.compile(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*|__)?(?P<label>"
        + "|".join(map(re.escape, LABELS))
        + r"):(?:\*\*|__)?[ \t]*(?P<inline>[^\r\n]*)$"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return {"Caption": text.strip()}
    sections = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        inline = match.group("inline").strip()
        following = text[match.end() : end].strip()
        value = "\n".join(part for part in (inline, following) if part)
        if value:
            sections[match.group("label")] = value
    return sections


def _split_top_level_clauses(text: str) -> list[str]:
    """Split comma/semicolon clauses without breaking parenthesized item lists."""
    clauses, start, depth = [], 0, 0
    for index, char in enumerate(text):
        depth += char == "("
        depth -= char == ")"
        if depth == 0 and char in ",;":
            clauses.append(text[start:index].strip())
            start = index + 1
    clauses.append(text[start:].strip())
    return [clause for clause in clauses if clause]


def _filter_accessory_clause(clause: str) -> str:
    def filter_for_list(match: re.Match) -> str:
        items = re.split(r"\s*,\s*|\s+(?:and|or)\s+", match.group(1), flags=re.I)
        kept = [item.strip() for item in items if item.strip() and not PERSON_ACCESSORY_RE.search(item)]
        return f"(for {', '.join(kept)})" if kept else ""

    had_for_list = bool(re.search(r"\(\s*for\s+", clause, re.I))
    clause = re.sub(r"\(\s*for\s+([^()]*)\)", filter_for_list, clause, flags=re.I).strip()
    if had_for_list and not re.search(r"\(\s*for\s+", clause, re.I):
        return ""
    return "" if PERSON_ACCESSORY_RE.search(clause) else clause


def _filter_person_accessories(content: str) -> str:
    kept = []
    for line in content.splitlines():
        sentences = re.split(r"(?<=[.!?])\s+", line.strip())
        for sentence in sentences:
            clauses = [_filter_accessory_clause(part) for part in _split_top_level_clauses(sentence)]
            filtered = ", ".join(clause for clause in clauses if clause).strip()
            if filtered:
                if sentence.rstrip().endswith(('.', '!', '?')) and not filtered.endswith(('.', '!', '?')):
                    filtered += sentence.rstrip()[-1]
                kept.append(filtered)
    return "\n".join(kept).strip()


def adapt_caption(text: str, no_person_accessories: bool = False) -> str:
    sections = split_sections(text)
    sections.pop("Uncertainty", None)
    if no_person_accessories:
        # Captions do not always keep carried items in Accessories; filter every labelled
        # section clause-by-clause while leaving the raw caption untouched for diagnosis.
        for label, content in tuple(sections.items()):
            value = _filter_person_accessories(content)
            if value:
                sections[label] = value
            else:
                sections.pop(label)
    parts = [PROMPT_PREFIX]
    if no_person_accessories:
        parts.append(NO_PERSON_ACCESSORY_PREFIX)
    parts.extend(f"{label}: {sections[label]}" for label in LABELS if label in sections)
    if "Caption" in sections:
        parts.append(sections["Caption"])
    result = "\n\n".join(parts)
    for pattern, replacement in SOFTENERS:
        result = pattern.sub(replacement, result)
    return result.strip()


def encode_image(image, max_image_mb: float) -> bytes:
    if not hasattr(image, "shape") or len(image.shape) != 4:
        raise ValueError("IMAGE must be a ComfyUI BHWC tensor.")
    if int(image.shape[0]) != 1:
        raise ValueError("Image batches are not supported; connect exactly one IMAGE.")
    pixels = (image[0].detach().cpu().clamp(0, 1) * 255).byte().numpy()
    if pixels.shape[-1] not in (1, 3, 4):
        raise ValueError("IMAGE must have 1, 3, or 4 channels.")
    if pixels.shape[-1] == 1:
        pixels = pixels[:, :, 0]
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    data = output.getvalue()
    limit = int(max_image_mb * 1024 * 1024)
    if len(data) > limit:
        raise ValueError(
            f"Encoded image is {len(data) / 1024 / 1024:.1f} MiB; limit is {max_image_mb:.1f} MiB."
        )
    return data


def extract_output_text(payload: dict) -> str:
    parts = []
    reasoning = []
    finish_reasons = []
    for choice in payload.get("choices", []):
        if isinstance(choice.get("finish_reason"), str):
            finish_reasons.append(choice["finish_reason"])
        message = choice.get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        elif isinstance(content, list):
            parts.extend(
                item["text"].strip()
                for item in content
                if isinstance(item.get("text"), str) and item["text"].strip()
            )
        for key in ("reasoning", "reasoning_content"):
            if isinstance(message.get(key), str) and message[key].strip():
                reasoning.append(message[key].strip())
    if parts:
        return "\n".join(parts)
    if reasoning:
        suffix = f" finish_reason={','.join(finish_reasons)}" if finish_reasons else ""
        raise RuntimeError(f"NVIDIA returned reasoning but no final caption;{suffix}.")
    raise RuntimeError("NVIDIA response did not contain final text output.")


def generate_caption(
    image_bytes: bytes, model: str, api_key: str, timeout: float, max_tokens: int
) -> str:
    image_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ],
            }
        ],
        "temperature": 0.1,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        GENERATE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    return extract_output_text(_request_json(request, timeout, api_key))


class NvidiaNimOutfitCaption:
    """Caption one outfit image and return both raw QA text and a VTON-ready prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Native combo stays a dropdown while the browser refresh replaces its choices.
                "model": (_initial_model_choices(),),
                "custom_model": ("STRING", {"default": "", "multiline": False}),
                "api_key_env": ("STRING", {"default": DEFAULT_ENV, "multiline": False}),
                "timeout_seconds": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 600.0, "step": 1.0}),
                "max_image_mb": ("FLOAT", {"default": 14.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 65536, "step": 1}),
                "no_person_accessories": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("raw_caption", "vton_prompt")
    FUNCTION = "caption"
    CATEGORY = "image/captioning"

    def caption(
        self,
        image,
        model: str,
        custom_model: str,
        api_key_env: str,
        timeout_seconds: float,
        max_image_mb: float,
        max_tokens: int,
        no_person_accessories: bool,
        unique_id: str | None = None,
    ):
        api_key = _SESSION_KEYS.get(str(unique_id), "") or os.environ.get(api_key_env.strip(), "")
        if not api_key:
            raise RuntimeError(f"Enter a session API key or set {api_key_env.strip() or DEFAULT_ENV}.")
        model_id = resolve_model(model, custom_model)
        raw_caption = generate_caption(
            encode_image(image, max_image_mb), model_id, api_key, timeout_seconds, max_tokens
        )
        return raw_caption, adapt_caption(raw_caption, no_person_accessories)


NODE_CLASS_MAPPINGS = {"NvidiaNimOutfitCaption": NvidiaNimOutfitCaption}
NODE_DISPLAY_NAME_MAPPINGS = {"NvidiaNimOutfitCaption": "NVIDIA NIM Outfit Caption"}


# ComfyUI loads server first; avoiding a fresh server import keeps standalone checks lightweight.
if "server" in sys.modules:
    from aiohttp import web
    from server import PromptServer

    _prompt_server = getattr(PromptServer, "instance", None)
else:
    _prompt_server = None

if _prompt_server is not None:

    @_prompt_server.routes.post("/nvidia-nim-outfit-caption/session")
    async def configure_session(request):
        data = await request.json()
        node_id = str(data.get("node_id", "")).strip()
        if not node_id or len(node_id) > 100:
            return web.json_response({"error": "Invalid node id.", "models": [], "stale": True}, status=400)
        api_key = str(data.get("api_key", "")).strip()
        if api_key:
            _SESSION_KEYS[node_id] = api_key
        else:
            _SESSION_KEYS.pop(node_id, None)
        env_name = str(data.get("api_key_env", DEFAULT_ENV)).strip() or DEFAULT_ENV
        resolved_key = _SESSION_KEYS.get(node_id, "") or os.environ.get(env_name, "")
        try:
            timeout = max(1.0, min(float(data.get("timeout_seconds", 15.0)), 600.0))
        except (TypeError, ValueError):
            timeout = 15.0
        if not resolved_key:
            cache = _read_cache()
            cached = rank_models(
                load_catalog(), cache["models"], cache["image_model_names"]
            )
            return web.json_response(
                {"models": cached, "stale": True, "error": f"Enter a session key or set {env_name}."}
            )
        return web.json_response(available_models(resolved_key, timeout))


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
