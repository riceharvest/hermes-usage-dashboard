"""OpenRouter pricing lookup — cheapest per-token rates, cached locally.

Hermes does not persist real costs (actual_cost_usd is NULL in state.db), so
the dashboard prices each (provider, model) by looking the model up on
OpenRouter's /api/v1/models and using its per-token input / output /
cache_read / cache_write rates. Only models that have an OpenRouter route get
priced; everything else is reported as n/a so we never fabricate a number.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE_PATH = os.path.join(
    os.path.expanduser(os.environ.get("HERMES_HOME", "~/.hermes")),
    "cache",
    "openrouter_pricing.json",
)
CACHE_TTL = 24 * 3600  # refresh once a day


@dataclass(frozen=True)
class Price:
    prompt: float          # $/token non-cached input
    completion: float      # $/token output
    cache_read: float      # $/token cached input
    cache_write: float     # $/token cache write (falls back to prompt when absent)
    source: str = "openrouter"


_lock = threading.Lock()
_cache: Optional[dict[str, Price]] = None
_cache_loaded_at: float = 0.0


def _fetch_models() -> dict[str, Price]:
    req = urllib.request.Request(
        OPENROUTER_MODELS_URL, headers={"User-Agent": "hermes-usage-dashboard"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    out: dict[str, Price] = {}
    for m in payload.get("data", []):
        mid = m.get("id")
        if not mid:
            continue
        p = m.get("pricing", {}) or {}
        try:
            prompt = float(p.get("prompt", "0") or "0")
            completion = float(p.get("completion", "0") or "0")
            cache_read = float(p.get("input_cache_read", "0") or "0")
            # OpenRouter has no explicit cache_write field in most entries;
            # cache writes are billed at the prompt rate.
            cache_write = float(p.get("input_cache_write", "0") or "0") or prompt
        except (TypeError, ValueError):
            continue
        out[mid] = Price(prompt, completion, cache_read, cache_write, "openrouter")
    return out


def _load_cache_file() -> Optional[dict[str, Price]]:
    try:
        if not os.path.exists(CACHE_PATH):
            return None
        with open(CACHE_PATH) as f:
            raw = json.load(f)
        return {k: Price(**v) for k, v in raw.get("prices", {}).items()}
    except Exception:
        return None


def _save_cache_file(prices: dict[str, Price]) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(
                {"prices": {k: dict(v._asdict()) for k, v in prices.items()}},
                f,
            )
    except Exception:
        pass


def get_prices(force: bool = False) -> dict[str, Price]:
    """Return {openrouter_model_id: Price}, using local cache when fresh."""
    global _cache, _cache_loaded_at
    with _lock:
        now = time.time()
        if not force and _cache is not None and (now - _cache_loaded_at) < CACHE_TTL:
            return _cache
        # try cache file
        if not force and _cache is None:
            cached = _load_cache_file()
            if cached is not None:
                _cache = _ensure_free_entries(cached)
                _cache_loaded_at = now
                return _cache
        try:
            prices = _fetch_models()
            if prices:
                # Unreleased experimental models (owl/elephant-alpha) had no
                # catalog entry; user confirmed they were free -> expose a $0
                # price so their usage is attributed rather than dropped.
                for fid in ("openrouter/owl-alpha", "openrouter/elephant-alpha"):
                    if fid not in prices:
                        prices[fid] = Price(0.0, 0.0, 0.0, 0.0, "unreleased-free")
                _cache = prices
                _cache_loaded_at = now
                _save_cache_file(prices)
                return _cache
        except Exception:
            # network failed — fall back to stale cache if we have one
            if _cache is not None:
                return _cache
            cached = _load_cache_file()
            if cached is not None:
                _cache = _ensure_free_entries(cached)
                _cache_loaded_at = now
                return _cache
        return _cache or {}


def _ensure_free_entries(prices: dict[str, Price]) -> dict[str, Price]:
    """Guarantee unreleased-free $0 entries exist (catalog may omit them)."""
    for fid in ("openrouter/owl-alpha", "openrouter/elephant-alpha"):
        if fid not in prices:
            prices[fid] = Price(0.0, 0.0, 0.0, 0.0, "unreleased-free")
    return prices


# Curated model-string -> OpenRouter id aliases. These cover native-provider
# names and local GGUF filenames that don't auto-normalize. Values are the
# actual OpenRouter model ids (verified against /api/v1/models). Local GGUF
# files are priced at their normal OpenRouter equivalent (user: "take the
# pricing of the normal OR version").
STATIC_ALIASES = {
    # free tiers -> price them anyway (OR free ids exist, cost $0; if not, the
    # paid equivalent is used so the usage is attributed rather than dropped)
    "deepseek-v4-flash-free": "deepseek/deepseek-v4-flash",
    "nemotron-3-super-120b-a12b:free": "nvidia/nemotron-3-super-120b-a12b:free",
    # kimi: "2.7 is just that, remove the code" -> OR id keeps -code
    "kimi-k2.7-code": "moonshotai/kimi-k2.7-code",
    "kimi-k2.7": "moonshotai/kimi-k2.7-code",
    "kimi-coding/kimi-k2.7-code": "moonshotai/kimi-k2.7-code",
    "kimi-for-coding/kimi-k2.7-code": "moonshotai/kimi-k2.7-code",
    # mimo v2 deprecated -> current OR id
    "mimo-v2-flash": "xiaomi/mimo-v2.5",
    "xiaomi/mimo-v2-flash": "xiaomi/mimo-v2.5",
    # gemini (provider was "unknown")
    "gemini-3-flash": "google/gemini-3-flash-preview",
    # qwen3.5 -> qwen3.6 (user: "for qwen3.5 there is a qwen3.6 now")
    "qwen3.5-36b": "qwen/qwen3.6-35b-a3b",
    "qwen3.5:0.8b": "qwen/qwen3.6-flash",
    "qwen35": "qwen/qwen3.6-plus",
    "qwen/qwen3.5": "qwen/qwen3.6-plus",
    "Qwen3.5-9B-UD-Q4_K_XL.gguf": "qwen/qwen3.6-27b",
    "Qwen3.5-9B-Q3_K_M.gguf": "qwen/qwen3.6-27b",
    "hermes-qwen3.5-35b-a3b": "qwen/qwen3.6-35b-a3b",
    "Qwen/Qwen3.5-36B": "qwen/qwen3.6-35b-a3b",
    "kimi-for-coding": "moonshotai/kimi-k2.7-code",
    # local GGUF -> normal OR version pricing
    "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf": "qwen/qwen3.6-35b-a3b",
    "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf": "google/gemma-4-26b-a4b-it",
    "LFM2.5-8B-A1B-Q4_K_M.gguf": "liquid/lfm-2.5-1.2b-instruct:free",
    "LFM-2.5-8B-1B-Hermes-Tuned-Q4KXL.gguf": "liquid/lfm-2.5-1.2b-instruct:free",
    "carnice-9b": None,  # local, no OR equivalent
    "ornith-9b-mtp-q4km": None,  # no OR equivalent -> stays n/a
    "local-model": None,  # unknown local
    "GPT-5.3-Codex-Spark": "openai/gpt-5.5",  # best-effort codex-equivalent
    "nemotron-3-nano:4b": "nvidia/nemotron-3-nano-9b-v2:free",
    # unreleased experimental (owl/elephant) -> free, explicitly $0
    "openrouter/owl-alpha": "openrouter/owl-alpha",
    "openrouter/elephant-alpha": "openrouter/elephant-alpha",
    "openrouter:openrouter/elephant-alpha": "openrouter/elephant-alpha",
    # free variants -> paid equivalent
    "minimax-m2.5-free": "minimax/minimax-m2.5",
    "mimo-v2-pro-free": "xiaomi/mimo-v2.5-pro",
    "MiMo-V2-Pro-Free": "xiaomi/mimo-v2.5-pro",
    "arcee-ai/trinity-large-preview:free": "arcee-ai/trinity-large-thinking",
    "anthropic/claude-sonnet-4-20250514": "anthropic/claude-sonnet-4",
    "nemotron-3-nano:4b": "nvidia/nemotron-3-nano-30b-a3b:free",
}


def _normalize_model(model: str) -> list[str]:
    """Yield candidate OpenRouter ids for a Hermes model string.

    Order: curated STATIC_ALIASES first (exact), then normalized candidates
    (lowercased, provider-prefixed, prefix-rewritten, every known OR provider
    tried). Returns [] only for models with no possible route.
    """
    m = (model or "").strip()
    if not m:
        return []

    cands: list[str] = []

    # 1) exact curated alias (may be None -> explicit n/a)
    if m in STATIC_ALIASES:
        alias = STATIC_ALIASES[m]
        if alias is None:
            return []
        return [alias]

    # 2) -free suffix -> strip and resolve the paid equivalent
    if m.endswith("-free") or m.endswith(":free"):
        base = m[:-len("-free")] if m.endswith("-free") else m[:-len(":free")]
        cands.append(base)
        cands.append(base + ":free")

    low = m.lower()
    cands.append(low)
    if "/" in low:
        bare = low.split("/", 1)[1]
        cands.append(bare)
    else:
        bare = low

    # native-provider prefix rewrites
    prefix_map = {
        "x-ai/": "x-ai/", "xai/": "x-ai/",
        "openai-codex/": "openai/", "codex/": "openai/",
        "minimax/": "minimax/", "kimi/": "moonshot/",
        "kimi-for-coding/": "moonshot/", "kimi-coding/": "moonshot/",
        "deepseek/": "deepseek/", "qwen/": "qwen/",
        "xiaomi/": "xiaomi/", "google/": "google/",
        "anthropic/": "anthropic/",
    }
    for native_prefix, or_prefix in prefix_map.items():
        if low.startswith(native_prefix):
            cands.append(or_prefix + bare)

    # try the bare name under every known OR provider
    for p in ("openai", "anthropic", "google", "deepseek", "minimax",
              "moonshot", "x-ai", "qwen", "xiaomi", "liquid", "nvidia",
              "moonshotai", "meta-llama", "nousresearch"):
        cands.append(f"{p}/{bare}")

    seen = set()
    out = []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def load_overrides(path: Optional[str] = None) -> dict[str, str]:
    """Load user model->OpenRouter-id price overrides from JSON."""
    if path is None:
        path = os.path.join(
            os.path.expanduser(os.environ.get("HERMES_HOME", "~/.hermes")),
            "cache",
            "usage_dashboard_overrides.json",
        )
    try:
        with open(path) as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in data.get("overrides", {}).items()}
    except Exception:
        return {}


def save_overrides(overrides: dict[str, str], path: Optional[str] = None) -> None:
    if path is None:
        path = os.path.join(
            os.path.expanduser(os.environ.get("HERMES_HOME", "~/.hermes")),
            "cache",
            "usage_dashboard_overrides.json",
        )
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"overrides": overrides}, f, indent=2)
    except Exception:
        pass


def price_for(
    model: Optional[str],
    prices: dict[str, Price],
    overrides: Optional[dict[str, str]] = None,
) -> Optional[Price]:
    """Resolve a Hermes model string to an OpenRouter Price.

    Applies user overrides first (a model string -> OpenRouter id chosen in
    the UI), then curated STATIC_ALIASES, then normalized candidates.
    Returns None only when the model truly has no OpenRouter route and no
    override was set.
    """
    if not model:
        return None
    if overrides:
        or_id = overrides.get(model)
        if or_id is not None:
            return prices.get(or_id)  # may be None if id invalid -> treated as n/a
    for cand in _normalize_model(model):
        p = prices.get(cand)
        if p is not None:
            return p
    return None


def compute_cost(
    price: Optional[Price],
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> Optional[float]:
    """Cost in USD using cached vs non-cached input separately. None if no price."""
    if price is None:
        return None
    non_cached_input = max(0, input_tokens - cache_read_tokens)
    cost = (
        non_cached_input * price.prompt
        + cache_read_tokens * price.cache_read
        + cache_write_tokens * price.cache_write
        + output_tokens * price.completion
    )
    return cost


if __name__ == "__main__":
    pr = get_prices(force=True)
    print(f"loaded {len(pr)} openrouter prices")
    for mid in ["openai/gpt-5.5", "deepseek/deepseek-v4-flash", "minimax/minimax-m2.7"]:
        print(mid, pr.get(mid))
