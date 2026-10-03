"""Client for the teacher model.

Talks to Ollama through its native API when it detects Ollama (that is the only way to reliably switch a
model's "thinking" on/off, fix the context size and read GPU/CPU placement), and to any other
OpenAI-compatible server (llama.cpp server, LM Studio, vLLM) through /v1/chat/completions.
"""
from __future__ import annotations

import threading
import time

import httpx

from .verifiers import split_think


class TeacherError(RuntimeError):
    pass


def _root(base_url: str) -> str:
    base = str(base_url).rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


class Teacher:
    def __init__(self, cfg: dict):
        t = cfg["teacher"]
        self.base_url = str(t["base_url"]).rstrip("/")
        self.root = _root(self.base_url)
        self.model = t["model"]
        self.api_key = t.get("api_key") or "none"
        self.temperature = float(t.get("temperature", 0.7))
        self.top_p = float(t.get("top_p", 0.95))
        self.max_tokens = int(t.get("max_tokens", 2048))
        self.num_ctx = int(t.get("num_ctx") or 0)          # Ollama context window; 0 = server default
        self.default_think = t.get("think")                 # None = let the caller / model decide
        self.timeout = float(t.get("timeout", 600))
        self.retries = int(t.get("retries", 3))
        self.extra_body = dict(t.get("extra_body") or {})
        self.api = str(t.get("api") or "auto").lower()       # auto | ollama | openai
        self._think_supported: bool | None = None
        self._lock = threading.Lock()
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.timeout, connect=15),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def close(self):
        self._client.close()

    # ------------------------------------------------------------------ backend detection
    def is_ollama(self) -> bool:
        with self._lock:
            if self.api == "auto":
                try:
                    r = self._client.get(f"{self.root}/api/version", timeout=5)
                    self.api = "ollama" if r.status_code == 200 and "version" in r.json() else "openai"
                except Exception:
                    self.api = "openai"
            return self.api == "ollama"

    def list_models(self) -> list[str]:
        r = self._client.get(f"{self.base_url}/models", timeout=15)
        r.raise_for_status()
        return [m.get("id") for m in r.json().get("data", [])]

    def gpu_placement(self) -> dict | None:
        """Ollama only: how much of the loaded teacher sits in VRAM (anything on CPU is ~5-10x slower)."""
        if not self.is_ollama():
            return None
        try:
            r = self._client.get(f"{self.root}/api/ps", timeout=10)
            for m in r.json().get("models", []):
                if m.get("name") == self.model or m.get("model") == self.model:
                    size, vram = int(m.get("size") or 0), int(m.get("size_vram") or 0)
                    return {"size_gb": round(size / 1e9, 2), "vram_gb": round(vram / 1e9, 2),
                            "gpu_fraction": round(vram / size, 3) if size else None,
                            "context": m.get("context_length")}
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------ chat
    def chat(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None,
             think: bool | None = None) -> dict:
        temp = self.temperature if temperature is None else temperature
        toks = max_tokens or self.max_tokens
        think = self.default_think if think is None else think
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                if self.is_ollama():
                    return self._chat_ollama(messages, temp, toks, think)
                return self._chat_openai(messages, temp, toks)
            except (httpx.HTTPError, TeacherError, KeyError, ValueError) as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(min(30, 2 ** attempt * 2))
        raise TeacherError(f"teacher request failed after {self.retries + 1} tries: {last_err}")

    def _chat_ollama(self, messages, temp, toks, think) -> dict:
        options = {"temperature": temp, "top_p": self.top_p, "num_predict": toks}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        body: dict = {"model": self.model, "messages": messages, "stream": False, "options": options, "keep_alive": "30m"}
        if think is not None and self._think_supported is not False:
            body["think"] = bool(think)
        extra = dict(self.extra_body)
        options.update(extra.pop("options", {}) or {})
        body.update(extra)
        t0 = time.perf_counter()
        r = self._client.post(f"{self.root}/api/chat", json=body)
        if r.status_code == 400 and "think" in body and "think" in r.text.lower():
            self._think_supported = False  # model has no thinking switch - ask again without it
            body.pop("think")
            r = self._client.post(f"{self.root}/api/chat", json=body)
        elapsed = time.perf_counter() - t0
        if r.status_code >= 400:
            raise TeacherError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        msg = data.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("thinking") or ""
        if not reasoning and ("<think>" in content or "</think>" in content):
            reasoning, content = split_think(content)
        gen_tokens = int(data.get("eval_count") or 0)
        gen_secs = (data.get("eval_duration") or 0) / 1e9
        return {
            "content": content.strip(),
            "reasoning": reasoning.strip(),
            "finish_reason": "length" if data.get("done_reason") == "length" else (data.get("done_reason") or "stop"),
            "usage": {"completion_tokens": gen_tokens, "prompt_tokens": int(data.get("prompt_eval_count") or 0)},
            "seconds": round(elapsed, 3),
            "gen_tokens_per_second": round(gen_tokens / gen_secs, 1) if gen_secs else None,
        }

    def _chat_openai(self, messages, temp, toks) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": temp, "top_p": self.top_p,
                "max_tokens": toks, "stream": False}
        body.update(self.extra_body)
        t0 = time.perf_counter()
        r = self._client.post(f"{self.base_url}/chat/completions", json=body)
        elapsed = time.perf_counter() - t0
        if r.status_code >= 400:
            raise TeacherError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if not reasoning and ("<think>" in content or "</think>" in content):
            reasoning, content = split_think(content)
        usage = data.get("usage") or {}
        n = usage.get("completion_tokens")
        return {
            "content": content.strip(),
            "reasoning": (reasoning or "").strip(),
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "seconds": round(elapsed, 3),
            "gen_tokens_per_second": round(n / elapsed, 1) if n and elapsed else None,
        }


def check_teacher(cfg: dict) -> dict:
    """Used by the CLI and the UI 'Test speed' button."""
    t = Teacher(cfg)
    out: dict = {"base_url": t.base_url, "model": t.model}
    try:
        try:
            out["models"] = t.list_models()
        except Exception as e:
            out["models"] = []
            out["models_error"] = str(e)
        t.retries = 0
        prompt = [{"role": "user", "content": "Write a Python function that checks whether a number is prime."}]
        res = t.chat(prompt, temperature=0, max_tokens=256, think=False)
        out.update({
            "ok": True,
            "api": "ollama" if t.is_ollama() else "openai",
            "reply": res["content"][:200],
            "has_reasoning": bool(res["reasoning"]),
            "seconds": res["seconds"],
            "tokens_per_second": res.get("gen_tokens_per_second"),
        })
        place = t.gpu_placement()
        if place:
            out["gpu"] = place
            if place.get("gpu_fraction") is not None and place["gpu_fraction"] < 0.97:
                out["warning"] = (f"only {place['gpu_fraction']:.0%} of the teacher fits in VRAM - the rest runs on the CPU, "
                                  "which makes it several times slower. A smaller teacher (e.g. qwen3.5:4b) fits entirely.")
    except Exception as e:
        out.update({"ok": False, "error": str(e)})
    finally:
        t.close()
    return out


def unload_teacher(cfg: dict) -> bool:
    """Free the teacher's VRAM before training. Works for Ollama (keep_alive=0); other servers must be stopped by hand."""
    t = cfg["teacher"]
    try:
        r = httpx.post(f"{_root(t['base_url'])}/api/generate", json={"model": t["model"], "keep_alive": 0}, timeout=20)
        return r.status_code < 400
    except Exception:
        return False
