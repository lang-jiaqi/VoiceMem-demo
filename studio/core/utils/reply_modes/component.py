"""Local reasoning-depth classification; VoiceMem owns memory eligibility."""
from __future__ import annotations

import asyncio
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from studio.harness.reply_modes.policy import SYSTEM, EXAMPLES

FAST = "fast"
MEDIUM = "medium"
SLOW = "slow"
THINKING_LEVELS = (FAST, MEDIUM, SLOW)
_DEFAULT_ROUTER_REPO = "Qwen/Qwen3-0.6B"

def _local_router_ready(path: Path) -> bool:
    """Return whether a local snapshot has config, tokenizer, and weights."""
    has_tokenizer = any((path / name).is_file() for name in (
        "tokenizer.json", "tokenizer.model", "vocab.json"))
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    return (path / "config.json").is_file() and has_tokenizer and has_weights

_ROUTE_LABEL = re.compile(
    r"(即时|记忆|深思|fast|medium|slow)", re.IGNORECASE)
@dataclass(frozen=True)
class ThinkingDecision:
    """Normalized three-way reply route selected for one confirmed user turn."""

    level: str
    raw: str = ""

    @property
    def reasoning_effort(self) -> str:
        # Memory retrieval is not chain-of-thought. Only the slow route enables it.
        return {FAST: "none", MEDIUM: "none", SLOW: "high"}[self.level]

    @property
    def reply_mode(self) -> str:
        return {FAST: "direct", MEDIUM: "memory", SLOW: "memory_cot"}[self.level]

    @property
    def display_name(self) -> str:
        return {FAST: "instant", MEDIUM: "mem", SLOW: "mem+cot"}[self.level]

def parse_level(output: str, fallback: str = FAST) -> ThinkingDecision:
    """Parse a model label, returning an explicit safe fallback on bad output."""
    match = _ROUTE_LABEL.search(output or "")
    labels = {
        "即时": FAST,
        "记忆": MEDIUM,
        "深思": SLOW,
        "fast": FAST,
        "medium": MEDIUM,
        "slow": SLOW,
    }
    level = labels.get(match.group(1).lower(), fallback) if match else fallback
    if level not in THINKING_LEVELS:
        level = FAST
    return ThinkingDecision(level=level, raw=(output or "").strip())

def _router_download_progress_class():
    """Return a lazy tqdm class whose output survives concise Studio logging."""
    from tqdm.auto import tqdm

    class RouterDownloadProgress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.update(
                desc="[status] Router 下载",
                disable=False,
                dynamic_ncols=True,
                file=sys.stdout,
                unit="文件",
            )
            super().__init__(*args, **kwargs)

    return RouterDownloadProgress

class QwenThinkingRouter:
    """Classify ordinary versus deep reasoning with serialized local inference.

    The public class name is retained for existing callers. Memory eligibility
    belongs to VoiceMem and is composed with depth by the Studio routing adapter.
    Only text and bounded conversation context participate in depth caching.
    """

    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        from studio.paths import MODELS
        local = MODELS / "reply-router/Qwen3-0.6B"
        configured = model
        self._local_model_dir = local
        self._download_default = not configured and not _local_router_ready(local)
        self.model_name = configured or str(local)
        self.device_name = device or ''
        self._tokenizer = None
        self._model = None
        self._device = None
        self._load_lock = threading.Lock()
        self.history_messages = 4
        self.history_chars = 320
        self._cache: dict[tuple[str, str], ThinkingDecision] = {}

    def _ensure_model_source(self) -> str:
        """Download the default router with visible progress when it is absent."""
        if not self._download_default:
            return self.model_name

        from huggingface_hub import snapshot_download

        destination = self._local_model_dir
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[status] 三级回复路由缺失，开始下载 {_DEFAULT_ROUTER_REPO} "
              f"→ {destination}", flush=True)
        snapshot_download(
            repo_id=_DEFAULT_ROUTER_REPO,
            local_dir=str(destination),
            tqdm_class=_router_download_progress_class(),
        )
        if not _local_router_ready(destination):
            raise FileNotFoundError(f"router download incomplete: {destination}")
        self.model_name = str(destination)
        self._download_default = False
        print(f"[status] 三级回复路由下载完成：{destination}", flush=True)
        return self.model_name

    def _load(self):
        if self._model is not None:
            return self._tokenizer, self._model, self._device
        with self._load_lock:
            if self._model is not None:
                return self._tokenizer, self._model, self._device
            self._ensure_model_source()
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
            model.to(device).eval()
            self._tokenizer, self._model, self._device = tokenizer, model, device
            print(f"[thinking] router ready: {self.model_name} on {device}", flush=True)
        return self._tokenizer, self._model, self._device

    def _predict(self, system: str, examples, prompt: str) -> str:
        tokenizer, model, device = self._load()
        messages = [{"role": "system", "content": system}]
        for example, label in examples:
            messages.extend((
                {"role": "user", "content": example},
                {"role": "assistant", "content": label},
            ))
        messages.append({"role": "user", "content": prompt})
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(device)
        import torch

        from voicemem.utils.torch_lock import TORCH_LOCK

        with TORCH_LOCK, torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        output = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return output.strip()

    def _context_prompt(self, text: str, history, prefetch_hint: bool) -> str:
        role_names = {"user": "用户", "assistant": "助手"}
        remaining = max(0, getattr(self, "history_chars", 320))
        count = max(0, getattr(self, "history_messages", 4))
        selected = list(history or [])[-count:] if count else []
        share = max(1, remaining // len(selected)) if selected else 0
        lines = []
        for message in reversed(selected):
            if remaining <= 0:
                break
            content = " ".join(str(message.get("content") or "").split())
            if not content:
                continue
            content = content[:min(remaining, share)]
            remaining -= len(content)
            role = role_names.get(str(message.get("role")), "上下文")
            lines.append(f"{role}: {content}")
        lines.reverse()
        history_text = "\n".join(lines) if lines else "无"
        return f"最近对话：\n{history_text}\n当前用户：{text.strip()}"

    def classify(self, text: str, memory_prefetch_hint: bool = False,
                 history=None) -> ThinkingDecision:
        """Classify depth off-loop; the legacy memory hint cannot affect this decision."""
        prompt = self._context_prompt(text, history, memory_prefetch_hint)
        key = (text, prompt)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        def remember(decision: ThinkingDecision) -> ThinkingDecision:
            if len(self._cache) >= 256:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = decision
            return decision

        output = self._predict(
            SYSTEM,
            EXAMPLES,
            prompt,
        )
        label = (output or '').strip().strip('。.!！')
        if label not in {'是', '否'}:
            print('[thinking] 深思判定输出无效，保持普通推理；不改变记忆资格', flush=True)
        decision = ThinkingDecision(SLOW if label == '是' else FAST, raw=label)
        return remember(decision)

    async def classify_async(self, text: str, memory_prefetch_hint: bool = False,
                             history=None) -> ThinkingDecision:
        """Run model inference outside the asyncio/WebSocket thread."""
        return await asyncio.to_thread(
            self.classify, text, memory_prefetch_hint, history)

    def warmup(self) -> ThinkingDecision:
        """Load weights and prime depth classification before accepting input."""
        return self.classify("介绍一下向量数据库", False)
