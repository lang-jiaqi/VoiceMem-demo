"""Studio routing implementation."""
import asyncio
import time
from studio.core.utils.reply_modes.initialize import DIRECT, thinking_router
from studio.core.utils.reply_modes.component import FAST, MEDIUM, SLOW, ThinkingDecision
from voicemem import gate
from voicemem.memory_api import build_memory_context
from studio.core.utils.contracts.component import Pending

class Routing:
    async def _ensure_pending_memory(self, pending: Pending, memory_vm) -> None:
        """Populate memory after a final route upgrades a speculative shallow turn."""
        if (pending.route == gate.DEEP
                and getattr(pending.result, "search_mode", "gated") != "gated"):
            return

        def search():
            from voicemem.leftbrain.query_embedding import query_embedding_scope
            with query_embedding_scope():
                classification = memory_vm.classify(pending.text)
                return memory_vm.search(
                    pending.text,
                    slots=classification.slots,
                    entities=classification.entities,
                    emotion=pending.emotion,
                )

        pending.result = await asyncio.to_thread(search)
        pending.memory_context = build_memory_context(pending.result)
        pending.route = gate.DEEP
        pending.replay = self._replay_id(pending.text, pending.result)

    async def route_pending_thinking(self, pending: Pending, memory_vm=None,
                                     history=None) -> Pending:
        """Combine VoiceMem memory eligibility with local reasoning depth off-loop."""
        if not self._THINKING_ROUTER_ON:
            return pending
        started = time.monotonic()
        memory_vm = memory_vm or self.vm
        memory_required = gate.needs_memory(pending.route)
        try:
            decision = await thinking_router().classify_async(
                pending.text, history=history)
        except Exception as exc:
            print(f"[thinking] 深思判断不可用（{type(exc).__name__}），保留 VoiceMem 记忆资格", flush=True)
            decision = ThinkingDecision(FAST, 'unavailable')
        decision = ThinkingDecision(
            SLOW if decision.level == SLOW else (MEDIUM if memory_required else FAST), decision.raw)
        classified = time.monotonic()
        memory_ms = 0.0
        pending.reply_mode = decision.reply_mode
        if pending.stranger:
            # Speaker privacy outranks routing: never expose the owner's retrieved data.
            from voicemem.stream import empty_result
            pending.result = empty_result()
            pending.memory_context = ""
            pending.replay = ""
            pending.route = gate.SHALLOW
        elif decision.reply_mode == DIRECT:
            from voicemem.stream import empty_result
            pending.result = empty_result()
            pending.memory_context = ""
            pending.replay = ""
            pending.route = gate.SHALLOW
        else:
            memory_started = time.monotonic()
            try:
                await self._ensure_pending_memory(pending, memory_vm)
            except Exception as exc:
                # The selected route still reaches the provider with an explicit
                # no-memory directive rather than silently degrading to instant.
                pending.route = gate.DEEP
                pending.memory_context = ""
                print(f"[route] memory retrieval failed: {type(exc).__name__}: {exc}",
                      flush=True)
            memory_ms = (time.monotonic() - memory_started) * 1000
        print(f"[route] {decision.display_name} → {decision.reply_mode}"
              f" / reasoning={decision.reasoning_effort} "
              f"(model={(classified - started) * 1000:.0f}ms"
              f" memory={memory_ms:.0f}ms"
              f" total={(time.monotonic() - started) * 1000:.0f}ms)", flush=True)
        return pending
