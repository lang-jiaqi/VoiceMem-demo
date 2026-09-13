"""Session-scoped policy for backchannels and end-of-turn handoffs."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from studio.harness.reply_modes.policy import MEMORY_COT

from .backchannel import Backchannel

class TurnPhase(str, Enum):
    """Observable phases of one conversational turn."""

    LISTENING = "listening"
    COMMITTED = "committed"
    FILLING = "filling"
    REPLYING = "replying"

class HandoffKind(str, Enum):
    """Ways to bridge the user's end of turn into the main reply."""

    DIRECT = "direct"
    CACHED_ACK = "cached_ack"
    LLM_FILLER = "llm_filler"

@dataclass(frozen=True)
class HandoffDecision:
    kind: HandoffKind
    reason: str
    expected_wait_s: float

@dataclass
class TurnTakingStateMachine:
    """Own turn-taking state and make transport-independent handoff decisions."""

    backchannel: Backchannel = field(default_factory=Backchannel)
    short_ack_after_s: float = 0.8
    long_filler_after_s: float = 1.5
    initial_wait_s: float = 1.0
    estimate_weight: float = 0.35
    echo_window_s: float = 4.0
    phase: TurnPhase = field(default=TurnPhase.LISTENING, init=False)
    _expected_wait_s: float = field(init=False)
    _recent_emissions: list[tuple[float, str]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._expected_wait_s = max(0.0, self.initial_wait_s)

    @property
    def expected_wait_s(self) -> float:
        return self._expected_wait_s

    @property
    def completed_turns(self) -> int:
        return self.backchannel.completed_turns

    def begin_user_turn(self) -> None:
        self.phase = TurnPhase.LISTENING
        self.backchannel.reset_turn()

    def commit_user_turn(self) -> None:
        self.backchannel.complete_turn()
        self.phase = TurnPhase.COMMITTED

    def offer_backchannel(self, **signals) -> str | None:
        if self.phase is not TurnPhase.LISTENING:
            return None
        return self.backchannel.offer(**signals)

    def prepare_cached_ack(self, *, text: str, emotion: str = "",
                           lang: str = "zh", available=None,
                           now: float | None = None) -> str | None:
        if self.phase is not TurnPhase.COMMITTED:
            return None
        return self.backchannel.choose(
            text=text, emotion=emotion, lang=lang, available=available, now=now)

    def record_emission(self, token: str, *, committed: bool = False,
                        now: float | None = None) -> None:
        """Record audio that was actually sent for cooldown and echo rejection."""
        at = now if now is not None else time.monotonic()
        if committed:
            self.backchannel.mark_emitted(token, at)
        self._recent_emissions.append((at, token))
        self._prune_emissions(at)

    def recent_agent_text(self, now: float | None = None) -> str:
        at = now if now is not None else time.monotonic()
        self._prune_emissions(at)
        return "".join(token for _, token in self._recent_emissions)

    def decide_handoff(self, *, main_audio_ready: bool, reply_mode: str,
                       cached_ack_available: bool,
                       expected_wait_s: float | None = None,
                       spoken: bool = True) -> HandoffDecision:
        """Choose direct speech, cached audio, or an LLM-generated work filler."""
        wait = (self._expected_wait_s if expected_wait_s is None
                else max(0.0, expected_wait_s))
        if not spoken:
            return HandoffDecision(HandoffKind.DIRECT, "text_turn", wait)
        if main_audio_ready:
            return HandoffDecision(HandoffKind.DIRECT, "main_audio_ready", wait)
        if reply_mode == MEMORY_COT:
            return HandoffDecision(HandoffKind.LLM_FILLER, "long_work", wait)
        if cached_ack_available and wait >= self.short_ack_after_s:
            return HandoffDecision(HandoffKind.CACHED_ACK, "reply_not_ready", wait)
        return HandoffDecision(HandoffKind.DIRECT, "short_wait", wait)

    def start_handoff(self, decision: HandoffDecision) -> None:
        self.phase = (TurnPhase.FILLING
                      if decision.kind is not HandoffKind.DIRECT
                      else TurnPhase.REPLYING)

    def start_reply(self) -> None:
        self.phase = TurnPhase.REPLYING

    def finish_reply(self) -> None:
        self.phase = TurnPhase.LISTENING

    def observe_first_audio(self, seconds: float) -> None:
        """Update the session estimate used by the next handoff decision."""
        sample = max(0.0, float(seconds))
        weight = min(1.0, max(0.0, self.estimate_weight))
        self._expected_wait_s = weight * sample + (1.0 - weight) * self._expected_wait_s

    def _prune_emissions(self, now: float) -> None:
        self._recent_emissions[:] = [
            item for item in self._recent_emissions
            if now - item[0] < self.echo_window_s
        ]
