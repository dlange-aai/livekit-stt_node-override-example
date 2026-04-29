"""
Backchannel STT filter — mixin form.

Wraps ``Agent.stt_node`` to drop transcript events whose tokens are all
known backchannels (e.g. "mhm", "yeah", "okay") while the agent is
speaking, plus a short grace window after. Sits *upstream* of LK's
``audio_recognition``: a dropped event is as if STT never emitted it,
so no buffer accumulates, no preempt-generation kicks off, no
interrupt fires.

Why a mixin: ``stt_node`` is a method on ``Agent``, so the natural way
to wrap it is via subclass + MRO. A standalone function would have to
monkey-patch the bound method, which breaks ``super()`` chains and
doesn't compose with other ``stt_node`` wrappers. (Compare
``short_utterance_buffer.install_short_utterance_filter``, which is a
function because *its* integration point — ``session.on(...)`` — is
event subscription, not method override.)

Drop into your project as a single file. The mixin imports only
``livekit.{rtc, agents}`` + stdlib — no other deps. Compose with your
own ``Agent`` subclass:

    class MyAgent(BackchannelSTTFilterMixin, Agent):
        ...

The mixin's ``stt_node`` is found first via MRO and forwards the
non-filtered events to ``Agent.default.stt_node``.
"""

from __future__ import annotations

import logging
import string
import time
from collections.abc import AsyncIterable

from livekit import rtc
from livekit.agents import Agent, stt
from livekit.agents.voice import ModelSettings


# Edit for your domain. "yes" / "no" deliberately omitted — in a
# booking flow a bare "yes" is a real confirmation.
BACKCHANNELS = frozenset({
    "mhm", "mm", "mmhm", "mmhmm",
    "uh", "uhhuh", "huh",
    "um", "umm", "uhm",
    "er", "erm",
    "hmm", "hm",
    "ah", "oh",
    "yeah", "yep", "yup",
    "okay", "ok",
    "right", "alright", "gotcha",
})

_TRANSCRIPT_TYPES = {
    stt.SpeechEventType.INTERIM_TRANSCRIPT,
    stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
    stt.SpeechEventType.FINAL_TRANSCRIPT,
}

_PUNCT_STRIP = str.maketrans("", "", string.punctuation)

log = logging.getLogger("backchannel_stt_filter")


def _is_all_backchannel(text: str) -> bool:
    # True only if the transcript has content AND every token is a known
    # filler. One non-filler word anywhere ("yeah I want the suite") flips
    # this to False so real intents are never dropped.
    tokens = text.lower().translate(_PUNCT_STRIP).split()
    return bool(tokens) and all(tok in BACKCHANNELS for tok in tokens)


class BackchannelSTTFilterMixin:
    # Keep filtering this long after agent_state flips off "speaking".
    # Covers AssemblyAI's late FINAL_TRANSCRIPT — without this, a filler
    # uttered right before TTS ends can still commit as a user turn a
    # moment later.
    _FILTER_GRACE_S: float = 1.0

    # Class-level default; set per-instance on first observation of
    # agent_state == "speaking".
    _last_speaking_at: float = 0.0

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ):
        # Filter event stream before it reaches downstream consumers
        # (audio_recognition -> turn detection, interrupt gates,
        # preemptive generation). Dropping a transcript event here means
        # it's as if the STT never emitted it.
        async for ev in Agent.default.stt_node(self, audio, model_settings):
            if self._should_drop(ev):
                text = ev.alternatives[0].text if ev.alternatives else ""
                log.info(
                    "event_filtered transcript=%r ev_type=%s agent_state=%s",
                    text, ev.type, self.session.agent_state,
                )
                continue
            yield ev

    def _should_drop(self, ev: stt.SpeechEvent) -> bool:
        # Three gates, cheapest first:
        #   1. Filter while agent is speaking OR within grace — past
        #      grace, a bare "yeah" is a real confirmation.
        #   2. Only touch transcript events. START/END_OF_SPEECH carry
        #      no text and the downstream state machine needs them.
        #   3. Drop only when the transcript is entirely filler.
        now = time.monotonic()
        if self.session.agent_state == "speaking":
            self._last_speaking_at = now
        elif now - self._last_speaking_at > self._FILTER_GRACE_S:
            return False
        if ev.type not in _TRANSCRIPT_TYPES:
            return False
        text = ev.alternatives[0].text if ev.alternatives else ""
        return _is_all_backchannel(text)
