"""
Hotel Booking Voice Agent — minimal demo
========================================

Same LiveKit + AssemblyAI + Cartesia + OpenAI stack as a typical
production voice agent, stripped of recording, logging, and tool calls
so the only thing on display is ``stt_node`` filtering.

The filter drops STT events whose transcript is entirely backchannel /
filler tokens (e.g. "mhm", "yeah", "okay") while the agent is speaking
(plus a short grace window after). Without the filter, a mid-speech
"um" commits as a one-word user turn and the LLM dutifully replies
to it.

VAD is deliberately disabled. With ``turn_detection="stt"`` +
``vad=None``, ``agent_state`` stays ``"speaking"`` through TTS and
the filter's "are we speaking?" gate answers reliably. Interrupts
still come from committed STT turns (``on_end_of_turn``), which run
after ``stt_node`` yields — so a real user intent is caught by
AssemblyAI's end-of-turn signal but a filler we dropped never makes
it that far.
"""

from __future__ import annotations

import string
import time
from collections.abc import AsyncIterable
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, stt
from livekit.agents.voice import ModelSettings
from livekit.plugins import (
    assemblyai,
    cartesia,
    openai,
)

# Uncomment for telephony calls that need noise cancellation.
# Requires the `livekit-plugins-noise-cancellation` package.
# from livekit.plugins import noise_cancellation

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


# Backchannels / fillers to drop while the agent is speaking.
# "yes" / "no" are deliberately omitted — they're real confirmations.
# Edit this set to match your domain.
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


def _is_all_backchannel(text: str) -> bool:
    # True only if the transcript has content AND every token is a known
    # filler. One non-filler word anywhere ("yeah I want the suite") flips
    # this to False so real intents are never dropped.
    tokens = text.lower().translate(_PUNCT_STRIP).split()
    return bool(tokens) and all(tok in BACKCHANNELS for tok in tokens)


HOTEL_SYSTEM_PROMPT = """\
You are the front desk at The Grandview Hotel in downtown San Francisco. \
Keep replies short and conversational — one or two sentences. Follow the \
caller's lead: answer what they ask, don't lecture.

Greeting: "I'm from The Grandview Hotel, how can I help?"

Hotel facts (only mention what's asked):
- 850 Market Street, SF. Check-in 3pm, check-out 11am.
- Standard $199/night, Deluxe $299, Suite $499. WiFi and breakfast included.
- Rooftop pool, fitness center, spa, restaurant, valet parking ($45/night).
- Pet-friendly in Standard and Deluxe ($50/night). Airport shuttle $35 each way.

If the caller wants to book, collect one item at a time — don't read the \
whole list up front: name, check-in date, check-out date, guests, room \
type, phone, email, credit card number, expiration, CVV. Read back the \
details, confirm, then wrap up with "You're all set — a confirmation \
email is on its way."

If the caller interrupts or corrects you, go with it. Never restart the \
booking from the top unless they ask.\
"""


class HotelBookingAgent(Agent):
    # Keep filtering for this long after agent_state flips off "speaking".
    # Covers the latency between TTS finishing and AssemblyAI's late
    # FINAL_TRANSCRIPT arriving — without this, a filler uttered right
    # before TTS ends can still commit as a user turn a moment later.
    _FILTER_GRACE_S = 1.0

    def __init__(self) -> None:
        super().__init__(instructions=HOTEL_SYSTEM_PROMPT)
        self._last_speaking_at = 0.0

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ):
        # Wraps the default STT pipeline and filters the event stream before
        # it reaches downstream consumers (audio_recognition -> turn detection,
        # interrupt gates, preemptive generation). Dropping a transcript event
        # here means it's as if the STT never emitted it: no accumulation into
        # the turn buffer, no interrupt, no preempt. Everything else is
        # forwarded untouched.
        async for ev in Agent.default.stt_node(self, audio, model_settings):
            if self._should_drop(ev):
                continue
            yield ev

    def _should_drop(self, ev: stt.SpeechEvent) -> bool:
        # Three gates, cheapest first:
        #   1. Filter while the agent is speaking OR within a short
        #      grace after — past the grace, a bare "yeah" is a real
        #      confirmation and must flow through.
        #   2. Only touch transcript events (INTERIM / PREFLIGHT / FINAL).
        #      START_OF_SPEECH / END_OF_SPEECH carry no text and the
        #      downstream state machine needs them.
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


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=assemblyai.STT(
            model="u3-rt-pro",
            min_turn_silence=100,
            max_turn_silence=1000,
            vad_threshold=0.3
        ),
        tts=cartesia.TTS(model="sonic-3", voice="607167f6-9bf2-473c-accc-ac7b3b66b30b"),
        llm=openai.LLM(model="gpt-4.1-nano"),
        # VAD is disabled — with turn_detection="stt", agent_state
        # stays "speaking" through TTS, so the filter's speaking-gate
        # answers reliably and fillers are dropped before they can
        # trigger an interrupt.
        vad=None,
        turn_handling={
            "turn_detection": "stt",
            "endpointing": {"min_delay": 1.0, "max_delay": 4.0},
            "interruption": {
                "enabled": True,
                "resume_false_interruption": True,
                "false_interruption_timeout": 1.5,
            },
        },
    )

    await session.start(
        room=ctx.room,
        agent=HotelBookingAgent()
    )

    await session.generate_reply(
        instructions='Say exactly: "I\'m from The Grandview Hotel, how can I help?"'
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
