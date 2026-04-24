"""
Hotel Booking Voice Agent — minimal demo
========================================

Same LiveKit + AssemblyAI + Cartesia + OpenAI stack as a typical
production voice agent, stripped of recording, logging, and tool calls
so the only thing on display is ``stt_node`` filtering.

The filter drops STT events whose transcript is entirely backchannel /
filler tokens (e.g. "mhm", "yeah", "okay") while the agent is speaking.
Downstream turn detection, interrupt gates, and preemptive generation
never see those events, so the agent doesn't get cut off mid-sentence.
Events pass through unchanged when the agent is not speaking — a bare
"yeah" at listening time still lands normally.

Note on VAD: this demo sets ``vad=None`` on purpose. With VAD enabled
and ``interruption.enabled=True``, Silero's speaking signal pauses TTS
and flips ``session.agent_state`` to ``"listening"`` before the STT
transcript reaches our filter — so the "speaking" branch never fires
and the filter can't do its job. Turn detection still works fine
without VAD via ``EnglishModel``; interrupts come from committed STT
turns (``on_end_of_turn``), which run after ``stt_node`` yields.
"""

from __future__ import annotations

import string
from collections.abc import AsyncIterable
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, room_io, stt
from livekit.agents.voice import ModelSettings
from livekit.plugins import (
    assemblyai,
    cartesia,
    noise_cancellation,
    openai,
)
from livekit.plugins.turn_detector.english import EnglishModel

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
    def __init__(self) -> None:
        super().__init__(instructions=HOTEL_SYSTEM_PROMPT)

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
        #   1. Only filter while the agent is actively speaking — at
        #      listening / thinking time a bare "yeah" is a real
        #      confirmation and must flow through.
        #   2. Only touch transcript events (INTERIM / PREFLIGHT / FINAL).
        #      START_OF_SPEECH / END_OF_SPEECH carry no text and the
        #      downstream state machine needs them.
        #   3. Drop only when the transcript is entirely filler.
        if self.session.agent_state != "speaking":
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
            min_turn_silence=200,
            max_turn_silence=1500,
            vad_threshold=0.3,
            keyterms_prompt=[
                "SpringHill",
                "TownePlace",
                "WoodSpring",
                "DoubleTree",
                "Hilton",
                "Marriott",
                "Bonvoy",
                "suite",
                "valet",
            ],
        ),
        tts=cartesia.TTS(model="sonic-3", voice="607167f6-9bf2-473c-accc-ac7b3b66b30b"),
        llm=openai.LLM(model="gpt-4.1-nano"),
        # VAD is deliberately disabled — see module docstring. Turn
        # detection is handled by EnglishModel below; interrupts come
        # out of the STT-commit path (on_end_of_turn).
        vad=None,
        turn_handling={
            "turn_detection": EnglishModel(),
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
        agent=HotelBookingAgent(),
        # Uncomment for telephony calls that need noise cancellation.
        # Requires the `livekit-plugins-noise-cancellation` package.
        # room_options=room_io.RoomOptions(
        #     audio_input=room_io.AudioInputOptions(
        #         noise_cancellation=noise_cancellation.BVCTelephony(),
        #     ),
        # ),
    )

    await session.generate_reply(
        instructions='Say exactly: "I\'m from The Grandview Hotel, how can I help?"'
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="hotel-booking-agent",
        )
    )
