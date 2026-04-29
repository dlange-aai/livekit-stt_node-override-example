"""
Hotel Booking Voice Agent — buffer-clearing variant
===================================================

Same LiveKit + AssemblyAI + Cartesia + OpenAI stack as the
``stt_node`` override demo, but uses a different technique: listen on
``user_input_transcribed`` and clear LiveKit's internal
audio_recognition buffers in-place whenever a short utterance lands
while the agent is speaking.

Strategy
--------
The ``stt_node`` override drops events upstream of audio_recognition,
gated on a hand-curated backchannel list. This handler lets events
accumulate normally, then wipes the buffers between LK's transcript
emit and its interrupt-gate evaluation — gating purely on the
configured interruption ``min_words``. Same net effect (no interrupt
fires for short utterances) without maintaining a filler list per
domain.

Tradeoff: reaches into private LK attributes
(``_activity._audio_recognition._audio_*``,
``_cancel_preemptive_generation``). Names may shift between
livekit-agents minor versions — pin your version. Confirmed against
``livekit-agents>=1.5``.

The portable filter — the only thing you'd copy into your own LiveKit
app — lives in ``filters/short_utterance_buffer.py``. Everything else
in this file is hotel-demo glue you'd replace with your own.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so ``from filters.X import Y`` works
# regardless of where you launch this script from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.plugins import (
    assemblyai,
    cartesia,
    openai,
)

from filters.short_utterance_buffer import install_short_utterance_filter

# Uncomment for telephony calls that need noise cancellation.
# Requires the `livekit-plugins-noise-cancellation` package.
# from livekit.plugins import noise_cancellation

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


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


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=assemblyai.STT(
            model="u3-rt-pro",
            min_turn_silence=100,
            max_turn_silence=1000,
            vad_threshold=0.3,
        ),
        tts=cartesia.TTS(model="sonic-3", voice="607167f6-9bf2-473c-accc-ac7b3b66b30b"),
        llm=openai.LLM(model="gpt-4.1-nano"),
        # VAD off — with turn_detection="stt", agent_state stays
        # "speaking" through TTS, so the filter's speaking-gate answers
        # reliably and short utterances can be cleared before they
        # trigger an interrupt.
        vad=None,
        turn_handling={
            "turn_detection": "stt",
            "endpointing": {"min_delay": 1.0, "max_delay": 4.0},
            "interruption": {
                "enabled": True,
                "resume_false_interruption": True,
                "false_interruption_timeout": 1.5,
                # Anything shorter than this gets the buffer wiped while
                # the agent is speaking. Default is 0 (off), so we set
                # it explicitly to give the filter something to gate on.
                "min_words": 2,
            },
        },
    )

    install_short_utterance_filter(session)

    await session.start(room=ctx.room, agent=HotelBookingAgent())

    await session.generate_reply(
        instructions='Say exactly: "I\'m from The Grandview Hotel, how can I help?"'
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
