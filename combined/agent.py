"""
Hotel Booking Voice Agent — combined variant
============================================

Runs both interruption-mitigation strategies on a single agent:

1. **Backchannel STT filter** (``BackchannelSTTFilterMixin``) — wraps
   ``Agent.stt_node`` and drops transcript events whose tokens are all
   known fillers ("mhm", "um", "yeah", …). This catches the common
   case upstream of ``audio_recognition`` so no buffer accumulates and
   no preempt-generation kicks off.

2. **Short-utterance buffer-clearer**
   (``install_short_utterance_filter``) — listens on
   ``user_input_transcribed`` and wipes LK's accumulation buffers when
   the utterance is below ``interruption.min_words``. This mops up
   *unknown* short utterances the filler list doesn't cover (a new
   stutter, a cough that transcribed as "k", a bare one-word probe
   like "wait").

Layered defense: filter 1 stops what we can name, filter 2 catches
what we can't. They overlap on short fillers (filter 1 wins because it
runs upstream), but the combined coverage is broader than either
alone.

The portable filters — the modules you'd copy into your own LiveKit
app — live in ``filters/``. Everything else in this file is
hotel-demo glue you'd replace with your own.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root on sys.path so ``from filters.X import Y`` works regardless
# of the launch directory — the demo lives in a subfolder, the portable
# filter modules don't.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.plugins import (
    assemblyai,
    cartesia,
    openai,
)

from filters.backchannel_stt import BackchannelSTTFilterMixin
from filters.debug_logging import install_session_event_logging, setup_demo_logging
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


class HotelBookingAgent(BackchannelSTTFilterMixin, Agent):
    # Mixin must come before Agent so MRO finds its stt_node first.
    def __init__(self) -> None:
        super().__init__(instructions=HOTEL_SYSTEM_PROMPT)


async def entrypoint(ctx: agents.JobContext):
    setup_demo_logging()
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
        # VAD off + turn_detection="stt" is the recommended setup for
        # both filters. See README "Why this setup is optimal".
        vad=None,
        turn_handling={
            "turn_detection": "stt",
            "endpointing": {"min_delay": 1.0, "max_delay": 4.0},
            "interruption": {
                "enabled": True,
                "resume_false_interruption": True,
                "false_interruption_timeout": 1.5,
                # Default is 0 (gate disabled). The buffer-clearer's
                # word-count gate reads this, so it must be set
                # explicitly to take effect.
                "min_words": 2,
            },
        },
    )

    install_session_event_logging(session)
    install_short_utterance_filter(session)

    await session.start(room=ctx.room, agent=HotelBookingAgent())

    await session.generate_reply(
        instructions='Say exactly: "I\'m from The Grandview Hotel, how can I help?"'
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
