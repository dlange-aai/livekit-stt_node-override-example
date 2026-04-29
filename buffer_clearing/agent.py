"""
Hotel Booking Voice Agent — buffer-clearing variant
===================================================

Same LiveKit + AssemblyAI + Cartesia + OpenAI demo as
``stt_node_override/agent.py`` (identical hotel front-desk persona,
identical session config). The *only* difference is how short user
utterances are kept from interrupting the agent mid-speech.

Strategy
--------
Listen for ``user_input_transcribed`` and, while the agent is speaking,
wipe the three transcript buffers LiveKit's interrupt + preempt gates
read from. With the buffer empty, the next gate check after a short
"mhm" / "um" sees nothing accumulated and doesn't trip an interrupt.

This is Ryan Seams' approach — a simpler alternative to the
``stt_node`` override: instead of filtering events upstream of
audio_recognition, we react after they've landed and reach into LK's
private state to undo the accumulation. Tradeoff: we couple to LK
private API (``_audio_transcript`` / ``_audio_interim_transcript`` /
``_audio_preflight_transcript`` / ``_cancel_preemptive_generation``)
so this example pins ``livekit-agents>=1.5``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.plugins import (
    assemblyai,
    cartesia,
    openai,
)

# Uncomment for telephony calls that need noise cancellation.
# Requires the `livekit-plugins-noise-cancellation` package.
# from livekit.plugins import noise_cancellation

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger("buffer_clearing_agent")


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


def install_short_utterance_filter(session: AgentSession) -> None:
    # Drops short utterances spoken while the agent is talking by clearing
    # LK's accumulated transcript state before the interrupt gate reads it.
    #
    # Why all three buffers:
    #   * _audio_transcript          — cumulative committed finals; read by
    #                                  _interrupt_by_audio_activity to decide
    #                                  whether the user has said enough to
    #                                  interrupt.
    #   * _audio_interim_transcript  — current interim chunk; concatenated
    #                                  with the above to form current_transcript,
    #                                  which the gate actually reads.
    #   * _audio_preflight_transcript — buffer + current preflight; used by
    #                                  on_preemptive_generation to decide
    #                                  whether to spin up a speculative reply.
    #
    # Leaving any of these populated lets the next gate check tip over even
    # though the *current* utterance is short.
    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev) -> None:
        # Gate 1 — only filter while the agent is producing TTS. When the
        # agent is idle, short answers like "yes" / "no" are real turns and
        # must pass through.
        if session.agent_state != "speaking":
            return

        # Gate 2 — let utterances at or above the configured interruption
        # threshold pass. Real intents that happen to be short ("the suite")
        # still interrupt, by design.
        word_count = len(ev.transcript.split())
        min_words = session.options.interruption["min_words"]
        if word_count >= min_words:
            return

        # Reach into LK's private state. getattr+default keeps us safe if
        # the session is mid-init or torn down.
        activity = getattr(session, "_activity", None)
        recognition = getattr(activity, "_audio_recognition", None) if activity else None
        if recognition is None:
            return
        recognition._audio_transcript = ""
        recognition._audio_interim_transcript = ""
        recognition._audio_preflight_transcript = ""

        # If LK already kicked off a preemptive LLM call on the preflight of
        # this short utterance, abort it before TTS starts. Best-effort —
        # if TTS already began, we accept the small race.
        cancel = getattr(activity, "_cancel_preemptive_generation", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                log.debug("_cancel_preemptive_generation failed", exc_info=True)

        log.info(
            "buffer_cleared transcript=%r words=%d is_final=%s agent_state=%s",
            ev.transcript, word_count, ev.is_final, session.agent_state,
        )


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
        # Same vad=None + turn_detection="stt" config as the stt_node_override
        # example. user_input_transcribed dispatches before LK's interrupt
        # gate runs, so clearing the buffers in our handler reliably
        # pre-empts the gate.
        vad=None,
        turn_handling={
            "turn_detection": "stt",
            "endpointing": {"min_delay": 1.0, "max_delay": 4.0},
            "interruption": {
                "enabled": True,
                "resume_false_interruption": True,
                "false_interruption_timeout": 1.5,
                # Anything shorter than this gets the buffer wiped while
                # the agent is speaking. Default is 0 (off), so we set it
                # explicitly to give the filter something to gate on.
                "min_words": 2,
            },
        },
    )

    install_short_utterance_filter(session)

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
