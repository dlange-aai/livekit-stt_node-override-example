"""
Short-utterance buffer-clearing filter.

Listens on ``user_input_transcribed`` and, while the agent is speaking,
wipes the three transcript buffers LK's interrupt + preempt gates read
from. With the buffers empty, the next gate check after a short
utterance ("mhm" / "um") sees nothing accumulated and doesn't trip.

Why a function (and not a mixin): the integration point here is
``session.on("user_input_transcribed")`` — runtime event subscription
on an already-instantiated ``AgentSession``. Nothing to override; just
a callback to attach. The natural shape for "take an object, register
behavior on it" is ``install_X(obj)``. (Compare
``backchannel_stt.BackchannelSTTFilterMixin``, which is a class
because *its* integration point — ``Agent.stt_node`` — is a method.)

Drop into your project as a single file. Imports only
``livekit.agents.AgentSession`` typing + stdlib — no other deps.

Couples to LK private API: ``_activity._audio_recognition._audio_*`` and
``_cancel_preemptive_generation``. Pin your ``livekit-agents`` version.
Confirmed against ``livekit-agents>=1.5``.
"""

from __future__ import annotations

import logging

from livekit.agents import AgentSession


log = logging.getLogger("short_utterance_buffer_filter")


def install_short_utterance_filter(session: AgentSession) -> None:
    # Drops short utterances spoken while the agent is talking by
    # clearing LK's accumulated transcript state before the interrupt
    # gate reads it.
    #
    # Why all three buffers:
    #   * _audio_transcript          — cumulative committed finals; read
    #                                  by _interrupt_by_audio_activity
    #                                  to decide whether the user has
    #                                  said enough to interrupt.
    #   * _audio_interim_transcript  — current interim chunk;
    #                                  concatenated with the above to
    #                                  form current_transcript.
    #   * _audio_preflight_transcript — buffer + current preflight; used
    #                                  by on_preemptive_generation to
    #                                  decide whether to spin up a
    #                                  speculative reply.
    #
    # Leaving any populated lets the next gate check tip over even though
    # the *current* utterance is short.
    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev) -> None:
        # Gate 1 — only filter while the agent is producing TTS. When
        # idle, short answers like "yes" / "no" are real turns.
        if session.agent_state != "speaking":
            return

        # Gate 2 — let utterances at or above the configured threshold
        # pass. Real intents that happen to be short ("the suite")
        # still interrupt, by design.
        word_count = len(ev.transcript.split())
        min_words = session.options.interruption["min_words"]
        if word_count >= min_words:
            return

        # getattr+default keeps us safe if the session is mid-init or
        # torn down.
        activity = getattr(session, "_activity", None)
        recognition = getattr(activity, "_audio_recognition", None) if activity else None
        if recognition is None:
            return
        recognition._audio_transcript = ""
        recognition._audio_interim_transcript = ""
        recognition._audio_preflight_transcript = ""

        # If LK already kicked off a preemptive LLM call on the
        # preflight of this short utterance, abort it before TTS
        # starts. Best-effort — if TTS already began, we accept the
        # small race.
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
