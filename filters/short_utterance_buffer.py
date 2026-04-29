"""
Short-utterance buffer-clearing filter.

Listens on ``user_input_transcribed`` and, while the agent is speaking,
wipes the three transcript buffers LK's interrupt + preempt gates read
from. The handler runs synchronously inside ``audio_recognition``'s
``on_interim_transcript`` / ``on_final_transcript`` hook (line 728 /
line 822 in audio_recognition.py 1.5.6), *before* LK's
``_interrupt_by_audio_activity`` reads ``current_transcript`` — so the
immediate TTS-pause path bails on ``min_words``.

The wipe does not, however, keep ``_audio_transcript`` empty for EOU
detection: audio_recognition appends the FINAL's transcript at line 743
*after* the hook returns, so by EOU time the buffer holds the latest
filler. Suppression of the turn commit comes from
``agent_activity.on_end_of_turn``'s own ``min_words`` check
(agent_activity.py:1871-1883). The wipe's value is preventing
``_audio_transcript`` from accumulating multiple short fillers across
uncommitted turns — without it, two consecutive short fillers sum past
the threshold and break through.

Why a function (and not a mixin): the integration point here is
``session.on("user_input_transcribed")`` — runtime event subscription
on an already-instantiated ``AgentSession``. Nothing to override; just
a callback to attach. (Compare
``backchannel_stt.BackchannelSTTFilterMixin``, which is a class
because *its* integration point — ``Agent.stt_node`` — is a method.)

Drop into your project as a single file. Imports only
``livekit.agents.AgentSession`` typing + stdlib — no other deps.

Couples to LK private API: ``_activity._audio_recognition._audio_*`` and
``_cancel_preemptive_generation``. Requires
``interruption.min_words >= 1`` (the demo uses 2). Pin your
``livekit-agents`` version. Confirmed against ``livekit-agents>=1.5``.
"""

from __future__ import annotations

import logging

from livekit.agents import AgentSession


log = logging.getLogger("short_utterance_buffer_filter")


def install_short_utterance_filter(session: AgentSession) -> None:
    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev) -> None:
        word_count = len(ev.transcript.split())
        min_words = session.options.interruption["min_words"]
        log.debug(
            "event_received transcript=%r is_final=%s words=%d agent_state=%s min_words=%d",
            ev.transcript, ev.is_final, word_count, session.agent_state, min_words,
        )

        if session.agent_state != "speaking":
            log.debug("kept reason='agent idle' agent_state=%s", session.agent_state)
            return

        if word_count >= min_words:
            log.debug(
                "kept reason='word_count >= min_words' words=%d min_words=%d",
                word_count, min_words,
            )
            return

        # getattr+default keeps us safe if the session is mid-init or
        # mid-teardown — _activity / _audio_recognition can be None.
        activity = getattr(session, "_activity", None)
        recognition = getattr(activity, "_audio_recognition", None) if activity else None
        if recognition is None:
            return

        log.debug(
            "buffer_snapshot before_wipe transcript=%r interim=%r preflight=%r",
            recognition._audio_transcript,
            recognition._audio_interim_transcript,
            recognition._audio_preflight_transcript,
        )

        recognition._audio_transcript = ""
        recognition._audio_interim_transcript = ""
        recognition._audio_preflight_transcript = ""

        # Best-effort: aborts any in-flight preemptive LLM call kicked
        # off on this short utterance's preflight. Wrapped because the
        # private method could change shape on a minor-version bump.
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
