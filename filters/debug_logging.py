"""
Demo-only logging helpers.

Two functions, intended to be called from the three demo ``agent.py``
entrypoints in this repo to make filter behavior easy to follow when
testing in a terminal:

- ``setup_demo_logging()`` configures stdlib ``logging`` with a
  human-readable single-line format, turns on DEBUG for the two filter
  loggers, and quiets transport-layer noise.
- ``install_session_event_logging(session)`` taps the LiveKit
  ``AgentSession`` event bus and prints a tagged one-liner per event
  (state changes, transcripts, conversation items, false interruptions,
  speech created, tools, close).

Neither is required to use the portable filters in
``filters/backchannel_stt.py`` and ``filters/short_utterance_buffer.py``.
They live here so the filter modules themselves stay clean — anyone
copying those into their own project doesn't pull in demo-grade
verbose logging.
"""

from __future__ import annotations

import logging

from livekit.agents import AgentSession


_FORMAT = "%(asctime)s.%(msecs)03d %(name)-30s %(message)s"
_DATEFMT = "%H:%M:%S"

_configured = False


def setup_demo_logging(level: int = logging.INFO) -> None:
    # Idempotent: ``dev`` mode reloads the entrypoint module on file
    # change but the Python process persists, so this can be called
    # multiple times in the same run.
    global _configured
    if not _configured:
        logging.basicConfig(level=level, format=_FORMAT, datefmt=_DATEFMT)
        _configured = True

    # Filter modules log decisions at DEBUG by default — enable them
    # so you can see why each event was kept or dropped.
    logging.getLogger("backchannel_stt_filter").setLevel(logging.DEBUG)
    logging.getLogger("short_utterance_buffer_filter").setLevel(logging.DEBUG)
    logging.getLogger("livekit_session_events").setLevel(logging.INFO)

    # Quiet transport / framework chatter so the signal we care about
    # isn't drowned out. Bump these back up if you're debugging LK
    # internals.
    for noisy in ("livekit", "livekit.agents", "httpx", "httpcore", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def install_session_event_logging(session: AgentSession) -> None:
    # Register before any other handlers (e.g. install_short_utterance_filter)
    # so the log line for an event prints BEFORE downstream handlers
    # mutate state.
    log = logging.getLogger("livekit_session_events")

    def _safe(fn):
        # Event payload shapes have shifted between livekit-agents
        # minor versions. We don't want a missed attribute to crash the
        # session — just log the failure and move on.
        def wrapped(ev):
            try:
                fn(ev)
            except Exception:
                log.debug("event_handler_failed", exc_info=True)
        return wrapped

    @session.on("agent_state_changed")
    @_safe
    def _on_agent_state(ev) -> None:
        log.info(
            "STATE agent: %-12s -> %s",
            getattr(ev, "old_state", "?"),
            getattr(ev, "new_state", "?"),
        )

    @session.on("user_state_changed")
    @_safe
    def _on_user_state(ev) -> None:
        log.info(
            "STATE user:  %-12s -> %s",
            getattr(ev, "old_state", "?"),
            getattr(ev, "new_state", "?"),
        )

    @session.on("user_input_transcribed")
    @_safe
    def _on_transcribed(ev) -> None:
        transcript = getattr(ev, "transcript", "")
        words = len(transcript.split())
        log.info(
            "EVENT transcribed: %r final=%s words=%d agent_state=%s",
            transcript,
            getattr(ev, "is_final", "?"),
            words,
            session.agent_state,
        )

    @session.on("conversation_item_added")
    @_safe
    def _on_item_added(ev) -> None:
        item = getattr(ev, "item", None)
        role = getattr(item, "role", "?")
        content = getattr(item, "content", None) or []
        text = " ".join(c if isinstance(c, str) else "<non-text>" for c in content)
        log.info("ITEM  added: role=%s text=%r", role, text)

    @session.on("agent_false_interruption")
    @_safe
    def _on_false_interruption(ev) -> None:
        log.info("RESUME false interruption — agent resuming")

    @session.on("speech_created")
    @_safe
    def _on_speech_created(ev) -> None:
        log.info(
            "SPEECH created: source=%s user_initiated=%s",
            getattr(ev, "source", "?"),
            getattr(ev, "user_initiated", "?"),
        )

    @session.on("function_tools_executed")
    @_safe
    def _on_tools(ev) -> None:
        calls = getattr(ev, "function_calls", None) or []
        names = [getattr(c, "name", "?") for c in calls]
        log.info("TOOLS executed: %s", names)

    @session.on("metrics_collected")
    @_safe
    def _on_metrics(ev) -> None:
        # DEBUG only — these fire frequently (one per STT/LLM/TTS/EOU
        # operation) and would drown out the rest of the stream at INFO.
        metrics = getattr(ev, "metrics", None)
        log.debug("METRICS %s", type(metrics).__name__ if metrics else "?")

    @session.on("close")
    @_safe
    def _on_close(ev) -> None:
        log.info(
            "SESSION closed reason=%s error=%s",
            getattr(ev, "reason", "?"),
            getattr(ev, "error", None),
        )
