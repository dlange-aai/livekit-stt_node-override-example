# LiveKit short-utterance interruption filters — two approaches

<!-- TODO(rename): the GitHub repo + local dir is still
livekit-stt_node-override-example, which now misleads. Rename to something
like livekit-short-utterance-interruption-filters once both examples land. -->

Three minimal [LiveKit Agents](https://github.com/livekit/agents) voice
agents that solve the same problem different ways: stop a caller's
**short backchannel utterances** ("mhm", "um", "yeah") from interrupting
the agent mid-speech.

All three examples use the same hotel front-desk demo so the only thing
that differs between them is the filter mechanism. Pick whichever fits
your codebase and constraints — or run both together (Strategy 3) for
layered coverage.

## Layout

```
filters/                       # Portable filter modules — drop these into
├── backchannel_stt.py         # your own LiveKit project. Each file imports
└── short_utterance_buffer.py  # only livekit + stdlib, nothing from this repo.

stt_node_override/agent.py     # Strategy 1 demo — backchannel STT filter only.
buffer_clearing/agent.py       # Strategy 2 demo — buffer-clearer only.
combined/agent.py              # Strategy 3 demo — both filters layered.
```

Each `agent.py` is a self-contained, runnable hotel-front-desk demo:
prompt + agent class + session config + entrypoint, all in one file.
The only cross-file import is the filter you're showcasing — that's
the point of the repo. Read `agent.py` top-to-bottom and you can see
the entire setup; copy `filters/<your-pick>.py` into your own project
to apply the same strategy.

## Stack

- **LiveKit Agents** — voice orchestration
- **AssemblyAI Universal-Streaming (`u3-rt-pro`)** — STT + end-of-turn detection
- **Cartesia (`sonic-3`)** — TTS
- **OpenAI (`gpt-4.1-nano`)** — LLM

## The three strategies at a glance

|                       | `stt_node` override                                                  | Buffer clearing                                                                       | Combined                                                                                          |
|-----------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Where it sits         | Wraps `Agent.stt_node`, drops events upstream of `audio_recognition` | Listens on `user_input_transcribed`, clears LK private buffers                        | Both, layered                                                                                     |
| What it filters on    | A configured set of backchannel tokens (`"mhm"`, `"um"`, …)          | Word count below `interruption.min_words` (anything shorter is dropped while talking) | Tokens upstream + word count downstream                                                           |
| LK internals touched  | None — `stt_node` is a public override point                         | Three private attrs + one private method                                              | Same private surface as Strategy 2                                                                |
| Best for              | Domains with a known finite filler list                              | Domains where any short utterance during agent speech should not interrupt            | When you want both: known fillers stopped cheaply at the source, novel short utterances mopped up |
| Risk                  | An unknown short filler ("aha") slips through                        | Couples to LK private API — pin `livekit-agents`                                      | Inherits Strategy 2's private-API coupling                                                        |
| File                  | [`stt_node_override/agent.py`](./stt_node_override/agent.py)         | [`buffer_clearing/agent.py`](./buffer_clearing/agent.py)                              | [`combined/agent.py`](./combined/agent.py)                                                        |

All three strategies rely on the same LK config: `vad=None` +
`turn_detection="stt"`. With VAD off, the only interrupt path is a
committed STT turn, which every strategy sits upstream of.

## Strategy 1 — `stt_node` override

Demo: [`stt_node_override/agent.py`](./stt_node_override/agent.py)
· Portable filter: [`filters/backchannel_stt.py`](./filters/backchannel_stt.py)

Override `Agent.stt_node` and drop short backchannel-only events from
the STT stream before they reach LiveKit's downstream interrupt /
preempt / turn-detection machinery.

### Why this setup is optimal

The config is **`vad=None` + `turn_detection="stt"` + `stt_node`
override**. That combination is what makes the filter work reliably;
changing any one of the three introduces timing races or competing
interrupt paths that can let fillers slip through.

**VAD off.** Silero VAD triggers on *any* user audio — fillers,
coughs, background noise — and pauses TTS at the audio-activity
layer, before our transcript-level filter ever sees the event. Even
if we later drop the transcript, the pause is already done. With
`vad=None`, the only interrupt path is a committed STT turn, which
the filter controls.

**Turn detection from STT.** With `turn_detection="stt"`, the
interrupt fires off AssemblyAI's end-of-turn signal riding on the
FINAL_TRANSCRIPT. Because our filter sits *upstream* of
audio_recognition, dropping the FINAL for a filler drops the
end-of-turn signal with it — no turn commits, no interrupt, nothing.
If we used a local turn-detection model instead, it would see the
committed text and make its own call; with `stt` we get a single,
inspectable event to gate on.

**`stt_node` override.** The override is the only place in the
pipeline that can intercept STT events *before* they enter the turn
buffer. Dropping an event here is equivalent to the STT never
emitting it — no buffer accumulation, no preempt-generation draft,
no interrupt.

With all three together, the filler flow is:

1. Agent is speaking. `agent_state == "speaking"`.
2. User emits "um" mid-sentence.
3. AssemblyAI emits INTERIM + FINAL for "um".
4. Our filter drops every event (gate 1 is open — we're speaking,
   or within the 1 s grace window that follows).
5. Nothing reaches audio_recognition → no turn commits → agent
   finishes its sentence normally. "um" never existed.

For a real intent ("I'd like the suite"), the filter's gate 3 sees
a non-filler token, returns `False`, and the event flows through —
AssemblyAI's end-of-turn commits the turn, the agent interrupts and
responds.

The tradeoff you're buying into: interrupts wait for AssemblyAI to
finalize the turn (bound by `max_turn_silence`, up to ~1 s), so
barge-in is a beat slower than it would be with VAD. For
customer-support / booking-style flows, that latency is invisible.
If you need sub-100 ms barge-in, you'll need VAD back on plus
per-utterance latching in the filter to handle the late-final race
(not shown in this demo).

### How the filter works

```python
class HotelBookingAgent(Agent):
    _FILTER_GRACE_S = 1.0  # keep filtering this long after TTS ends

    def __init__(self) -> None:
        super().__init__(instructions=...)
        self._last_speaking_at = 0.0

    async def stt_node(self, audio, model_settings):
        async for ev in Agent.default.stt_node(self, audio, model_settings):
            if self._should_drop(ev):
                continue
            yield ev

    def _should_drop(self, ev):
        now = time.monotonic()
        if self.session.agent_state == "speaking":
            self._last_speaking_at = now
        elif now - self._last_speaking_at > self._FILTER_GRACE_S:
            return False                 # gate 1: past grace, pass through
        if ev.type not in _TRANSCRIPT_TYPES:
            return False                 # gate 2: only transcript events
        text = ev.alternatives[0].text if ev.alternatives else ""
        return _is_all_backchannel(text) # gate 3: every token a filler
```

Three gates, cheapest first:

1. **Grace window.** Filter while `agent_state == "speaking"` or
   within `_FILTER_GRACE_S` of the last "speaking" observation. The
   grace covers AssemblyAI's FINAL_TRANSCRIPT latency — a filler
   uttered just before TTS ends can still be finalized a few hundred
   ms after `agent_state` flipped to `"listening"`. Tradeoff: a bare
   `"yeah"` uttered within 1 s of the agent finishing also gets
   filtered. Tune `_FILTER_GRACE_S` down if that's too aggressive.
2. **Transcript-only.** `START_OF_SPEECH` / `END_OF_SPEECH` carry no
   text and the downstream state machine needs them, so we only
   inspect transcript events.
3. **All-filler check.** `_is_all_backchannel` normalizes (lowercase,
   strip punctuation), tokenizes on whitespace, and returns `True`
   only if **every** token is in `BACKCHANNELS`. A single real word
   anywhere in the transcript (`"yeah I want the suite"`) flips the
   result to `False`, so legitimate intents never get dropped — even
   when they start with a filler.

### Tuning the filler list

`BACKCHANNELS` in [`filters/backchannel_stt.py`](./filters/backchannel_stt.py)
is a `frozenset` of lowercase tokens. Edit for your domain:

- **`"yes"` / `"no"` are deliberately omitted** — in a booking flow a bare
  "yes" is a real confirmation. Add them if your flow never expects
  standalone yes/no.
- `"okay"` / `"ok"` are borderline; kept in because in telephony they're
  overwhelmingly used as backchannels. Remove if "ok" is a confirmation
  in your flow.
- Hyphenated forms like `"uh-huh"` / `"mm-hmm"` are handled by the
  punctuation-stripping step — the set just needs `"uhhuh"` / `"mmhmm"`.

## Strategy 2 — Buffer clearing

Demo: [`buffer_clearing/agent.py`](./buffer_clearing/agent.py)
· Portable filter: [`filters/short_utterance_buffer.py`](./filters/short_utterance_buffer.py)

Listen on `session.on("user_input_transcribed")` and, while the agent is
speaking, wipe the three transcript buffers LK's interrupt + preempt
gates read from. With the buffers empty, the next gate check after a
short utterance sees no accumulated text and doesn't trip an interrupt.

### How the filter works

```python
def install_short_utterance_filter(session):
    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev):
        if session.agent_state != "speaking":
            return
        word_count = len(ev.transcript.split())
        if word_count >= session.options.interruption["min_words"]:
            return

        activity = getattr(session, "_activity", None)
        recognition = getattr(activity, "_audio_recognition", None) if activity else None
        if recognition is None:
            return
        recognition._audio_transcript = ""
        recognition._audio_interim_transcript = ""
        recognition._audio_preflight_transcript = ""

        cancel = getattr(activity, "_cancel_preemptive_generation", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass
```

Two gates, then surgery on LK's private state:

1. **Agent state.** Only filter while `agent_state == "speaking"`.
   When the agent is idle, short utterances ("yes", "no") are real
   turns and must pass through.
2. **Word count.** Compare against `session.options.interruption["min_words"]`
   (LK's own threshold for STT-driven interruption). Anything at or
   above that count is a real interrupt and we leave it alone.

Once the gates open, three buffers get cleared because each backs a
different decision path inside LK's `audio_recognition`:

- `_audio_transcript` — cumulative committed finals; read by
  `_interrupt_by_audio_activity` to decide whether the user has said
  enough to interrupt.
- `_audio_interim_transcript` — current interim chunk; concatenated
  with the above to form `current_transcript`, which the gate
  actually reads.
- `_audio_preflight_transcript` — buffer + current preflight; used
  by `on_preemptive_generation` to decide whether to spin up a
  speculative LLM reply.

If any of the three still has stale text from a prior backchannel,
the next gate check trips on it. Wiping all three keeps the buffers
in sync.

`_cancel_preemptive_generation()` is a best-effort tidy-up: if LK
already kicked off a speculative LLM call on the preflight of this
short utterance, this aborts it before TTS starts. Wrapped in a
`try/except` because the method is private and could change shape
on a minor-version bump.

### Why this works (timing)

Inside LK, `agent_activity.on_interim_transcript` /
`on_final_transcript` dispatches the `user_input_transcribed` event
**before** running `_interrupt_by_audio_activity`. Our handler runs
first, wipes the accumulated buffers, then LK's gate runs against an
empty buffer and doesn't trip. No interrupt fires; agent keeps
talking.

### Tradeoffs vs. the `stt_node` override

- **Simpler to drop in.** No `stt_node` subclass, no backchannel set
  to maintain — just an event handler.
- **Coarser.** The decision is purely "how many words?", not "is it
  a filler?". A short real intent ("the suite") still interrupts as
  long as it clears `min_words` — tune that knob to your flow.
- **Couples to LK private API.** `_audio_transcript`,
  `_audio_interim_transcript`, `_audio_preflight_transcript`, and
  `_cancel_preemptive_generation` are all leading-underscore names.
  Pin your `livekit-agents` version (`>=1.5` here, but ideally a
  fully pinned `==`) and re-test on upgrade.
- **Confirmed limitation:** legitimate ≥`min_words` utterances still
  interrupt, which is the intended behavior. If you need to suppress
  those too, add phrase-matching logic inside the same handler.

## Strategy 3 — Combined (both layered)

Demo: [`combined/agent.py`](./combined/agent.py)
· Portable filters: [`filters/backchannel_stt.py`](./filters/backchannel_stt.py),
[`filters/short_utterance_buffer.py`](./filters/short_utterance_buffer.py)

Run both filters on the same agent. The backchannel STT filter sits
upstream and drops *known* fillers from the STT stream before they
ever reach `audio_recognition`. The buffer-clearer sits downstream
and mops up any *unknown* short utterance the filler list doesn't
cover, by wiping LK's accumulation buffers before the interrupt gate
reads them.

```python
from filters.backchannel_stt import BackchannelSTTFilterMixin
from filters.short_utterance_buffer import install_short_utterance_filter

class MyAgent(BackchannelSTTFilterMixin, Agent):
    ...  # your prompt, tools, etc.

session = AgentSession(
    ...,
    turn_handling={
        ...,
        "interruption": {..., "min_words": 2},
    },
)
install_short_utterance_filter(session)
await session.start(room=ctx.room, agent=MyAgent())
```

### Why use both

The two filters cover non-overlapping failure modes:

- **Strategy 1 alone** misses an unknown short word — a stutter
  fragment ("k…"), a regional filler ("aha"), a one-word probe
  ("wait?") — anything not in `BACKCHANNELS`. That word commits as
  a turn and interrupts.
- **Strategy 2 alone** filters by length, not by content. It catches
  the unknowns Strategy 1 misses, but only after the event has
  already accumulated into `audio_recognition` and possibly kicked
  off a speculative LLM call (which `_cancel_preemptive_generation`
  then has to abort, with a small race window).
- **Combined**: Strategy 1 stops everything in `BACKCHANNELS` *before*
  preempt-gen can fire (fast path, race-free), and Strategy 2 cleans
  up whatever slipped past as a short non-filler.

They overlap on short fillers — Strategy 1 drops the event upstream
so Strategy 2's handler never fires for those — but the overlap is
harmless. The cost of running both is one event-handler registration
plus one MRO lookup per STT event.

### When *not* to use combined

If your domain has a closed filler vocabulary you can fully enumerate
(call-center scripts, narrow IVR-style flows), Strategy 1 alone keeps
you on the public API and avoids the private-attr coupling Strategy 2
brings in. Combined makes sense when you need belt-and-suspenders
coverage and have already accepted the `livekit-agents` version pin.

## Setup

Requires Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with your keys
```

You'll need accounts / API keys from:

- [LiveKit Cloud](https://cloud.livekit.io) (or a self-hosted server)
- [AssemblyAI](https://assemblyai.com)
- [Cartesia](https://cartesia.ai)
- [OpenAI](https://platform.openai.com)

`.env` lives at the repo root and is shared by all three examples.

## Run

Pick one strategy and run its `agent.py` in dev mode (reloads on save):

```bash
# Strategy 1
python stt_node_override/agent.py dev

# Strategy 2
python buffer_clearing/agent.py dev

# Strategy 3 — both filters layered
python combined/agent.py dev
```

Then connect a client. Easiest path is LiveKit's [agents
playground](https://agents-playground.livekit.io/): pick the project
whose `LIVEKIT_URL` / key / secret are in your `.env` and click
*Connect* — the worker auto-dispatches into the new room.

## Try it out

All three strategies should exhibit the same external behavior on these
inputs (via different mechanisms — tail the log to see which filter
fired):

| You say (mid-agent-speech)    | Agent behavior                                    |
|-------------------------------|---------------------------------------------------|
| "mhm"                         | keeps talking — filtered                          |
| "Um,"                         | keeps talking — filtered                          |
| "yeah yeah"                   | keeps talking — filtered                          |
| "yeah I'd like the suite"     | stops & responds — real intent, passes through    |
| "suite please"                | stops & responds — real intent, passes through    |
| (>1s after agent stops) "mhm" | responds — agent idle, no filtering applies       |

When the backchannel STT filter drops something it logs
`event_filtered transcript=...`. When the buffer-clearer drops
something it logs `buffer_cleared transcript=...`. In `combined/`,
both log lines appear depending on which filter caught the utterance.

## Files

- **`filters/backchannel_stt.py`** — portable Strategy 1 filter
  (`BackchannelSTTFilterMixin`, `BACKCHANNELS`). Drop into your project.
- **`filters/short_utterance_buffer.py`** — portable Strategy 2 filter
  (`install_short_utterance_filter`). Drop into your project.
- **`stt_node_override/agent.py`** — Strategy 1 self-contained demo.
- **`buffer_clearing/agent.py`** — Strategy 2 self-contained demo.
- **`combined/agent.py`** — Strategy 3 self-contained demo.
- **`.env.example`** — required API keys (shared by all three examples).
- **`requirements.txt`** — pinned to the LiveKit plugin families the
  demos use.

## License

MIT — see [`LICENSE`](./LICENSE).
