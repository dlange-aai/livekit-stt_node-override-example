# LiveKit short-utterance interruption filters — three strategies

Three minimal [LiveKit Agents](https://github.com/livekit/agents) voice
agents that solve the same problem different ways: stop a caller's
**short backchannel utterances** ("mhm", "um", "yeah") from interrupting
the agent mid-speech.

All three examples use the same hotel front-desk demo so the only thing
that differs between them is the filter mechanism. Pick whichever fits
your codebase and constraints — or layer both filters together
(Strategy 3) for broader coverage.

## Layout

```
filters/                       # Portable filter modules — drop the one
├── backchannel_stt.py         # you want into your own LiveKit project.
├── short_utterance_buffer.py  # Each filter imports only livekit +
├── debug_logging.py           # stdlib. debug_logging.py is demo-only.
└── __init__.py

stt_node_override/agent.py     # Strategy 1 demo — backchannel STT filter only.
buffer_clearing/agent.py       # Strategy 2 demo — buffer-clearer only.
combined/agent.py              # Strategy 3 demo — both filters layered.
```

Each `agent.py` is a self-contained, runnable hotel-front-desk demo:
prompt + agent class + session config + entrypoint, all in one file.
The only cross-file imports are the filter(s) the demo showcases plus
the demo-only `debug_logging` helpers. Read `agent.py` top-to-bottom
and you can see the entire setup; copy `filters/<your-pick>.py` into
your own project to apply the same strategy.

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

The recommended LK config for all three strategies is `vad=None` +
`turn_detection="stt"`. With VAD off, the only interrupt path is a
committed STT turn, which every strategy sits upstream of. See
[Why this setup is optimal](#why-this-setup-is-optimal) for the
rationale, and [Alternative LiveKit configurations](#alternative-livekit-configurations)
for VAD-on / LK-turn-detector setups that also work — they require
one extra knob, and each strategy reaches the same outcome through
a different mechanism.

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

The filter would also block a local turn-detection model
(`MultilingualModel`, `EnglishModel`) from seeing the text — its
classifier reads `_audio_transcript`, which our filter prevents
from accumulating. So the reason to prefer `stt` here isn't about
what the classifier sees. It's structural: `turn_detection="stt"`
lets us run `vad=None`, because end-of-turn rides on the STT event
stream itself. A local turn-detection model has no end-of-turn
trigger without VAD, so it forces VAD on — which adds the
audio-activity interrupt path the filter can't see (see
Alternative configurations below).

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
If you need sub-100 ms barge-in, see
[Alternative LiveKit configurations](#alternative-livekit-configurations)
below.

### How the filter works

The portable filter is a mixin — drop it ahead of `Agent` in your
class's MRO and its `stt_node` runs first on every STT event:

```python
from filters.backchannel_stt import BackchannelSTTFilterMixin

class HotelBookingAgent(BackchannelSTTFilterMixin, Agent):
    def __init__(self) -> None:
        super().__init__(instructions=...)
```

The mixin itself, condensed:

```python
class BackchannelSTTFilterMixin:
    _FILTER_GRACE_S: float = 1.0   # keep filtering this long after TTS ends
    _last_speaking_at: float = 0.0

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

### Event sequence

When the user says "um" while the agent is speaking:

1. **AssemblyAI emits `INTERIM_TRANSCRIPT("um")`.**
   - Filter sees `agent_state == "speaking"` (gate 1) ✓, transcript
     event (gate 2) ✓, all-filler (gate 3) ✓ → drops event with
     `continue`, never yields it.
   - **Doesn't run downstream:** `audio_recognition._on_stt_event`
     for this event, the `_audio_interim_transcript = ev.text` write,
     `agent_activity.on_interim_transcript`, the
     `user_input_transcribed` event emit, and
     `_interrupt_by_audio_activity`.

2. **AssemblyAI emits `FINAL_TRANSCRIPT("um")`.**
   - Same gates → dropped.
   - **Doesn't run downstream:** `audio_recognition._on_stt_event`
     for this event, the `_audio_transcript += " um"` append at
     `audio_recognition.py:743`,
     `agent_activity.on_final_transcript`,
     `user_input_transcribed`, `_interrupt_by_audio_activity`,
     preempt-generation kickoff.

3. **AssemblyAI emits `END_OF_SPEECH`.**
   - Not in `_TRANSCRIPT_TYPES` → gate 2 returns `False` → filter
     yields the event through.
   - `audio_recognition._on_stt_event` runs the END_OF_SPEECH branch.
   - With `turn_detection="stt"`: calls `_run_eou_detection`.
   - `_run_eou_detection` reads `_audio_transcript` — empty (nothing
     was ever appended) → early return at the
     `not self._audio_transcript` check.
   - **Doesn't run:** `on_end_of_turn` hook, turn commit, LLM call,
     TTS interrupt.

**Net effect on the LiveKit pipeline:** for the duration of the
"um", LK behaves as if AssemblyAI never produced any transcript
at all. The filter consumes the events at the source, so
audio_recognition never reads them, the LK internal buffers never
populate, no `user_input_transcribed` event ever fires, and
nothing subscribed to transcript events (telemetry, transcripts
panel, observers, recording layers) is notified. END_OF_SPEECH
still flows through, but `_run_eou_detection` finds the buffer
empty and bails — no turn commits, no LLM call, no TTS interrupt.
Agent TTS plays out uninterrupted.

For a real intent ("I'd like the suite"), gate 3 returns `False`
on the first non-filler token. INTERIM and FINAL flow through
normally, `_audio_transcript` accumulates as expected,
`user_input_transcribed` fires, END_OF_SPEECH triggers
`_run_eou_detection` against a non-empty buffer, the turn
commits, and the agent interrupts.

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

- `_audio_transcript` — accumulated finals for the *current
  uncommitted user turn*. LK only clears this when a turn commits
  (`audio_recognition.py:1044`), so short fillers that fail
  `on_end_of_turn`'s own `min_words` check stay in the buffer and
  pile up across consecutive utterances.
- `_audio_interim_transcript` — current interim chunk; concatenated
  with the above to form `current_transcript`, which
  `_interrupt_by_audio_activity` reads.
- `_audio_preflight_transcript` — buffer + current preflight; used
  by `on_preemptive_generation` to decide whether to spin up a
  speculative LLM reply.

LK already has its own `min_words` gates — one in
`_interrupt_by_audio_activity` (blocks the immediate TTS-pause) and
one in `on_end_of_turn` (blocks the turn commit / LLM call). Both
read transcripts that are *cumulative across the user's turn*. The
buffer-clearer's job is to keep `_audio_transcript` from
accumulating short fillers across multiple un-committed FINALs:
without it, two consecutive `"yeah"` + `"um"` finals sum to
`"yeah um"` (2 words) and trip both gates even though each
individual utterance is below threshold.

`_cancel_preemptive_generation()` is a best-effort tidy-up: if LK
already kicked off a speculative LLM call on the preflight of this
short utterance, this aborts it before TTS starts. Wrapped in a
`try/except` because the method is private and could change shape
on a minor-version bump.

### Event sequence

When the user says "um" while the agent is speaking (with
`min_words = 2`):

1. **AssemblyAI emits `INTERIM_TRANSCRIPT("um")`.**
   - `stt_node` yields the event unchanged (no filter on this
     strategy).
   - `audio_recognition._on_stt_event` enters the INTERIM branch
     (`audio_recognition.py:821`). The hook fires *first*, the
     buffer write happens *after*.
   - `agent_activity.on_interim_transcript` runs.
   - **`user_input_transcribed(transcript="um", is_final=False)` is
     emitted** — telemetry, transcripts panel, recording layers all
     see it.
   - **Our handler runs:** `agent_state == "speaking"` ✓,
     `word_count = 1 < min_words = 2` ✓ → wipes all three
     `_audio_*` buffers, calls `_cancel_preemptive_generation`.
   - Control returns to `agent_activity.on_interim_transcript`,
     which calls `_interrupt_by_audio_activity`. It reads
     `current_transcript` (empty after the wipe), `len(split_words("")) = 0 < 2`
      → bails at the `min_words` check (`agent_activity.py:1593-1601`).
     No TTS pause, no interrupt.
   - After the hook returns, audio_recognition runs
     `self._audio_interim_transcript = ev.alternatives[0].text` at
     line 828, so `_audio_interim_transcript = "um"` after this
     step.

2. **AssemblyAI emits `FINAL_TRANSCRIPT("um")`.**
   - `stt_node` yields it.
   - `audio_recognition._on_stt_event` enters the FINAL branch.
     The hook at line 728 fires *before* the buffer append at
     line 743.
   - `agent_activity.on_final_transcript` runs.
   - **`user_input_transcribed(transcript="um", is_final=True)` is
     emitted** — observers see this too.
   - **Our handler runs again:** wipes the three `_audio_*`
     buffers (clearing the `"um"` `_audio_interim_transcript` left
     by step 1), cancels preempt.
   - `_interrupt_by_audio_activity` reads `current_transcript = ""` →
     bails on `min_words`.
   - **After the hook returns**, audio_recognition continues at
     line 743: `self._audio_transcript += " um"` → so
     `_audio_transcript = "um"` once FINAL processing completes
     (our wipe doesn't survive the append). `_audio_interim_transcript`
     and `_audio_preflight_transcript` are then re-zeroed by LK
     itself at lines 747-748.

3. **AssemblyAI emits `END_OF_SPEECH`.**
   - With `turn_detection="stt"`: `_run_eou_detection` runs
     (`audio_recognition.py:852`).
   - `_audio_transcript = "um"` — **not empty**, so the
     `if self._stt and not self._audio_transcript ...: return`
     guard at line 916 does *not* short-circuit.
   - `_run_eou_detection` schedules `_bounce_eou_task`, which after
     the endpointing delay calls
     `agent_activity.on_end_of_turn(new_transcript="um")`.
   - **`on_end_of_turn` has its own `min_words` gate**
     (`agent_activity.py:1871-1883`): `len(split_words("um")) = 1 < 2`
      → calls `_cancel_preemptive_generation`, returns `False`.
     The turn is NOT committed.
   - **Doesn't run:** `_user_turn_completed_task`, LLM call, TTS
     interrupt.

**Net effect on the LiveKit pipeline:** transcript events flow
through audio_recognition normally — the data path is unchanged
from a no-filter setup. Two `user_input_transcribed` events fire
(one interim, one final), and anything subscribed to that event
(telemetry, transcripts panel, observers, recording layers)
records the filler. `_audio_transcript` ends up holding the
latest filler (`"um"`) — our wipe doesn't keep it empty because
audio_recognition appends *after* the hook returns. What the
wipe *does* prevent is **accumulation**: a follow-up `"uh"` final
arrives with `_audio_transcript = ""` (we wiped it again),
audio_recognition appends → `_audio_transcript = "uh"` instead of
`"um uh"`. With min_words=2 in place, both LK gates
(`_interrupt_by_audio_activity` and `on_end_of_turn`) keep
returning early on the single-word transcript, so no interrupt
fires and no turn commits. Without the wipe, two short fillers in
a row would sum past the threshold and break through.

For a real intent ("I'd like the suite"), `word_count >= min_words`
on the first INTERIM. Our handler returns early at gate 2 without
wiping. Buffers accumulate normally, `_interrupt_by_audio_activity`
reads non-empty `current_transcript`, the gate trips, TTS pauses,
and the agent interrupts.

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

### Event sequence

With both filters installed, transcript events pass through two
checkpoints — Strategy 1's `stt_node` filter at the source, and
Strategy 2's `user_input_transcribed` handler downstream of
audio_recognition. Whether the event reaches the second
checkpoint depends on what Strategy 1 decides at the first.

**Scenario A: known filler ("um").**

1. AssemblyAI emits `INTERIM_TRANSCRIPT("um")`.
   - The stt_node filter's gates pass: agent speaking ✓,
     transcript event ✓, all tokens in `BACKCHANNELS` ✓ → `continue`,
     event dropped.
   - audio_recognition never reads the event. The INTERIM-handling
     code in `audio_recognition._on_stt_event` doesn't run, no
     buffer write happens, `user_input_transcribed` doesn't fire,
     and Strategy 2's handler — registered on
     `user_input_transcribed` — is never invoked.
   - `_interrupt_by_audio_activity` doesn't run either.
2. AssemblyAI emits `FINAL_TRANSCRIPT("um")` — same drop, same
   result.
3. AssemblyAI emits `END_OF_SPEECH`.
   - Filter passes it through (not a transcript type).
   - audio_recognition runs `_run_eou_detection`. `_audio_transcript`
     is empty (no FINAL ever appended), early-returns at the
     `not self._audio_transcript` guard.
   - No turn commit, no LLM call.

**Net for known fillers:** Strategy 1 owns the case end-to-end;
Strategy 2 never sees it. Zero observable downstream side effects.

**Scenario B: unknown short utterance ("aha", not in
`BACKCHANNELS`).**

1. AssemblyAI emits `INTERIM_TRANSCRIPT("aha")`.
   - The stt_node filter's gate 3 returns `False` (token not in
     `BACKCHANNELS`) → filter yields the event through.
   - audio_recognition enters the INTERIM branch — hook fires
     *before* the buffer write at line 828.
   - `agent_activity.on_interim_transcript` runs and emits
     `user_input_transcribed(is_final=False)` — observers see it.
   - **Strategy 2's handler runs** on that event:
     `agent_state == "speaking"` ✓,
     `word_count = 1 < min_words = 2` ✓ → wipes all three
     `_audio_*` buffers, calls `_cancel_preemptive_generation`.
   - Control returns; `_interrupt_by_audio_activity` reads
     `current_transcript` (empty after the wipe) → bails on
     `min_words`.
   - After the hook returns, audio_recognition sets
     `_audio_interim_transcript = "aha"` (line 828).
2. AssemblyAI emits `FINAL_TRANSCRIPT("aha")`.
   - Filter yields it. Hook fires (line 728) *before* the append
     (line 743).
   - `user_input_transcribed(is_final=True)` fires. Strategy 2's
     handler wipes again. Interrupt gate reads empty → bails.
   - After the hook returns, audio_recognition runs
     `self._audio_transcript += " aha"` → `_audio_transcript = "aha"`,
     and re-zeroes the interim/preflight buffers.
3. AssemblyAI emits `END_OF_SPEECH`.
   - `_run_eou_detection` runs. `_audio_transcript = "aha"` (not
     empty), so no early-return. It schedules
     `_bounce_eou_task`, which calls
     `on_end_of_turn(new_transcript="aha")`.
   - `on_end_of_turn`'s `min_words` gate
     (`agent_activity.py:1871-1883`):
     `len(split_words("aha")) = 1 < 2` → returns `False`. Turn NOT
     committed.

**Net for unknown short utterances:** the event reaches
audio_recognition, populates `_audio_transcript = "aha"`, and emits
two `user_input_transcribed` events that observers see. No
interrupt fires (the buffer-clearer's wipe blanks `current_transcript`
before `_interrupt_by_audio_activity` reads it), and `on_end_of_turn`
short-circuits on its own `min_words` check, so no turn commits and
the agent keeps talking. The end state differs from Scenario A:
`_audio_transcript` retains `"aha"` until the next user input wipes
it (or a turn commits and clears it).

For a real intent ("I'd like the suite"), Strategy 1's gate 3
returns `False` (non-filler tokens present), Strategy 2's gate 2
returns early on `word_count >= min_words`, and neither filter
does anything. The transcript flows through normally,
`_audio_transcript` accumulates, `_run_eou_detection` runs
against the populated buffer, `on_end_of_turn` sees ≥ `min_words`
words and commits the turn, and the agent interrupts.

### When *not* to use combined

If your domain has a closed filler vocabulary you can fully enumerate
(call-center scripts, narrow IVR-style flows), Strategy 1 alone keeps
you on the public API and avoids the private-attr coupling Strategy 2
brings in. Combined makes sense when you need belt-and-suspenders
coverage and have already accepted the `livekit-agents` version pin.

## Alternative LiveKit configurations

The three demos all run `vad=None` + `turn_detection="stt"`. That's
the simplest correct setup — see
[Why this setup is optimal](#why-this-setup-is-optimal). Two other
LK configurations also work, but each filter strategy needs its
own treatment because they hook different parts of the
`audio_recognition` pipeline. The shared gotcha across both
alternative configs: an `interruption.min_words` floor — `>= 1`
for Strategy 1, `>= 2` for Strategy 2 and 3. Per-strategy details
below.

### VAD on + LK turn-detector model

Use `livekit-plugins-turn-detector`'s `MultilingualModel` (or
`EnglishModel`) for end-of-turn classification, with Silero VAD
for audio-activity detection:

```python
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

session = AgentSession(
    ...,
    vad=silero.VAD.load(),
    turn_detection=MultilingualModel(),
    turn_handling={
        ...,
        "interruption": {..., "min_words": 1},  # required — see below
    },
)
```

Why pick this over the recommended setup: faster barge-in (VAD
reacts in tens of ms vs AssemblyAI's `max_turn_silence` of ~1 s),
and a model-driven turn classifier with broader language support.

There are two LK gates that can suppress a short utterance — and
the two strategies hit them differently:

```python
# audio_recognition.py
def _run_eou_detection(self, chat_ctx, skip_reply=False):
    if self._stt and not self._audio_transcript and self._turn_detection_mode != "manual":
        # stt enabled but no transcript yet
        return

# agent_activity.py — fires from inside _bounce_eou_task
def on_end_of_turn(self, info):
    ...
    if (
        ...
        and self._session.options.interruption["min_words"] > 0
        and len(split_words(info.new_transcript, ...)) < self._session.options.interruption["min_words"]
    ):
        self._cancel_preemptive_generation()
        return False  # turn NOT committed
```

Strategy 1 hits the first gate (drops the FINAL upstream so
`_audio_transcript` never accumulates and EOU early-returns).
Strategy 2 hits the second (lets `_audio_transcript` accumulate
the latest filler, then `on_end_of_turn` short-circuits on
`min_words`). Read each section below carefully — these are not
the same mechanism in two outfits.

#### How Strategy 1 (`stt_node` override) is affected

The filter intercepts the FINAL_TRANSCRIPT *upstream* of
`audio_recognition`. The `_audio_transcript += transcript` append
inside `audio_recognition.on_final_transcript` never executes for
filtered events, so the buffer stays empty by virtue of nothing
ever being appended. By the time `_run_eou_detection` runs (on VAD
END_OF_SPEECH in this config), the buffer is empty because no
FINAL ever made it into the pipeline.

The new piece compared to the recommended config: a VAD-driven
interrupt path now exists. `on_vad_inference_done` can call
`_interrupt_by_audio_activity` on bare audio (a cough, a breath)
*before* any transcript exists — the `stt_node` filter never sees
this event because there's no STT event to filter. To plug the
hole, set `interruption.min_words >= 1`. The `min_words` check
inside `_interrupt_by_audio_activity` reads the current transcript
and bails when it's empty (or below the threshold). With that knob
set, an empty-transcript VAD blip exits early and TTS keeps
playing.

#### How Strategy 2 (buffer-clearer) is affected

The order of operations is the same as the recommended config: on
each FINAL, the audio_recognition hook fires *first* (line 728),
the buffer-clearer wipes the three `_audio_*` buffers and calls
`_cancel_preemptive_generation`, then `_interrupt_by_audio_activity`
runs against an empty `current_transcript` and bails on
`min_words`. After the hook returns, audio_recognition appends the
filler at line 743 — so `_audio_transcript = "um"` by the time
EOU runs. The early-return in `_run_eou_detection` does *not*
fire; suppression comes from `on_end_of_turn`'s own `min_words`
check, exactly as in the recommended config. The wipe's purpose is
unchanged: prevent `_audio_transcript` from accumulating multiple
short fillers across uncommitted turns.

What changes from the recommended config: the buffer-clearer's
existing word-count gate already uses
`session.options.interruption["min_words"]`, so the requirement
is to set it to at least `2` (which the demo already does). Below
`2`, the gate is permissive and won't fire on the short utterances
you're trying to suppress; at `2` it covers the same VAD-blip hole
Strategy 1 needs `>= 1` for, *and* gives the word-count gate a
useful threshold. As with Strategy 1, the VAD-blip path itself is
covered by `_interrupt_by_audio_activity`'s own `min_words`
check — same line of code, doing double duty.

#### How Strategy 3 (combined) is affected

Strategy 1 catches everything in the filler vocabulary *before* it
ever appends to `_audio_transcript`, so for known fillers the
buffer-clearer doesn't even fire — Strategy 1's drop suppresses
the `user_input_transcribed` event. For known fillers, the
suppression mechanism is the EOU early-return on empty
`_audio_transcript`. The buffer-clearer only runs for short
non-filler utterances Strategy 1 didn't recognize (a stutter, a
regional filler); for those, `_audio_transcript` ends up holding
the latest filler and `on_end_of_turn`'s `min_words` check
returns `False`.

The `min_words` requirement is the union of the two: `>= 2` to
satisfy Strategy 2's word-count gate, which also covers the
VAD-blip hole Strategy 1 needs `>= 1` for.

### VAD on + `turn_detection="vad"`

Same setup but with no turn-detector model loaded — end-of-turn
comes from VAD silence rather than the model. The per-strategy
mechanics are identical to the MultilingualModel case above:
Strategy 1 still drops upstream so `_run_eou_detection`
early-returns on empty `_audio_transcript`; Strategy 2 still
relies on `on_end_of_turn`'s `min_words` check downstream of the
classifier-less EOU path. The `min_words` floors carry over
unchanged (`>= 1` for Strategy 1; `>= 2` for Strategies 2 and 3).
Cheaper to run, but you lose the model's smarter classification
(and any multi-language support).

### What doesn't work in any config

`turn_detection="manual"` and `turn_detection="realtime_llm"` both
bypass `audio_recognition`'s transcript path entirely — neither
filter has anything to gate on. If your design needs one of those,
the short-utterance check has to live in your application logic.

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

The strategies agree on most inputs but diverge on multi-word backchannels —
that's the whole reason Strategy 3 exists. Tail the log to see which filter
fired.

| You say (mid-agent-speech)    | Strategy 1   | Strategy 2 (`min_words=2`) | Strategy 3   |
|-------------------------------|--------------|----------------------------|--------------|
| "mhm"                         | keeps talking | keeps talking              | keeps talking |
| "Um,"                         | keeps talking | keeps talking              | keeps talking |
| "yeah yeah"                   | keeps talking | **interrupts** (2 ≥ 2)     | keeps talking |
| "yeah I'd like the suite"     | interrupts   | interrupts                 | interrupts   |
| "suite please"                | interrupts   | interrupts                 | interrupts   |
| (>1s after agent stops) "mhm" | responds     | responds                   | responds     |

When the backchannel STT filter drops something it logs
`event_filtered transcript=...`. When the buffer-clearer drops something it
logs `buffer_cleared transcript=...`. In `combined/`, either log line may
appear depending on which filter caught the utterance.

### Reading the logs

Each demo's `entrypoint` calls `setup_demo_logging()` and
`install_session_event_logging(session)` from
[`filters/debug_logging.py`](./filters/debug_logging.py). Together they
give a tail-friendly view of the whole pipeline:

- **`STATE agent: …`** / **`STATE user: …`** — every agent / user state
  transition (initializing → listening → thinking → speaking → …).
- **`EVENT transcribed: …`** — every `user_input_transcribed` event LK
  fires, with `final=`, `words=`, and the current `agent_state=`.
- **`ITEM added: …`** — committed turns added to the chat history.
- **`RESUME false interruption …`** — LK resumed TTS after deciding the
  interrupt was false.
- **`SPEECH created: …`** / **`TOOLS executed: …`** / **`SESSION closed …`**
  — speech handles, tool runs, session shutdown.

The filter loggers run at DEBUG inside the demos, so you also get a
`transcript_seen` line per STT event and a `kept reason='…'` line per
non-drop decision — the symmetric counterpart to `event_filtered` /
`buffer_cleared`. Drop `setup_demo_logging()` (or change its `level=`
argument) if you want quieter output.

The helper is demo-only — it's not imported by either portable filter
module, so dropping `filters/backchannel_stt.py` or
`filters/short_utterance_buffer.py` into your own project pulls in zero
extra logging machinery.

## Files

- **`filters/backchannel_stt.py`** — portable Strategy 1 filter
  (`BackchannelSTTFilterMixin`, `BACKCHANNELS`). Drop into your project.
- **`filters/short_utterance_buffer.py`** — portable Strategy 2 filter
  (`install_short_utterance_filter`). Drop into your project.
- **`filters/debug_logging.py`** — demo-only logging helpers
  (`setup_demo_logging`, `install_session_event_logging`). Not needed
  by the portable filters.
- **`stt_node_override/agent.py`** — Strategy 1 self-contained demo.
- **`buffer_clearing/agent.py`** — Strategy 2 self-contained demo.
- **`combined/agent.py`** — Strategy 3 self-contained demo.
- **`.env.example`** — required API keys (shared by all three examples).
- **`requirements.txt`** — pinned to the LiveKit plugin families the
  demos use.

## License

MIT — see [`LICENSE`](./LICENSE).
