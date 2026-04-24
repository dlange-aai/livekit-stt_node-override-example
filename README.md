# livekit-stt_node-override-example

A minimal [LiveKit Agents](https://github.com/livekit/agents) voice agent
that filters out backchannel / filler utterances ("mhm", "yeah", "okay")
**while the agent is speaking**, so the caller's throat-clears and
acknowledgements don't derail the agent into treating them as real
user turns.

The technique: override `Agent.stt_node` and drop short backchannel-only
events from the STT stream before they reach LiveKit's downstream
interrupt / preempt / turn-detection machinery.

The demo is a hotel front-desk agent, but the filter is domain-agnostic;
copy the `HotelBookingAgent.stt_node` / `_should_drop` / `BACKCHANNELS`
block into your own agent as-is.

## Stack

- **LiveKit Agents** — voice orchestration
- **AssemblyAI Universal-Streaming (`u3-rt-pro`)** — STT + end-of-turn detection
- **Cartesia (`sonic-3`)** — TTS
- **OpenAI (`gpt-4.1-nano`)** — LLM

## Why this setup is optimal for filtering backchannels

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

## How the filter works

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

`BACKCHANNELS` in `agent.py` is a `frozenset` of lowercase tokens. Edit
for your domain:

- **`"yes"` / `"no"` are deliberately omitted** — in a booking flow a bare
  "yes" is a real confirmation. Add them if your flow never expects
  standalone yes/no.
- `"okay"` / `"ok"` are borderline; kept in because in telephony they're
  overwhelmingly used as backchannels. Remove if "ok" is a confirmation
  in your flow.
- Hyphenated forms like `"uh-huh"` / `"mm-hmm"` are handled by the
  punctuation-stripping step — the set just needs `"uhhuh"` / `"mmhmm"`.

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

## Run

Development mode (reloads on save):

```bash
python agent.py dev
```

Then connect a client. Easiest path is LiveKit's [agents
playground](https://agents-playground.livekit.io/): pick the project
whose `LIVEKIT_URL` / key / secret are in your `.env` and click
*Connect* — the worker auto-dispatches into the new room.

## Try it out

Once connected, experiment with these patterns:

| You say (mid-agent-speech)    | Agent behavior                                  |
|-------------------------------|-------------------------------------------------|
| "mhm"                         | keeps talking — dropped by filter               |
| "Um,"                         | keeps talking — dropped by filter               |
| "yeah yeah"                   | keeps talking — every token is a filler         |
| "yeah I'd like the suite"     | stops & responds — real intent, passes through  |
| "suite please"                | stops & responds — no fillers in the transcript |
| (>1s after agent stops) "mhm" | responds — past the grace window, filter off    |

## Files

- **`agent.py`** — the entire demo; filter + session config in one file.
- **`.env.example`** — required API keys.
- **`requirements.txt`** — pinned to the LiveKit plugin families the
  demo uses.

## License

MIT — see [`LICENSE`](./LICENSE).
