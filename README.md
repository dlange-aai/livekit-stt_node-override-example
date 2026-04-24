# livekit-stt_node-override-example

A minimal [LiveKit Agents](https://github.com/livekit/agents) voice agent
that filters out backchannel / filler utterances ("mhm", "yeah", "okay")
**while the agent is speaking**, so the caller's throat-clears and
acknowledgements don't cut the agent off mid-sentence.

The technique: override `Agent.stt_node` and drop short backchannel-only
events from the STT stream before they reach LiveKit's downstream
interrupt / preempt / turn-detection machinery. No monkey-patching of
LiveKit internals — just the public extensibility hook.

The demo is a hotel front-desk agent, but the filter is domain-agnostic;
copy the `HotelBookingAgent.stt_node` / `_should_drop` / `BACKCHANNELS`
block into your own agent as-is.

## Stack

- **LiveKit Agents** — voice orchestration
- **AssemblyAI Universal-Streaming (`u3-rt-pro`)** — STT
- **Cartesia (`sonic-3`)** — TTS
- **OpenAI (`gpt-4.1-nano`)** — LLM
- **LiveKit `EnglishModel`** — turn detection

## How the filter works

```python
class HotelBookingAgent(Agent):
    async def stt_node(self, audio, model_settings):
        async for ev in Agent.default.stt_node(self, audio, model_settings):
            if self._should_drop(ev):
                continue
            yield ev

    def _should_drop(self, ev):
        if self.session.agent_state != "speaking":
            return False                 # gate 1: only filter during TTS
        if ev.type not in _TRANSCRIPT_TYPES:
            return False                 # gate 2: only transcript events
        text = ev.alternatives[0].text if ev.alternatives else ""
        return _is_all_backchannel(text) # gate 3: every token a filler
```

`_is_all_backchannel` normalizes (lowercase, strip punctuation), tokenizes
on whitespace, and returns `True` only if **every** token is in the
`BACKCHANNELS` set. A single real word anywhere in the transcript
(`"yeah I want the suite"`) flips the result to `False`, so legitimate
intents never get dropped — even when they start with a filler.

Dropping an event in `stt_node` is equivalent to the STT never emitting
it: no accumulation into the turn buffer, no preemptive-generation draft,
no interrupt. Everything else flows through untouched.

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

## Why `vad=None`?

Silero VAD is deliberately disabled in this demo. With VAD **on** and
`interruption.enabled=True`, the sequence on a backchannel goes:

1. User starts making any audible sound.
2. Silero fires its speaking event → LiveKit's
   `_interrupt_by_audio_activity` runs → TTS is paused.
3. `session.agent_state` flips `"speaking"` → `"listening"`.
4. AssemblyAI finishes recognizing → emits the transcript.
5. Our filter reads `agent_state` → sees `"listening"` → passes the
   event through.

The VAD-triggered pause races ahead of the STT event, so the `"speaking"`
branch of the filter never fires. Turn detection works fine without
VAD — `EnglishModel` commits turns from the STT + end-of-turn signal —
and interrupts still happen, they just fall out of the normal
`on_end_of_turn` commit path which runs **after** `stt_node` yields.

If your use case genuinely needs VAD-based interrupts, this particular
filter won't be sufficient on its own.

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

Then dial into the agent. The quickest path is LiveKit's SIP integration
— follow [LiveKit's telephony
docs](https://docs.livekit.io/agents/start/telephony/) to hook a phone
number to this worker. You can also connect a browser client via
LiveKit's [agents playground](https://agents-playground.livekit.io/).

## Try it out

Once connected, experiment with these patterns:

| You say (mid-agent-speech)    | Agent behavior                                  |
|-------------------------------|-------------------------------------------------|
| "mhm"                         | keeps talking — dropped by filter               |
| "Um,"                         | keeps talking — dropped by filter               |
| "yeah yeah"                   | keeps talking — every token is a filler         |
| "yeah I'd like the suite"     | stops & responds — real intent, passes through  |
| "suite please"                | stops & responds — no fillers in the transcript |
| (after agent stops) "mhm"     | responds — `agent_state == "listening"`, filter |
|                               |   doesn't fire, legitimate user turn            |

## Files

- **`agent.py`** — the entire demo; filter + session config in one file.
- **`.env.example`** — required API keys.
- **`requirements.txt`** — pinned to the LiveKit plugin families the
  demo uses.

## License

MIT — see [`LICENSE`](./LICENSE).
