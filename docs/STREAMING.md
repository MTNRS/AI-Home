# Foundation HTTP streaming

`POST /api/v1/chat/stream` accepts the same authenticated request body as
`/api/v1/chat/completions`: messages, optional approved `model_version`, and
explicit `use_knowledge`. Only the Foundation backend is supported. Existing
non-streaming and external-provider paths remain available independently.

The response is `text/event-stream`, with one JSON object per SSE `data:` event:

- `metadata`: actual selected model UUID and retrieved references.
- `delta`: generated text in `content` (UTF-8 decoded chunks, not guaranteed tokens).
- `done`: chunk count, character count and generation elapsed seconds.
- `error`: generic `busy` or `inference_failed`, with no private prompt or local path.

Authorization/configuration failures before streaming use HTTP errors. Runtime
failures after response headers use an error event, not a fabricated completion.
A client must require `done` before treating the answer as complete; an error or
connection close may follow partial text. No `[DONE]` compatibility sentinel is
emitted. This is the AI@Home event protocol, not a claim of OpenAI SSE compatibility.

The first metadata event follows checkpoint loading and optional retrieval/context
validation. Knowledge packing is shared with non-streaming inference and retains
the same provenance and context-fit checks. Non-RAG mode retains the existing
context truncation policy. Registry version resolution is per request and never
changes global activation.

A bounded queue holds at most 16 events. One worker performs generation and owns
the same process-local inference lock as non-streaming chat. Disconnect/cancellation
sets a stop event; the worker checks it between generation steps and during output
backpressure. The lock is released only when work exits. Cancellation cannot
interrupt checkpoint loading or an in-flight tensor kernel; those must finish
before resources can be released. There is no cross-process lock, cancellation-by-ID
endpoint, or service-wide admission controller yet.

Validation includes real trained-checkpoint generation through the authenticated
ASGI streaming route with knowledge references, and transport lifecycle tests for
cancellation, lock ownership, bounded queue, busy rejection and private-error
redaction. TestClient buffers the HTTP response, so this is not a network latency
benchmark or proxy disconnect test. Full suite: 202 passed, 4 PostgreSQL skipped.
Desktop incremental rendering and its stop button are now available as an
explicit option, described below.


## Cancellable client and real loopback verification

`ai_home.chat_stream.stream_server_chat` is the background-thread client consumer.
It validates metadata ordering and requested model UUID, emits text callbacks,
requires a final event with matching character/chunk counts, bounds accumulated
text to 65,536 characters, and rejects incomplete/error streams. Its cancellation
event also interrupts a wait for response headers via an asynchronous HTTP task;
context cleanup closes the connection. Callbacks must enqueue UI work rather than
mutate Tk widgets directly. The helper is used by the optional desktop streaming path described below.

A real loopback test now starts Uvicorn on an ephemeral localhost TCP port, trains
and loads a real tiny checkpoint, authenticates an account, receives a generated
chunk with this client, cancels the connection, observes server cancellation and
then completes a second ordinary inference request. To ensure the first request
is still running when cancelled, the test holds the real generator after a real
chunk until cancellation arrives. It does not replace model output with simulated
text. The server is shut down and joined after the test. This establishes the
HTTP disconnect path on this host, not Internet/proxy behavior or a latency bound
for arbitrary CUDA kernels. The earlier buffered TestClient tests remain separate.


## Desktop progressive text and stop control

Enable “Texto progresivo (servidor Foundation)” in provider routing while using
the AI@Home server. The existing full-response mode stays available by leaving it
disabled. Direct/external provider routing rejects this option explicitly.

A read-only, four-line preview shows the latest 2,000 characters during generation.
The worker sends fragments through the desktop event queue; only the Tk thread
updates widgets. Full text is retained by the bounded client consumer until done.
“Detener generación” requests cancellation and clears the preview. The request
remains active until its cancellation/success callback resolves, avoiding overlap
in that conversation. Late completion after stop is discarded.

Only successfully completed text is appended as an assistant message. Partial
text is never included in subsequent prompts or consented contribution payloads.
The user question remains in history after cancellation, so no completed assistant
answer is implied. Errors clear the preview and use the existing error display;
intentional cancellation uses a status message instead of an error popup.

Clear-chat, profile transitions and application close request cancellation.
Conversation ID plus cancellation-object identity guard queued fragments against
cross-conversation display. Profile changes reset streaming selection. Completed
streams retain the same source references, model-version pinning and contribution
restrictions as non-streaming chat. These controls apply to server streaming, not
to stopping legacy non-streaming provider requests.

Validation: 212 passed, 4 PostgreSQL skipped. A hidden real Windows Tk check fed
isolated UI fixture fragments through the queue, rendered the preview, stopped,
then delivered a late completion and verified that history contained only the
question. Clear-chat reset was also checked. Real model/network behavior remains
covered by the separate actual-checkpoint HTTP loopback test. Full visual layout
review and production proxy testing remain outstanding.


## Generation controls

Both chat routes accept an optional `generation` object:

```json
{
  "messages": [{"role": "user", "content": "Hola"}],
  "generation": {
    "max_new_tokens": 64,
    "temperature": 0.0,
    "top_k": 40,
    "top_p": 0.95,
    "repetition_penalty": 1.0,
    "seed": 42
  }
}
```

Omitted values retain the existing sampler defaults. The token count defaults to
the server setting; an explicit count above that setting is rejected (422), not
silently clamped. Global request bounds are 1–4096 new tokens, temperature 0–5,
top-k 0–100000, top-p greater than 0 and at most 1, repetition penalty 1–10,
and integer seed 0 through 2^63−1. Booleans/string coercions, non-finite floats,
unknown fields and fractional integer controls are rejected.

Temperature zero is greedy decoding. A seed controls stochastic sampling on the
selected device; reproducibility across different devices/runtime versions is not
promised. The full-response `generation` field and streaming metadata return the
effective values. These controls are Foundation-only; specifying them for an
external backend returns an explicit error. Desktop sampling controls are available through the optional dialog below;
existing defaults remain unchanged unless customized.

Actual-checkpoint tests compare direct inference, full HTTP response and SSE with
identical greedy settings and retrieved context. They assert equal generated
text, matching reported settings and token-budget rejection on both routes.


### Desktop generation dialog

“Ajustes de generación Foundation” opens the temperature, top-k, top-p,
repetition penalty, seed and token-limit controls. Enable custom settings and
apply; leave the token limit empty to retain the server limit. Disabling custom
settings restores server defaults. Values use the same strict bounded schema as
the API, after parsing text input into numbers. Invalid inputs leave the current
settings unchanged and display an error.

Both complete-response and streaming clients forward the selected settings.
Each request captures its own copy; editing while a request runs affects only the
next request. These controls require server routing and the Foundation backend;
they are not silently applied to direct/external providers. Settings are session
local, not stored on disk, and reset on profile changes. The actual server token
limit remains authoritative and oversized requests return an error.

Validation: 229 passed, 4 PostgreSQL skipped. HTTP-client tests cover forwarding
for both routes and malformed/out-of-range input. A hidden Windows Tk test invoked
the real dialog controls, applied temperature zero and verified account reset.
