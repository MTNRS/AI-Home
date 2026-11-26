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
Desktop incremental rendering and its stop button remain pending; current desktop
chat continues to use the non-streaming route.
