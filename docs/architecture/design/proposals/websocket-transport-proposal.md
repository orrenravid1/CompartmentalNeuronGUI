---
title: WebSocket Transport Proposal
summary: Add a WebSocket transport so the backend session can run in a separate process or machine (e.g. WSL) while the vispy frontend runs on Windows. Also establishes the language-agnostic wire protocol needed for future non-Python frontends such as Unity.
---

# WebSocket Transport Proposal

Status: proposal

This document captures the plan for adding a WebSocket-based transport alongside the existing `PipeTransport`. It is not a settled decision yet. When the implementation stabilizes and the durable lesson is clear, the final doctrine should move into [Design Decisions](../decisions.md).

For the current high-level priority, see [Roadmap](../roadmap.md). For the deferred summary entry, see [Backlog](../backlog.md#remote-frontend-alternate-transport).

## Problem

`PipeTransport` uses `multiprocessing.Pipe` or a thread queue, both of which require the session and the frontend to live in the same OS environment. That makes it impossible to run the backend on WSL (Linux, no vispy, no display) and the frontend on Windows, even though WSL2 and the Windows host share a loopback-style network interface.

The immediate concrete need: NEURON simulations and other Linux-native backends need to run in WSL. VisPy renders on the Windows host. Pipes cannot bridge that boundary.

A secondary goal is to establish the wire protocol foundation that a future non-Python frontend (e.g. Unity) would need. That frontend replaces VisPy entirely and implements its own WebSocket client in C#; the backend server is the same in both cases.

## Goals

- Let the backend session run in WSL with no display, no Qt, no vispy.
- Let the vispy frontend on Windows connect to that backend via WebSocket over the WSL2 loopback.
- Add a `WebSocketTransport` to the vispy frontend that presents the same interface as `PipeTransport` and integrates with the Qt event loop.
- Add a standalone `run_backend_server()` entry point that runs a session loop with no Qt dependency.
- Design the wire protocol to be language-agnostic from the start so a Unity C# client can implement it later without forcing a serialization redesign.
- Keep every existing `Session` subclass unchanged.

## Non-Goals For This Phase

- Building the Unity frontend.
- Authentication or TLS (loopback/trusted-LAN use case only for now).
- Multi-client broadcasting (one backend, one frontend).
- Automatic reconnection or fault recovery beyond basic error surfacing.
- Changing the session protocol types or the `Session` / `BufferedSession` interface.

## The Two Extensibility Axes

Transport and frontend are independent axes and must stay that way.

**Axis 1 — transport:** how backend and frontend communicate (pipes, WebSocket, ...).
`AppSpec.transport` and `WebSocketTransport` live here. This axis is about the
vispy frontend's connection to wherever the backend runs.

**Axis 2 — frontend:** what renders the scene (vispy, Unity, browser, ...).
A Unity frontend replaces `VispyFrontendWindow` entirely. It does not use
`AppSpec` at all; it implements its own WebSocket client and speaks the same
wire protocol directly.

The connection between the two axes is the wire protocol. If the protocol is
language-agnostic, axis 2 can be exercised by any client language. If it is
Python-specific, axis 2 is blocked.

## Design

### 1. `AppSpec.transport` field

Add a `transport` field to `AppSpec`. The frontend uses it when present instead of constructing `PipeTransport` from `session`:

```python
@dataclass(slots=True)
class AppSpec:
    scene: Scene | None = None
    session: Any = None
    transport: Any = None       # new — pass one of session or transport, not both
    interaction_target: Any = None
    title: str | None = None
    diagnostics: DiagnosticsSpec | None = None
```

Frontend construction logic in `VispyFrontendWindow.__init__`:

- `transport` given → use it directly, skip `PipeTransport` construction
- `session` given, no `transport` → construct `PipeTransport(session)` as today
- neither → static scene, no polling

`WebSocketTransport.__init__` must not create the `QWebSocket` yet — it defers
that to `start()`, called inside the window constructor after the Qt application
exists. This matches the pattern `PipeTransport` already uses for subprocess
spawning.

Authoring shape:

```python
# Windows frontend
run_app(AppSpec(transport=WebSocketTransport("localhost", 8765)))

# WSL backend (no Qt, no vispy)
run_backend_server(MySession, host="0.0.0.0", port=8765)
```

### 2. `Transport` protocol

Define a structural `Transport` protocol so both transports are explicitly typed
without requiring inheritance:

```python
class Transport(Protocol):
    def start(self) -> None: ...
    def poll_updates(self) -> list[SessionUpdate]: ...
    def send_command(self, command: SessionCommand) -> None: ...
    def stop(self) -> None: ...
```

`PipeTransport` already satisfies this. `WebSocketTransport` implements it. The
`AppSpec.transport` field is typed `Transport | None`.

### 3. `WebSocketTransport` (frontend side)

A `QObject` that wraps `QWebSocket` from `PyQt6.QtWebSockets`. Signal-based,
integrates with the Qt event loop, no additional thread required.

Responsibilities:
- `start()` — create `QWebSocket`, connect signals, call `open(url)`
- On `binaryMessageReceived` signal → decode payload, append to internal queue
- `poll_updates()` — drain the queue and return updates (same call site as today)
- `send_command(cmd)` → encode and call `QWebSocket.sendBinaryMessage()`
- `stop()` → send `StopSession`, close the socket

The existing 60 Hz `QTimer` in the frontend drives `_poll_transport()` unchanged.
`WebSocketTransport.poll_updates()` just drains from an internal `deque` instead
of a pipe.

### 4. `run_backend_server()` (WSL side)

A standalone entry point with no Qt, no vispy dependency:

```python
def run_backend_server(
    session_source: SessionSource,
    host: str = "0.0.0.0",
    port: int = 8765,
) -> None:
```

Internally: asyncio + `websockets` library. Accepts one connection, runs the
same session loop as `_session_process` in `pipe.py`, sends serialized updates,
receives and dispatches serialized commands. When the connection closes or
`StopSession` arrives, shuts down cleanly.

The session loop logic is almost identical to `_session_process`. The main
difference is that sends and receives go through the WebSocket rather than
`Pipe.send` / `Pipe.recv`.

### 5. Serialization

**Pickle-first.** For the WSL-to-Windows vispy use case, use `pickle` as the
initial codec. The network hop (WSL2 loopback) is the bottleneck, not
serialization speed. Pickle of a numpy array and a proper binary codec are
comparable in payload size; the marginal difference is not measurable at
simulation frequencies. Pickle also requires zero codec development to ship the
first working implementation.

The WSL-to-Windows use case is trusted and same-environment (same Python
version, same numpy version), so the normal pickle safety concerns do not apply.

**Swappable codec seam.** Isolate encode and decode behind two functions from
day one so swapping to msgpack later is a one-file change:

```python
# in websocket transport module
def _encode(obj) -> bytes:
    return pickle.dumps(obj)

def _decode(data: bytes):
    return pickle.loads(data)
```

**Language-agnostic path (for Unity).** When a non-Python frontend is needed,
replace the codec with msgpack + numpy extension. `MessagePack-CSharp` is a
mature Unity library. msgpack with a numpy extension stores dtype + shape + raw
bytes — comparable size to pickle for array data, and natively decodable in C#.
JSON + base64 is viable for control/metadata-only messages but unsuitable for
dense array fields.

The decision to switch codecs is entirely isolated to the two functions above.
No changes to protocol types, session logic, or the frontend dispatch path.

### 6. WSL2 networking

WSL2 runs behind a virtual NAT adapter. The recommended pattern:

- backend binds `0.0.0.0:PORT` in WSL
- frontend connects to `localhost:PORT` on Windows

Windows 11 with recent WSL2 auto-proxies `localhost` to the WSL2 virtual
interface. If that fails, the WSL2 IP is available via `cat /etc/resolv.conf`
(nameserver line) and can be passed as a config argument to `WebSocketTransport`.

No special networking setup is required for the standard case.

### 7. Optional: Socket.IO variant

Socket.IO is WebSocket with a thin envelope layer on top: named events,
automatic reconnection with exponential backoff, a built-in heartbeat, and
optional room/namespace routing. It has been used successfully in prior work for
similar backend-to-frontend bridging and is worth documenting as an explicit
alternative to the raw `websockets` implementation.

**What it adds over raw WebSocket:**

- Auto-reconnection — directly resolves the reconnection open question without
  any custom retry logic.
- Named events (`socket.emit("update", data)` vs raw binary frames) — cleaner
  message routing if the protocol gains multiple message kinds at the socket
  level, though the current design handles multiplexing through payload type
  rather than socket events.
- Built-in heartbeat / ping-pong — no manual keepalive implementation needed.
- Room support — not needed now, but relevant if multi-client broadcasting is
  ever added.

**Server side (WSL):** `python-socketio` with `aiohttp` or `uvicorn`/`starlette`.
The session loop is the same; only the send/receive calls change.

**Client side (Windows/Qt) tradeoff:** This is the complication. `QWebSocket`
speaks raw WebSocket; it does not implement the Socket.IO handshake and envelope
protocol. Two options:

- Run `python-socketio`'s asyncio client in a daemon thread, feeding received
  updates into a `queue.Queue` that `poll_updates()` drains on the Qt thread.
  This works but adds a thread and an asyncio event loop alongside the Qt event
  loop.
- Implement the Socket.IO handshake on top of `QWebSocket` manually. Not
  recommended — it duplicates what `python-socketio` already provides.

The thread-based client option is straightforward and has no correctness risk.
The cost is a slightly more complex `WebSocketTransport` compared to the pure
`QWebSocket` path.

**For Unity:** Several maintained Socket.IO client libraries exist for Unity
(`SocketIOUnity`, `socket.io-unity`). They handle reconnection, namespaces, and
typed event callbacks natively in C#. This is a meaningful advantage if Unity is
the target frontend — it removes the need to implement reconnection and keepalive
in C# from scratch.

**Recommendation:** Start with raw `websockets` + `QWebSocket` for the initial
vispy/WSL implementation. It is the simpler path, and the only open question it
leaves unanswered is reconnection. If reconnection turns out to be important in
practice, or if a Unity client is the next frontend, switching the server to
`python-socketio` and the client to the thread-backed asyncio variant is a
contained change that does not touch anything above the transport layer.

## Effort Estimate

| Piece | Effort |
|---|---|
| Swappable codec (pickle, two functions) | half day |
| `WebSocketTransport` (QWebSocket-based) | 1 day |
| `run_backend_server()` (asyncio + websockets) | 1 day |
| `AppSpec.transport` field + `Transport` protocol | half day |
| WSL connection config + example | half day |
| **Total** | **~4 days** |

## Tradeoffs and Risks

- Pickle as initial codec is fast to ship and safe for the trusted loopback case,
  but it creates a hidden dependency on both sides using the same Python and numpy
  versions. This is acceptable for the WSL use case, but must be replaced before
  any untrusted or cross-version deployment.
- `QWebSocket` is synchronous-friendly (signal-based) but adds `PyQt6.QtWebSockets`
  as a frontend dependency. This module ships with PyQt6 on Windows; no extra
  install is required.
- `websockets` (asyncio) adds a backend dependency. It is a small, stable library
  with no C extensions, appropriate for a WSL server with no display.
- A single-connection server is the simplest correct model for now. Multi-client
  support is a separate problem and should not be designed in prematurely.
- `AppSpec` now has two mutually exclusive session-source fields (`session` and
  `transport`). The mutual exclusivity should be enforced at runtime with a clear
  error message rather than silently preferring one over the other.

## Open Questions

- Should `WebSocketTransport` attempt reconnection when the connection drops,
  or surface a fatal error and stop the poll timer as `PipeTransport` does on
  worker death? Switching to Socket.IO (see section 7) would resolve this
  without any custom retry logic.
- Should `run_backend_server()` accept a `startup_scene` hook so the Windows
  frontend can display a loading scene before the first `SceneReady` arrives, as
  the current `startup_scene` classmethod does for `PipeTransport`?
- What port should be the default? 8765 is conventional for Python WebSocket
  examples and unassigned by IANA for this range.

## References

- [Session Protocol](../../session-protocol.md)
- [VisPy Frontend](../../vispy-frontend.md)
- [Roadmap](../roadmap.md)
- [Backlog](../backlog.md)
