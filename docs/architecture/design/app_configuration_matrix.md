# CompNeuroVis App Configuration Matrix

A taxonomy of every valid app configuration. Use this to drive architectural decisions — if a proposed refactor can't express a row in this matrix, it's the wrong abstraction.

---

## Dimensions

Every configuration is a point in this space:

| Dimension | Options |
|---|---|
| **Backend environment** | Same process, same-machine subprocess, WSL, remote cloud/server |
| **Frontend environment** | Same process, same-machine subprocess, remote browser, remote notebook |
| **Frontend renderer** | Vispy/Qt, Jupyter notebook (ipywidgets), Unity, Web (WebGL/Three.js), Headless |
| **Transport** | In-process queue, OS pipe, WebSocket, Shared memory |
| **Topology** | 1B:1F, 1B:NF (broadcast), NB:1F (aggregation), NB:MF (mesh) |
| **Interaction role** | Full (owner), Observer (read-only), Partial (constrained controls) |
| **Data source** | Live simulation, Replay, Static/one-shot, External stream |
| **Authoring API** | Inline sugar, Attach API, RunSpec, Bespoke |

---

## Topology Catalog

Named topologies used in the matrix below.

| ID | Name | Description |
|---|---|---|
| **T1** | Local single-process | Backend + frontend + orchestrator all in one Python process |
| **T2** | Local multiprocess | Backend subprocess + frontend main process, same machine, OS pipe |
| **T3** | Local thread | Backend daemon thread + frontend in same process (notebook pattern) |
| **T4** | Remote 1:1 | Backend and frontend in separate environments, WebSocket |
| **T5** | Broadcast 1:N | One backend, multiple frontend observers (teacher + students) |
| **T6** | Aggregation N:1 | Multiple backends feeding one frontend (multi-region, multi-cell) |
| **T7** | Mesh N:M | Multiple backends, multiple frontends, arbitrary routing |

---

## The Matrix

### Row key

- ✅ Implemented and tested
- 🔧 Implemented, gaps known (see Notes)
- 🔜 Designed, not implemented
- ❌ Not yet designed
- N/A Not applicable

---

### Vispy/Qt Frontend

| Topology | Backend env | Transport | Authoring | Status | Notes |
|---|---|---|---|---|---|
| T1 | Same process | In-process queue | RunSpec | ✅ | `run_app(RunSpec)` with `inprocess_transport` |
| T2 | Subprocess | OS pipe | RunSpec | ✅ | `run_app(RunSpec)` with `pipe_transport` + `ActorProcess` |
| T2 | Subprocess | OS pipe | Inline sugar | 🔧 | `inline.show()` / `attach().show()` — works but bypasses `RunSpec` |
| T2 | Subprocess | OS pipe | Attach API | 🔧 | Same bypass as inline |
| T4 | WSL | WebSocket | RunSpec | 🔜 | `run_as_backend` + `run_as_frontend` stubs exist, transport not built |
| T4 | Remote server | WebSocket | RunSpec | 🔜 | Same |
| T5 | Subprocess | OS pipe + broadcast | RunSpec | ❌ | Teacher controls Qt; students observe (Qt or other) |
| T6 | Multi-subprocess | OS pipes | RunSpec | ❌ | Multiple backends feeding one Qt frontend |

---

### Notebook Frontend (ipywidgets, VS Code / JupyterLab / classic Jupyter)

| Topology | Backend env | Transport | Authoring | Status | Notes |
|---|---|---|---|---|---|
| T3 | Same process (thread) | In-process queue | Attach API | 🔧 | `attach().show_notebook()` — works; `NotebookFrontendHost` now uses `AppRuntime` |
| T3 | Same process (thread) | In-process queue | Inline sugar | ❌ | `inline.show_notebook()` not yet designed |
| T3 | Same process (thread) | In-process queue | RunSpec | ❌ | No `RunSpec` path to notebook frontend yet |
| T4 | WSL | WebSocket | Attach API | 🔜 | Core motivation for keeping `AppRuntime` alive in notebook path |
| T4 | WSL | WebSocket | RunSpec | 🔜 | Depends on WebSocket transport |
| T4 | Remote server | WebSocket | RunSpec | 🔜 | Same |
| T5 | Subprocess | Any | RunSpec | ❌ | Teacher notebook (or Qt) + student notebooks as observers |

---

### Unity Frontend

| Topology | Backend env | Transport | Authoring | Status | Notes |
|---|---|---|---|---|---|
| T4 | Python subprocess | WebSocket | Bespoke | ❌ | Unity C# receives field updates; Python runs sim |
| T4 | Remote server | WebSocket | Bespoke | ❌ | Same, backend off-machine |
| T5 | Python subprocess | WebSocket broadcast | Bespoke | ❌ | One sim, Unity + notebook observers |

---

### Web Frontend (browser)

| Topology | Backend env | Transport | Authoring | Status | Notes |
|---|---|---|---|---|---|
| T4 | Python subprocess | WebSocket | Bespoke | ❌ | Browser WebGL renderer + Python sim |
| T4 | Remote server | WebSocket | Bespoke | ❌ | Fully remote |

---

### Headless / Data Export

| Topology | Backend env | Transport | Authoring | Status | Notes |
|---|---|---|---|---|---|
| T1 | Same process | None | RunSpec | 🔜 | Replay backend + file export frontend |
| T1 | Same process | None | RunSpec | 🔜 | Batch run, no UI |

---

### Special Configurations

| Config | Description | Status | Notes |
|---|---|---|---|
| **Static data viewer** | No simulation; renders pre-existing Field/Geometry data | ✅ | Replay backend + Vispy frontend |
| **Bespoke app** | Full custom app (e.g. NeuroML editor) using compneurovis primitives | ❌ | No sugar API; raw `BackendBase + FrontendBase + AppSpec` |
| **Classroom (T5 teacher/student)** | Teacher owns full-control session; students connect as observers with constrained `PartialInteractionRole` | ❌ | Requires 1:N broadcast transport + role-scoped `InteractionCatalog` |
| **Multi-backend aggregation (T6)** | e.g. C. elegans pharynx (muscle physics) + neural model feeding one frontend | ❌ | Multiple `BackendBase` actors, router/aggregator needed |
| **Physics + neuroscience (T6)** | Separate physics and neural backends, shared visualisation | ❌ | Same as above |
| **External data stream** | Frontend observes a live stream not generated by a CompNeuroVis backend | ❌ | Backend adapter wrapping e.g. BrainFlow, Lab Streaming Layer |

---

## Architectural Gaps Exposed by This Matrix

### Gap 1 — Inline/Attach bypass `RunSpec`

Inline and attach use subprocess-by-re-run, which isn't expressible as a picklable `ActorSpec.host_source`. They wire `AppRuntime + BackendHost + VispyFrontendHost` manually, duplicating `run_app()` logic.

**Fix candidate:** `ScriptRerunBackendProcess(script_path, endpoint)` — a `Startable` that spawns via `runpy.run_path`. Inline/attach then compile to a `RunSpec`.

### Gap 2 — `run_app()` is blocking; notebook frontend is non-blocking

`runtime.wait()` blocks on the foreground actor (Qt). A notebook frontend has no foreground actor — it starts an asyncio task and returns a widget. `run_app()` has no path for this.

**Fix candidate:** `start_app(RunSpec) -> AppHandle` that starts all actors and returns a handle. `AppHandle.widget` gives the notebook widget; `AppHandle.wait()` blocks for Qt. `run_app()` becomes `start_app(spec).wait()`.

### Gap 3 — 1:N broadcast transport not built

Teacher/student and multi-observer topologies require a transport that fans out updates to N endpoints. The current `pipe_transport` / `inprocess_transport` are strictly 1:1.

**Fix candidate:** `BroadcastTransport` — one inbound, N outbound endpoints. Frontend role distinguishes Full vs. Observer.

### Gap 4 — No WebSocket transport

`run_as_backend` / `run_as_frontend` exist as stubs. The WSL→notebook scenario, Unity frontend, and all remote topologies block on this.

### Gap 5 — No N-backend aggregation

Multiple backends feeding one frontend requires a router actor or a compound backend. No design exists yet.

---

## Priority Order (suggested)

1. **Gaps 1+2** — unify inline/attach/notebook under `RunSpec` + `start_app()`. Enables a single canonical execution model.
2. **Gap 4** — WebSocket transport. Unlocks the entire remote quadrant of the matrix.
3. **Gap 3** — Broadcast transport. Enables classroom / observer scenarios.
4. **Gap 5** — N-backend aggregation. Enables physics + neural model compositions.
