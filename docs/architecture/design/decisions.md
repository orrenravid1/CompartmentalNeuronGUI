---
title: Design Decisions
summary: Settled architectural decisions and the lessons that motivated them. Changed rarely and deliberately.
---

# Design Decisions

Settled architectural choices and the evidence behind them. This document is changed rarely and deliberately — new insights start as notes in `roadmap.md` or `backlog.md` and are elevated here after review.

For active planning and priorities, see [Roadmap](roadmap.md). For deferred ideas, see [Backlog](backlog.md).

---

## Cross-Platform Launch Behavior

**Decision:**
- User-facing launch code should work the same on Windows, Linux, and macOS.
- `run_app(...)` must protect against spawned child imports internally.
- `if __name__ == "__main__":` is allowed in user scripts, but should not be required by the library just to make examples work.
- Session construction should be lazy by default for worker-backed apps. The recommended public launch pattern is to pass a session class or top-level zero-argument factory, not an already-created session instance.
- Eager session instances should not be supported for worker-backed apps.
- Frontend-only interaction state should be a separate concern from backend session construction. The canonical architecture should not rely on instantiating a second full session in the UI process just to recover custom interactions.

**Why:**
Worker-side error routing only applies to code that runs inside the worker. If a user script constructs a session eagerly at module scope, constructor side effects and exceptions happen before worker error handling can help. On Windows, multiprocessing spawn requires `if __name__ == "__main__":` guards in user scripts — but the library should not impose that requirement. The chosen default is session-side semantic interaction handling (`InvokeAction`, `KeyPressed`, `EntityClicked`) with session-emitted `StatePatch` / `Status` responses, rather than a second UI-process session.

---

## Frontend Invalidation

**Decision:**
- Whole-window refreshes are too coarse for performance-sensitive scenes.
- The frontend should invalidate only the affected targets.
- Protocol and scene updates should follow the same rule: send only the information required by the affected targets, not broad bundled refreshes by default.
- The cost model should be opt-in: if a backend or frontend wants broader updates, it should ask for them explicitly.
- Current explicit targets: controls, morphology, surface visual, surface axes, surface slice overlay, and per-view line plots.

**Why:**
Real-scene profiling showed that slice changes should update only the derived line plot and slice overlay — not the surface mesh, axes, or unrelated panels. Surface rendering must distinguish geometry updates, color/data updates, axes updates, and overlay updates separately. Long-lived visuals and renderer-side caches are required for good performance; rebuilding them on every change was the original bottleneck.

---

## Protocol Granularity and History Separation

**Decision:**
- High-throughput rendering workflows should default to need-to-know updates.
- `FieldAppend` and `ScenePatch` should be the normal path when they can express the change correctly.
- `FieldReplace` remains valid but is the explicit broader-cost path, not the default.
- Latest-state display and captured history are different concerns and should not be forced into the same default storage and update path.
- `HistoryCaptureMode`: `ON_DEMAND` by default; `FULL` as an explicit opt-in for all-entity history capture.
- Display roles must be field-generic rather than voltage-specific. Default field ids are role-based conventions (`segment_display`, `segment_history`), not claims that the displayed quantity is inherently voltage.
- The intended model: arbitrary display fields for current morphology/surface coloring, arbitrary history fields for retained traces or replay — with voltage treated as one common preset, not the defining concept.

**Why:**
Live simulation backends should not resend full trace history on every update. Incremental live data belongs in typed append-style updates, with the frontend owning the displayed rolling history. The original NEURON viewer had a useful split — latest morphology values streamed on the fly, trace history recorded only when opted into — and that split should return in typed form. Batching currently does double duty as throughput control and perceived smoothness control; those concerns should eventually separate (see roadmap). Full history for every entity is valuable for click-later trace inspection and replay, but should be configurable rather than imposed on every live session.

---

## Architectural Automation

**Decision:**
- Important vocabulary and protocol taxonomy decisions should not live only in prose.
- If the repo decides a term is retired or canonical, that decision should be encoded in machine-readable checks (`docs/architecture/invariants.json`).
- Breaking terminology changes should prefer immediate convergence plus automated detection of stale names over compatibility aliases that let drift persist unnoticed.
- For deliberate taxonomy changes, the preferred policy is: remove the old term, encode it as banned in invariants, regenerate derived docs, let checks surface any missed sites.

**Why:**
Compatibility aliases can hide incomplete architectural migrations by keeping tests green while docs, skills, or generated references remain semantically stale. Experience showed that letting aliases persist meant stale terminology lived undetected across multiple files and surfaces simultaneously. Automated enforcement is the only reliable fix once the codebase grows beyond what manual review can catch consistently.

---

## Explicit Frontend Host/Panel Naming

**Decision:**
- When a visible frontend region has both a host widget and an inner rendering/control widget, public seams should name both layers explicitly.
- Avoid ambiguous singular convenience handles once multiple panels or wrapper hosts exist.
- Prefer plural collections and explicit lookup helpers for repeated regions, for example `line_plot_host_panels`, `line_plot_panels`, and `line_plot_panel(view_id)`.

**Why:**
Once line plots and controls adopted the same framed host-wrapper pattern as 3-D views, old singular convenience names stopped saying whether callers meant the visible panel chrome or the inner widget that actually plots data or owns controls. Keeping both names hid that distinction and made multi-panel layouts harder to reason about. Explicit host/panel naming keeps tests, docs, and future host abstractions aligned.

---

## Generic Layout System

**Decision:**
- Layout should be fully generic and composable, with one default arrangement rather than hard-coded app categories.
- The user-facing model should be closer to Blender, Unity, or Unreal: a default layout that works immediately, customizable panel arrangement when needed, no assumption that a 3D viewport is always primary.
- The current 2D-only collapse behavior is a temporary step, not the target design.
- Multi-series line plots are a first-class need within the generic layout system, not an edge case.
- 3D hosting should be a swappable layout concern: `ViewSpec` describes what to render; a separate host layer describes whether views use independent canvases, shared canvases, or other strategies. The current one-view-one-canvas behavior is a host implementation, not the permanent architectural assumption.

**Why:**
A narrowly simplified abstraction can still be wrong if it is not composable. A user who starts with "plot a few live traces and expose sliders" must still be able to add morphology or a surface later without changing conceptual frameworks. The right simplification is a feature-based authoring layer over the common core model, not separate specialized app categories with different mental models. The current splitter-and-fixed-slots approach encodes a temporary model that will need to be replaced once Phase 2 authoring work begins in earnest.

---

## Startup Layout Behavior

**Decision:**
- A live app should not visibly start in a fallback layout and then jump to the intended layout if the initial structure is already knowable.
- If layout and views are known before the session starts, provide a startup `Scene` up front via `@classmethod startup_scene(cls) -> Scene | None`.

**Why:**
Visible startup jumps signal that layout knowledge is fragmented across the app construction path and create a poor first impression. The startup scene hook keeps the startup path generic and avoids hand-building `AppSpec(scene=...)` in each app script.

---

## Public Interaction API

**Decision:**
- The internal architecture may use tools/controllers/manipulators, but the default user-facing API should not require users to think in those terms.
- For the intended audience, custom interactions should be expressible with a few small callbacks and strong defaults.
- The framework should expose semantic interaction hooks: action/button invocation, key press, clicked morphology entity.
- For worker-backed apps, these hooks should live on the session and be driven by semantic commands rather than frontend-only callback objects.
- Per-app interaction policy should stay outside core renderer/transport logic.

**Why:**
Profiling real app authoring (external pharynx workflow, signaling cascade) showed that if a user has to ask "does this method go in the frontend object or the session object or it breaks pipes?", the authoring model has failed. The intended audience is closer to SciPy, matplotlib, NEURON, PyTorch, and Plotly users than engine/tool authors. Splitting per-app logic across frontend and backend classes just to make pipes work is a sign that the public API is still too exposed.

---

## Public Authoring Surface

**Decision:**
- The simplified public API must stay feature-composable.
- High-level helpers should assemble the same underlying `Scene + Session + Frontend + Transport` model rather than introducing separate app families.
- Backend choice, feature choice, and layout choice must stay orthogonal. A NEURON-backed signaling cascade app with controls and traces but no morphology is a valid first-class app shape.
- The preferred user experience: declare controls, declare exposed series/fields, declare views/layout features, opt into built-in capabilities such as reset or pause/play, provide model lifecycle hooks — the library owns scene assembly, history buffering, and protocol packaging.
- Adding morphology, surfaces, or extra plots to an existing app should mean adding declarations, not rewriting the app around a different abstraction.
- Backend-specific helpers such as `build_neuron_app(...)` are acceptable as transitional conveniences, but they should not become the conceptual boundary of the library. The long-term model should read like "choose a backend, then add features" rather than "pick an app type."

**Why:**
A narrowly simplified abstraction can still be wrong if it is not composable. Backend-labeled helpers can become misleading if they imply a default visualization mode — `build_neuron_app(...)` should not imply that morphology is always shown. Real examples (signaling cascade, external pharynx workflow) that still read like framework plumbing after simplification are evidence that the public API is not done yet.

---

## Technical: NumPy Masked Divide

`np.divide(..., where=...)` without `out=...` can leave masked entries undefined and produce warnings. Use explicit output buffers for geometry normalization paths.
