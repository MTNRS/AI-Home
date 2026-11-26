# Desktop visual review — 2026-11-26

Inspected the live Windows Tk application through the computer-use screenshot and
input API at its default 900×700 client size. The review process used empty
settings and memory loaders and a distinct “AI@Home Visual Review” title. No
account was authenticated, credentials edited, or network action submitted.

Observed defects before changes:

- The memory explanation consumed the chat control row and clipped the stop label.
- Provider settings extended below the viewport; server/retrieval controls were
  inaccessible in the fixed page.
- The conversation canvas minimum requested height squeezed the composer and
  status bar as new controls increased the required height.

Implemented and visually rechecked:

- The memory explanation wraps on its own row; stop and source buttons show fully.
- Provider, account and node settings use vertically scrollable pages with a
  visible scrollbar and mouse-wheel support. Wrapped labels track page width.
- The chat canvas has a flexible minimum height, preserving composer space.
- Status is allocated before the expanding page stack so it remains visible.
- Scrolling the provider page reveals both server-routing and retrieval controls,
  plus the explanatory text at the bottom.

The corrected chat and provider pages were observed in the running application.
The isolated review window was then closed and its absence verified. The full
CUDA-enabled test suite passed: 229 passed, 4 PostgreSQL tests skipped.

Limits: this was not a complete visual review of every dialog, minimum-size layout,
DPI scaling, screen readers, or authenticated workflows. Account and node pages
share the corrected scroll container, but their complete visual content was not
individually inspected. Earlier hidden-widget tests remain functional evidence,
not substitutes for visual inspection.

## Correction editor (2026-11-26)

Inspected the real Tk correction dialog from commit `a112a2b` on Windows using
a development fixture with four authored Spanish messages, no account credentials
and an unused loopback endpoint. At its initial 800×620 client size, the original
prefix, replacement editor, both unchecked consent/review boxes, explanation and
submit button were visible and legible. Selecting assistant message 1 instead of
3 visibly removed the later turns and populated the corresponding replacement.
Attempting submission with both boxes unchecked displayed “Confirma consentimiento
y revisión.” No contribution was transmitted in this visual review. The separate
functional API/DB smoke remains the evidence for successful submission.

Resize attempts did not change the observed window size, so minimum-size behavior
and DPI scaling remain unverified. Long-content scrolling and accessibility also
remain outside this review. The fixture window was closed and its absence checked.

## Distributed job viewer (2026-11-26)

Inspected the real Tk viewer from revision `6a25e3f` with a development-only
62-job fixture. No real account, network request or contribution was involved.
At the default 860×640 client size, all four navigation buttons, job rows and the
selected job's full constraints/explanations were legible. Clicking Más antiguos
visibly changed page 1 (50 jobs) to page 2 (12 jobs), enabled Más recientes and
disabled Más antiguos at the end.

A second fixture window started at the supported minimum 640×480 client size.
Navigation and the job list remained visible. The detail pane required scrolling;
scrolling revealed the complete freshness/device/profile explanation and the
warning that queued does not identify a cause. There was no inaccessible clipped
control or text in these observed states. Both windows were closed and their
absence checked; their fixture processes also exited.

This covers those two window sizes at the current Windows display configuration,
not every DPI, screen reader, keyboard-only path or error dialog. The records were
synthetic UI fixtures, not evidence of completed distributed jobs. The separate
HTTP/CPU/CUDA experiments remain the evidence for real work. No screenshot of a
real user's account or private work was captured.

## Named control navigation (2026-11-26)

Inspected the new Sección selector on the development app with empty settings and
no account. At 900×700, the ten named sections appeared in the dropdown and
selecting TRAINING opened the real page with train/resume, inspection and consented
contribution entry points. Chat's controls and composer remained visible. A second
window started at the minimum 720×560; its selector, original shortcuts, stop/
reference controls, composer and status remained accessible. Both windows were
closed and absence verified. No training/network/account action was triggered.

All ten routes and the existing training form were additionally checked with
hidden real-Tk widgets. The visual check does not cover every section/dialog,
other DPI settings, keyboard-only navigation or screen readers.
