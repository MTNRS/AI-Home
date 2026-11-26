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
