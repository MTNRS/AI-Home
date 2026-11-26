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

## 2026-11-26 — Ventana de inferencia visual

Inspección directa mediante Computer Use de la ventana real `visual_desktop` en
760×560 y en su mínimo 600×450 (área cliente). En ambos tamaños se observaron
completos el aviso de privacidad, selectores checkpoint/PNG-JPEG, instrucción,
CPU/CUDA, Generar/Cancelar, estado y área de respuesta con scrollbar. No se
observaron solapamientos ni controles inaccesibles. El primer snapshot estaba
oculto por otra app; se activó la ventana correcta y solo se evaluaron las capturas
posteriores. El intento de redimensionar por arrastre no cambió su tamaño, por lo
que se abrió otra instancia de prueba configurada al mínimo antes de inspeccionarla.

Fixtures locales sin cuenta ni red: artifacts/ui-review/visual.py y
visual-minimum.py. Se cerraron por la UI; ambos procesos terminaron con código 0.
No se modificó el diseño porque la inspección no encontró un defecto en estos
estados. No se revisaron otros DPI, lector de pantalla, diálogos de archivos,
mensajes de error ni una respuesta larga durante esta inspección. Las pruebas
funcionales de generación y aislamiento están documentadas aparte en VISION_PREFIX.

## Panel de nodo ampliado — revisión 2026-11-26

Revisión nativa Windows del panel DISTRIBUTED COMPUTE tras a1e2850, con
preferencias vacías y sin iniciar sesión ni registrar un nodo. Se inspeccionó
la ventana real a 720x560 (mínimo) y 900x700 (habitual). En ambos tamaños se
ven completas las casillas de consentimiento, entrenamiento y medición HTTP,
el selector CPU/CUDA y los botones iniciar, detener y revocar. Textos legibles,
sin solapamientos observados; el tamaño mínimo ajusta los párrafos en varias
líneas. La selección inicial mantiene entrenamiento y medición desactivados.

Un intento de ampliar mediante arrastre fue rechazado por límites de coordenadas
del controlador; se abrió una segunda instancia con geometría 900x700 para
inspeccionarla. Ambas ventanas se cerraron y sus procesos finalizaron con código
0. Esta revisión cubre el DPI actual; no otros escalados, lectores de pantalla,
mensajes de error o todas las combinaciones de controles. Los smokes CPU/CUDA
separados aportan la evidencia funcional.
