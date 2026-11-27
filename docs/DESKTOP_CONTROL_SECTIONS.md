# Desktop control sections

The persistent **Sección** selector exposes the ten requested areas. Existing
Chat/Conectar IA/Cuenta/Colaborar shortcuts remain available; switching them also
updates the section indicator. Existing dialogs and their validation, privacy,
background work and cancellation paths are reused.

| Section | Working entry points |
| --- | --- |
| FOUNDATION MODEL | Checkpoint architecture/configuration/parameter inspection |
| TRAINING | Train/resume, inspect training/checkpoint, consented contributions |
| EVALUATION | Execute a suite, open evaluation reports |
| EVOLUTION | Controlled experiment, open experiment results |
| KNOWLEDGE | Explore reviewed index/sources, build a new index |
| DISTRIBUTED COMPUTE | Existing node controls and private paginated job viewer |
| MODELS | Provider/model/version selection, local registry approval/activation/rollback, checkpoint inspection |
| CHAT | Existing conversation view and real selected backend |
| PRIVACY | Profile memory, contributions, account/consent controls |
| CREDITS | Authenticated balance, transactions and inference limit |

Section descriptions are navigation guidance, not live performance statistics.
Metrics remain in their real checkpoint/report/server-backed dialogs. The selector
does not start training, grant consent, authenticate, approve a model, alter a
credit limit or connect an external provider. Account-gated actions retain their
existing requirements. This is navigation consolidation, not evidence of new
model capabilities or a complete integrated dashboard.

Validation: `python scripts/verify_desktop_sections.py` exercises real hidden Tk
widgets, routes all ten entries and opens the existing training form without
starting training. It uses empty development settings and no account. Visual
inspection checked the selector, Chat and Training at 900×700, and selector/chat
controls/composer at 720×560. Existing dialogs were not all visually re-reviewed;
DPI/accessibility coverage remains incomplete. Windows/controller evidence is in
[DESKTOP_VISUAL_REVIEW.md](DESKTOP_VISUAL_REVIEW.md).

## Installed-wheel navigation gate

The functional script accepts `--require-installed` to reject imports outside
`site-packages`. It passed again on Windows using the isolated non-editable wheel
and Python `-I`, rather than the development checkout. Linux Foundation CI now
installs Tk/Xvfb and runs the same script with an isolated interpreter on a virtual
display after installing the wheel. This is a widget/routing test, not a screenshot
review; visual claims still come only from the explicit Windows observations.
The first remote execution passed on revision `2969fa1`: [Foundation run
45827171081](https://github.com/MTNRS/AI-Home/actions/runs/45827171081). Its log
confirms all ten real-Tk routes and opening the training form. The same run passed
291 tests (15 optional PostgreSQL/CUDA omissions), real training/packaging,
installed API and distributed-worker checks, and artifact upload.
