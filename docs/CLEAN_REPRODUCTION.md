# Clean installation reproduction — 2026-11-26

Verified source revision `1f12fd8` in a detached worktree, initially clean, using a
new Windows Python 3.11.9 virtual environment with no system site-packages.
Installed PyTorch 2.15.0+cpu from its CPU wheel index and the project with `[llm]`
through a normal wheel build/install (not editable). No development dependencies
or existing CUDA environment were used. `pip check` returned no broken requirements.

The smoke and package commands ran from outside the checkout with Python `-I`,
which ignores `PYTHONPATH` and excludes the current directory from module lookup.
`ai_home.__file__` pointed to the new environment's `Lib/site-packages`. Training,
resume, checkpoint loading, generation, packaging and verification all exited 0.
The wheel build created an untracked `build/` directory in the checkout; tracked
source files remained unchanged. This is environment isolation on the existing
Windows host, not a new machine or offline installation.

Reproduction sequence (PowerShell, replace locations with new paths):

```powershell
git worktree add --detach clean-source 1f12fd8
python -m venv clean-venv
clean-venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
clean-venv\Scripts\python.exe -m pip install './clean-source[llm]'
clean-venv\Scripts\python.exe -I -m ai_home.foundation.smoke --output-dir clean-run --device cpu
$revision = git -C clean-source rev-parse HEAD
clean-venv\Scripts\python.exe -I -m ai_home.foundation.smoke_package build clean-run nano.zip --revision $revision --license clean-source/LICENSE
clean-venv\Scripts\python.exe -I -m ai_home.foundation.smoke_package verify nano.zip
clean-venv\Scripts\python.exe -m pip check
```

Actual results are in [CLEAN_REPRODUCTION_CPU.json](CLEAN_REPRODUCTION_CPU.json):
27,808 parameters, step 8, embedding L2 change since resume 0.0236474276,
validation loss 5.6156215147. The generated text is not coherent conversation.
The ZIP SHA256 is `1fc610da75c5ffd1591744d845ebe96ca0d95e5e3281e24eb4dd36342864e70c`.
This is a mechanical reproduction, not a quality improvement.

The [Linux CI run](https://github.com/MTNRS/AI-Home/actions/runs/45819653096)
also built, verified and uploaded `foundation-nano-smoke-package` (artifact
15117526133). Downloaded that artifact and verified its inner ZIP with the same
isolated Windows installation. The inner ZIP SHA256 is
`80e359780d2af2cbd7d62411970bf02f5c16aab63e9bdd040367e1f88c4b50d0`.
This verifies cross-platform loading/metadata integrity, not byte-identical
training, hardware equivalence or provenance authentication. Actions artifacts
have retention limits; this is not a permanent release asset.

PyTorch emitted an optional NumPy initialization warning because NumPy was not
installed; this workflow completed without it. Dependency ranges are not a locked
reproducible environment. Local installation logs and `pip freeze` are retained
under ignored `artifacts/reproduction-*`. CUDA reproduction, API/database startup,
GUI installation, registry approval and full distributed deployment were not
retested in this fresh environment. Existing separate tests do not turn this
narrow installation check into a complete platform acceptance audit.

## Installed API and packaged migrations

The subsequent migration-packaging change moves the existing Alembic history into
`ai_home.migrations` without changing revision IDs or migration operations. Both
the repository's `alembic.ini` and the installed command use this one history.
After installing the updated wheel, migration no longer requires the source tree:

```powershell
$env:AI_HOME_DATABASE_URL = 'sqlite+pysqlite:///./development.db'
$env:AI_HOME_ENVIRONMENT = 'development'
python -m ai_home.migrate upgrade
python -m ai_home.migrate current
python -m ai_home.migrate check
```

`upgrade` explicitly applies all pending revisions. It is not run automatically at
API startup. Existing deployment backup/change procedures still apply. The CLI
uses the existing Settings/environment configuration and does not print secrets.

Verified the updated non-editable wheel in the isolated CPU environment using
`python -I scripts/verify_installed_api.py --output-dir artifacts/installed-api --checkpoint artifacts/reproduction-run-v1/foundation-nano.pt`.
The script creates a new local SQLite database exclusively, applies packaged
migrations, checks schema drift and starts a real loopback Uvicorn server. It
checks 401 without authentication, registers/logs in a synthetic account, performs
real Foundation completion and SSE inference, and stops the server. Credentials
are generated in memory and excluded from its report. It clears inherited app
environment settings for this fixture; it does not use production services.

[Observed report](INSTALLED_API_SMOKE.json). The reusable script was also run
successfully with a second isolated database. Full local regression suite:
292 passed, 5 optional PostgreSQL tests skipped. A migration test upgrades an
existing revision 0008 database to 0009 from outside the checkout, repeats upgrade
idempotently and checks schema drift. Linux CI now runs the installed-wheel HTTP
check too; this addition still needs its remote result. GUI installation and full
distributed deployment remain outside this fresh-environment verification.

## Installed distributed runtime

The subsequent distributed smoke uses the packaged Alembic history and schema
check instead of `Base.metadata.create_all`. Its explicit disposable Settings are
passed to Alembic, so an unrelated environment database URL cannot redirect the
migration. A regression test verifies that the environment-selected database is
not created when explicit Settings target a different database.

The coordinator and worker subprocesses inherit Python `-I` when the controller
runs isolated. From outside the checkout, with the updated wheel installed:

```powershell
clean-venv\Scripts\python.exe -I -m ai_home.foundation.distributed_smoke --output-dir distributed-run --rounds 2 --reward-credits 3 --worker-devices cpu cpu --restart-coordinator
```

[Actual Windows CPU report](INSTALLED_DISTRIBUTED_SMOKE.json): 28,192 parameters,
four worker processes, two rounds, 26.57 seconds, embedding L2 change 0.1838996708.
Two worker accounts each earned six synthetic credits for verified gradients;
the requester finished with zero available/reserved credits. Four separate
coordinator invocations returned waiting/waiting/committed/committed, preserving
job IDs and the committed checkpoint on retry. The smoke compares restored
weights against its uninterrupted reference and rejects duplicate jobs.

Linux CI now runs this installed-wheel experiment and publishes the API and
distributed JSON reports, excluding disposable databases and credentials. The new
remote result remains to be checked. This is real process collaboration on one
host, not independent physical machines or evidence about residential bandwidth,
latency, coordinator service failover, or useful language quality. Production
PostgreSQL/authentication and the GUI remain separate deployment checks.


## Installed distributed evaluation planner

Revision `a945f23` was built and installed as a non-editable wheel into the
existing isolated Windows CPU reproduction environment. Wheel SHA256:
`8b949db98e2ba91dc5001f238308616e2909e30e835b149c7b9853e365510c3c`.
With PyTorch `2.15.0+cpu`, `python -I scripts/verify_desktop_evaluation_plan.py
--require-installed --checkpoint artifacts/evaluation-dataset-cli-v1/model.pt
--dataset artifacts/evaluation-dataset-cli-v1/holdout.json
--output artifacts/evaluation-dataset-cli-v1/installed-desktop-plan.json` exited 0.
The source checkpoint had been trained using the separate CUDA environment;
the installed CPU planner loaded it and generated a plan for its own runtime.

The real Tk check created the plan with explicit consents, switched profiles
during an active task, and discarded late success/error callbacks. The deliberate
late success wrote its local plan as documented. Imports resolved to site-packages
and pip check found no broken requirements. The pre-existing optional NumPy
warning remained; it did not prevent the checks. This reuses an existing virtual
environment and host, not a new-machine reproduction or visual layout review.
Remote validation of this revision was still running when checked.
