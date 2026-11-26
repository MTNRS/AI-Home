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
