# Findings

## Robustness Review Findings

- The previous runtime assumed `/bin/zsh` existed, which makes command execution brittle on slimmer environments; shell selection must be discovered at runtime.
- A three-step shell fallback is sufficient for this project: `zsh` -> `bash` -> system-default shell when neither binary is found.
- The previous documentation over-emphasized `uv`; a robust runtime should treat `uv` as preferred, not required.
- Install-command auto-detection works best when it separates package-manager choice from project-layout choice:
  - `uv sync` for `pyproject.toml` when `uv` exists
  - `python -m pip install -r requirements.txt` when requirements are present
  - `python -m pip install -e .` for editable local installs
- Using `sys.executable -m pip` is more portable than relying on a `pip` executable path, especially in virtualenv-heavy environments.
- Environment detection should be visible in more than one place: runtime command records, `status`, contract docs, and handoff docs.
- The fallback logic is worth testing at the unit level because the failure mode is environmental and easy to miss in normal local development.
- The robustness work did not change the exactness contract; it only made the harness more resilient to host-environment differences.
