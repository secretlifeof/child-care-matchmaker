This workspace is configured to use Ruff as a code-action on save.

Recommended local install (macOS Homebrew + pipx):

1. Install pipx via Homebrew (if you don't have pipx already):

   brew install pipx
   pipx ensurepath

   After running `pipx ensurepath` you may need to restart your terminal or VS Code for PATH changes to apply.

2. Install ruff using pipx:

   pipx install ruff

3. Test ruff from the terminal:

   ruff --version
   ruff check --fix src/matchmaker/database/complex_queries.py

VS Code tips:
- Install the "Ruff" extension (charliermarsh.ruff) and the Python extension (ms-python.python).
- The extension will use the `ruff` binary found on your PATH. Ensure VS Code was restarted after installing pipx and running `pipx ensurepath`.
- The workspace setting enables `source.fixAll.ruff` on save which applies fixes (including --fix behaviour).

If you'd rather pin ruff per-project, add it to your dev environment (poetry/hatch/venv) and point VS Code to that interpreter under "Python: Select Interpreter".
