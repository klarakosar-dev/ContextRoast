# Contributing

Use Python 3.11+ and install the development dependencies in a virtual environment:

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest --cov=contextroast --cov-branch --cov-fail-under=95
python -m build
python -m twine check dist/*.whl dist/*.tar.gz
```

Keep changes focused. Add regression tests for changes to selection, scoring, validation or accounting. Tests must stay offline; use `httpx.MockTransport` for API tests. Update the relevant guide when changing a dataset or report format.

Preserve explicit consent for network calls and Python execution. Keep token estimates distinct from reported usage, and missing prices distinct from zero.

Bug reports should include the command, Python/ContextRoast versions, expected behaviour and a small synthetic reproduction. Remove credentials, private prompts and personal data. Use a GitHub noreply commit address if you want to keep your email private. Report vulnerabilities through [the security policy](SECURITY.md).
