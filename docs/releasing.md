# Releasing

Releases are manual. CI never publishes packages or GitHub releases, and there is no PyPI upload configured.

1. Update the version in `pyproject.toml` and `src/contextroast/__init__.py`, add changelog entries and `docs/releases/vVERSION.md`. Use a new version for each release; do not replace existing tags or assets.
2. Merge the changes after CI passes: Python 3.11–3.14, lint, coverage, package installation, dependency audit and secret scan. The FAQ holdout must exit 1 with `deployment_ready: false`.
3. From the default branch, run **Draft release** in GitHub Actions with the version number, without `v`.
4. Review the draft's notes, wheel, source archive and `SHA256SUMS`. The draft targets the exact workflow commit. Publish it in GitHub when ready.

The workflow runs only on the default branch, refuses an existing release or tag, checks the version, downloads the packages from successful CI for that exact commit, and creates a prerelease draft. It uses the repository's Actions token and makes no paid model calls.

## First public launch

Before changing visibility, review the source, release archives and Git history. Removing a file from the current tree does not remove it from old commits or releases. Commit author emails are public metadata.

When the repository becomes public:

- Enable private vulnerability reporting and confirm the link in `SECURITY.md` opens the reporting form.
- Protect `main`: require pull requests, resolved conversations, an up-to-date branch and passing `test (3.11)` through `test (3.14)`, `package`, `dependencies` and `secrets` checks. Require no approval count while there is only one maintainer. Block force pushes and deletion, including for administrators.
- Enable Dependabot alerts, secret scanning and push protection.
- Check the README, image and installation from a logged-out session before announcing the release.

GitHub Free enables branch protection and private vulnerability reporting for public repositories; these settings cannot be fully checked while this repository is private.
