# Security

Report vulnerabilities privately through [GitHub's reporting form](https://github.com/newapple77/ContextRoast/security/advisories/new). Include the affected version, impact and a minimal reproduction without real credentials or customer data. If the form is unavailable, open an issue requesting a private contact; leave out vulnerability details.

Security fixes target the latest release. Older alpha versions are unsupported.

## Data and execution

- The mock runs offline. ContextRoast has no telemetry.
- The API provider sends instructions, case inputs and selected context to the endpoint you configure. Expected answers stay in the local scorer.
- Python entrypoints run unsandboxed imports and functions. They can read files, contact services, print sensitive data and incur charges. `--allow-paid` is execution consent, not a spending cap.
- Detailed JSON/Markdown reports may contain inputs, expected answers and predictions. Keep them, private datasets and credentials out of commits. The SVG card exports aggregate metrics only.
