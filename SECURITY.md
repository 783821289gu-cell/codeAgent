# Security and public-demo scope

RepoPilot is a local coding-agent demonstration for trusted repositories. Public source code
does not imply a hardened execution service or permission to process confidential repositories.

- Shell commands execute as the current OS user. Regex command checks, timeouts, and file-path
  checks are defense in depth, not a sandbox. Do not expose this CLI as an unauthenticated service.
- File reads, tool outputs, repository snippets, and session history may reach the configured
  model provider or PostgreSQL. `.gitignore` prevents ordinary Git publication; it is not an
  access-control policy for the agent. Keep production secrets outside the target workspace.
- `run_shell` side effects are not fully tracked like `create_file` / `replace_text`; the current
  completion gate records the latest test-tool status and Reviewer decision, not a signed proof
  of code correctness. Tests and reviews can miss defects.
- Store provider keys and database passwords only in local environment variables or `.env`.
  The example connection strings are placeholders. Use a separate development database role.
- Existing `artifacts/` are curated synthetic login demos. New artifacts are ignored by default:
  inspect outputs for source code, credentials, personal paths, and real user data before sharing.
- The publication scan did not identify production credentials or real customer datasets in
  tracked content. Runtime databases, virtual environments, and model weights are excluded.
  Paths in retained reports are anonymized demonstration paths, not the original machine paths.

The publication review scanned reachable Git history with Gitleaks and compared tracked content
against local credential values, including PDF/DOCX text where present. This is a bounded
publication review, not a penetration test or a guarantee that no vulnerability exists.

Report a suspected credential leak privately to the repository owner; do not paste the secret
into a public issue. Rotate the credential before discussing remediation publicly.

## Dependency audit: 2026-09-17

`pip-audit` against the installed development environment reported known advisories for
`httpx2==2.10.0` and `transformers==4.57.6`. Representative IDs are
`PYSEC-2026-3846` (HTTP decompression resource exhaustion), `PYSEC-2026-2289`
(untrusted model configuration code execution), and `PYSEC-2026-3929` (model/tokenizer
save-path traversal). Some Transformers entries have no fixed version recorded by the feed.
These findings have **not** been resolved; passing tests do not make these dependencies safe.

Only use trusted, pinned model files and trusted local repositories. Do not download arbitrary
model repositories or expose this application as a public execution service. Updating across
the Transformers major-version boundary requires compatibility and retrieval regression tests;
this audit did not silently upgrade the local environment. Dependency ranges in `pyproject.toml`
are not a reproducible lockfile, so audit the resolved environment on each installation.
