# RepoPilot login bug demo

This intentionally small WSGI API contains a missing-user login bug. Its baseline test passes, but
requesting `/login/unknown` is converted to HTTP 500 by the WSGI boundary.

From the RepoPilot project root, run the complete agent workflow with:

```powershell
$env:OPENAI_API_KEY = "..."
.\.venv\Scripts\repopilot.exe run `
  --workspace .\demo\bug_repo `
  "Fix the login endpoint returning HTTP 500 for an unknown user and add a regression test."
```

The expected behavior is a focused production change, a regression test for the missing user, a
passing test run, an independent Reviewer report, durable memory/checkpoint state, and a trace.

