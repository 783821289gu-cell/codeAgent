# Repository history and archived runs

This repository was prepared for GitHub publication on 2026-09-10.

The supplied project directory and its adjacent `codeAgent.zip` contained no
top-level `.git` directory. Consequently, no original main-project commit history
was available to publish. The initial commit on `main` records the current project
snapshot; it does not reconstruct or backdate development commits.

The 14 nested Demo/Evaluation repositories contained the same original baseline
commit, `historical-commit` (`baseline: login endpoint with missing-user bug`). That original
commit and its history are preserved unchanged on the `archive/demo-baseline`
branch. These nested repositories' current working files are included as ordinary
files on `main`, rather than as submodules.

The `artifacts/` directory preserves available run evidence, including traces,
reports, test output, Git diffs, evaluation results, and the context-optimization
token comparison. Uncommitted demo repairs remain available in the recorded
workspace snapshots and diffs; they are not represented as historical commits.

Local credentials (`.env`), virtual environments, caches, build output, and local
runtime databases are excluded. `.env.example` remains as the configuration
template.
