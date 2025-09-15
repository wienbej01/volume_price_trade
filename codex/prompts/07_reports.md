GOAL
Finalize reporting helpers, tidy docs, ensure repo is clean.

FILE EDITS
1) `scripts/make_report.py`
   - Bundle metrics PNGs/HTML/CSV into a single report directory; print the path

2) `PROJECT_PLAN.md` and `CHANGELOG.md`
   - Update progress and note the implemented features

3) `codex/state/project_state.yaml`
   - M6.status = done
   - Optionally M7.status = done after a scale-out dry run
   - runs += {id: "run-0007", step: "M6-done", timestamp: "<UTC>", notes: "Reports & checks"}

COMMANDS
- `bash scripts/run_checks.sh`

ACCEPTANCE
- All checks pass; report-making script works; state updated.
