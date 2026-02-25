# Code Review: analyze_and_report.yml

**Review Date:** 2026-02-25
**Reviewer:** Claude Code
**File:** `.github/workflows/analyze_and_report.yml`

---

## Executive Summary

This workflow is a straightforward, readable CI pipeline that runs stock analysis
after OHLC data is downloaded and commits results back to the repository. The
overall intent is clear and the conditional guard (`workflow_run` + conclusion
check) is a solid pattern. However, there are three categories of concern:
a **silent failure mode** caused by `tee` masking exit codes, **non-deterministic
builds** from unpinned dependencies, and **supply-chain risk** from floating
action tags. None are showstoppers, but they will cause painful debugging
sessions when they surface in production.

---

## Artifacts Produced

This workflow produces **three outputs** across two persistence mechanisms:

### 1. GitHub Actions Artifact (ephemeral, 30-day retention)
| Field | Value |
|-------|-------|
| Name | `analysis-results-{run_id}` |
| Path | `data/results/it/analysis_results.parquet` |
| Size | ~500 MB |
| Retention | 30 days |
| Trigger | Every successful run |

**Purpose:** The analysis results Parquet is too large to commit to git, so it
is stored as a workflow artifact downloadable from the Actions UI. It is purged
after 30 days — there is no long-term persistence for this file.

### 2. Git-Committed Files (permanent, in the repository)
| File | Description |
|------|-------------|
| `data/results/it/trading_dashboard.xlsx` | Excel workbook with trading signals / dashboard |
| `data/results/it/trading_report.txt` | Plain-text report captured via `tee` from `trading_report.py` stdout |

These are committed with message `chore: update trading dashboard and report [skip ci]`
and pushed to `main`. They persist in git history indefinitely.

---

## Findings

### 🔴 Critical Issues (Count: 1)

#### Issue 1: `tee` silently masks Python script exit codes
**Severity:** Critical
**Category:** Correctness / Observability
**Lines:** 42

**Description:**
```yaml
run: python trading_report.py | tee data/results/it/trading_report.txt
```
In a shell pipe `A | B`, the exit code of the entire pipeline is the exit code
of **the last command** (`tee`), not `A`. If `trading_report.py` crashes or
exits with code 1, `tee` will still exit 0, the step will be marked **green**,
and the workflow will proceed to commit a corrupt or empty report file.

**Impact:**
- A broken report is silently committed to `main`
- The workflow appears to succeed when it has actually failed
- Downstream consumers of `trading_report.txt` receive corrupt data with no alert

**Recommendation:**
Enable `pipefail` in the shell so that a pipe fails if any component fails.

**Proposed Solution:**
```yaml
- name: Generate trading report
  shell: bash
  run: |
    set -euo pipefail
    python trading_report.py | tee data/results/it/trading_report.txt
```

`set -euo pipefail` ensures:
- `-e`: exit immediately on error
- `-u`: fail on unset variables
- `-o pipefail`: propagate pipe component failures

---

### 🟠 High Priority Issues (Count: 2)

#### Issue 2: Unpinned pip dependencies cause non-deterministic builds
**Severity:** High
**Category:** Maintainability / Reliability
**Lines:** 34

**Description:**
```yaml
pip install yfinance pandas openpyxl pyarrow joblib tqdm
```
Every run installs the latest available versions of six packages. A breaking
API change upstream (e.g., pandas 3.x) will silently break the workflow with
no clear signal that a dependency changed.

**Impact:**
- Analysis may produce different results on Monday vs. Tuesday from different
  library versions
- Debugging regressions is hard because the environment is not reproducible
- `requirements.txt` exists in the repo (per CLAUDE.md) but is ignored here

**Recommendation:**
Install from `requirements.txt` instead of listing packages inline. Add
`pip caching` to speed up runs.

**Proposed Solution:**
```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.11'
    cache: 'pip'           # cache ~/.cache/pip between runs

- name: Install dependencies
  run: |
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install algoshort-0.1.1-py3-none-any.whl
```

---

#### Issue 3: Wheel version in workflow differs from CLAUDE.md
**Severity:** High
**Category:** Correctness / Documentation
**Lines:** 35

**Description:**
- `CLAUDE.md` documents: `algoshort-0.1.0-py3-none-any.whl`
- This workflow installs: `algoshort-0.1.1-py3-none-any.whl`

Either the CLAUDE.md is outdated or the workflow references a wheel version
that may not exist in the repository. This creates confusion about which version
of the private package is in use.

**Impact:**
- Local dev environments may use a different `algoshort` version than CI
- If `algoshort-0.1.1` is not in the repo, the CI install step will fail silently
  until the next developer tries a fresh run

**Recommendation:**
Align the version referenced in `CLAUDE.md` and the workflow. Define the wheel
filename as a single constant (e.g., an env var at the workflow level).

**Proposed Solution:**
```yaml
env:
  ALGOSHORT_WHEEL: algoshort-0.1.1-py3-none-any.whl

# ... then in the step:
run: pip install ${{ env.ALGOSHORT_WHEEL }}
```
Update CLAUDE.md to match.

---

### 🟡 Medium Priority Issues (Count: 2)

#### Issue 4: Action tags are floating (supply-chain risk)
**Severity:** Medium
**Category:** Security
**Lines:** 24, 27, 47

**Description:**
All `uses:` directives reference floating tags (`@v4`, `@v5`) rather than
pinned commit SHAs. A compromised tag in a third-party action (e.g.,
`actions/upload-artifact`) could silently run malicious code in CI with
`contents: write` permissions (which here can push to main).

**Current:**
```yaml
uses: actions/checkout@v4
uses: actions/setup-python@v5
uses: actions/upload-artifact@v4
```

**Recommendation:**
Pin each action to a full SHA. Use Dependabot to keep them updated.

**Proposed Solution:**
```yaml
uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683  # v4.2.2
uses: actions/setup-python@0b93645e9fea7318ecaed2b359559ac225c90a2  # v5.3.0
uses: actions/upload-artifact@65c4c4a1ddee5b72f698fdd19549f0f0fb45cf08  # v4.6.0
```

Add to `.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

---

#### Issue 5: Artifact upload runs unconditionally — loses debug data on failure
**Severity:** Medium
**Category:** Observability
**Lines:** 46–51

**Description:**
If `analyze_stock.py` (line 38) fails, the job stops and the artifact upload
step never runs. Any partial `analysis_results.parquet` written before the crash
is lost. This makes post-mortem debugging much harder.

**Impact:**
- On failure, no Parquet is available to inspect what signals were computed
- Developers must re-run the full workflow (90 min timeout) to reproduce

**Recommendation:**
Mark the upload step with `if: always()` so it runs even after a failure.

**Proposed Solution:**
```yaml
- name: Upload analysis parquet as artifact
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: analysis-results-${{ github.run_id }}
    path: data/results/it/analysis_results.parquet
    if-no-files-found: warn   # don't fail if the file wasn't produced
    retention-days: 30
```

---

### 🟢 Low Priority Issues (Count: 2)

#### Issue 6: No pip cache = slow installs on every run
**Severity:** Low
**Category:** Performance
**Lines:** 27–35

**Description:**
Without `cache: 'pip'` on `actions/setup-python`, every run downloads and
installs all packages from scratch. With six packages plus the private wheel,
this likely adds 2–3 minutes per run unnecessarily.

**Recommendation:**
Add `cache: 'pip'` to `actions/setup-python` (shown in Issue 2's solution).

---

#### Issue 7: Output directory assumed to pre-exist
**Severity:** Low
**Category:** Robustness
**Lines:** 42, 57–58

**Description:**
Both the `tee` command and `git add` assume `data/results/it/` already exists.
If the analysis scripts fail to create it, both steps will silently fail or
produce confusing errors. There is no explicit `mkdir -p` guard.

**Recommendation:**
Add an explicit directory creation step before scripts run, or rely on the
scripts themselves and document the invariant.

```yaml
- name: Ensure output directory exists
  run: mkdir -p data/results/it
```

---

## Positive Observations

- **Clear trigger logic:** The `workflow_run` + conclusion guard correctly
  prevents running analysis on a failed download. The comment explains the
  `workflow_dispatch` override well.
- **`tee` usage is explained:** The comment on line 41 correctly documents
  what `tee` does — good for future maintainers.
- **Large file handled correctly:** Using a workflow artifact instead of
  committing the 500 MB Parquet is the right architectural choice.
- **`[skip ci]` tag prevents loops:** The commit message includes `[skip ci]`
  to avoid triggering further CI runs from the bot commit.
- **`|| echo "No changes"` pattern:** Gracefully handles the no-op case where
  data hasn't changed.

---

## Action Plan

### Phase 1: Critical Fix (Immediately)
- [ ] Add `set -euo pipefail` to the `Generate trading report` step

### Phase 2: High Priority (This sprint)
- [ ] Pin `algoshort` wheel version to an env var; sync CLAUDE.md to match
- [ ] Switch `pip install` inline packages to `pip install -r requirements.txt`

### Phase 3: Medium Priority (Next sprint)
- [ ] Add `if: always()` + `if-no-files-found: warn` to artifact upload step
- [ ] Pin action SHAs and add `dependabot.yml` for automated updates

### Phase 4: Low Priority (Backlog)
- [ ] Add `cache: 'pip'` to speed up dependency installs
- [ ] Add explicit `mkdir -p data/results/it` guard step

---

## Technical Debt Estimate

| Metric | Value |
|--------|-------|
| Total Issues | 7 (1 critical, 2 high, 2 medium, 2 low) |
| Estimated Fix Time | 2–3 hours |
| Risk Level | Medium (critical failure mode is latent but will surface) |
| Recommended Refactor | No — incremental fixes sufficient |

---

## References

- [GitHub Docs: `workflow_run` event](https://docs.github.com/en/actions/writing-workflows/choosing-when-your-workflow-runs/events-that-trigger-workflows#workflow_run)
- [Bash `pipefail` option](https://www.gnu.org/software/bash/manual/bash.html#The-Set-Builtin)
- [Hardening GitHub Actions (supply chain)](https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions#using-third-party-actions)
- [Pinning actions to SHAs](https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions#using-third-party-actions)
