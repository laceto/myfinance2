# Testing Rules — myfinance2

## Framework & Structure

- Use `pytest` (not unittest)
- One `describe`-style class or module per unit under test
- Test names: `test_<behavior>_<given_context>` (e.g., `test_returns_empty_df_when_no_tickers`)
- Arrange → Act → Assert pattern, one assertion per logical outcome

## TDD Cycle — Mandatory

1. **Red**: write the test first — run it and confirm it fails
2. **Green**: write minimum code to make it pass — no more, no less
3. **Refactor**: clean up with the safety net in place

A test that cannot fail proves nothing. Never write implementation first.

## Coverage Requirements

- Happy path (primary behavior)
- Edge cases: empty DataFrame, single row, missing columns, NaN values
- Failure modes: bad ticker, network error, malformed Parquet

## Finance-Specific Rules

- Never use live network calls in unit tests — mock `YFinanceDataHandler`
- Use deterministic fixture DataFrames with known OHLC values
- Test signal logic against hand-calculated expected outputs
- Verify no lookahead bias: signals at time `t` must only use data ≤ `t`

## No Implementation Changes While Testing

- Do NOT modify source code to force a test to pass with a workaround
- If a bug is found → create a task to fix it separately
- If a missing feature is found → document it, don't implement inline

## When Done

→ Run `pytest` and confirm all tests pass
→ If any fail → switch to fixing mode: identify root cause, make minimal fix, re-run
→ Do not push test changes without matching implementation passing
→ READ: git-rules.md before committing
