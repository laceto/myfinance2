# Git Rules — myfinance2

## Commit Messages

- Use Conventional Commits format: `type(scope): description`
- Common types: `feat`, `fix`, `chore`, `refactor`, `test`, `docs`, `ci`
- Keep the subject line under 72 characters
- Body: explain **why**, not what (the diff shows what)

## Strict Rules

- **NEVER** include `Co-Authored-By: Claude` in commit messages
- Use `[skip ci]` suffix only for data-only commits (Parquet updates) — not for code changes
- Do not force-push to `main`
- Do not amend published commits

## When Done
→ STOP. Do not push unless the user explicitly asks.
