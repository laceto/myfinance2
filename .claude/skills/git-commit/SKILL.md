---
name: git-commit
description: 'Create standardized, semantic git commits using the Conventional Commits specification. Use when asked to "commit changes", "save my work", "stage and commit", "create a commit", or when making atomic commits with proper type/scope/description format. Supports feat, fix, docs, refactor, test, perf, build, ci, chore, and revert types.'
---

# Git Commit with Conventional Commits

## Overview

Create standardized, semantic git commits using the Conventional Commits specification.
Analyze the actual diff to determine appropriate type, scope, and message.

## When to Use This Skill

- User asks to "commit", "save changes", or "stage and commit"
- User wants a conventional/semantic commit message
- User needs help choosing the right commit type or scope
- User wants to commit specific files with a proper message

## Prerequisites

- Git installed and available in PATH
- Current directory is inside a git repository
- Files to commit are either staged or ready to stage

## Conventional Commit Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

## Commit Types

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature or capability | `feat(auth): add OAuth2 login` |
| `fix` | Bug fix | `fix(parser): handle empty input` |
| `docs` | Documentation only | `docs(readme): update install steps` |
| `style` | Formatting, whitespace (no code change) | `style: fix indentation` |
| `refactor` | Code change that neither fixes nor adds | `refactor(api): extract helper function` |
| `perf` | Performance improvement | `perf(query): add index lookup` |
| `test` | Adding or fixing tests | `test(utils): add edge case coverage` |
| `build` | Build system or dependencies | `build: upgrade to R 4.3` |
| `ci` | CI/CD configuration | `ci: add GitHub Actions workflow` |
| `chore` | Maintenance tasks | `chore: update .gitignore` |
| `revert` | Revert a previous commit | `revert: undo feature X` |

## Workflow

1. **Analyze Diff**
   - Run `git status` and `git diff` to understand changes
   - Identify the logical units of work

2. **Stage Files**
   - Stage specific files by name (prefer over `git add .`)
   - Never commit secrets (`.env`, `credentials.json`, private keys)

3. **Generate Commit Message**
   - **Type**: What kind of change is this? (see table above)
   - **Scope**: What area/module is affected? (optional but recommended)
   - **Description**: One-line summary (present tense, imperative mood, <72 chars)

4. **Execute Commit**
   - Use HEREDOC for multi-line messages to preserve formatting

5. **Verify**
   - Run `git status` to confirm commit succeeded

## Best Practices

- **One logical change per commit** — atomic, reviewable units
- **Present tense**: "add" not "added"
- **Imperative mood**: "fix bug" not "fixes bug"
- **Reference issues**: `Closes #123`, `Refs #456`
- **Keep description under 72 characters**
- **Body for context**: Explain *why*, not just *what*

## Git Safety Protocol

- NEVER update git config
- NEVER run destructive commands (`--force`, `reset --hard`) without explicit request
- NEVER skip hooks (`--no-verify`) unless user asks
- NEVER force push to `main`/`master`
- NEVER add `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>` to commit messages
- If commit fails due to hooks, fix and create NEW commit (don't amend)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Pre-commit hook fails | Fix the issues reported, re-stage files, create a NEW commit (don't amend) |
| Nothing to commit | Check `git status` - files may need staging or there are no changes |
| Commit message rejected | Ensure format matches `type(scope): description` with valid type |
| Merge conflict blocking commit | Resolve conflicts first with `git mergetool` or manual edit, then stage |
| Accidentally staged wrong file | Use `git restore --staged <file>` to unstage before committing |
| Need to undo last commit | Use `git reset --soft HEAD~1` to undo commit but keep changes staged |
