---
name: breakdown
description: 'Recursively decomposes large, complex, or ambiguous tasks into smaller independently solvable sub-problems. Use when a task feels too big, too hard, or unclear — coding problems, system design, debugging, refactoring, migrations, or any multi-step work. Keeps breaking down until every piece is concrete and actionable, then solves them in order (A -> A1 -> A2 -> A3 -> B).'
---

# Problem Breakdown

Decompose large or hard problems into a sequence of small, independently solvable steps. Repeat recursively until every leaf is concrete and actionable.

## When to Use This Skill

- The task is large, vague, or hard to start
- A coding problem has multiple unknowns at once
- A refactor, migration, or feature spans many files or systems
- You feel stuck or unsure where to begin
- The user says "break this down", "this is too big", "where do I start", or "help me plan this"

## The Core Rule

**Never go from A to B directly if B feels hard.**
Go A → A1 → A2 → A3 → B.
If any Aₙ still feels hard, break it down again.
Stop only when every step is: small, concrete, and immediately executable.

## Workflow

### Step 1: Understand the Goal
Restate the end goal in one sentence.
Identify: inputs, outputs, constraints, unknowns.

### Step 2: First Split
Divide the problem into 2–5 sequential or parallel sub-problems.
Label them A1, A2, A3, ...
Each sub-problem should be independently completable.

### Step 3: Assess Each Sub-problem
For each Aₙ ask:
- Can I solve this right now without solving anything else first?
- Is it small enough to fit in a single focused session?
- Do I know exactly what "done" looks like?

If **yes** to all three → it is a leaf. Keep it.
If **no** to any → break it down further into Aₙ.1, Aₙ.2, ...

### Step 4: Recurse Until Leaves
Repeat Step 3 on every non-leaf node.
Typical depth: 2–3 levels. Stop when every node is a leaf.

### Step 5: Order and Solve
Arrange leaves in dependency order.
Solve each leaf in sequence, confirming completion before moving on.

## Output Format

Present the breakdown as a numbered tree, then solve leaf by leaf:

```
Goal: <one sentence>

A1  <sub-problem>
  A1.1  <leaf>
  A1.2  <leaf>
A2  <sub-problem>
  A2.1  <leaf>
  A2.2  <leaf>
A3  <leaf>

Starting with A1.1 ...
```

## Sizing Rules

| Size signal | Action |
|---|---|
| "implement X feature" (multi-file) | Split by layer or concern |
| "refactor this module" | Split by: characterize → test → change → verify |
| "debug this bug" | Split by: reproduce → isolate → fix → confirm |
| "migrate to new API" | Split by: diff interfaces → adapt callsites → update tests → remove old |
| Single function change | Leaf — solve directly |

## Example

**Input:** "Add authentication to the app"

```
Goal: Add JWT-based authentication to the app.

A1  Define the auth contract
  A1.1  List all routes that need protection
  A1.2  Define the JWT payload shape and expiry
A2  Implement token issuance
  A2.1  Write login endpoint (returns JWT on valid credentials)
  A2.2  Write token validation middleware
A3  Protect routes
  A3.1  Apply middleware to protected routes
  A3.2  Return 401 on invalid/missing token
A4  Test
  A4.1  Unit test token validation
  A4.2  Integration test: protected route rejects unauthenticated request

Starting with A1.1 — listing routes that need protection...
```

## Rules

- Leaves must be **concrete**: a specific file, function, test, or command.
- Leaves must be **small**: completable in one focused step.
- Leaves must be **verifiable**: you know when they are done.
- Never skip levels — if a step is hard, it is not a leaf yet.
- Solve leaves in order; do not jump ahead.
