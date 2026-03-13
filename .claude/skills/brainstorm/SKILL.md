---
name: brainstorm
description: 'Runs a structured brainstorming session for any idea, project, feature, problem, or creative request. Use when the user says "brainstorm", "let''s think through", "help me explore", "I have an idea", "what could I build", "how should I approach", or provides a vague description and wants to develop it. Surfaces assumptions, explores multiple angles, generates creative alternatives, identifies risks and constraints, and proposes concrete next steps.'
---

# Brainstorm

A facilitated brainstorming session that takes any seed idea and develops it into a structured set of perspectives, alternatives, risks, and actionable next steps.

## When to Use This Skill

- User says "brainstorm", "let's think through this", "help me explore an idea"
- User provides a project description and wants to flesh it out
- User has a vague concept and needs help making it concrete
- User wants alternatives or creative angles on a problem
- User wants to sanity-check assumptions before committing to a direction

## Session Structure

When invoked, run the following phases **in order**, outputting each as a labeled section. Do **not** skip phases. Do **not** converge prematurely — diverge first.

---

### Phase 1 — Reframe & Clarify

Restate the idea in your own words to confirm understanding. Then reframe it as:
- A **problem statement** ("How might we…")
- A **goal statement** ("The desired outcome is…")
- A **constraint statement** ("The boundaries are…")

If the input is ambiguous, state your assumed interpretation explicitly before continuing.

---

### Phase 2 — Angles & Dimensions

Explore the idea across at least **5 distinct lenses**. Choose the most relevant from this list (or invent better ones for the context):

| Lens | Question to ask |
|---|---|
| User / Audience | Who benefits? Who is harmed? Who is ignored? |
| Technical | What does this require to build or implement? |
| Data & Information | What data is needed, produced, or at risk? |
| Business / Value | What is the value exchange? Who pays, who gains? |
| Time & Sequencing | What must happen first? What can be deferred? |
| Failure & Edge Cases | How does this break? What's the worst-case? |
| Analogies | What existing system or domain is this most like? |
| Second-order effects | What happens after the obvious outcome? |

For each lens, write 2–4 concrete observations, not just questions.

---

### Phase 3 — Alternatives & Variations

Generate **at least 5 distinct alternatives** to the original idea. Use these forcing functions:

- **Invert it** — What if you did the opposite?
- **Simplify radically** — What is the smallest version that still delivers value?
- **Scale wildly** — What if this had 1000x more users / data / scope?
- **Remove the constraint** — What would you do if the hardest constraint didn't exist?
- **Steal from another domain** — How does finance / biology / games / logistics solve this?

Label each alternative clearly and give it a one-sentence rationale.

---

### Phase 4 — Hidden Assumptions

List **5–8 assumptions** embedded in the original idea that, if false, would materially change the approach. Format as:

> "We are assuming that **[X]**, but if **[not-X]** were true, we would need to **[different approach]**."

---

### Phase 5 — Risks & Constraints

Identify the top risks across three categories:

- **Execution risks** — things that could go wrong while building/doing this
- **Adoption risks** — things that could prevent it from being used or accepted
- **Strategic risks** — ways this could backfire or create worse problems

Rate each risk **High / Medium / Low** and suggest one mitigation per risk.

---

### Phase 6 — Recommended Next Steps

Propose **3–5 concrete, sequenced next steps**. Each step must:
- Be actionable (start with a verb)
- Have a clear deliverable or decision point
- Indicate who or what is needed

End with a single **"Key open question"** — the one question that, if answered, would most accelerate progress.

---

## Tone & Style

- Be direct and opinionated. Label speculation clearly ("Hypothesis:", "Speculation:").
- Prefer concrete examples over abstract principles.
- When two options are genuinely equal, say so. When one is clearly better, say that too.
- Keep each section tight. Depth over breadth within each point.
- Do not ask clarifying questions at the start — make reasonable assumptions, state them, and proceed. The user can correct after seeing the output.
