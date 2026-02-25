---
name: code-review
description: 'Performs comprehensive code review and generates detailed findings reports in markdown format. Use when asked to review code, analyze code quality, find issues, assess maintainability, identify cognitive debt, check for security vulnerabilities, evaluate performance, or create an action plan for code improvements. Creates reports named {filename}_findings_plan.md with prioritized issues and actionable recommendations.'
---

# Code Review Skill

Conducts thorough code reviews with a focus on cognitive debt prevention, maintainability, performance, and security. Generates structured markdown reports with findings, severity ratings, and actionable improvement plans.

## When to Use This Skill

Use this skill when you need to:
- Review code files for quality issues
- Identify cognitive debt and maintainability problems
- Find security vulnerabilities or performance bottlenecks
- Generate formal code review reports
- Create action plans for code improvements
- Assess codebase health before refactoring
- Document technical debt for team planning

**Trigger phrases:**
- "Review this code"
- "Analyze [filename]"
- "Find issues in..."
- "Check code quality"
- "Create a code review report"
- "What's wrong with this code?"
- "How can I improve this file?"

## Review Methodology

### Analysis Dimensions

The review examines code across these dimensions:

1. **Cognitive Debt**
   - Hidden complexity and implicit assumptions
   - Lack of encapsulation and unclear boundaries
   - Missing documentation for non-obvious behavior
   - Clever code that sacrifices readability

2. **Maintainability**
   - Code organization and modularity
   - Naming clarity and consistency
   - DRY violations (duplication)
   - Single Responsibility Principle adherence
   - Error handling robustness

3. **Performance**
   - Algorithmic efficiency (O(n) analysis)
   - Memory usage patterns
   - Unnecessary computations
   - Database query efficiency
   - Caching opportunities

4. **Security**
   - Input validation gaps
   - SQL injection risks
   - XSS vulnerabilities
   - Authentication/authorization issues
   - Sensitive data exposure
   - Dependency vulnerabilities

5. **Testing & Observability**
   - Test coverage gaps
   - Testability issues
   - Logging and debugging support
   - Error messages clarity
   - Monitoring hooks

### Severity Levels

Issues are classified by severity:

| Level | Icon | Criteria | Action Timeline |
|-------|------|----------|-----------------|
| **Critical** | 🔴 | Security vulnerability, data loss risk, production blocker | Immediate |
| **High** | 🟠 | Major cognitive debt, performance degradation, scalability issue | This sprint |
| **Medium** | 🟡 | Moderate complexity, maintainability concern, minor performance hit | Next sprint |
| **Low** | 🟢 | Style inconsistency, minor refactoring opportunity, documentation gap | Backlog |

## Step-by-Step Workflow

### Step 1: Receive Review Request

When user requests a code review:
1. Identify the file(s) to review
2. Read the file using the Read tool
3. Understand the context (what the code does, its role in the system)

### Step 2: Perform Multi-Dimensional Analysis

For each code file, analyze:

1. **High-Level Architecture**
   - Overall structure and organization
   - Separation of concerns
   - Dependencies and coupling

2. **Function/Method Level**
   - Single responsibility adherence
   - Input validation
   - Error handling
   - Edge cases coverage

3. **Code Patterns**
   - Anti-patterns usage
   - Best practices alignment
   - Language-specific idioms

4. **Documentation & Comments**
   - Public API documentation
   - Complex logic explanation
   - TODO/FIXME markers

### Step 3: Generate Findings Report

Create a markdown file named `{filename}_findings_plan.md` with this structure:

```markdown
# Code Review: {filename}

**Review Date:** {current_date}
**Reviewer:** Claude Code
**File:** `{filepath}`

## Executive Summary

{1-2 paragraph overview of code quality, main concerns, and overall health}

## Findings

### 🔴 Critical Issues (Count: X)

#### Issue 1: {Title}
**Severity:** Critical
**Category:** {Security/Performance/Correctness}
**Lines:** {line_numbers}

**Description:**
{What the issue is and why it's critical}

**Current Code:**
```{language}
{problematic code snippet}
```

**Impact:**
- {Specific consequence 1}
- {Specific consequence 2}

**Recommendation:**
{Detailed fix with rationale}

**Proposed Solution:**
```{language}
{corrected code example}
```

---

### 🟠 High Priority Issues (Count: X)

{Same structure as above}

---

### 🟡 Medium Priority Issues (Count: X)

{Same structure as above}

---

### 🟢 Low Priority Issues (Count: X)

{Same structure as above}

---

## Positive Observations

- {What's done well}
- {Good patterns used}
- {Strengths to maintain}

## Action Plan

### Phase 1: Critical Fixes (Week 1)
- [ ] Fix {critical issue 1}
- [ ] Fix {critical issue 2}

### Phase 2: High Priority (Week 2-3)
- [ ] Refactor {component}
- [ ] Add {missing functionality}

### Phase 3: Medium Priority (Sprint 2)
- [ ] Improve {aspect}
- [ ] Extract {component}

### Phase 4: Low Priority (Backlog)
- [ ] Polish {detail}
- [ ] Document {behavior}

## Technical Debt Estimate

- **Total Issues:** {count}
- **Estimated Fix Time:** {hours} hours
- **Risk Level:** {Low/Medium/High}
- **Recommended Refactor:** {Yes/No}

## References

- {Link to relevant best practices}
- {Link to security guidelines}
- {Link to performance optimization guide}
```

### Step 4: Present Findings

After generating the report:
1. Use the Write tool to save the markdown file
2. Provide a summary to the user highlighting:
   - Total issues found by severity
   - Most critical concerns
   - Quick wins for immediate improvement
   - Link to the full report

## Review Best Practices

### Do's ✅

- **Be specific:** Reference exact line numbers and code snippets
- **Provide context:** Explain WHY something is an issue
- **Offer solutions:** Don't just criticize, show better alternatives
- **Balance critique:** Acknowledge good practices too
- **Prioritize ruthlessly:** Not everything needs immediate fixing
- **Consider trade-offs:** Recognize valid design choices even if not ideal

### Don'ts ❌

- **Don't be vague:** Avoid "this could be better" without specifics
- **Don't bikeshed:** Focus on substantial issues, not minor style preferences
- **Don't assume malice:** Code is often constrained by time/knowledge
- **Don't overload:** 20+ critical issues means you need to focus on top 3-5
- **Don't ignore context:** What works in one domain might not in another

## Language-Specific Considerations

### R Packages
- Check DESCRIPTION dependencies
- Validate roxygen2 documentation
- Verify S3/S4 method consistency
- Test coverage via testthat
- R CMD check compliance

### Python
- PEP 8 compliance
- Type hints usage
- docstring completeness
- pytest coverage
- Security: SQL injection, eval() usage

### JavaScript/TypeScript
- ESLint/TypeScript errors
- Async/await patterns
- Memory leaks (closures, event listeners)
- Security: XSS, prototype pollution

### SQL
- Query optimization (EXPLAIN ANALYZE)
- Index usage
- N+1 query patterns
- SQL injection vectors

## Example: Quick Review Output

```markdown
# Code Review: apply_BE.R

**Review Date:** 2026-02-24
**File:** `R/apply_BE.R`

## Executive Summary

The code implements best estimate deflator application with good input
validation but has moderate cognitive debt due to implicit matrix operations
and limited error context.

## Findings

### 🟠 High Priority Issues (Count: 2)

#### Issue 1: Silent dimension mismatch in matrix operations
**Severity:** High
**Category:** Correctness/Maintainability
**Lines:** 139-140

**Description:**
Matrix sweep operations can fail with non-conformable arrays when
BE_value length mismatches incidence_rate, but error message doesn't
explain root cause.

**Impact:**
- Difficult debugging for users
- Hidden assumptions about input alignment
- Cryptic "non-conformable arrays" errors

**Recommendation:**
Add explicit dimension checks with clear error messages before matrix ops.

**Proposed Solution:**
```r
# Before matrix operations
if (ncol(inci_mat) != ncol(be_mat)) {
  cli::cli_abort(c(
    "Dimension mismatch in BE calculation.",
    "x" = "Found {ncol(inci_mat)} incidence columns but {ncol(be_mat)} BE columns.",
    "i" = "Each incidence rate needs a corresponding BE value column."
  ))
}
```

---

## Action Plan

### Phase 1: High Priority (This Sprint)
- [ ] Add dimension validation before matrix operations
- [ ] Improve error messages with context

### Phase 2: Medium Priority (Next Sprint)
- [ ] Extract matrix sweep logic to named function
- [ ] Add examples for multi-column usage

## Technical Debt Estimate

- **Total Issues:** 5 (0 critical, 2 high, 2 medium, 1 low)
- **Estimated Fix Time:** 4-6 hours
- **Risk Level:** Medium
- **Recommended Refactor:** No - incremental improvements sufficient
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Report too long | Focus on top 3-5 issues per severity level |
| Unclear findings | Add more code context and specific line numbers |
| No actionable plan | Break down large refactorings into smaller tasks |
| Missing severity | Use the severity criteria table to classify |
| User disagrees | Acknowledge valid constraints; adjust recommendations |

## Tips for Effective Reviews

1. **Read the code twice:** First for understanding, second for issues
2. **Start with architecture:** High-level structure before line-by-line
3. **Question assumptions:** "Why is this done this way?"
4. **Check boundaries:** What happens at edges, nulls, empties?
5. **Think about users:** How will this code be called? What errors are likely?
6. **Consider testing:** Is this code testable? How would you test it?
7. **Look for patterns:** Repeated issues suggest systemic problems

## Output Format Validation

Before finalizing the report, ensure:
- [ ] Filename follows `{original_name}_findings_plan.md` format
- [ ] All severity icons are present (🔴🟠🟡🟢)
- [ ] Each issue has: title, severity, category, lines, description, impact, recommendation
- [ ] Code snippets are properly formatted with language tags
- [ ] Action plan has checkbox items with time estimates
- [ ] Technical debt section has all fields filled
- [ ] Executive summary is 1-2 paragraphs max

## References

- [Cognitive Load Theory](https://en.wikipedia.org/wiki/Cognitive_load)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Clean Code Principles](https://github.com/ryanmcdermott/clean-code-javascript)
- [Google Style Guides](https://google.github.io/styleguide/)
