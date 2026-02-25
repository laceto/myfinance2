# Projection Registry - Quick Reference Cheat Sheet

> Print this and keep at your desk while implementing

---

## 🚀 Quick Start (Copy-Paste Ready)

```r
library(pricingr)
library(dplyr)

# 1. Define configs
qx_table <- pricingr::qx %>% filter(year == 2006)

configs <- list(
  axa_cqs_death = projection_config(
    template = "cqs",
    name = "axa_cqs_death",
    ceding = "axa",
    product = "cqs",
    risk = "death",
    mortality_table = qx_table,
    BE_fraction = 1,
    sum_insured_type = "fixed"
  )
)

# 2. Build & dispatch
registry <- build_projection_registry(configs)
dispatcher <- create_registry_dispatcher(registry)

# 3. Run
results <- project_policies(portfolio, dispatcher, .policy_id = "policy_id")
```

---

## 📋 Function Quick Reference

| Task | Function | Example |
|------|----------|---------|
| Create registry | `create_projection_registry()` | `create_projection_registry(ceding = "exact", product = "exact")` |
| Add rule | `register_projection()` | `register_projection(registry, ceding = "axa", proj_fn = my_fn)` |
| Create dispatcher | `create_registry_dispatcher()` | `dispatcher <- create_registry_dispatcher(registry)` |
| Batch build | `build_projection_registry()` | `registry <- build_projection_registry(configs)` |
| Create config | `projection_config()` | `projection_config(template = "cqs", ceding = "axa", ...)` |
| Validate configs | `validate_projection_configs()` | `validate_projection_configs(configs)` |
| List templates | `list_projection_templates()` | `list_projection_templates()` |
| Register template | `register_projection_template()` | `register_projection_template("name", factory_fn)` |

---

## 🏗️ Configuration Template

```r
projection_config(
  # ─── REQUIRED ───────────────────────────────
  template = "cqs",              # Template name
  name = "unique_config_name",   # For debugging

  # ─── DIMENSIONS (for matching) ──────────────
  ceding = "axa",
  product = "cqs",
  risk = "death",

  # ─── TEMPLATE-SPECIFIC ──────────────────────
  mortality_table = qx_table,
  BE_fraction = 1,
  sum_insured_type = "fixed"     # or "outstanding_debt"
)
```

---

## 🔀 Dimension Types

### Exact Match
```r
create_projection_registry(
  ceding = "exact",
  product = "exact"
)
# Matches: ceding == "axa" AND product == "cqs"
```

### Range Match
```r
create_projection_registry(
  ceding = "exact",
  uw_year = c("uw_year_min", "uw_year_max")
)
# Matches: uw_year >= uw_year_min AND uw_year <= uw_year_max
```

---

## 🃏 Wildcards

| Pattern | Meaning | Priority Impact |
|---------|---------|-----------------|
| `ceding = "axa"` | Match only "axa" | +0 |
| `ceding = NA` | Match any value | -10 |
| `uw_year_min = NA` | No lower bound | -5 |
| `uw_year_max = NA` | No upper bound | -5 |

**Auto Priority**: `100 - (NA_count × 10)`

```r
# Specific: priority = 100
ceding = "axa", product = "cqs", risk = "death"

# Partial: priority = 90
ceding = "axa", product = "cqs", risk = NA

# Catch-all: priority = 70
ceding = NA, product = NA, risk = NA
```

---

## 🎯 CQS Template Parameters

| Parameter | Type | Values | Required |
|-----------|------|--------|----------|
| `mortality_table` | data.frame | Columns: current_age, gender, qx | ✓ |
| `BE_fraction` | numeric | 1, 10, etc. | ✓ |
| `sum_insured_type` | character | "fixed", "outstanding_debt" | ✓ |

### Required Model Point Columns

**Always required:**
- `duration_months`, `inception_date`, `entry_age`, `gender`, `BE`

**For `sum_insured_type = "fixed"`:**
- `sum_insured`

**For `sum_insured_type = "outstanding_debt"`:**
- `monthly_loan_pay`, `monthly_loan_interest`

---

## 🛠️ Custom Template Skeleton

```r
make_my_template <- function(config) {
  # 1. Validate
  required <- c("param1", "param2")
  missing <- setdiff(required, names(config))
  if (length(missing) > 0) {
    cli::cli_abort("Missing: {.val {missing}}")
  }

  # 2. Capture
  force(config)

  # 3. Return closure
  function(mp_one) {
    mp_one %>%
      setup_projection(...) %>%
      # Use config$param1, config$param2
      mutate(result = config$param1 * value)
  }
}

# Register
register_projection_template("my_template", make_my_template)
```

---

## ⚠️ Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `"Column X not found"` | Column name mismatch | Use same names in registry and portfolio |
| `"No matching rule"` | No rule matches | Add catch-all rule with all `NA` |
| `"Invalid registry"` | Wrong object type | Use `create_projection_registry()` |
| `"Missing template"` | Template not registered | Check `list_projection_templates()` |

---

## 🔍 Debugging Commands

```r
# Check registry contents
print(registry)

# Check what matched
match <- find_matching_projection(registry, portfolio[1, ])
print(match$proj_fn_name)

# Test projection directly
proj_fn <- match$proj_fn
result <- proj_fn(portfolio[1, ])

# Check column names
names(portfolio)
names(attr(registry, "col_meta"))
```

---

## 📁 Recommended File Structure

```
R/
├── zzz.R                      # Template registration
├── template_cqs.R             # CQS template
├── template_term.R            # Term template (if custom)
├── configs.R                  # get_production_configs()
└── registry.R                 # build_production_registry()

inst/configs/
├── production.yaml
└── test.yaml

tests/testthat/
├── test-templates.R
└── test-registry.R
```

---

## ✅ Pre-Flight Checklist

- [ ] Configs have unique `name` fields
- [ ] Configs have `template` field
- [ ] Configs have all dimension fields (ceding, product, risk)
- [ ] Template is registered (`list_projection_templates()`)
- [ ] Portfolio columns match registry dimensions
- [ ] Catch-all rule added (if needed)
- [ ] `validate_projection_configs()` passes

---

## 📊 Typical Workflow

```
┌─────────────────┐
│  Define Configs │
│  (YAML or R)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Validate      │  validate_projection_configs()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Build Registry │  build_projection_registry()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Create Dispatch │  create_registry_dispatcher()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Run Projections │  project_policies(portfolio, dispatcher)
└─────────────────┘
```

---

*Cheat Sheet v1.0.0 - Keep this handy!*
