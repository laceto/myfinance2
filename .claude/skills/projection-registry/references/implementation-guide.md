# Projection Registry System - Implementation Guide

> **Target Audience**: R Software Developers implementing reinsurance portfolio modeling
> **Prerequisite**: Familiarity with pricingr core functions and tidyverse
> **Version**: 1.0.0

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Implementation Roadmap](#implementation-roadmap)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Configuration Management](#configuration-management)
5. [Custom Template Development](#custom-template-development)
6. [Testing Strategy](#testing-strategy)
7. [Debugging Guide](#debugging-guide)
8. [Production Deployment](#production-deployment)
9. [Common Pitfalls](#common-pitfalls)
10. [Checklist](#checklist)

---

## Quick Start

### Minimal Working Example

```r
library(pricingr)
library(dplyr)

# 1. Define configurations
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

# 2. Build registry
registry <- build_projection_registry(configs)

# 3. Create dispatcher
dispatcher <- create_registry_dispatcher(registry)

# 4. Run projections
results <- project_policies(portfolio, dispatcher, .policy_id = "policy_id")
```

---

## Implementation Roadmap

### Phase 1: Assessment (Day 1)

```
┌─────────────────────────────────────────────────────────────────────┐
│  □ Inventory existing projection functions                          │
│  □ Identify dimension axes (ceding, product, risk, uw_year, etc.)  │
│  □ Map variations between projections                               │
│  □ Identify shared vs. unique logic                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Template Design (Days 2-3)

```
┌─────────────────────────────────────────────────────────────────────┐
│  □ Group projections by shared structure                            │
│  □ Design template parameters for each group                        │
│  □ Implement template factory functions                             │
│  □ Unit test templates with sample configs                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 3: Configuration (Days 4-5)

```
┌─────────────────────────────────────────────────────────────────────┐
│  □ Define configuration schema                                      │
│  □ Create configs for all ceding/product/risk combinations          │
│  □ Validate configurations                                          │
│  □ Set up configuration version control                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 4: Integration (Days 6-7)

```
┌─────────────────────────────────────────────────────────────────────┐
│  □ Build production registry                                        │
│  □ Integration test with sample portfolios                          │
│  □ Compare results against legacy implementation                    │
│  □ Performance benchmark                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 5: Deployment (Day 8+)

```
┌─────────────────────────────────────────────────────────────────────┐
│  □ Document configurations                                          │
│  □ Set up CI/CD for config changes                                  │
│  □ Train team on new workflow                                       │
│  □ Monitor production runs                                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Implementation

### Step 1: Audit Existing Projections

Create an inventory spreadsheet:

| Function Name | Ceding | Product | Risk | Key Differences |
|---------------|--------|---------|------|-----------------|
| `axa_cqs_death` | AXA | CQS | Death | BE_fraction=1, fixed SI |
| `axa_cqs_loe` | AXA | CQS | LOE | BE_fraction=10, fixed SI |
| `cf_cqs_death` | CF | CQS | Death | BE_fraction=1, outstanding debt |
| `cf_cqs_loe` | CF | CQS | LOE | BE_fraction=10, outstanding debt |

**Questions to answer:**
- Which dimensions vary? (ceding, product, risk, uw_year)
- What parameters differ? (tables, factors, methods)
- Is the projection structure the same or different?

### Step 2: Identify Template Candidates

Group projections by **structural similarity**:

```
Template "cqs":
  - axa_cqs_death (BE_fraction=1, sum_insured_type="fixed")
  - axa_cqs_loe (BE_fraction=10, sum_insured_type="fixed")
  - cf_cqs_death (BE_fraction=1, sum_insured_type="outstanding_debt")
  - cf_cqs_loe (BE_fraction=10, sum_insured_type="outstanding_debt")

Template "term_life":
  - axa_term_death (lapse_table=axa_lapse, commission=0.25)
  - cf_term_death (lapse_table=cf_lapse, commission=0.30)
```

**Rule of thumb**: If projections share >70% of code, they belong to the same template.

### Step 3: Define Template Parameters

For each template, list the parameterizable elements:

```r
# CQS Template Parameters
# -----------------------
# mortality_table: data.frame   - Mortality rates by age/gender
# BE_fraction: numeric          - Best estimate multiplier (1, 10, etc.)
# sum_insured_type: character   - "fixed" or "outstanding_debt"
```

### Step 4: Implement Template Factory

```r
# R/template_cqs.R (already implemented in pricingr)

make_cqs_projection <- function(config) {
  # 1. Validate required fields
  required_fields <- c("mortality_table", "BE_fraction", "sum_insured_type")
  missing <- setdiff(required_fields, names(config))
  if (length(missing) > 0) {
    cli::cli_abort("Missing: {.val {missing}}")
  }

  # 2. Validate field values
  valid_si_types <- c("fixed", "outstanding_debt")
  if (!config$sum_insured_type %in% valid_si_types) {
    cli::cli_abort("Invalid sum_insured_type: {.val {config$sum_insured_type}}")
  }

  # 3. Force evaluation to capture in closure
  force(config)

  # 4. Return projection closure
  function(mp_one) {
    result <- mp_one %>%
      setup_projection(
        n_months = duration_months,
        start_date = inception_date,
        entry_age = entry_age
      ) %>%
      set_assumption(
        table = config$mortality_table,
        joinvars = c("current_age", "gender"),
        joinvars_table = c("current_age", "gender")
      ) %>%
      apply_BE(
        incidence_rate = c(qx),
        incidence_rate_fraction = c(1000),
        BE_value = c(BE),
        BE_fraction = config$BE_fraction,
        newvar = c(qx_BE)
      ) %>%
      apply_monthly_rate(
        rate = c(qx_BE),
        method = "exponential",
        decrement = "yes",
        newvar = c(m_qx_BE)
      ) %>%
      proj_lx(c(m_qx_BE), newvar = c(lx_boy, lx_eom)) %>%
      proj_num(
        incidence_rate = c(m_qx_BE),
        lx = lx_boy,
        newvar = c(claim_num)
      )

    # Conditional logic based on config
    if (config$sum_insured_type == "fixed") {
      result <- result %>%
        proj_claim(
          num_decrement = c(claim_num),
          sum_insured = c(sum_insured),
          newvar = c(claim_amount)
        )
    } else if (config$sum_insured_type == "outstanding_debt") {
      result <- result %>%
        proj_outstanding_debt(
          monthly_pay = monthly_loan_pay,
          monthly_loan_interest = monthly_loan_interest,
          n_months = duration_months,
          months = c(months_bom, months_eom),
          newvar = c(outstanding_bom, outstanding_eom)
        ) %>%
        proj_claim(
          num_decrement = c(claim_num),
          sum_insured = c(outstanding_bom),
          newvar = c(claim_amount)
        )
    }

    # Add traceability
    result$projection_config <- config$name
    result
  }
}
```

### Step 5: Register Template

If creating a custom template outside pricingr:

```r
# In your package's zzz.R or startup script
.onLoad <- function(libname, pkgname) {
  pricingr::register_projection_template("my_term", make_term_projection)
  pricingr::register_projection_template("my_annuity", make_annuity_projection)
}
```

Or at runtime:

```r
# Register before building registry
register_projection_template("custom_product", make_custom_projection)
```

### Step 6: Create Configurations

```r
# R/configs.R or inst/configs/production.R

get_production_configs <- function() {
  qx_2006 <- pricingr::qx %>% filter(year == 2006)
  qx_2020 <- load_mortality_table("tables/qx_2020.rds")

  list(
    # AXA Products
    axa_cqs_death = projection_config(
      template = "cqs",
      name = "axa_cqs_death",
      ceding = "axa",
      product = "cqs",
      risk = "death",
      mortality_table = qx_2006,
      BE_fraction = 1,
      sum_insured_type = "fixed"
    ),

    axa_cqs_loe = projection_config(
      template = "cqs",
      name = "axa_cqs_loe",
      ceding = "axa",
      product = "cqs",
      risk = "loe",
      mortality_table = qx_2006,
      BE_fraction = 10,
      sum_insured_type = "fixed"
    ),

    # CF Products
    cf_cqs_death = projection_config(
      template = "cqs",
      name = "cf_cqs_death",
      ceding = "cf",
      product = "cqs",
      risk = "death",
      mortality_table = qx_2006,
      BE_fraction = 1,
      sum_insured_type = "outstanding_debt"
    ),

    cf_cqs_loe = projection_config(
      template = "cqs",
      name = "cf_cqs_loe",
      ceding = "cf",
      product = "cqs",
      risk = "loe",
      mortality_table = qx_2006,
      BE_fraction = 10,
      sum_insured_type = "outstanding_debt"
    )

    # Add more...
  )
}
```

### Step 7: Build and Test Registry

```r
# Validate first
configs <- get_production_configs()
validate_projection_configs(configs)

# Build registry
registry <- build_projection_registry(configs)

# Inspect
print(registry)

# Test matching
test_mp <- tibble(
  ceding = "axa",
  product = "cqs",
  risk = "death",
  policy_id = 1,
  entry_age = 40,
  gender = "M",
  duration_months = 12,
  inception_date = lubridate::ymd("2023-01-01"),
  sum_insured = 100000,
  BE = 0.8
)

match <- find_matching_projection(registry, test_mp)
print(match$proj_fn_name)  # Should be "axa_cqs_death"
```

### Step 8: Create Dispatcher and Run

```r
# Create dispatcher
dispatcher <- create_registry_dispatcher(registry)

# Run on full portfolio
results <- project_policies(
  mp_df = production_portfolio,
  proj_fn = dispatcher,
  .progress = TRUE,
  .policy_id = "policy_id",
  .on_error = "warn"
)

# Check for errors
if (!is.null(attr(results, "errors"))) {
  errors <- attr(results, "errors")
  warning(sprintf("Failed policies: %d", length(errors)))
}
```

---

## Configuration Management

### Option 1: R Code (Recommended for <50 configs)

```r
# inst/configs/production.R
get_production_configs <- function() {
  list(
    axa_cqs_death = projection_config(...),
    axa_cqs_loe = projection_config(...),
    ...
  )
}
```

**Pros**: Type safety, IDE completion, version control friendly
**Cons**: Requires R knowledge to modify

### Option 2: YAML (Recommended for >50 configs or non-R users)

```yaml
# inst/configs/production.yaml
axa_cqs_death:
  template: cqs
  ceding: axa
  product: cqs
  risk: death
  mortality_table: qx_2006   # Reference name
  BE_fraction: 1
  sum_insured_type: fixed

axa_cqs_loe:
  template: cqs
  ceding: axa
  product: cqs
  risk: loe
  mortality_table: qx_2006
  BE_fraction: 10
  sum_insured_type: fixed
```

```r
# Load with table resolution
mortality_tables <- list(
  qx_2006 = pricingr::qx %>% filter(year == 2006),
  qx_2020 = readRDS("tables/qx_2020.rds")
)

configs <- load_projection_configs(
  "inst/configs/production.yaml",
  mortality_tables = mortality_tables
)
```

**Pros**: Non-R users can edit, easy diff/review
**Cons**: Requires YAML knowledge, table references add complexity

### Option 3: Spreadsheet → YAML Pipeline

```
Excel/Google Sheets → Export CSV → R script → YAML
```

```r
# scripts/generate_configs.R
library(tidyverse)
library(yaml)

config_data <- read_csv("configs.csv")

configs <- config_data %>%
  split(.$config_name) %>%
  map(~ as.list(.x))

write_yaml(configs, "inst/configs/generated.yaml")
```

---

## Custom Template Development

### Template Anatomy

```r
make_<product>_projection <- function(config) {

  # ═══════════════════════════════════════════════════════════════════
  # SECTION 1: VALIDATION
  # ═══════════════════════════════════════════════════════════════════

  # Required fields

  required <- c("field1", "field2", "field3")
  missing <- setdiff(required, names(config))
  if (length(missing) > 0) {
    cli::cli_abort("Missing required config fields: {.val {missing}}")
  }

  # Value validation
  if (!config$field1 %in% c("valid1", "valid2")) {
    cli::cli_abort("Invalid field1: {.val {config$field1}}")
  }

  # ═══════════════════════════════════════════════════════════════════
  # SECTION 2: CAPTURE CONFIG
  # ═══════════════════════════════════════════════════════════════════

  force(config)  # CRITICAL: Forces evaluation, captures in closure

  # ═══════════════════════════════════════════════════════════════════
  # SECTION 3: RETURN PROJECTION CLOSURE
  # ═══════════════════════════════════════════════════════════════════

  function(mp_one) {

    # ─────────────────────────────────────────────────────────────────
    # 3.1 SETUP PHASE
    # ─────────────────────────────────────────────────────────────────
    result <- mp_one %>%
      setup_projection(
        n_months = duration_months,
        start_date = inception_date,
        entry_age = entry_age
      )

    # ─────────────────────────────────────────────────────────────────
    # 3.2 ASSUMPTION JOINS
    # ─────────────────────────────────────────────────────────────────
    result <- result %>%
      set_assumption(
        table = config$mortality_table,
        joinvars = c("current_age", "gender"),
        joinvars_table = c("current_age", "gender")
      )

    # ─────────────────────────────────────────────────────────────────
    # 3.3 RATE CALCULATIONS
    # ─────────────────────────────────────────────────────────────────
    result <- result %>%
      apply_BE(...) %>%
      apply_monthly_rate(...)

    # ─────────────────────────────────────────────────────────────────
    # 3.4 SURVIVORSHIP
    # ─────────────────────────────────────────────────────────────────
    result <- result %>%
      proj_lx(...)

    # ─────────────────────────────────────────────────────────────────
    # 3.5 CLAIM CALCULATIONS (CONDITIONAL)
    # ─────────────────────────────────────────────────────────────────
    if (config$benefit_type == "type_a") {
      result <- result %>% calculate_benefit_a(...)
    } else {
      result <- result %>% calculate_benefit_b(...)
    }

    # ─────────────────────────────────────────────────────────────────
    # 3.6 METADATA
    # ─────────────────────────────────────────────────────────────────
    result$projection_config <- config$name
    result$template_version <- "1.0.0"

    result
  }
}
```

### Registration Pattern

```r
# In your package's R/zzz.R
.onLoad <- function(libname, pkgname) {
  # Register all custom templates
  pricingr::register_projection_template("term_life", make_term_life_projection)
  pricingr::register_projection_template("whole_life", make_whole_life_projection)
  pricingr::register_projection_template("annuity", make_annuity_projection)
}
```

---

## Testing Strategy

### Unit Tests for Templates

```r
# tests/testthat/test-template-cqs.R

test_that("make_cqs_projection validates required fields", {
  expect_error(
    make_cqs_projection(list()),
    "Missing"
  )
})

test_that("make_cqs_projection validates sum_insured_type", {
  expect_error(
    make_cqs_projection(list(
      mortality_table = data.frame(),
      BE_fraction = 1,
      sum_insured_type = "invalid"
    )),
    "Invalid sum_insured_type"
  )
})

test_that("CQS projection runs for fixed sum insured", {
  qx <- pricingr::qx %>% filter(year == 2006)

  proj_fn <- make_cqs_projection(list(
    name = "test",
    mortality_table = qx,
    BE_fraction = 1,
    sum_insured_type = "fixed"
  ))

  mp <- tibble(
    policy_id = 1,
    entry_age = 40,
    gender = "M",
    duration_months = 12,
    inception_date = lubridate::ymd("2023-01-01"),
    sum_insured = 100000,
    BE = 0.8
  )

  result <- proj_fn(mp)

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 12)
  expect_true("claim_amount" %in% names(result))
})
```

### Integration Tests for Registry

```r
# tests/testthat/test-registry-integration.R

test_that("registry dispatches to correct projection", {
  configs <- example_projection_configs()
  registry <- build_projection_registry(configs, .verbose = FALSE)
  dispatcher <- create_registry_dispatcher(registry)

  # Test AXA routing
  mp_axa <- tibble(
    ceding = "axa", product = "cqs", risk = "death",
    policy_id = 1, entry_age = 40, gender = "M",
    duration_months = 12, inception_date = lubridate::ymd("2023-01-01"),
    sum_insured = 100000, BE = 0.8
  )

  result <- dispatcher(mp_axa)
  expect_equal(result$projection_config[1], "axa_cqs_death")

  # Test CF routing
  mp_cf <- mp_axa
  mp_cf$ceding <- "cf"
  mp_cf$monthly_loan_pay <- 1000
  mp_cf$monthly_loan_interest <- 0.005

  result <- dispatcher(mp_cf)
  expect_equal(result$projection_config[1], "cf_cqs_death")
})
```

### Regression Tests

```r
# tests/testthat/test-regression.R

test_that("new implementation matches legacy results", {
  # Run legacy
  legacy_result <- legacy_axa_cqs_death(test_mp)

  # Run new
  configs <- get_production_configs()
  registry <- build_projection_registry(configs["axa_cqs_death"], .verbose = FALSE)
  dispatcher <- create_registry_dispatcher(registry)
  new_result <- dispatcher(test_mp)

  # Compare key columns
  expect_equal(
    legacy_result$claim_amount,
    new_result$claim_amount,
    tolerance = 1e-10
  )
})
```

---

## Debugging Guide

### Problem: "No matching projection rule found"

**Diagnosis**:
```r
# Check model point values
mp <- portfolio[1, ]
print(mp[c("ceding", "product", "risk")])

# Check registry rules
print(registry)

# Check if column names match
registry_dims <- names(attr(registry, "col_meta"))
mp_cols <- names(mp)
setdiff(registry_dims, mp_cols)  # Missing in mp
```

**Common causes**:
1. Column name mismatch (e.g., `ceding_company` vs `ceding`)
2. Value mismatch (e.g., "AXA" vs "axa")
3. Missing catch-all rule

### Problem: "Column not found in model point"

**Diagnosis**:
```r
# The error message shows available columns
# Compare with registry dimensions
attr(registry, "col_meta")
```

**Fix**: Either rename portfolio columns or recreate registry with matching dimension names.

### Problem: Wrong projection selected

**Diagnosis**:
```r
# Check which rule matched
match <- find_matching_projection(registry, mp[1, ])
print(match$proj_fn_name)
print(match$priority)

# Check all rules
print(registry)
```

**Common causes**:
1. Priority misconfiguration
2. Overly broad wildcard rule
3. Missing specific rule

### Problem: Projection function error

**Diagnosis**:
```r
# Test projection directly
proj_fn <- find_matching_projection(registry, mp[1, ])$proj_fn
debug(proj_fn)
proj_fn(mp[1, ])
```

---

## Production Deployment

### Recommended Package Structure

```
mycompany.reinsurance/
├── DESCRIPTION
├── NAMESPACE
├── R/
│   ├── zzz.R                    # Template registration
│   ├── templates/
│   │   ├── template_cqs.R
│   │   ├── template_term.R
│   │   └── template_annuity.R
│   ├── configs.R                # get_production_configs()
│   └── registry.R               # build_production_registry()
├── inst/
│   └── configs/
│       ├── production.yaml
│       ├── test.yaml
│       └── uat.yaml
├── tests/
│   └── testthat/
│       ├── test-templates.R
│       ├── test-configs.R
│       └── test-regression.R
└── man/
```

### Environment-Specific Registries

```r
# R/registry.R

#' @export
build_production_registry <- function(env = Sys.getenv("ENV", "production")) {
  config_file <- switch(env,
    production = "production.yaml",
    uat = "uat.yaml",
    test = "test.yaml",
    stop("Unknown environment: ", env)
  )

  config_path <- system.file("configs", config_file, package = "mycompany.reinsurance")
  configs <- load_projection_configs(config_path, mortality_tables = get_mortality_tables())

  build_projection_registry(configs)
}
```

---

## Common Pitfalls

### 1. Forgetting `force(config)`

```r
# WRONG - config may not be captured correctly
make_bad_template <- function(config) {
  function(mp) {
    mp %>% mutate(x = config$value)  # May fail!
  }
}

# CORRECT
make_good_template <- function(config) {
  force(config)  # Force evaluation
  function(mp) {
    mp %>% mutate(x = config$value)
  }
}
```

### 2. Column Name Mismatches

```r
# Portfolio has "ceding_company"
# Registry expects "ceding"
# Result: "Column not found" error

# Fix: Use consistent naming
registry <- create_projection_registry(
  ceding_company = "exact",  # Match portfolio column
  product = "exact",
  risk = "exact"
)
```

### 3. Missing Catch-All Rule

```r
# If no rule matches, dispatcher throws error
# Always add a catch-all for safety

registry <- register_projection(registry,
  ceding = NA, product = NA, risk = NA,
  proj_fn = proj_default,
  proj_fn_name = "default"
)
```

### 4. Modifying Registry After Creating Dispatcher

```r
# WRONG - dispatcher captured old registry
dispatcher <- create_registry_dispatcher(registry)
registry <- register_projection(registry, ...)  # New rule not seen!

# CORRECT - create dispatcher after all registrations
registry <- register_projection(registry, ...)
dispatcher <- create_registry_dispatcher(registry)
```

---

## Checklist

### Pre-Implementation

- [ ] Inventoried all existing projection functions
- [ ] Identified dimension axes (ceding, product, risk, etc.)
- [ ] Grouped projections by structural similarity
- [ ] Designed template parameters

### Implementation

- [ ] Implemented template factory functions
- [ ] Added template validation
- [ ] Used `force(config)` in all templates
- [ ] Registered templates (in `zzz.R` or at runtime)
- [ ] Created configurations for all products
- [ ] Validated configurations with `validate_projection_configs()`

### Testing

- [ ] Unit tests for template validation
- [ ] Unit tests for projection output
- [ ] Integration tests for registry routing
- [ ] Regression tests against legacy implementation
- [ ] Performance benchmarks

### Deployment

- [ ] Documented all configurations
- [ ] Set up environment-specific configs (test/uat/prod)
- [ ] Created monitoring for failed projections
- [ ] Trained team on new workflow

---

*Implementation Guide v1.0.0 - pricingr Projection Registry System*
