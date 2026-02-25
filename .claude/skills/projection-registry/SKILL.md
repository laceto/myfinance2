---
name: projection-registry
description: 'Builds projection registry dispatchers for pricingr reinsurance modeling. Use when asked to create a registry, set up projection routing, configure multi-dimensional dispatching, build projection configs, create CQS projections, route by ceding company, product, risk, or underwriting year. Implements tibble-based rule matching with priority-based selection for insurance cash flow projections.'
---

# Projection Registry Builder

Creates projection registry dispatchers for pricingr's multi-dimensional routing system. Enables configuration-driven projection selection based on ceding company, product type, risk category, and underwriting year ranges.

## When to Use This Skill

Use this skill when you need to:
- **Set up projection routing** for multiple ceding companies or products
- **Create projection registries** with dimension-based matching
- **Build projection configurations** for CQS, term life, or custom products
- **Route projections** based on ceding, product, risk, or time dimensions
- **Batch process portfolios** with varying projection logic
- **Manage reinsurance rules** separate from projection code
- **Create dispatchers** for use with `project_policies()`

**Trigger phrases:**
- "Create a projection registry"
- "Set up CQS projections"
- "Build a dispatcher for [product/ceding]"
- "Route projections by [dimension]"
- "Configure projection matching"
- "Add projection rules"
- "Create projection configs"

## Core Concepts

### Architecture

```
Configuration → Factory → Registry → Dispatcher → Execution
     ↓             ↓          ↓           ↓            ↓
projection_   create_    register_   create_     project_
  config()    proj_fn()  projection() registry_   policies()
                                     dispatcher()
```

### Key Components

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **Config** | Parameters for projection | Template name, dimensions, params | `projection_config` list |
| **Registry** | Rule storage | Dimension specs | Empty `projection_registry` tibble |
| **Rule** | Mapping criteria → function | Config + proj_fn | Registry row |
| **Dispatcher** | Router function | Registry | Function: `mp → tibble` |

### Dimension Types

**Exact Match:**
- Column: `ceding = "exact"`
- Matches: `ceding == "axa"` or `ceding == NA` (wildcard)

**Range Match:**
- Columns: `uw_year = c("uw_year_min", "uw_year_max")`
- Matches: `uw_year >= min AND uw_year <= max` (NA = unbounded)

## Quick Start Workflow

### Step 1: Create Configurations

```r
library(pricingr)
library(dplyr)

# Load mortality table
qx_table <- pricingr::qx %>% filter(year == 2006)

# Define projection configs
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
  ),

  axa_cqs_loe = projection_config(
    template = "cqs",
    name = "axa_cqs_loe",
    ceding = "axa",
    product = "cqs",
    risk = "loe",
    mortality_table = qx_table,
    BE_fraction = 10,
    sum_insured_type = "fixed"
  ),

  # Add catch-all default
  default = projection_config(
    template = "cqs",
    name = "default",
    ceding = NA,
    product = NA,
    risk = NA,
    mortality_table = qx_table,
    BE_fraction = 1,
    sum_insured_type = "fixed"
  )
)
```

### Step 2: Build Registry

```r
# Validate configs first (optional but recommended)
validate_projection_configs(configs)

# Build registry
registry <- build_projection_registry(
  configs,
  dimensions = c("ceding", "product", "risk")
)

# Inspect registry
print(registry)
```

### Step 3: Create Dispatcher

```r
# Create dispatcher closure
dispatcher <- create_registry_dispatcher(registry)

# Test with single model point
test_mp <- tibble(
  policy_id = 1,
  ceding = "axa",
  product = "cqs",
  risk = "death",
  inception_date = lubridate::ymd("2023-01-01"),
  duration_months = 12L,
  entry_age = 40L,
  gender = "M",
  sum_insured = 100000,
  BE = 0.8
)

result <- dispatcher(test_mp)
```

### Step 4: Batch Process Portfolio

```r
# Run on full portfolio
results <- project_policies(
  mp_df = portfolio,
  proj_fn = dispatcher,
  .policy_id = "policy_id",
  .progress = TRUE
)
```

## Detailed Workflows

### Workflow 1: Basic Registry (Single Dimension)

**Scenario:** Route by ceding company only

```r
# 1. Create registry
registry <- create_projection_registry(ceding = "exact")

# 2. Register projection for AXA
registry <- register_projection(registry,
  ceding = "axa",
  proj_fn = function(mp) {
    mp %>%
      setup_projection(
        n_months = duration_months,
        start_date = inception_date,
        entry_age = entry_age
      ) %>%
      mutate(source = "axa_projection")
  },
  proj_fn_name = "proj_axa"
)

# 3. Register default for all others
registry <- register_projection(registry,
  ceding = NA,  # Wildcard
  proj_fn = function(mp) {
    mp %>%
      setup_projection(
        n_months = duration_months,
        start_date = inception_date,
        entry_age = entry_age
      ) %>%
      mutate(source = "default_projection")
  },
  proj_fn_name = "proj_default"
)

# 4. Create dispatcher
dispatcher <- create_registry_dispatcher(registry)
```

### Workflow 2: Multi-Dimensional Registry

**Scenario:** Route by ceding, product, and risk

```r
# 1. Create registry with multiple dimensions
registry <- create_projection_registry(
  ceding = "exact",
  product = "exact",
  risk = "exact"
)

# 2. Register specific rules (highest priority)
registry <- register_projection(registry,
  ceding = "axa", product = "cqs", risk = "death",
  proj_fn = proj_axa_cqs_death,
  proj_fn_name = "axa_cqs_death"
)

# 3. Register partial wildcards (medium priority)
registry <- register_projection(registry,
  ceding = "axa", product = "cqs", risk = NA,
  proj_fn = proj_axa_cqs_default,
  proj_fn_name = "axa_cqs_default"
)

# 4. Register catch-all (lowest priority)
registry <- register_projection(registry,
  ceding = NA, product = NA, risk = NA,
  proj_fn = proj_default,
  proj_fn_name = "default"
)

# Priority auto-calculated:
# axa_cqs_death: 100 (no wildcards)
# axa_cqs_default: 90 (1 wildcard)
# default: 70 (3 wildcards)
```

### Workflow 3: Range Dimensions (Underwriting Year)

**Scenario:** Different logic for policies written before/after 2023

```r
# 1. Create registry with range dimension
registry <- create_projection_registry(
  ceding = "exact",
  uw_year = c("uw_year_min", "uw_year_max")
)

# 2. Register new business (2023+)
registry <- register_projection(registry,
  ceding = "axa",
  uw_year_min = 2023L,
  uw_year_max = NA,  # Unbounded upper
  proj_fn = function(mp) {
    mp %>%
      setup_projection(...) %>%
      mutate(
        assumptions = "new_table",
        cession_rate = 0.30
      )
  },
  proj_fn_name = "axa_2023_plus"
)

# 3. Register legacy (before 2023)
registry <- register_projection(registry,
  ceding = "axa",
  uw_year_min = NA,  # Unbounded lower
  uw_year_max = 2022L,
  proj_fn = function(mp) {
    mp %>%
      setup_projection(...) %>%
      mutate(
        assumptions = "legacy_table",
        cession_rate = 0.25
      )
  },
  proj_fn_name = "axa_legacy"
)
```

### Workflow 4: Using Projection Configs (Recommended)

**Scenario:** Configuration-driven approach with CQS template

```r
# 1. Load reference data
qx_2006 <- pricingr::qx %>% filter(year == 2006)

# 2. Create configs
configs <- list(
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

  cf_cqs_death = projection_config(
    template = "cqs",
    name = "cf_cqs_death",
    ceding = "cf",
    product = "cqs",
    risk = "death",
    mortality_table = qx_2006,
    BE_fraction = 1,
    sum_insured_type = "outstanding_debt"
  )
)

# 3. Build registry from configs
registry <- build_projection_registry(configs)

# 4. Create dispatcher
dispatcher <- create_registry_dispatcher(registry)
```

### Workflow 5: YAML Configuration

**Scenario:** Load configs from YAML file

**File: `configs/production.yaml`**
```yaml
axa_cqs_death:
  template: cqs
  ceding: axa
  product: cqs
  risk: death
  mortality_table: qx_2006  # Reference name
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

**R code:**
```r
# 1. Prepare mortality table references
mortality_tables <- list(
  qx_2006 = pricingr::qx %>% filter(year == 2006)
)

# 2. Load configs from YAML
configs <- load_projection_configs(
  "configs/production.yaml",
  mortality_tables = mortality_tables
)

# 3. Build and dispatch
registry <- build_projection_registry(configs)
dispatcher <- create_registry_dispatcher(registry)
```

## CQS Template Reference

### Required Configuration Fields

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `template` | character | `"cqs"` | Template identifier |
| `name` | character | Unique ID | For traceability |
| `ceding` | character | Company code | Dimension value |
| `product` | character | Product code | Dimension value |
| `risk` | character | Risk code | Dimension value |
| `mortality_table` | data.frame | Columns: `current_age`, `gender`, `qx` | Mortality rates |
| `BE_fraction` | numeric | `1`, `10`, etc. | Best estimate divisor |
| `sum_insured_type` | character | `"fixed"` or `"outstanding_debt"` | Claim calculation method |

### Required Model Point Columns

**Always required:**
- `duration_months`, `inception_date`, `entry_age`, `gender`, `BE`

**For `sum_insured_type = "fixed"`:**
- `sum_insured`

**For `sum_insured_type = "outstanding_debt"`:**
- `monthly_loan_pay`, `monthly_loan_interest`

### Output Columns Added

- `qx_BE`, `m_qx_BE` (adjusted mortality rates)
- `lx_boy`, `lx_eom` (survivorship)
- `claim_num`, `claim_amount` (claims)
- `outstanding_bom`, `outstanding_eom` (if applicable)
- `projection_config` (config name for audit trail)

## Priority System

Priority determines which rule wins when multiple rules match:

**Automatic Priority:**
```
priority = 100 - (number_of_NA_wildcards × 10)
```

**Examples:**

| Rule | NA Count | Auto Priority |
|------|----------|---------------|
| `ceding="axa", product="cqs", risk="death"` | 0 | 100 |
| `ceding="axa", product="cqs", risk=NA` | 1 | 90 |
| `ceding="axa", product=NA, risk=NA` | 2 | 80 |
| `ceding=NA, product=NA, risk=NA` | 3 | 70 |

**Manual Override:**
```r
registry <- register_projection(registry,
  ceding = NA, product = NA, risk = NA,
  proj_fn = proj_special_default,
  priority = 999L  # Force highest priority
)
```

## Testing and Validation

### Pre-Flight Validation

```r
# Check configs before building
tryCatch({
  validate_projection_configs(configs)
  message("✓ Configs valid")
}, error = function(e) {
  stop("Config validation failed: ", e$message)
})
```

### Test Matching Logic

```r
# Check which rule matches for each portfolio row
test_portfolio <- portfolio %>%
  head(5) %>%
  rowwise() %>%
  mutate(
    match = list(find_matching_projection(registry, pick(everything()))),
    matched_rule = match$proj_fn_name,
    priority = match$priority
  ) %>%
  ungroup()

print(test_portfolio %>% select(ceding, product, risk, matched_rule, priority))
```

### Verify Dispatcher Output

```r
# Test dispatcher on single policy
test_mp <- portfolio[1, ]
result <- dispatcher(test_mp)

# Verify output structure
stopifnot(
  is.data.frame(result),
  nrow(result) == test_mp$duration_months,
  "claim_amount" %in% names(result)
)
```

## Common Patterns

### Pattern 1: Company-Specific Variants

```r
configs <- list(
  # AXA uses fixed sum insured
  axa_cqs_death = projection_config(
    template = "cqs",
    ceding = "axa",
    product = "cqs",
    risk = "death",
    ...,
    sum_insured_type = "fixed"
  ),

  # CF uses outstanding debt
  cf_cqs_death = projection_config(
    template = "cqs",
    ceding = "cf",
    product = "cqs",
    risk = "death",
    ...,
    sum_insured_type = "outstanding_debt"
  )
)
```

### Pattern 2: Risk-Specific BE Fractions

```r
configs <- list(
  # Death: BE_fraction = 1
  axa_cqs_death = projection_config(
    ...,
    risk = "death",
    BE_fraction = 1
  ),

  # Loss of employment: BE_fraction = 10
  axa_cqs_loe = projection_config(
    ...,
    risk = "loe",
    BE_fraction = 10
  )
)
```

### Pattern 3: Time-Based Rule Changes

```r
registry <- create_projection_registry(
  ceding = "exact",
  uw_year = c("uw_year_min", "uw_year_max")
)

# New terms for 2023+
registry <- register_projection(registry,
  ceding = "axa",
  uw_year_min = 2023L,
  uw_year_max = NA,
  proj_fn = proj_axa_new_treaty
)

# Legacy terms before 2023
registry <- register_projection(registry,
  ceding = "axa",
  uw_year_min = NA,
  uw_year_max = 2022L,
  proj_fn = proj_axa_legacy_treaty
)
```

## Creating a Registry from Custom Projection Functions

This workflow shows how to create a registry from a custom projection function (like `pricing_model_cqs`). The key principle is **explicit assumption passing** - assumptions are captured in closures rather than relying on global variables.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Registry Creation Flow                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. PRICING MODEL FUNCTION (accepts assumptions as parameters)          │
│     pricing_model_xxx(df, extract_mp, BE_qx, rx, uix)                  │
│                           ↑                                             │
│                           │ passes assumptions                          │
│                           │                                             │
│  2. TEMPLATE FACTORY (captures assumptions in closure)                  │
│     make_xxx_projection(config) → function(mp) { ... }                 │
│                           ↑                                             │
│                           │ extracts from config                        │
│                           │                                             │
│  3. PROJECTION CONFIG (stores assumptions)                              │
│     projection_config(BE_qx = table, rx = table, ...)                  │
│                           ↑                                             │
│                           │ loaded at registry build time               │
│                           │                                             │
│  4. ASSUMPTION TABLES (loaded once)                                     │
│     load_assumptions() → BE_qx, rx, uix                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Create the Pricing Model Function

The pricing model function must accept assumption tables as parameters instead of using global variables.

**File: `models/pricing_model_cqs.R`**

```r
#' CQS Pricing Model
#'
#' @param df Single-row model point tibble
#' @param extract_mp If TRUE, return full projection; if FALSE, aggregate
#' @param BE_qx Best estimate mortality adjustment table
#' @param rx Lapse rate table
#' @param uix Unemployment incidence table
#' @return Projected cash flows tibble
pricing_model_cqs <- function(df, extract_mp = FALSE, BE_qx, rx, uix) {

  # Define output columns

  columns_to_select <- c("id_mp", "cover", "run", "uy", "years",
                         "single_premium_amount", "DC_claim_amount",
                         "LAPSE_claim_amount", "reserve_amount_bom")

  tryCatch({
    # Build projection pipeline using pricingr utilities
    projected_mp <- df |>
      dplyr::slice(1) |>
      pricingr::expand_df(n_months = pol_duration_months) |>
      pricingr::proj_time(start_date = start_date, n_months = pol_duration_months) |>
      pricingr::proj_age(years = years, entry_age = life1_age_at_entry,
                         newvar = current_age, months2bd = 12) |>

      # Join assumption tables (passed as parameters, not global!)
      pricingr::set_assumption(
        table = qx,  # Base mortality (global or passed)
        joinvars = c("current_age", "life1_gender"),
        joinvars_table = c("current_age", "gender")
      ) |>
      pricingr::set_assumption(
        table = BE_qx,  # ← Passed parameter
        joinvars = c("years", "life1_gender", "year_assumption_mortality"),
        joinvars_table = c("years", "gender", "year_assumption_mortality")
      ) |>
      pricingr::set_assumption(
        table = rx,  # ← Passed parameter
        joinvars = c("years", "assumption_lapse"),
        joinvars_table = c("years", "assumption_lapse")
      ) |>
      pricingr::set_assumption(
        table = uix,  # ← Passed parameter
        joinvars = c("current_age", "year_assumption_uix"),
        joinvars_table = c("current_age", "year_assumption_uix")
      ) |>

      # Apply BE adjustments and calculate rates
      pricingr::apply_BE(
        incidence_rate = c(qx),
        incidence_rate_fraction = c(1000),
        BE_value = c(BE),
        BE_fraction = c(1),
        newvar = c(qx_BE)
      ) |>
      pricingr::apply_monthly_rate(
        rate = c(qx_BE, rix, uix),
        method = c("exponential", "exponential", "exponential"),
        decrement = c("yes", "yes", "yes"),
        newvar = c(m_qx_BE_DC, m_rix_BE_LAPSE, m_uix_BE_ILOE)
      ) |>

      # Project survivorship and claims
      pricingr::proj_lx(c(m_qx_BE_DC, m_rix_BE_LAPSE, m_uix_BE_ILOE),
                        newvar = c(lx_bom, lx_eom)) |>
      pricingr::proj_claim(...) |>
      pricingr::proj_reserve_sp(...)

    # Return full or aggregated output
    if (extract_mp) {
      output <- projected_mp
    } else {
      output <- projected_mp |>
        pricingr::map_reduce_by_var(
          sum_vars = c('ends_with("amount")'),
          first_vars = c('ends_with("_bom")'),
          last_vars = c('ends_with("_eom")'),
          id_mp, cover, run, uy, years
        ) |>
        dplyr::select(dplyr::all_of(columns_to_select))
    }

    return(output)

  },
  # Error handler: re-throw with context (don't swallow errors!)
  error = function(e) {
    cli::cli_abort(
      c("Projection failed for id_mp = {unique(df$id_mp)}",
        "x" = conditionMessage(e)),
      parent = e
    )
  })
}
```

**Key points:**
- All assumption tables are function parameters (`BE_qx`, `rx`, `uix`)
- Error handler re-throws with `cli::cli_abort()` - never swallow errors!
- Returns tibble (not string) on success

### Step 2: Create the Template Factory

The template factory captures assumptions from the config in a closure and passes them to the pricing model.

**File: `axa_registry.R`**

```r
#' CQS Template Factory
#'
#' Creates a projection function for CQS products.
#'
#' @param config List with fields:
#'   - BE_qx_cqs: Best estimate mortality table
#'   - rx_cqs: Lapse rate table
#'   - uix_age_cqs: Unemployment incidence table
#'   - extract_mp: Logical, whether to return full projection
#'
#' @return A projection function (closure with assumptions bound)
make_cqs_projection <- function(config) {

  # Capture assumptions from config in closure
  extract_mp <- if (is.null(config$extract_mp)) FALSE else config$extract_mp
  BE_qx_cqs <- config$BE_qx_cqs
  rx_cqs <- config$rx_cqs
  uix_age_cqs <- config$uix_age_cqs

  # Return projection function with assumptions bound

  function(mp) {
    # Validate tables were provided
    if (is.null(BE_qx_cqs) || is.null(rx_cqs) || is.null(uix_age_cqs)) {
      cli::cli_abort(c(
        "CQS assumptions not provided in config",
        "x" = "BE_qx_cqs, rx_cqs, or uix_age_cqs tables missing",
        "i" = "Add assumption tables to projection_config()"
      ))
    }

    # Call pricing model with captured assumptions
    pricing_model_cqs(
      mp,
      extract_mp = extract_mp,
      BE_qx = BE_qx_cqs,
      rx = rx_cqs,
      uix = uix_age_cqs
    )
  }
}

# Register the template
projectionregistry::register_projection_template("cqs", make_cqs_projection)
```

**Key points:**
- `config$BE_qx_cqs` is captured in the closure at registration time
- The returned function only takes `mp` as argument
- Validation happens at projection time, with helpful error messages

### Step 3: Load Assumption Tables

Load assumptions once and make them available for configs.

```r
#' Load assumption tables
load_assumptions <- function(path = "data-raw/assumptions") {

  cli::cli_h2("Loading Assumption Tables")

  # Best Estimate mortality
  mortality_raw <- readr::read_tsv(file.path(path, "Mortality.txt"))

  BE_qx_cqs <<- mortality_raw %>%
    tidyr::pivot_longer(cols = c("Male", "Female"), names_to = "gender") %>%
    dplyr::rename(years = Years, BE = value, cover = Cover) %>%
    dplyr::mutate(
      gender = dplyr::if_else(gender == "Male", "M", "F"),
      years = years - 1,
      year_assumption_mortality = 2026
    ) %>%
    dplyr::filter(cover == "CQS")

  # Lapse rates
  lapse_raw <- readr::read_tsv(file.path(path, "Lapse.txt"))

  rx_cqs <<- lapse_raw %>%
    dplyr::filter(Cover == "CQS") %>%
    dplyr::select(years = Years, rix = Lapse) %>%
    dplyr::mutate(years = years - 1, assumption_lapse = "be")

  # Unemployment incidence
  uix_age_cqs <<- readr::read_tsv(file.path(path, "uix_age_cqs.txt")) %>%
    dplyr::rename(current_age = AGE, year_assumption_uix = UY)

  cli::cli_alert_success("Loaded BE_qx_cqs ({nrow(BE_qx_cqs)} rows)")
  cli::cli_alert_success("Loaded rx_cqs ({nrow(rx_cqs)} rows)")
  cli::cli_alert_success("Loaded uix_age_cqs ({nrow(uix_age_cqs)} rows)")
}
```

### Step 4: Create Projection Configs

Configs include the assumption tables, which get captured in closures.

```r
#' Create projection configurations
create_configs <- function() {

  configs <- list(

    # CQS - Credit Security
    cqs = projectionregistry::projection_config(
      template = "cqs",
      name = "cqs",

      # Routing dimensions
      cover = "CQS",
      ceding = "AXA",
      product = "CQS",

      # Assumption tables (captured in closure!)
      BE_qx_cqs = BE_qx_cqs,
      rx_cqs = rx_cqs,
      uix_age_cqs = uix_age_cqs,

      # Other parameters
      risk = "DC_LAPSE_LOE",
      extract_mp = FALSE,
      description = "Credit Security - death, lapse, unemployment"
    )
  )

  return(configs)
}
```

### Step 5: Build Registry and Dispatcher

```r
# 1. Load assumptions
load_assumptions()

# 2. Create configs (assumptions are captured here)
configs <- create_configs()

# 3. Build registry
registry <- projectionregistry::build_projection_registry(
  configs = configs,
  dimensions = c("cover"),  # Route by cover
  .verbose = TRUE
)

# 4. Create dispatcher
dispatcher <- projectionregistry::create_registry_dispatcher(registry)
```

### Step 6: Run Projections

```r
# Test single model point
single_mp <- portfolio[1, ]
result <- dispatcher(single_mp)

# Batch process with project_policies
results <- pricingr::project_policies(
  mp_df = portfolio,
  proj_fn = dispatcher,
  .policy_id = "id_mp",
  .on_error = "warn",  # ← Note: .on_error, not .error_action!
  .progress = TRUE
)
```

### Complete Example: Adding a New Product

To add a new product (e.g., LOE), follow this checklist:

```r
# ═══════════════════════════════════════════════════════════════
# CHECKLIST: Adding a New Product to Registry
# ═══════════════════════════════════════════════════════════════

# □ Step 1: Create pricing model function
#   File: models/pricing_model_loe.R
#   - Accept assumptions as parameters
#   - Use cli::cli_abort() in error handler

# □ Step 2: Create template factory
#   In: registry.R
#   - Capture assumptions from config
#   - Validate before calling pricing model
#   - Pass assumptions to pricing model

# □ Step 3: Register template
#   projectionregistry::register_projection_template("loe", make_loe_projection)

# □ Step 4: Load assumptions
#   In: load_assumptions()
#   - Load LOE-specific tables
#   - Apply transformations (years-1, gender mapping, etc.)

# □ Step 5: Add config
#   In: create_configs()
#   - Include all assumption tables
#   - Set routing dimensions

# □ Step 6: Test
#   - Test dispatcher directly: dispatcher(single_mp)
#   - Test with project_policies()
#   - Verify assumption tables are bound correctly
# ═══════════════════════════════════════════════════════════════
```

### Anti-Patterns to Avoid

```r
# ❌ WRONG: Relying on global variables in pricing model
pricing_model_bad <- function(df) {
  df |>
    set_assumption(table = BE_qx_cqs) |>  # Global lookup - fragile!
    ...
}

# ❌ WRONG: Swallowing errors
tryCatch({
  ...
},
error = function(e) {
  return(paste0("error at ", id))  # Returns string, not tibble!
})

# ❌ WRONG: Wrong parameter name in project_policies
project_policies(
  mp_df = df,
  proj_fn = dispatcher,
  .error_action = "warn"  # ← WRONG! Should be .on_error
)

# ✅ CORRECT: Explicit assumption passing
pricing_model_good <- function(df, BE_qx, rx) {
  df |>
    set_assumption(table = BE_qx) |>  # Passed explicitly
    set_assumption(table = rx) |>
    ...
}

# ✅ CORRECT: Re-throw errors with context
tryCatch({
  ...
},
error = function(e) {
  cli::cli_abort(
    c("Failed for id_mp = {id}", "x" = conditionMessage(e)),
    parent = e
  )
})

# ✅ CORRECT: Use .on_error parameter
project_policies(
  mp_df = df,
  proj_fn = dispatcher,
  .on_error = "warn"  # ← Correct!
)
```

## Debugging Guide

### Issue: "No matching projection rule found"

**Cause:** No rule matches the model point's dimension values.

**Diagnosis:**
```r
# Check model point values
mp <- portfolio[1, ]
print(mp[c("ceding", "product", "risk")])

# Check registry rules
print(registry)

# Check dimension column names
registry_cols <- names(attr(registry, "col_meta"))
mp_cols <- names(mp)
setdiff(registry_cols, mp_cols)  # Missing in mp
```

**Fix:**
1. Add catch-all rule with all `NA` dimensions
2. Fix column name mismatch
3. Fix value mismatch (check case sensitivity)

### Issue: "Column not found in model point"

**Cause:** Portfolio missing dimension column required by registry.

**Fix:**
```r
# Option 1: Add column to portfolio
portfolio$risk <- "death"

# Option 2: Recreate registry with correct dimensions
registry <- create_projection_registry(
  ceding_company = "exact",  # Match portfolio column name
  product_type = "exact"
)
```

### Issue: Wrong projection selected

**Cause:** Lower-priority rule matches instead of expected rule.

**Diagnosis:**
```r
# Check which rule matched
match <- find_matching_projection(registry, mp[1, ])
print(paste("Matched:", match$proj_fn_name, "Priority:", match$priority))

# Check all matching rules manually
registry %>%
  filter(
    (is.na(ceding) | ceding == mp$ceding[1]) &
    (is.na(product) | product == mp$product[1])
  ) %>%
  select(ceding, product, proj_fn_name, priority) %>%
  arrange(desc(priority))
```

**Fix:**
1. Add more specific rule with higher priority
2. Use manual priority override
3. Check for overly broad wildcard rules

### Issue: Projection function error

**Cause:** Config parameters don't match model point columns.

**Diagnosis:**
```r
# Test projection directly
proj_fn <- find_matching_projection(registry, mp[1, ])$proj_fn
debug(proj_fn)
result <- proj_fn(mp[1, ])
```

**Common causes:**
- Missing required MP columns (e.g., `sum_insured`)
- Incorrect column data types
- Missing mortality table age/gender combinations

## Best Practices

### ✅ Do's

1. **Always add a catch-all rule**
   ```r
   register_projection(registry,
     ceding = NA, product = NA, risk = NA,
     proj_fn = proj_default
   )
   ```

2. **Validate before building**
   ```r
   validate_projection_configs(configs)
   ```

3. **Use meaningful names**
   ```r
   proj_fn_name = "axa_cqs_death_v2_2024"
   ```

4. **Test with sample data**
   ```r
   test_mp <- portfolio %>% head(1)
   result <- dispatcher(test_mp)
   ```

5. **Check which rules matched**
   ```r
   results %>%
     select(policy_id, projection_config) %>%
     distinct() %>%
     count(projection_config)
   ```

### ❌ Don'ts

1. **Don't modify registry after creating dispatcher**
   ```r
   # WRONG
   dispatcher <- create_registry_dispatcher(registry)
   registry <- register_projection(registry, ...)  # Too late!

   # CORRECT
   registry <- register_projection(registry, ...)
   dispatcher <- create_registry_dispatcher(registry)
   ```

2. **Don't use inconsistent column names**
   ```r
   # Portfolio has "ceding_company", registry expects "ceding"
   # Result: Column not found error
   ```

3. **Don't forget case sensitivity**
   ```r
   ceding = "AXA"  # vs "axa" - won't match!
   ```

4. **Don't skip validation**
   ```r
   # Can catch config issues early
   validate_projection_configs(configs)
   ```

## Output Format

The skill follows this pattern when creating a registry:

1. **Load dependencies** and reference data
2. **Create configurations** using `projection_config()`
3. **Validate** with `validate_projection_configs()`
4. **Build registry** with `build_projection_registry()`
5. **Create dispatcher** with `create_registry_dispatcher()`
6. **Test** on sample data
7. **Provide usage example** with `project_policies()`

## References

Bundled documentation in `/references/`:

1. **`registry-dispatcher.Rmd`** - Multi-dimensional registry vignette with examples
2. **`cheatsheet.md`** - Quick reference for common operations
3. **`architecture.md`** - System design and data flow diagrams
4. **`api-reference.md`** - Complete API documentation
5. **`implementation-guide.md`** - Step-by-step implementation roadmap

## Quick Reference Card

```r
# ═══════════════════════════════════════════════════════════════
# PROJECTION REGISTRY - QUICK REFERENCE
# ═══════════════════════════════════════════════════════════════

# 1. CREATE CONFIGS
configs <- list(
  name1 = projection_config(
    template = "cqs",
    name = "name1",
    ceding = "axa",
    product = "cqs",
    risk = "death",
    mortality_table = qx_table,
    BE_fraction = 1,
    sum_insured_type = "fixed"
  )
)

# 2. BUILD REGISTRY
registry <- build_projection_registry(configs)

# 3. CREATE DISPATCHER
dispatcher <- create_registry_dispatcher(registry)

# 4. RUN PROJECTIONS
results <- project_policies(portfolio, dispatcher, .policy_id = "policy_id")

# ═══════════════════════════════════════════════════════════════
```
