# Projection Registry System - API Reference

> **Target Audience**: R Software Developers
> **Version**: 1.0.0

---

## Table of Contents

1. [Registry Dispatcher API](#registry-dispatcher-api)
2. [Projection Factory API](#projection-factory-api)
3. [Registry Builder API](#registry-builder-api)
4. [Template API](#template-api)
5. [S3 Classes](#s3-classes)
6. [Error Catalog](#error-catalog)

---

## Registry Dispatcher API

**Source**: `R/registry_dispatcher.R`

### `create_projection_registry()`

Creates an empty tibble-based registry for storing projection rules.

**Signature**:
```r
create_projection_registry(...)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `...` | Named arguments | Column specifications. Each name becomes a dimension. Values: `"exact"` for exact matching, or `c("min_col", "max_col")` for range matching. |

**Returns**: `projection_registry` (S3 class inheriting from `tbl_df`)

**Examples**:
```r
# Exact match dimensions
registry <- create_projection_registry(
  ceding = "exact",
  product = "exact",
  risk = "exact"
)

# With range dimension (underwriting year)
registry <- create_projection_registry(
  ceding = "exact",
  product = "exact",
  uw_year = c("uw_year_min", "uw_year_max")
)
```

**Errors**:
- `"No columns specified"` - Called with no arguments

---

### `register_projection()`

Adds a projection rule to a registry.

**Signature**:
```r
register_projection(
  registry,
  ...,
  proj_fn,
  proj_fn_name = NULL,
  priority = NULL,
  .verbose = TRUE
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `registry` | projection_registry | Required | Registry created by `create_projection_registry()` |
| `...` | Named arguments | - | Dimension values for matching. Use `NA` for wildcard. |
| `proj_fn` | function | Required | Projection function `f(mp_one) → tibble` |
| `proj_fn_name` | character | `deparse(substitute(proj_fn))` | Human-readable name for debugging |
| `priority` | integer | Auto-calculated | Higher wins on tie. Auto: `100 - na_count * 10` |
| `.verbose` | logical | `TRUE` | Print registration message |

**Returns**: Updated `projection_registry` (invisibly)

**Priority Auto-Calculation**:
```
priority = 100 - (number_of_NA_dimensions * 10)

Examples:
  ceding="axa", product="cqs", risk="death" → priority = 100
  ceding="axa", product="cqs", risk=NA      → priority = 90
  ceding="axa", product=NA, risk=NA         → priority = 80
  ceding=NA, product=NA, risk=NA            → priority = 70
```

**Examples**:
```r
# Specific rule
registry <- register_projection(registry,
  ceding = "axa", product = "cqs", risk = "death",
  proj_fn = proj_axa_cqs_death,
  proj_fn_name = "axa_cqs_death"
)

# Wildcard rule (any risk)
registry <- register_projection(registry,
  ceding = "axa", product = "cqs", risk = NA,
  proj_fn = proj_axa_cqs_default
)

# Catch-all rule
registry <- register_projection(registry,
  ceding = NA, product = NA, risk = NA,
  proj_fn = proj_default
)

# Manual priority override
registry <- register_projection(registry,
  ceding = NA, product = NA, risk = NA,
  proj_fn = proj_high_priority_default,
  priority = 999L
)
```

**Errors**:
- `"Invalid registry"` - Not a `projection_registry` object
- `"Invalid projection function"` - `proj_fn` is not a function

---

### `find_matching_projection()`

Finds the best matching projection rule for a model point.

**Signature**:
```r
find_matching_projection(registry, mp)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `registry` | projection_registry | Registry with rules |
| `mp` | data.frame | Single-row model point |

**Returns**: List with:
- `proj_fn`: The matching projection function
- `proj_fn_name`: Name of the function
- `priority`: Priority of matched rule
- `row_index`: Row index in registry

Returns `NULL` if no match found.

**Matching Algorithm**:
```
For each dimension:
  - Exact: match if registry_value == mp_value OR registry_value is NA
  - Range: match if mp_value >= min AND mp_value <= max (NA = unbounded)

Among all matching rules, return the one with highest priority.
```

**Errors**:
- `"Missing column in model point"` - Required dimension column not found

---

### `create_registry_dispatcher()`

Creates a dispatcher function that routes model points to projections.

**Signature**:
```r
create_registry_dispatcher(registry)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `registry` | projection_registry | Registry with rules |

**Returns**: Function `f(mp) → tibble` that:
1. Finds matching projection rule
2. Executes projection function
3. Returns projection result

**Behavior**:
- Dispatcher captures registry at creation time (immutable)
- Throws error if no matching rule found

**Examples**:
```r
dispatcher <- create_registry_dispatcher(registry)

# Use with project_policies
results <- project_policies(portfolio, dispatcher, .policy_id = "policy_id")

# Or call directly
result <- dispatcher(single_model_point)
```

**Errors**:
- `"Invalid registry"` - Not a `projection_registry` object
- `"No matching projection rule found"` - At dispatch time, no rule matches

**Warnings**:
- `"Empty registry"` - Registry has no rules

---

### `print.projection_registry()`

Print method for projection registries.

**Signature**:
```r
## S3 method for class 'projection_registry'
print(x, ...)
```

**Output**:
```
── Projection Registry ─────────────────────────────────────────────────────────
ℹ Dimensions: "ceding (exact)", "product (exact)", and "risk (exact)"
ℹ Rules: 4
# A tibble: 4 × 4
  ceding product risk  proj_fn_name   priority
  <chr>  <chr>   <chr> <chr>             <int>
1 axa    cqs     death axa_cqs_death       100
2 axa    cqs     loe   axa_cqs_loe         100
3 cf     cqs     death cf_cqs_death        100
4 NA     NA      NA    default              70
```

---

## Projection Factory API

**Source**: `R/projection_factory.R`

### `register_projection_template()`

Registers a template factory function.

**Signature**:
```r
register_projection_template(name, factory_fn)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | character(1) | Unique template identifier (e.g., "cqs", "term") |
| `factory_fn` | function | Factory: `f(config) → f(mp) → tibble` |

**Returns**: `invisible(NULL)`

**Examples**:
```r
register_projection_template("custom", function(config) {
  force(config)
  function(mp_one) {
    mp_one %>%
      setup_projection(
        n_months = duration_months,
        start_date = inception_date,
        entry_age = entry_age
      ) %>%
      mutate(custom_value = config$multiplier * sum_insured)
  }
})
```

**Errors**:
- `"Invalid template name"` - Not a single character string
- `"Invalid factory function"` - Not a function

---

### `list_projection_templates()`

Returns names of all registered templates.

**Signature**:
```r
list_projection_templates()
```

**Returns**: Character vector of template names

**Examples**:
```r
list_projection_templates()
#> [1] "cqs"
```

---

### `get_projection_template()`

Retrieves a template factory function.

**Signature**:
```r
get_projection_template(name)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | character(1) | Template name |

**Returns**: Factory function

**Errors**:
- `"Unknown projection template"` - Template not registered

---

### `create_projection_fn()`

Creates a projection function from configuration.

**Signature**:
```r
create_projection_fn(config)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | list | Must contain `template` field, plus template-specific parameters |

**Returns**: Projection function `f(mp_one) → tibble`

**Examples**:
```r
config <- list(
  template = "cqs",
  name = "axa_cqs_death",
  mortality_table = qx_table,
  BE_fraction = 1,
  sum_insured_type = "fixed"
)

proj_fn <- create_projection_fn(config)
result <- proj_fn(model_point)
```

**Errors**:
- `"Invalid configuration"` - Not a list
- `"Missing template specification"` - No `template` field
- `"Unknown projection template"` - Template not registered

---

### `projection_config()`
Creates a structured projection configuration.

**Signature**:
```r
projection_config(template, name, ceding, product, risk, ...)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `template` | character | Template name (e.g., "cqs") |
| `name` | character | Unique configuration identifier |
| `ceding` | character | Ceding company identifier |
| `product` | character | Product type identifier |
| `risk` | character | Risk type identifier |
| `...` | - | Template-specific parameters |

**Returns**: `projection_config` (S3 class inheriting from `list`)

**Examples**:
```r
config <- projection_config(
  template = "cqs",
  name = "axa_cqs_death",
  ceding = "axa",
  product = "cqs",
  risk = "death",
  mortality_table = qx_2006,
  BE_fraction = 1,
  sum_insured_type = "fixed"
)
```

---

## Registry Builder API

**Source**: `R/registry_builder.R`

### `build_projection_registry()`

Batch-builds a registry from configuration list.

**Signature**:
```r
build_projection_registry(
  configs,
  dimensions = c("ceding", "product", "risk"),
  range_dimensions = NULL,
  .verbose = TRUE
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `configs` | named list | Required | List of `projection_config` or compatible lists |
| `dimensions` | character | `c("ceding", "product", "risk")` | Exact-match dimension names |
| `range_dimensions` | named list | `NULL` | Range dimensions: `list(uw_year = c("uw_year_min", "uw_year_max"))` |
| `.verbose` | logical | `TRUE` | Print progress messages |

**Returns**: `projection_registry` populated with all configs

**Examples**:
```r
configs <- list(
  axa_cqs_death = projection_config(...),
  axa_cqs_loe = projection_config(...),
  cf_cqs_death = projection_config(...)
)

# Basic usage
registry <- build_projection_registry(configs)

# Custom dimensions
registry <- build_projection_registry(
  configs,
  dimensions = c("ceding_company", "product_type", "risk_type")
)

# With range dimension
registry <- build_projection_registry(
  configs,
  dimensions = c("ceding", "product"),
  range_dimensions = list(uw_year = c("uw_year_min", "uw_year_max"))
)
```

**Errors**:
- `"Invalid configurations"` - Not a non-empty list
- `"All projections failed to register"` - Every config failed

---

### `validate_projection_configs()`

Validates configurations before building registry.

**Signature**:
```r
validate_projection_configs(
  configs,
  dimensions = c("ceding", "product", "risk")
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `configs` | named list | Required | Configurations to validate |
| `dimensions` | character | `c("ceding", "product", "risk")` | Required dimension fields |

**Returns**: `invisible(TRUE)` if valid

**Checks Performed**:
1. All configs are named
2. All configs have `template` field
3. All configs have required dimension fields
4. Template exists in registry
5. No duplicate dimension combinations

**Errors**:
- `"Configuration validation failed"` - With list of specific issues

**Examples**:
```r
# Validate before building
tryCatch({
  validate_projection_configs(configs)
  registry <- build_projection_registry(configs)
}, error = function(e) {
  message("Config validation failed: ", e$message)
})
```

---

### `load_projection_configs()`

Loads configurations from YAML file.

**Signature**:
```r
load_projection_configs(path, mortality_tables = list())
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | character | Required | Path to YAML file |
| `mortality_tables` | named list | `list()` | Tables to resolve by name |

**Returns**: Named list of configurations

**YAML Format**:
```yaml
axa_cqs_death:
  template: cqs
  ceding: axa
  product: cqs
  risk: death
  mortality_table: qx_2006    # Resolved from mortality_tables
  BE_fraction: 1
  sum_insured_type: fixed
```

**Examples**:
```r
mortality_tables <- list(
  qx_2006 = pricingr::qx %>% filter(year == 2006),
  qx_2020 = readRDS("tables/qx_2020.rds")
)

configs <- load_projection_configs(
  "configs/production.yaml",
  mortality_tables = mortality_tables
)
```

**Errors**:
- `"Configuration file not found"` - File doesn't exist
- `"Package 'yaml' is required"` - yaml package not installed
- `"Unknown mortality table reference"` - Table name not in `mortality_tables`

---

### `example_projection_configs()`

Returns example configurations for testing.

**Signature**:
```r
example_projection_configs(mortality_table = NULL)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mortality_table` | data.frame | `NULL` | Mortality table to use. If `NULL`, uses `pricingr::qx` filtered to year 2006. |

**Returns**: Named list with 4 example configs:
- `axa_cqs_death`
- `axa_cqs_loe`
- `cf_cqs_death`
- `cf_cqs_loe`

**Examples**:
```r
configs <- example_projection_configs()
registry <- build_projection_registry(configs)
```

---

## Template API

**Source**: `R/template_cqs.R`

### `make_cqs_projection()`

Factory function for CQS (Credit Protection Insurance) projections.

**Signature**:
```r
make_cqs_projection(config)
```

**Parameters** (via `config` list):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `mortality_table` | data.frame | Yes | Must have `current_age`, `gender`, `qx` columns |
| `BE_fraction` | numeric | Yes | Best estimate fraction (e.g., 1, 10) |
| `sum_insured_type` | character | Yes | `"fixed"` or `"outstanding_debt"` |
| `name` | character | No | Configuration name for traceability |

**Returns**: Projection function `f(mp_one) → tibble`

**Required Model Point Columns**:

| Column | Type | Description |
|--------|------|-------------|
| `duration_months` | integer | Projection duration |
| `inception_date` | Date | Policy start date |
| `entry_age` | integer | Age at inception |
| `gender` | character | "M" or "F" for mortality lookup |
| `BE` | numeric | Best estimate multiplier |
| `sum_insured` | numeric | For `sum_insured_type = "fixed"` |
| `monthly_loan_pay` | numeric | For `sum_insured_type = "outstanding_debt"` |
| `monthly_loan_interest` | numeric | For `sum_insured_type = "outstanding_debt"` |

**Output Columns Added**:

| Column | Description |
|--------|-------------|
| `qx_BE` | Best estimate adjusted mortality rate |
| `m_qx_BE` | Monthly mortality rate |
| `lx_boy` | Lives at beginning of year |
| `lx_eom` | Lives at end of month |
| `claim_num` | Number of claims |
| `claim_amount` | Claim amount |
| `outstanding_bom` | Outstanding debt at BOM (if applicable) |
| `outstanding_eom` | Outstanding debt at EOM (if applicable) |
| `projection_config` | Config name for traceability |

**Errors**:
- `"Missing required configuration fields"` - Missing `mortality_table`, `BE_fraction`, or `sum_insured_type`
- `"Invalid sum_insured_type"` - Not "fixed" or "outstanding_debt"

---

## S3 Classes

### `projection_registry`

**Inherits**: `tbl_df`, `tbl`, `data.frame`

**Columns**:
- Dimension columns (character for exact, numeric for range)
- `proj_fn` (list-column): Projection closures
- `proj_fn_name` (character): Function names
- `priority` (integer): Matching priority

**Attributes**:
- `col_meta` (list): Dimension metadata

### `projection_config`

**Inherits**: `list`

**Standard Fields**:
- `template` (character)
- `name` (character)
- `ceding` (character)
- `product` (character)
- `risk` (character)

**Template-Specific Fields**: Varies by template

---

## Error Catalog

### Registry Errors

| Error Message | Cause | Resolution |
|---------------|-------|------------|
| `"No columns specified"` | `create_projection_registry()` called with no args | Add dimension specifications |
| `"Invalid column specification"` | Invalid dimension spec | Use `"exact"` or `c("min", "max")` |
| `"Invalid registry"` | Wrong object type | Use `create_projection_registry()` |
| `"Invalid projection function"` | `proj_fn` not a function | Pass a function |
| `"Missing column in model point"` | MP missing dimension column | Add column or rename |
| `"No matching projection rule found"` | No rule matches MP | Add catch-all rule or specific rule |

### Factory Errors

| Error Message | Cause | Resolution |
|---------------|-------|------------|
| `"Invalid template name"` | Not single character | Pass single string |
| `"Invalid factory function"` | Not a function | Pass a function |
| `"Unknown projection template"` | Template not registered | Register template first |
| `"Invalid configuration"` | Config not a list | Pass a list |
| `"Missing template specification"` | No `template` field | Add `template` to config |

### Builder Errors

| Error Message | Cause | Resolution |
|---------------|-------|------------|
| `"Invalid configurations"` | Empty or non-list | Pass non-empty list |
| `"Configuration validation failed"` | Validation errors | Fix listed issues |
| `"All projections failed to register"` | Every config failed | Check error details |
| `"Configuration file not found"` | YAML file missing | Check path |
| `"Package 'yaml' is required"` | yaml not installed | `install.packages("yaml")` |
| `"Unknown mortality table reference"` | Table not in lookup | Add to `mortality_tables` |

### Template Errors (CQS)

| Error Message | Cause | Resolution |
|---------------|-------|------------|
| `"Missing required configuration fields"` | Missing required config | Add missing fields |
| `"Invalid sum_insured_type"` | Invalid value | Use "fixed" or "outstanding_debt" |

---

*API Reference v1.0.0 - pricingr Projection Registry System*
