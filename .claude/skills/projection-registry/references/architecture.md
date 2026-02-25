# Projection Registry System - Architecture Reference

> **Target Audience**: R Software Developers implementing reinsurance portfolio modeling
> **Version**: 1.0.0
> **Last Updated**: 2026-02-19

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Design Patterns](#design-patterns)
6. [File Structure](#file-structure)
7. [Dependencies](#dependencies)

---

## System Overview

The Projection Registry System implements a **rule-based dispatch pattern** for routing model points to appropriate projection functions based on multi-dimensional criteria. This architecture addresses the common reinsurance challenge of managing projections across multiple ceding companies, products, risks, and underwriting periods.

### Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| Tibble-based registry | Consistent with pricingr's `set_assumption()` lookup pattern; easy to inspect, export, audit |
| Closure-based dispatchers | Captures registry state at creation time; immutable dispatch behavior |
| Factory pattern for templates | Separates projection logic (templates) from business rules (configurations) |
| Priority-based matching | Enables specific rules to override general catch-all rules |
| Environment-based template registry | O(1) lookup; mutable at package load time only |

### System Boundaries

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRICINGR PACKAGE                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    PROJECTION REGISTRY SUBSYSTEM                        ││
│  │                                                                         ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 ││
│  │  │   Template   │  │   Factory    │  │   Registry   │                 ││
│  │  │   Registry   │──│   System     │──│   Builder    │                 ││
│  │  │  (zzz.R)     │  │              │  │              │                 ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 ││
│  │         │                 │                 │                          ││
│  │         ▼                 ▼                 ▼                          ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 ││
│  │  │   Template   │  │  Projection  │  │  Projection  │                 ││
│  │  │ Definitions  │  │   Configs    │  │   Registry   │                 ││
│  │  │(template_*.R)│  │              │  │              │                 ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 ││
│  │                                             │                          ││
│  │                                             ▼                          ││
│  │                                      ┌──────────────┐                 ││
│  │                                      │  Dispatcher  │                 ││
│  │                                      │   Closure    │                 ││
│  │                                      └──────────────┘                 ││
│  │                                             │                          ││
│  └─────────────────────────────────────────────┼───────────────────────────┘│
│                                                │                            │
│  ┌─────────────────────────────────────────────┼───────────────────────────┐│
│  │                    CORE PROJECTION ENGINE   │                           ││
│  │                                             ▼                           ││
│  │  project_policies() ──► dispatcher(mp) ──► proj_fn(mp) ──► results     ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Diagram

### Component Interaction Flow

```
                    ┌─────────────────────────────────┐
                    │         YAML Config             │
                    │    (optional external file)     │
                    └─────────────┬───────────────────┘
                                  │ load_projection_configs()
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION LAYER                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  projection_config()                                           │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │ {                                                        │  │  │
│  │  │   template: "cqs",                                       │  │  │
│  │  │   name: "axa_cqs_death",                                 │  │  │
│  │  │   ceding: "axa",                                         │  │  │
│  │  │   product: "cqs",                                        │  │  │
│  │  │   risk: "death",                                         │  │  │
│  │  │   mortality_table: <tibble>,                             │  │  │
│  │  │   BE_fraction: 1,                                        │  │  │
│  │  │   sum_insured_type: "fixed"                              │  │  │
│  │  │ }                                                        │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ build_projection_registry()
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FACTORY LAYER                                   │
│  ┌────────────────────────┐    ┌────────────────────────────────┐  │
│  │   Template Registry    │    │     create_projection_fn()     │  │
│  │   (PROJECTION_TEMPLATES│◄───│                                │  │
│  │   environment)         │    │  config ──► factory_fn(config) │  │
│  │                        │    │          ──► proj_fn closure   │  │
│  │  "cqs" ──► make_cqs_*  │    │                                │  │
│  │  "term" ──► make_term_*│    └────────────────────────────────┘  │
│  └────────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ register_projection()
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      REGISTRY LAYER                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  projection_registry (S3 class inheriting tibble)             │  │
│  │  ┌─────────┬─────────┬──────┬──────────┬───────────┬────────┐│  │
│  │  │ ceding  │ product │ risk │ proj_fn  │ proj_fn_  │priority││  │
│  │  │         │         │      │ (list)   │ name      │        ││  │
│  │  ├─────────┼─────────┼──────┼──────────┼───────────┼────────┤│  │
│  │  │ "axa"   │ "cqs"   │"death│ <closure>│axa_cqs_   │  100   ││  │
│  │  │ "axa"   │ "cqs"   │"loe" │ <closure>│axa_cqs_loe│  100   ││  │
│  │  │ "cf"    │ "cqs"   │"death│ <closure>│cf_cqs_    │  100   ││  │
│  │  │ NA      │ NA      │ NA   │ <closure>│default    │   70   ││  │
│  │  └─────────┴─────────┴──────┴──────────┴───────────┴────────┘│  │
│  │                                                               │  │
│  │  attr(registry, "col_meta") ──► dimension metadata            │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ create_registry_dispatcher()
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DISPATCH LAYER                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  dispatcher = function(mp) {                                  │  │
│  │    match <- find_matching_projection(registry, mp)            │  │
│  │    match$proj_fn(mp)                                          │  │
│  │  }                                                            │  │
│  │                                                               │  │
│  │  Closure captures: registry (immutable snapshot)              │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ project_policies(mp_df, dispatcher)
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EXECUTION LAYER                                 │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  for each row in mp_df:                                       │  │
│  │    1. dispatcher(mp_row)                                      │  │
│  │    2. find_matching_projection() evaluates all rules          │  │
│  │    3. Highest priority matching rule selected                 │  │
│  │    4. proj_fn(mp_row) executed                                │  │
│  │    5. Results accumulated                                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Template Registry (`R/projection_factory.R`)

**Purpose**: Stores factory functions that create projection closures from configuration.

**Implementation**:
```r
PROJECTION_TEMPLATES <- new.env(parent = emptyenv())
```

**Key Characteristics**:
- Uses R environment for O(1) lookup by template name
- `parent = emptyenv()` prevents accidental parent scope leakage
- Populated at package load time via `.onLoad()` in `zzz.R`
- Immutable after package initialization (by convention)

**API Surface**:
| Function | Visibility | Purpose |
|----------|------------|---------|
| `register_projection_template()` | Exported | Add template to registry |
| `list_projection_templates()` | Exported | Enumerate available templates |
| `get_projection_template()` | Internal/Exported | Retrieve factory function |

### 2. Projection Factory (`R/projection_factory.R`)

**Purpose**: Creates concrete projection functions by combining templates with configurations.

**Key Function**: `create_projection_fn(config)`

**Mechanics**:
1. Extracts `template` field from config
2. Retrieves factory function from `PROJECTION_TEMPLATES`
3. Invokes factory with config to produce closure
4. Closure captures config parameters in its environment

**Closure Structure**:
```r
# Factory creates:
function(mp_one) {
  # config captured in enclosing environment
  # config$mortality_table, config$BE_fraction, etc. available
  mp_one %>%
    setup_projection(...) %>%
    set_assumption(table = config$mortality_table, ...) %>%
    ...
}
```

### 3. Projection Registry (`R/registry_dispatcher.R`)

**Purpose**: Stores rules mapping dimension values to projection functions.

**Data Structure**: S3 class `projection_registry` inheriting from `tbl_df`

**Schema**:
```
registry[i, ] = {
  <dimension_cols>: character/numeric values (NA = wildcard),
  proj_fn: list-column containing closure,
  proj_fn_name: character identifier,
  priority: integer (higher = more specific)
}
```

**Metadata Attribute**:
```r
attr(registry, "col_meta") = list(
  <dim_name> = list(
    name = "<dim_name>",
    type = "exact" | "range",
    cols = c("<col1>", "<col2>") | "<col>"
  ),
  ...
)
```

### 4. Dispatcher (`R/registry_dispatcher.R`)

**Purpose**: Routes model points to projection functions based on registry rules.

**Created by**: `create_registry_dispatcher(registry)`

**Matching Algorithm** (in `find_matching_projection()`):
```
1. Initialize matches = TRUE for all registry rows
2. For each dimension:
   a. If exact: matches &= (is.na(reg_val) | reg_val == mp_val)
   b. If range: matches &= (is.na(min) | mp_val >= min) &
                          (is.na(max) | mp_val <= max)
3. Filter to matched rows
4. Select row with max(priority)
5. Return proj_fn from selected row
```

### 5. Registry Builder (`R/registry_builder.R`)

**Purpose**: Batch construction of registries from configuration lists.

**Key Functions**:
| Function | Purpose |
|----------|---------|
| `build_projection_registry()` | Iterates configs, creates proj_fns, registers rules |
| `validate_projection_configs()` | Pre-flight validation of config structure |
| `load_projection_configs()` | Deserializes YAML to config list |
| `example_projection_configs()` | Returns sample configs for testing |

---

## Data Flow

### Initialization Flow (Package Load)

```
.onLoad()
    │
    └──► register_projection_template("cqs", make_cqs_projection)
              │
              └──► PROJECTION_TEMPLATES[["cqs"]] <- make_cqs_projection
```

### Configuration Flow (User Code)

```
projection_config(template="cqs", ceding="axa", ...)
    │
    └──► list with class "projection_config"
              │
              └──► configs <- list(axa_cqs_death = <config>, ...)
                        │
                        └──► build_projection_registry(configs)
                                  │
                                  ├──► create_projection_registry(dimensions)
                                  │         │
                                  │         └──► empty tibble with col_meta
                                  │
                                  └──► for each config:
                                            │
                                            ├──► create_projection_fn(config)
                                            │         │
                                            │         ├──► get_projection_template("cqs")
                                            │         │
                                            │         └──► make_cqs_projection(config)
                                            │                   │
                                            │                   └──► closure capturing config
                                            │
                                            └──► register_projection(registry, ...)
                                                      │
                                                      └──► bind_rows(registry, new_rule)
```

### Execution Flow (Projection Run)

```
project_policies(mp_df, dispatcher)
    │
    └──► for each mp_row in mp_df:
              │
              └──► dispatcher(mp_row)
                        │
                        └──► find_matching_projection(registry, mp_row)
                                  │
                                  ├──► evaluate all rules against mp_row
                                  │
                                  ├──► filter to matching rules
                                  │
                                  └──► select highest priority
                                            │
                                            └──► proj_fn(mp_row)
                                                      │
                                                      └──► expanded projection tibble
```

---

## Design Patterns

### 1. Factory Pattern

**Where**: `projection_factory.R`

**Implementation**: Template functions (`make_cqs_projection`) are factories that take config and return projection closures.

```r
# Factory
make_cqs_projection <- function(config) {
  force(config)  # Capture config in closure environment

  # Product: projection function
  function(mp_one) {
    # Uses config$mortality_table, config$BE_fraction, etc.
    ...
  }
}
```

### 2. Registry Pattern

**Where**: `registry_dispatcher.R`

**Implementation**: Central tibble stores mappings from criteria to handlers (projection functions).

### 3. Strategy Pattern

**Where**: Dispatch mechanism

**Implementation**: Different projection strategies (proj_fns) are selected at runtime based on model point attributes.

### 4. Closure Pattern

**Where**: Dispatcher and projection functions

**Implementation**:
- Dispatcher captures registry at creation time
- Projection functions capture config at creation time
- Both are immune to subsequent modifications of source data

---

## File Structure

```
R/
├── zzz.R                    # Package initialization
│                            # - .onLoad() registers built-in templates
│
├── projection_factory.R     # Template registry and factory
│                            # - PROJECTION_TEMPLATES environment
│                            # - register_projection_template()
│                            # - list_projection_templates()
│                            # - get_projection_template()
│                            # - create_projection_fn()
│                            # - projection_config()
│
├── registry_dispatcher.R    # Projection registry and dispatch
│                            # - create_projection_registry()
│                            # - register_projection()
│                            # - find_matching_projection()
│                            # - create_registry_dispatcher()
│                            # - print.projection_registry()
│
├── registry_builder.R       # Batch registry construction
│                            # - build_projection_registry()
│                            # - validate_projection_configs()
│                            # - load_projection_configs()
│                            # - example_projection_configs()
│
└── template_cqs.R           # CQS product template
                             # - make_cqs_projection()
```

---

## Dependencies

### Internal Dependencies

| Component | Depends On |
|-----------|------------|
| `zzz.R` | `projection_factory.R`, `template_cqs.R` |
| `registry_builder.R` | `projection_factory.R`, `registry_dispatcher.R` |
| `create_projection_fn()` | `PROJECTION_TEMPLATES` |
| `build_projection_registry()` | `create_projection_fn()`, `register_projection()` |

### External Package Dependencies

| Package | Used By | Purpose |
|---------|---------|---------|
| `tibble` | Registry storage | Tibble data structure |
| `dplyr` | Registry operations, templates | `bind_rows()`, `filter()`, pipe |
| `cli` | Error handling | Formatted error messages |
| `yaml` | `load_projection_configs()` | YAML parsing (optional) |

### pricingr Core Dependencies

Templates depend on pricingr core projection functions:
- `setup_projection()`
- `set_assumption()`
- `apply_BE()`
- `apply_monthly_rate()`
- `proj_lx()`
- `proj_num()`
- `proj_claim()`
- `proj_outstanding_debt()`

---

## Performance Considerations

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Template lookup | O(1) | Environment-based |
| Rule matching | O(n × d) | n = rules, d = dimensions |
| Priority selection | O(n) | Linear scan of matches |
| Projection execution | O(m) | m = months in projection |

### Memory Considerations

- Each projection function closure captures its config (~KB per config)
- Registry tibble grows linearly with rule count
- Dispatcher closure captures entire registry snapshot

### Optimization Tips

1. **Reduce rule count**: Use wildcards (NA) for catch-all rules instead of enumerating
2. **Order dimensions**: Put most discriminating dimension first in `create_projection_registry()`
3. **Pre-filter portfolio**: If running subset, filter before `project_policies()`

---

*Document generated for pricingr projection registry system v1.0.0*
