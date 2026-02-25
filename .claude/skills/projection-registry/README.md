# Projection Registry Skill

A comprehensive skill for creating projection registry dispatchers in pricingr reinsurance modeling.

## What This Skill Does

Guides users through creating multi-dimensional projection routing systems that dispatch insurance cash flow projections based on:
- Ceding company
- Product type
- Risk category
- Underwriting year ranges

## Structure

```
projection-registry/
├── SKILL.md                          # Main skill instructions
├── README.md                         # This file
└── references/                       # Bundled documentation
    ├── registry-dispatcher.Rmd       # Vignette with examples
    ├── cheatsheet.md                 # Quick reference
    ├── architecture.md               # System design
    ├── api-reference.md              # Complete API docs
    └── implementation-guide.md       # Step-by-step guide
```

## Trigger Phrases

- "Create a projection registry"
- "Set up CQS projections"
- "Build a dispatcher"
- "Route projections by ceding/product/risk"
- "Configure projection matching"

## Key Features

- **Configuration-driven**: Define projections via configs, not code
- **Multi-dimensional routing**: Match on multiple dimensions (ceding, product, risk, time)
- **Priority-based selection**: Specific rules override general catch-all rules
- **Wildcard matching**: Use `NA` to match any value
- **Range matching**: Match numeric ranges (e.g., underwriting years)
- **Built-in CQS template**: Credit Protection Insurance projections ready to use

## Quick Example

```r
library(pricingr)

# 1. Create configs
configs <- list(
  axa_cqs_death = projection_config(
    template = "cqs",
    ceding = "axa",
    product = "cqs",
    risk = "death",
    mortality_table = qx_table,
    BE_fraction = 1,
    sum_insured_type = "fixed"
  )
)

# 2. Build registry & dispatcher
registry <- build_projection_registry(configs)
dispatcher <- create_registry_dispatcher(registry)

# 3. Run projections
results <- project_policies(portfolio, dispatcher)
```

## References

All reference documentation is bundled in the `references/` folder and accessible to Claude when this skill is activated.

## Version

1.0.0 - Initial release with CQS template support
