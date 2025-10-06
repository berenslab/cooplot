# cooplot

A new Python project.

## Installation

```bash
source .venv/bin/activate
uv pip install ".[dev]"
```

## Workflow

The library now separates scraping, aggregation, metrics, and visualization so
reporting queries can reuse the same data.

```python
from cooplot import (
    aggregate,
    build,
    cross_group_publications_summary,
    load_csv,
    scrape,
)

header, people = load_csv("people.csv")
publications = scrape(people)

# Merge publications by team and write per-team JSON snapshots
grouped = aggregate(publications, group_col="team")
summary = cross_group_publications_summary(grouped, year_from=2020)
print(summary["count"], "multi-team papers since 2020")

mats = build(grouped, ["2020-2024"])
```

When manually constructing author-level data, wrap it with `publications_from_data(...)` so `build` receives the accompanying metadata automatically.

Pass the matrices to `cooplot.viz.show` (or directly to
`cooplot.viz.plot_panels`) to render the visualizations.
