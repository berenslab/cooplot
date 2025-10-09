# cooplot

Analysis of co-op between members and subgroups.

## Example Gallery

## Cluster of Excellence – Machine Learning for Science

```python
import cooplot
palette = {
    "Life Science": "#61859e",
    "Norms": "#e0aa41",
    "Human Science": "#5fb4d0",
    "ML": "#bc3b2f",
    "Physical Science": "#608dd2",
}

_, people = cooplot.load_csv("excelclust.csv", delimiter=";")
pubs = cooplot.scrape(
    people,
    name_col="name",
    scholar_col="scholar_id",
    semantic_col="semantic_id",
    cache_dir=".cache/excelclust",
)
mats = cooplot.build(pubs, windows=["2014-2018", "2019-2023"], name_col="name", group_col="group")
fig = cooplot.show(mats, group_col="group", style="circle", heatmap_counts=True, palette=palette)
fig.savefig("../.github/excelclust.png", dpi=300)
```

![](.github/excelclust.png)

## AG Berens

```python
import cooplot
header, people = cooplot.load_csv("agberens.csv", delimiter=";")
pubs = cooplot.scrape(
    people,
    name_col="name",
    scholar_col="scholar_id",
    semantic_col="semantic_id",
    cache_dir=".cache/hai",
    drop_subtitle=False,
    fallback_semantic_if_empty=True,  # try Semantic if GS had 0 pubs
)


mats = cooplot.build(pubs, windows=["2016-2025"], name_col="name", group_col="group")
fig = cooplot.show(mats, group_col="group", style="both", heatmap_counts=True, figsize=(18,10))
```

![](.github/agberens.png)


## Installation

```bash
uv pip install -e ".[dev]"
```

## Usage

See the [example notebook](https://github.com/berenslab/cooplot/blob/main/notebooks/agberens.ipynb) for a complete usage example.