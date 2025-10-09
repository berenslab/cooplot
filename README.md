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

You can also generate a list of reference highlighting the cross-group collaboration:

```python
groups = cooplot.aggregate(pubs, name_col="name", cache_dir=".cache/agberens-group")
_ = cooplot.cross_group_publications(groups, year_from=2016, year_to=2025, out_path=".cache/agberens-coop.csv", enrich_crossref=True)
cooplot.cross_group_report(".cache/agberens-coop.csv", out_path=".cache/agberens-coop-ref.txt", verbose=True)
```

which will give you 

```
...

Schmors, L., Kotkat, A. H., Bauer, Y., Huang, Z., Crombie, D., Meyerolbersleben, L. S., Sokoloski, S., Berens, P., & Busse, L. (2025). Effects of corticothalamic feedback depend on visual responsiveness and stimulus type. IScience, 28(6), 112481. https://doi.org/10.1016/j.isci.2025.112481
Collaboration: All (Philipp Berens, Ziwei Huang) and Neural (Lisa Schmors, Sacha Sokoloski).

Gervelmeyer, J., Müller, S., Huang, Z., & Berens, P. (2025). Fundus Image Toolbox: A Python package for fundus image processing. Journal of Open Source Software, 10(108), 7101. https://doi.org/10.21105/joss.07101
Collaboration: All (Philipp Berens, Ziwei Huang) and MedML (Julius Gervelmeyer, Sarah Müller).

Schmidt, G., Heidrich, H., Berens, P., & Müller, S. (2025). Learning Disease State from Noisy Ordinal Disease Progression Labels. Medical Image Computing and Computer Assisted Intervention – MICCAI 2025, 284–293. https://doi.org/10.1007/978-3-032-04971-1_27
Collaboration: All (Philipp Berens) and MedML (Sarah Müller).

Ofosu Mensah, S., Djoumessi, K., & Berens, P. (2025). Prototype-Guided and Lightweight Adapters for Inherent Interpretation and Generalisation in Federated Learning. Medical Image Computing and Computer Assisted Intervention – MICCAI 2025, 464–473. https://doi.org/10.1007/978-3-032-04981-0_44
Collaboration: All (Philipp Berens) and MedML (Kerol Djoumessi, Samuel Ofosu Mensah).

Oesterle, J., Ran, Y., Stahr, P., Kerr, J. N. D., Schubert, T., Berens, P., & Euler, T. (2025). Task-specific regional circuit adaptations in distinct mouse retinal ganglion cells. Science Advances, 11(17). https://doi.org/10.1126/sciadv.adp7075
Collaboration: All (Philipp Berens) and Neural (Jonathan Oesterle).
```


## Installation

```bash
uv pip install cooplot
```

or

```bash
git clone git@github.com:berenslab/cooplot.git
cd cooplot
uv pip install -e ".[dev]"
```

## Usage

See the [example notebook](https://github.com/berenslab/cooplot/blob/main/notebooks/agberens.ipynb) for a complete usage example.