# cooplot/api.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from .aggregate import GroupedPublications, aggregate_publications
from .build import build_matrices
from .io import read_rows_with_header
from .metrics import cross_group_publications, cross_group_report as _cross_group_report
from .scrape import Publications, scrape_all
from .viz import plot_panels

PublicationInput = Publications | Dict[str, List[dict]]
GroupedInput = GroupedPublications | Dict[str, List[dict]]


def load_csv(csv_path: str | Path, delimiter: str = ";"):
    header, rows = read_rows_with_header(csv_path, delimiter=delimiter)
    return header, rows


def scrape(
    csv_rows: List[dict],
    *,
    name_col="name",
    scholar_col="scholar_id",
    semantic_col="semantic_id",
    cache_dir=".cache/cooplot",
    drop_subtitle=False,
    fallback_semantic_if_empty=False,
    group_col: str | None = None,
) -> Publications:
    return scrape_all(
        csv_rows,
        name_col=name_col,
        scholar_col=scholar_col,
        semantic_col=semantic_col,
        cache_dir=cache_dir,
        drop_subtitle=drop_subtitle,
        fallback_semantic_if_empty=fallback_semantic_if_empty,
        group_col=group_col,
    )


def publications_from_data(
    publications_by_author: Dict[str, List[dict]] | Iterable[tuple[str, List[dict]]],
    people: Iterable[dict],
    *,
    name_col: str = "name",
    group_col: str | None = None,
) -> Publications:
    """Wrap raw author publications with their metadata for downstream use."""

    return Publications.from_data(
        publications_by_author,
        people,
        name_col=name_col,
        group_col=group_col,
    )


def aggregate(
    publications: PublicationInput,
    people: Iterable[dict] | None = None,
    *,
    name_col: str = "name",
    group_col: str = "group",
    cache_dir: Path | str = ".cache/cooplot/groups",
    include_unlabeled: bool = True,
    save_json: bool = True,
    ensure_ascii: bool = False,
) -> GroupedPublications:
    """Group and deduplicate publications at the group level."""

    if isinstance(publications, Publications):
        pub_data = publications
        if people is None:
            people = pub_data.people_rows()
        publications = pub_data.mapping()
        if name_col == "name":
            name_col = pub_data.resolve_name_column(name_col)
        resolved_group = pub_data.resolve_group_column(group_col)
        if resolved_group is not None:
            group_col = resolved_group
    if people is None:
        raise ValueError("people list is required when aggregating publications")

    return aggregate_publications(
        publications,
        people,
        name_col=name_col,
        group_col=group_col,
        cache_dir=cache_dir,
        include_unlabeled=include_unlabeled,
        save_json=save_json,
        ensure_ascii=ensure_ascii,
    )


def build(
    pubs: PublicationInput | GroupedPublications,
    windows: List[str],
    *,
    people: List[dict] | None = None,
    name_col: str = "name",
    group_col: str | None = None,
):
    """Build co-authorship matrices from author or group level data.

    When ``pubs`` is a :class:`Publications` or :class:`GroupedPublications`
    instance the people metadata is already bundled and does not need to be passed
    explicitly. Supplying bare mappings still requires the corresponding people
    list so group assignments can be resolved.
    """

    if isinstance(pubs, Publications):
        pub_data = pubs
        if people is None:
            people = pub_data.people_rows()
        pubs = pub_data.mapping()
        if name_col == "name":
            name_col = pub_data.resolve_name_column(name_col)
        if group_col is None:
            group_col = pub_data.resolve_group_column(group_col)
    if isinstance(pubs, GroupedPublications):
        grouped = pubs
        if people is None:
            people = grouped.to_author_list(name_col)
        pubs = grouped.by_group
        group_col = group_col or "group"
    if people is None:
        raise ValueError("people list is required when building matrices")

    return build_matrices(
        pubs,
        people,
        windows,
        name_col=name_col,
        group_col=group_col,
    )


def show(
    mats,
    *,
    group_col=None,
    style=None,
    vmax=None,
    palette=None,
    cap_weights=None,
    counts_label="Shared coauthorships",
    legend_counts=True,
    legend_groups=True,
    heatmap_counts=False,
    figsize=None,
):
    if style is None:
        try:
            win_count = len(mats)
        except TypeError:
            win_count = len(list(mats.keys()))
        if win_count == 1:
            resolved_style = "both"
        else:
            resolved_style = "circle"
    else:
        resolved_style = style

    return plot_panels(
        mats,
        out_path=None,
        group_col=group_col,
        palette=palette,
        vmax=vmax,
        style=resolved_style,
        show=True,
        return_fig=True,
        cap_weights=cap_weights,
        counts_label=counts_label,
        legend_counts=legend_counts,
        legend_groups=legend_groups,
        heatmap_counts=heatmap_counts,
        figsize=figsize,
    )


def cross_group_report(*args, **kwargs):
    """Wrapper for :func:`cooplot.metrics.cross_group_report` for convenience."""
    return _cross_group_report(*args, **kwargs)
