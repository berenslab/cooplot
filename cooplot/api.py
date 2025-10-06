# cooplot/api.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .aggregate import GroupedPublications, aggregate_publications
from .build import build_matrices
from .io import read_rows_with_header
from .metrics import (
    cross_group_publications as metrics_cross_group_publications,
)
from .metrics import (
    cross_group_publications_by_window,
)
from .scrape import scrape_all
from .viz import plot_panels

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
) -> Dict[str, List[dict]]:
    return scrape_all(
        csv_rows,
        name_col=name_col,
        scholar_col=scholar_col,
        semantic_col=semantic_col,
        cache_dir=cache_dir,
        drop_subtitle=drop_subtitle,
        fallback_semantic_if_empty=fallback_semantic_if_empty,
    )


def aggregate(
    publications_by_author: Dict[str, List[dict]],
    people: Iterable[dict],
    *,
    name_col: str = "name",
    group_col: str = "group",
    cache_dir: Path | str = ".cache/cooplot/groups",
    include_unlabeled: bool = True,
    save_json: bool = True,
    ensure_ascii: bool = False,
) -> GroupedPublications:
    """Group and deduplicate publications at the group level."""

    return aggregate_publications(
        publications_by_author,
        people,
        name_col=name_col,
        group_col=group_col,
        cache_dir=cache_dir,
        include_unlabeled=include_unlabeled,
        save_json=save_json,
        ensure_ascii=ensure_ascii,
    )


def build(
    pubs: Dict[str, List[dict]] | GroupedPublications,
    windows: List[str],
    *,
    people: List[dict] | None = None,
    name_col: str = "name",
    group_col: str | None = None,
):
    """Build co-authorship matrices from author or group level data.

    When ``pubs`` is a :class:`GroupedPublications` the ``people`` list can be
    omitted. Otherwise supply the author records so groups can be resolved
    alongside the raw publications.
    """

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
    style="circle",
    vmax=None,
    palette=None,
    cap_weights=None,
    counts_label="Shared coauthorships",
    legend_counts=True,
    legend_groups=True,
    heatmap_counts=False,
    figsize=None,
):
    return plot_panels(
        mats,
        out_path=None,
        group_col=group_col,
        palette=palette,
        vmax=vmax,
        style=style,
        show=True,
        return_fig=True,
        cap_weights=cap_weights,
        counts_label=counts_label,
        legend_counts=legend_counts,
        legend_groups=legend_groups,
        heatmap_counts=heatmap_counts,
        figsize=figsize,
    )


def run_inline(
    csv_path: str | Path,
    *,
    windows: List[str],
    delimiter: str = ";",
    name_col="name",
    scholar_col="scholar_id",
    semantic_col="semantic_id",
    group_col: str | None = None,
    style: str = "circle",
    drop_subtitle: bool = False,
    fallback_semantic_if_empty: bool = False,
    cache_dir: str | Path = ".cache/cooplot",
    heatmap_counts: bool = False,
):
    header, people = load_csv(csv_path, delimiter=delimiter)
    pubs = scrape(
        people,
        name_col=name_col,
        scholar_col=scholar_col,
        semantic_col=semantic_col,
        cache_dir=cache_dir,
        drop_subtitle=drop_subtitle,
        fallback_semantic_if_empty=fallback_semantic_if_empty,
    )
    mats = build(
        pubs,
        windows,
        people=people,
        name_col=name_col,
        group_col=group_col,
    )
    return show(
        mats,
        group_col=group_col,
        style=style,
        heatmap_counts=heatmap_counts,
    )


def cross_group_coauthored(
    pubs_by_author: Dict[str, List[dict]],
    people: List[dict],
    windows: List[str],
    *,
    name_col: str = "name",
    group_col: str | None = None,
):
    """Return per-window metadata about titles with authors from multiple groups."""

    return cross_group_publications_by_window(
        pubs_by_author,
        people,
        windows,
        name_col=name_col,
        group_col=group_col,
    )


def cross_group_publications_grouped(
    grouped: GroupedInput,
    people: Optional[Iterable[dict]] = None,
    *,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    include_missing_year: bool = False,
    include_unlabeled: bool = False,
    min_group_count: int = 2,
) -> List[dict]:
    """Cross-group publication records based on grouped data."""

    return metrics_cross_group_publications(
        grouped,
        people,
        year_from=year_from,
        year_to=year_to,
        include_missing_year=include_missing_year,
        include_unlabeled=include_unlabeled,
        min_group_count=min_group_count,
    )


def cross_group_publications_summary(
    grouped: GroupedInput,
    people: Optional[Iterable[dict]] = None,
    *,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    include_missing_year: bool = False,
    include_unlabeled: bool = False,
    min_group_count: int = 2,
) -> Dict[str, object]:
    """Summarize cross-group publications with optional year filters."""

    publications = cross_group_publications_grouped(
        grouped,
        people,
        year_from=year_from,
        year_to=year_to,
        include_missing_year=include_missing_year,
        include_unlabeled=include_unlabeled,
        min_group_count=min_group_count,
    )
    summary: Dict[str, object] = {
        "count": len(publications),
        "publications": publications,
    }
    if year_from is not None:
        summary["year_from"] = year_from
    if year_to is not None:
        summary["year_to"] = year_to
    return summary
