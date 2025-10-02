# cooplot/api.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .build import build_matrices
from .io import read_rows_with_header
from .scrape import scrape_all
from .viz import plot_panels


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


def build(
    pubs_by_author: Dict[str, List[dict]],
    people: List[dict],
    windows: List[str],
    *,
    name_col="name",
):
    return build_matrices(pubs_by_author, people, windows, name_col=name_col)


def show(
    mats,
    people,
    *,
    group_col=None,
    style="circle",
    vmax=None,
    palette=None,
    cap_weights=None,
    counts_label="Shared coauthorships",
    legend_counts=True,
    legend_groups=True,
):
    return plot_panels(
        mats,
        out_path=None,
        people=people,
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
    mats = build(pubs, people, windows, name_col=name_col)
    return show(mats, people, group_col=group_col, style=style)
