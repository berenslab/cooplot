from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .aggregate import GroupedPublications
from .build import _prepare_labels_and_groups, _titles_for_windows

_UNLABELED = "Unlabeled"


def _ensure_group_mapping(
    grouped: GroupedPublications | Dict[str, List[dict]],
) -> Dict[str, List[dict]]:
    if isinstance(grouped, GroupedPublications):
        return grouped.by_group
    return grouped


def _lastname(name: str) -> str:
    return (name or "").strip().split()[-1].lower()


def _publication_sort_key(record: dict) -> Tuple[int, str]:
    year = record.get("year")
    norm_title = (record.get("norm_title") or record.get("title") or "").lower()
    year_key = year if isinstance(year, int) else -1
    return (year_key, norm_title)


@dataclass(frozen=True)
class CrossGroupSummary:
    publications: List[dict]

    @property
    def count(self) -> int:
        return len(self.publications)



def cross_group_publications_by_window(
    publications_by_author: Dict[str, List[dict]],
    people: List[dict],
    windows: List[str],
    *,
    name_col: str = "name",
    group_col: Optional[str] = None,
) -> Dict[str, List[dict]]:
    """Return window-indexed records of cross-group titles using author data."""

    labels, group_map = _prepare_labels_and_groups(people, name_col, group_col)
    if not group_map:
        return {win: [] for win in windows}

    title_sets_by_window = _titles_for_windows(publications_by_author, labels, windows)
    results: Dict[str, List[dict]] = {}

    for win in windows:
        title_sets = title_sets_by_window[win]
        cross_titles: Dict[str, dict] = {}
        for label in labels:
            titles = title_sets[label]
            if not titles:
                continue
            group_label = group_map.get(label, "Unlabeled")
            for title in titles:
                entry = cross_titles.setdefault(
                    title,
                    {
                        "title": title,
                        "groups": set(),
                        "authors": {},
                    },
                )
                entry["groups"].add(group_label)
                authors_for_group = entry["authors"].setdefault(group_label, set())
                authors_for_group.add(label)

        formatted: List[dict] = []
        for entry in cross_titles.values():
            groups = entry["groups"]
            if len(groups) < 2:
                continue
            sorted_groups = sorted(groups, key=str.lower)
            formatted.append(
                {
                    "title": entry["title"],
                    "groups": sorted_groups,
                    "authors": {
                        g: sorted(entry["authors"].get(g, set()), key=_lastname)
                        for g in sorted_groups
                    },
                }
            )

        formatted.sort(key=lambda rec: rec["title"].lower())
        results[win] = formatted

    return results

def cross_group_publications(
    grouped_publications: GroupedPublications | Dict[str, List[dict]],
    *,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    include_missing_year: bool = False,
    include_unlabeled: bool = False,
    min_group_count: int = 2,
) -> List[dict]:
    """Identify publications that include authors from multiple groups.

    Parameters
    ----------
    grouped_publications
        Output of :func:`cooplot.aggregate.aggregate_publications` or a compatible
        mapping of ``{group: [publication, ...]}`` records.
    year_from, year_to
        Inclusive bounds for publication years. ``None`` disables the respective
        filter. When a bound is provided, publications missing the ``year`` field
        are excluded unless ``include_missing_year`` is ``True``.
    include_missing_year
        If set, publications lacking a year are retained even when a year bound is
        applied.
    include_unlabeled
        When ``False`` (default) the special ``"Unlabeled"`` bucket is ignored in
        the cross-group computation.
    min_group_count
        Minimum distinct groups required for a record to be returned.

    Returns
    -------
    List[dict]
        Each dict contains ``title``, ``norm_title``, ``year``, ``groups`` and an
        ``authors`` mapping keyed by group name.
    """

    grouped = _ensure_group_mapping(grouped_publications)

    combined: Dict[str, dict] = {}

    for group_label, records in grouped.items():
        if group_label == _UNLABELED and not include_unlabeled:
            continue
        for record in records:
            norm_title = (record.get("norm_title") or "").strip()
            if not norm_title:
                continue
            year = record.get("year")
            if isinstance(year, int):
                if year_from is not None and year < year_from:
                    continue
                if year_to is not None and year > year_to:
                    continue
                year_key: Optional[int] = year
            else:
                if (year_from is not None or year_to is not None) and not include_missing_year:
                    continue
                year_key = None

            entry = combined.setdefault(
                norm_title,
                {
                    "title": record.get("title") or norm_title,
                    "norm_title": norm_title,
                    "groups": set(),
                    "authors": {},
                    "year_counts": {},
                },
            )
            if entry["title"] == entry["norm_title"] and record.get("title"):
                entry["title"] = record["title"]
            entry["groups"].add(group_label)
            authors_for_group = entry["authors"].setdefault(group_label, set())
            for author in record.get("authors", []):
                authors_for_group.add(author)
            counts = entry["year_counts"]
            counts[year_key] = counts.get(year_key, 0) + 1

    results: List[dict] = []
    for entry in combined.values():
        if len(entry["groups"]) < min_group_count:
            continue
        counts = entry.pop("year_counts")
        chosen_year: Optional[int] = None
        if counts:
            int_counts = [(year, freq) for year, freq in counts.items() if isinstance(year, int)]
            if int_counts:
                int_counts.sort(key=lambda item: (-item[1], -item[0]))
                chosen_year = int_counts[0][0]
            elif None in counts:
                chosen_year = None
        groups_sorted = sorted(entry["groups"], key=str.lower)
        authors_sorted = {
            group: sorted(entry["authors"].get(group, []), key=_lastname)
            for group in groups_sorted
        }
        results.append(
            {
                "title": entry["title"],
                "norm_title": entry["norm_title"],
                "year": chosen_year,
                "groups": groups_sorted,
                "authors": authors_sorted,
            }
        )

    results.sort(key=_publication_sort_key)
    return results


def cross_group_summary(
    grouped_publications: GroupedPublications | Dict[str, List[dict]],
    **kwargs,
) -> CrossGroupSummary:
    """Convenience wrapper returning a :class:`CrossGroupSummary`."""

    publications = cross_group_publications(
        grouped_publications,
        **kwargs,
    )
    return CrossGroupSummary(publications=publications)
