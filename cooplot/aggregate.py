from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_CACHE_DIR = Path(".cache/cooplot/groups")
_UNLABELED = "Unlabeled"
_slug_pattern = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class GroupedPublications:
    """Container for group-level publication data."""

    by_group: Dict[str, List[dict]]
    paths: Dict[str, Path]

    def sorted_groups(self) -> List[str]:
        return sorted(self.by_group.keys(), key=str.lower)

    def to_author_list(self, name_field: str = "name") -> List[dict]:
        """Return a minimal people list that can be fed into ``build``."""

        records: List[dict] = []
        for group in self.sorted_groups():
            records.append({name_field: group, "group": group})
        return records

    def exclude_groups(self, groups: Iterable[str]) -> "GroupedPublications":
        """Return a new instance without the specified groups."""

        to_remove = {str(group).strip() for group in groups if group is not None}
        to_remove.discard("")
        if not to_remove:
            return self

        filtered_by_group = {
            group: pubs
            for group, pubs in self.by_group.items()
            if group not in to_remove
        }
        filtered_paths = {
            group: path for group, path in self.paths.items() if group not in to_remove
        }
        return GroupedPublications(by_group=filtered_by_group, paths=filtered_paths)


def _lastname(name: str) -> str:
    return (name or "").strip().split()[-1].lower()


def _normalize_group(value: Optional[str]) -> str:
    if value is None:
        return _UNLABELED
    group = str(value).strip()
    return group if group else _UNLABELED


def _slugify(label: str) -> str:
    slug = _slug_pattern.sub("_", label.strip())
    slug = slug.strip("._")
    return slug or "group"


def _publication_key(publication: dict) -> Optional[Tuple[str, Optional[int]]]:
    norm_title = (publication.get("norm_title") or "").strip()
    if not norm_title:
        return None
    year = publication.get("year")
    if isinstance(year, int):
        return norm_title, year
    return norm_title, None


def _format_record(record: dict) -> dict:
    authors = sorted(record["authors"], key=_lastname)
    return {
        "title": record["title"],
        "norm_title": record["norm_title"],
        "year": record["year"],
        "authors": authors,
    }


def _publication_sort_key(record: dict) -> Tuple[int, str]:
    year = record.get("year")
    norm_title = (record.get("norm_title") or record.get("title") or "").lower()
    year_key = year if isinstance(year, int) else -1
    return (year_key, norm_title)


def aggregate_publications(
    publications_by_author: Dict[str, List[dict]],
    people: Iterable[dict],
    *,
    name_col: str = "name",
    group_col: str = "group",
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    include_unlabeled: bool = True,
    save_json: bool = True,
    ensure_ascii: bool = False,
) -> GroupedPublications:
    """Group publications by ``group_col`` and deduplicate by normalized title.

    Parameters
    ----------
    publications_by_author
        Mapping of author name to list of publication dicts, as returned by
        :func:`cooplot.scrape.scrape_all`.
    people
        Iterable of records describing each author. ``group_col`` is used to map
        authors onto the aggregation key.
    name_col
        Field name within ``people`` entries identifying each author.
    group_col
        Field name used to determine group membership.
    cache_dir
        Directory where group-level JSON files will be stored if ``save_json`` is
        ``True``. The directory is created if needed.
    include_unlabeled
        Whether authors without a ``group_col`` value should be collected under an
        ``"Unlabeled"`` bucket.
    save_json
        When ``True`` the grouped publication lists are written to individual
        JSON files under ``cache_dir``.
    ensure_ascii
        Passed through to :func:`json.dumps` so callers can enforce ASCII-only
        output if desired.

    Returns
    -------
    GroupedPublications
        Dataclass containing the grouped publication mapping and the optional
        cache file paths (empty when ``save_json`` is ``False``).
    """

    name_to_group: Dict[str, str] = {}
    for person in people:
        name_value = person.get(name_col)
        if not isinstance(name_value, str) or not name_value.strip():
            continue
        group_value = _normalize_group(person.get(group_col))
        name_to_group[name_value] = group_value

    grouped: Dict[str, Dict[Tuple[str, Optional[int]], dict]] = {}

    for author, publications in publications_by_author.items():
        group_label = name_to_group.get(author, _UNLABELED)
        if group_label == _UNLABELED and not include_unlabeled:
            continue

        bucket = grouped.setdefault(group_label, {})
        for publication in publications:
            key = _publication_key(publication)
            if key is None:
                continue
            norm_title, year = key
            title = publication.get("title") or norm_title
            entry = bucket.setdefault(
                key,
                {
                    "title": title,
                    "norm_title": norm_title,
                    "year": year,
                    "authors": set(),
                },
            )
            if entry["title"] == entry["norm_title"] and publication.get("title"):
                entry["title"] = publication["title"]
            entry["authors"].add(author)

    grouped_lists: Dict[str, List[dict]] = {}
    for group_label, records in grouped.items():
        formatted_records = [_format_record(rec) for rec in records.values()]
        formatted_records.sort(key=_publication_sort_key)
        grouped_lists[group_label] = formatted_records

    paths: Dict[str, Path] = {}
    if save_json:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        for group_label, records in grouped_lists.items():
            filename = f"{_slugify(group_label)}.json"
            out_path = cache_path / filename
            out_path.write_text(
                json.dumps(records, ensure_ascii=ensure_ascii, indent=2),
                encoding="utf-8",
            )
            paths[group_label] = out_path

    return GroupedPublications(by_group=grouped_lists, paths=paths)


def aggregate_cross_group_data(
    records: Iterable[dict],
    *,
    filter: Optional[str] = None,
) -> GroupedPublications:
    """Construct a :class:`GroupedPublications` from cross-group collaboration records.

    Parameters
    ----------
    records
        Iterable of dicts as returned by :func:`cooplot.metrics.cross_group_publications`
        (or compatible structure) where each record contains ``title``, ``norm_title``,
        ``year``, ``groups`` and ``authors`` entries.
    filter
        Optional string selecting records that contain identifiers. Supported values are
        ``"doi"``, ``"pubmed"`` (or ``"pubmed_id"`` / ``"pmid"``), and ``"identifier"``
        (alias ``"any"``) which keeps entries having either DOI or PubMed identifiers.
        When ``None`` (default) no filtering is applied.
    """

    filter_normalized = (filter or "").strip().lower()
    if filter_normalized and filter_normalized not in {
        "doi",
        "pubmed",
        "pubmed_id",
        "pmid",
        "identifier",
        "any",
    }:
        raise ValueError(
            "filter must be one of None, 'doi', 'pubmed', 'pubmed_id', 'pmid', "
            "'identifier', or 'any'",
        )

    def _has_doi(record: dict) -> bool:
        doi = record.get("doi") or record.get("DOI")
        return bool(isinstance(doi, str) and doi.strip())

    def _has_pubmed(record: dict) -> bool:
        pmid = record.get("pubmed_id") or record.get("pmid")
        return bool(isinstance(pmid, str) and pmid.strip())

    def _passes_filter(record: dict) -> bool:
        if not filter_normalized:
            return True
        if filter_normalized == "doi":
            return _has_doi(record)
        if filter_normalized in {"pubmed", "pubmed_id", "pmid"}:
            return _has_pubmed(record)
        if filter_normalized in {"identifier", "any"}:
            return _has_doi(record) or _has_pubmed(record)
        return True

    by_group: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            continue
        if not _passes_filter(record):
            continue
        title = record.get("title")
        norm_title = record.get("norm_title")
        if not isinstance(norm_title, str) or not norm_title.strip():
            continue
        year = record.get("year")
        authors_by_group = record.get("authors") or {}
        groups = record.get("groups") or list(authors_by_group.keys())
        if not groups:
            continue
        base = {
            "title": title or norm_title,
            "norm_title": norm_title,
            "year": year if isinstance(year, int) else None,
        }
        for group in groups:
            if not isinstance(group, str):
                continue
            group_name = group.strip()
            if not group_name:
                continue
            authors = authors_by_group.get(group) or authors_by_group.get(
                group_name, []
            )
            if not isinstance(authors, list):
                authors = list(authors)  # tolerate iterables/sets
            filtered_authors = [
                author
                for author in authors
                if isinstance(author, str) and author.strip()
            ]
            formatted = dict(base)
            formatted["authors"] = sorted(filtered_authors, key=_lastname)
            by_group[group_name].append(formatted)

    grouped_lists = {
        group: sorted(pubs, key=_publication_sort_key)
        for group, pubs in by_group.items()
    }
    return GroupedPublications(by_group=grouped_lists, paths={})
