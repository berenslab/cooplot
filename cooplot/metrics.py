from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

try:  # pragma: no cover - optional dependency should already be installed
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

from .aggregate import GroupedPublications
from .build import _prepare_labels_and_groups, _titles_for_windows

_UNLABELED = "Unlabeled"

_DOTENV_LOADED = False
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PubMedDetails:
    pubmed_id: Optional[str]
    doi: Optional[str]
    authors: Optional[List[str]]
    journal: Optional[str]
    title: Optional[str] = None
    year: Optional[int] = None


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


_RE_YEAR = re.compile(r"(19|20)\d{2}")


def _extract_year(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    match = _RE_YEAR.search(str(value))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


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
    out_path: str | Path | None = None,
    ensure_ascii: bool = False,
    enrich_pubmed: bool = False,
    ncbi_api_key: str | None = None,
    ncbi_email: str | None = None,
    ncbi_min_delay: Optional[float] = None,
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
    out_path
        Optional destination file. When provided, the extension determines the
        export format (``.json`` or ``.csv``).
    ensure_ascii
        Controls :func:`json.dumps(ensure_ascii=...)` when exporting to JSON.
    enrich_pubmed
        When ``True`` each record is augmented with ``pubmed_id`` and ``doi``
        values fetched from the NCBI PubMed E-utilities API. Network errors are
        ignored and represented as ``None`` values.
    ncbi_api_key
        Optional API key to include with NCBI requests. When ``None`` a value is
        resolved from the ``NCBI_API_KEY`` environment variable, allowing usage
        with ``.env`` files. When present the client respects the higher
        throughput limits supported by the service.
    ncbi_email
        Optional contact email forwarded to the NCBI endpoints per their usage
        guidelines. Falls back to the ``NCBI_EMAIL`` environment variable when
        not provided.
    ncbi_min_delay
        Minimum delay between successive NCBI requests in seconds. Defaults to
        ``0.34`` seconds without an API key or ``0.11`` seconds with a key.

    Returns
    -------
    List[dict]
        Each dict contains ``title``, ``norm_title``, ``year``, ``groups`` and an
        ``authors`` mapping keyed by group name. When ``enrich_pubmed`` is ``True``
        the records also provide ``pubmed_id``, ``doi``, ``pubmed_authors`` (NCBI
        author order) and ``pubmed_journal``.
    """

    grouped = _ensure_group_mapping(grouped_publications)

    out_file: Optional[Path] = None
    if out_path is not None:
        out_file = Path(out_path)
        if out_file.exists():
            existing = _load_cross_group_records(out_file)
            existing.sort(key=_publication_sort_key)
            return existing

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
                if (
                    year_from is not None or year_to is not None
                ) and not include_missing_year:
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
            int_counts = [
                (year, freq) for year, freq in counts.items() if isinstance(year, int)
            ]
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

    if enrich_pubmed and results:
        _ensure_env_loaded()
        resolved_api_key = ncbi_api_key or os.getenv("NCBI_API_KEY")
        resolved_email = ncbi_email or os.getenv("NCBI_EMAIL")
        lookup = _PubMedLookup(
            api_key=resolved_api_key,
            email=resolved_email,
            min_delay=ncbi_min_delay,
        )
        cache: Dict[Tuple[str, Optional[int]], _PubMedDetails] = {}
        empty_details = _PubMedDetails(None, None, None, None)
        for record in results:
            key = (record["norm_title"], record["year"])
            details = cache.get(key)
            if details is None:
                year_value = record.get("year")
                titles_to_try: List[str] = []
                title_value = record.get("title")
                if isinstance(title_value, str) and title_value.strip():
                    titles_to_try.append(title_value.strip())
                norm_value = record.get("norm_title")
                if (
                    isinstance(norm_value, str)
                    and norm_value.strip()
                    and norm_value.strip() not in {t.strip() for t in titles_to_try}
                ):
                    titles_to_try.append(norm_value.strip())

                for candidate in titles_to_try:
                    details = lookup.identifiers_for_title(candidate, year_value)
                    if details is not None:
                        break

                if details is None:
                    details = empty_details
                cache[key] = details
            record["pubmed_id"] = details.pubmed_id
            record["doi"] = details.doi
            record["pubmed_authors"] = details.authors
            record["pubmed_journal"] = details.journal

    if out_path is None:
        return results

    assert out_file is not None
    out_file.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_file.suffix.lower()

    if suffix == ".json":
        out_file.write_text(
            json.dumps(results, ensure_ascii=ensure_ascii, indent=2),
            encoding="utf-8",
        )
    elif suffix == ".csv":
        base_fields = ["title", "norm_title", "year", "groups", "authors"]
        include_pubmed = bool(results and "pubmed_id" in results[0])
        extra_fields = [
            "pubmed_id",
            "doi",
            "pubmed_authors",
            "pubmed_journal",
        ]
        fieldnames = base_fields + (extra_fields if include_pubmed else [])
        with out_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for record in results:
                row = {
                    "title": record["title"],
                    "norm_title": record["norm_title"],
                    "year": "" if record["year"] is None else record["year"],
                    "groups": ";".join(record["groups"]),
                    "authors": json.dumps(
                        record["authors"],
                        ensure_ascii=ensure_ascii,
                        separators=(",", ":"),
                    ),
                }
                if include_pubmed:
                    row["pubmed_id"] = record.get("pubmed_id") or ""
                    row["doi"] = record.get("doi") or ""
                    authors_list = record.get("pubmed_authors")
                    row["pubmed_authors"] = (
                        json.dumps(
                            authors_list,
                            ensure_ascii=ensure_ascii,
                            separators=(",", ":"),
                        )
                        if authors_list
                        else ""
                    )
                    row["pubmed_journal"] = record.get("pubmed_journal") or ""
                writer.writerow(row)
    else:
        raise ValueError(
            f"Unsupported export format for {out_file}. Expected .json or .csv",
        )
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


def cross_group_report(
    input_path: str | Path,
    *,
    citation_style: str = "apa",
    citation_locale: str = "en-US",
    out_path: str | Path | None = None,
    ensure_ascii: bool = False,
    pubmed_min_delay: Optional[float] = None,
    verbose: bool = False,
) -> str:
    """Generate a formatted collaboration report based on saved cross-group output.

    Parameters
    ----------
    input_path
        Path to the JSON or CSV file generated by :func:`cross_group_publications`.
    citation_style
        Citation style used when resolving DOIs via doi.org content negotiation.
        Examples include ``"apa"`` (default), ``"ieee"`` or ``"chicago-author-date"``.
    citation_locale
        Locale hint forwarded to doi.org when formatting bibliography entries.
    out_path
        Optional destination file. When provided the report text is also written
        to this location.
    ensure_ascii
        When ``True`` the generated text is normalised to ASCII before returning
        or writing to ``out_path``.
    pubmed_min_delay
        Optional override for the PubMed client rate limit delay. When ``None``
        the default behaviour (0.11s with API key, 0.34s otherwise) is used.
    verbose
        When ``True`` progress information and retrieved citations are printed to
        stdout as the report is generated.

    Returns
    -------
    str
        Multi-line report containing citations and collaboration summaries.
    """

    path = Path(input_path)
    records = _load_cross_group_records(path)
    if not records:
        report = ""
    else:
        printer: Optional[Callable[[str], None]] = print if verbose else None
        fetcher = _CitationFetcher(
            style=citation_style,
            locale=citation_locale,
            pubmed_min_delay=pubmed_min_delay,
            verbose=verbose,
            printer=printer,
        )
        entries: List[str] = []
        total = len(records)
        for index, record in enumerate(records, start=1):
            title = record.get("title") or record.get("norm_title") or "Untitled"
            doi = _clean_doi(record.get("doi") or record.get("DOI"))
            pmid = record.get("pubmed_id") or record.get("pmid")
            if not doi or not pmid:
                if printer:
                    printer(
                        f"Skipping record {index}/{total}: '{title}' (missing DOI or PubMed ID)",
                    )
                continue
            if printer:
                printer(f"Processing record {index}/{total}: {title}")
            citation = fetcher.citation_for(record)
            if not citation:
                citation = record.get("title") or record.get("norm_title") or "Untitled"
                if printer:
                    printer(f"Citation unavailable for '{title}'; using fallback text.")
            elif printer:
                printer(f"Retrieved citation for '{title}': {citation}")
            collab = _format_collaboration_line(record)
            entries.append(f"{citation}\n{collab}")
        report = "\n\n".join(entries)

    if ensure_ascii and report:
        report = report.encode("ascii", "ignore").decode("ascii")

    if out_path is not None:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        encoding = "ascii" if ensure_ascii else "utf-8-sig"
        out_file.write_text(report, encoding=encoding)

    return report


def _load_cross_group_records(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [_normalise_record(rec) for rec in data]
    if suffix == ".csv":
        records: List[dict] = []
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                record = {
                    "title": row.get("title", ""),
                    "norm_title": row.get("norm_title", ""),
                    "year": _safe_int(row.get("year")),
                    "groups": _split_groups(row.get("groups")),
                    "authors": _normalise_author_map(row.get("authors")),
                    "doi": row.get("doi") or None,
                    "pubmed_id": row.get("pubmed_id") or None,
                }
                pubmed_authors_raw = row.get("pubmed_authors")
                if pubmed_authors_raw:
                    try:
                        record["pubmed_authors"] = json.loads(pubmed_authors_raw)
                    except json.JSONDecodeError:
                        record["pubmed_authors"] = []
                pubmed_journal = row.get("pubmed_journal")
                if pubmed_journal:
                    record["pubmed_journal"] = pubmed_journal
                records.append(record)
        return records
    raise ValueError(f"Unsupported input format for {path}. Expected .json or .csv")


def _normalise_record(record: dict) -> dict:
    normalised = dict(record)
    normalised["year"] = _safe_int(record.get("year"))
    normalised["groups"] = list(record.get("groups", []))
    normalised["authors"] = _normalise_author_map(record.get("authors"))
    return normalised


def _split_groups(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _safe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        ivalue = int(str(value).strip())
        return ivalue
    except Exception:
        return None


def _normalise_author_map(raw: Optional[dict | str]) -> Dict[str, List[str]]:
    if raw is None:
        return {}
    data = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(data, dict):
        return {}
    normalised: Dict[str, List[str]] = {}
    for group, authors in data.items():
        if isinstance(authors, (list, tuple, set)):
            normalised[str(group)] = [str(name) for name in authors]
    return normalised


def _format_collaboration_line(record: dict) -> str:
    groups = record.get("groups") or []
    if not groups:
        return "Collaborated between (no groups listed)."
    authors_by_group = record.get("authors") or {}
    segments: List[str] = []
    for group in groups:
        names = authors_by_group.get(group) or []
        if names:
            segments.append(f"{group} ({', '.join(names)})")
        else:
            segments.append(group)
    return f"Collaborated between {_format_series(segments)}."


def _format_series(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _clean_doi(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    doi = str(raw).strip()
    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "http://dx.doi.org/",
        "https://dx.doi.org/",
    ]
    for prefix in prefixes:
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix) :]
            break
    if doi.lower().startswith("doi:"):
        doi = doi[4:].strip()
    return doi or None


def _format_summary_citation(details: _PubMedDetails) -> str:
    parts: List[str] = []
    if details.authors:
        parts.append(", ".join(details.authors))
    if details.year:
        parts.append(f"({details.year}).")
    if details.title:
        title = details.title.rstrip(".")
        parts.append(f"{title}.")
    if details.journal:
        parts.append(f"{details.journal}.")
    if details.pubmed_id:
        parts.append(f"https://pubmed.ncbi.nlm.nih.gov/{details.pubmed_id}/")
    return " ".join(parts).strip()


class _CitationFetcher:
    def __init__(
        self,
        *,
        style: str,
        locale: str,
        pubmed_min_delay: Optional[float],
        verbose: bool,
        printer: Optional[Callable[[str], None]],
    ) -> None:
        self.style = style
        self.locale = locale
        self._session = requests.Session()
        self._pubmed = _PubMedLookup(min_delay=pubmed_min_delay)
        self._doi_cache: Dict[str, Optional[str]] = {}
        self._pubmed_cache: Dict[str, Optional[str]] = {}
        self._printer = printer if verbose else None
        self._verbose = verbose

    def _emit(self, message: str) -> None:
        if self._printer is not None:
            try:
                self._printer(message)
            except Exception:
                pass

    def citation_for(self, record: dict) -> Optional[str]:
        doi = _clean_doi(record.get("doi") or record.get("DOI"))
        if doi:
            citation = self._fetch_via_doi(doi)
            if citation:
                return citation
        pmid = record.get("pubmed_id") or record.get("pmid")
        if pmid:
            citation = self._fetch_from_pubmed(str(pmid))
            if citation:
                return citation
        return None

    def _fetch_via_doi(self, doi: Optional[str]) -> Optional[str]:
        clean = _clean_doi(doi)
        if not clean:
            return None
        cache_key = clean.lower()
        if cache_key in self._doi_cache:
            return self._doi_cache[cache_key]
        headers = {
            "Accept": f"text/x-bibliography; style={self.style}; locale={self.locale}",
        }
        url = f"https://doi.org/{quote(clean, safe='/')}"
        try:
            response = self._session.get(url, headers=headers, timeout=20)
            if response.status_code == 200:
                citation = response.content.decode("utf-8", errors="replace").strip()
                self._doi_cache[cache_key] = citation
                if self._printer:
                    self._emit(f"Resolved DOI {clean} via doi.org")
                return citation
        except Exception:
            pass
        if self._printer:
            self._emit(f"Failed to resolve DOI {clean} via doi.org")
        self._doi_cache[cache_key] = None
        return None

    def _fetch_from_pubmed(self, pmid: str) -> Optional[str]:
        if not pmid:
            return None
        if pmid in self._pubmed_cache:
            return self._pubmed_cache[pmid]
        details = self._pubmed.summary_by_pmid(pmid)
        if details is None:
            self._pubmed_cache[pmid] = None
            if self._printer:
                self._emit(f"Unable to retrieve PubMed summary for PMID {pmid}")
            return None
        doi = _clean_doi(details.doi)
        if doi:
            citation = self._fetch_via_doi(doi)
            if citation:
                self._pubmed_cache[pmid] = citation
                return citation
        citation = _format_summary_citation(details)
        self._pubmed_cache[pmid] = citation
        if self._printer:
            self._emit(f"Formatted citation from PubMed summary for PMID {pmid}")
        return citation


class _PubMedLookup:
    """Lightweight PubMed helper that follows NCBI rate limits."""

    _SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    _SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        email: str | None = None,
        min_delay: Optional[float] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        _ensure_env_loaded()
        env_api_key = os.getenv("NCBI_API_KEY") or os.getenv("NCIB_API_KEY")
        self.api_key = api_key if api_key is not None else env_api_key
        self.email = email if email is not None else os.getenv("NCBI_EMAIL")
        default_delay = 0.11 if self.api_key else 0.34
        self.min_delay = default_delay if min_delay is None else max(min_delay, 0.0)
        self._last_request = 0.0
        self._session = session or requests.Session()
        if (
            api_key is None
            and not os.getenv("NCBI_API_KEY")
            and os.getenv("NCIB_API_KEY")
        ):
            _LOG.warning(
                "NCIB_API_KEY environment variable detected; please rename to NCBI_API_KEY for consistency",
            )
        if self.api_key:
            _LOG.info(
                "PubMed lookup will use NCBI API key with min_delay=%s", self.min_delay
            )
        else:
            _LOG.info(
                "PubMed lookup running without NCBI API key; min_delay=%s",
                self.min_delay,
            )

    def identifiers_for_title(
        self,
        title: Optional[str],
        year: Optional[int],
    ) -> _PubMedDetails | None:
        if not title:
            return None
        try:
            attempts: List[Tuple[str, Optional[int]]] = []
            attempts.append((title, year))
            if year is not None:
                attempts.append((title, None))
            seen: set[Tuple[str, Optional[int]]] = set()
            for attempt_title, attempt_year in attempts:
                key = (attempt_title, attempt_year)
                if key in seen:
                    continue
                seen.add(key)
                pubmed_id = self._search_pubmed(attempt_title, attempt_year)
                if not pubmed_id:
                    continue
                summary = self._fetch_summary(pubmed_id)
                if summary is not None:
                    return summary
        except Exception:
            return None
        return None

    def _search_pubmed(self, title: str, year: Optional[int]) -> Optional[str]:
        clean_title = title.replace('"', " ").strip()
        if not clean_title:
            return None

        def _build_params(term: str, use_title_field: bool) -> Dict[str, str]:
            params: Dict[str, str] = {
                "db": "pubmed",
                "retmode": "json",
                "retmax": "1",
                "sort": "relevance",
                "term": term,
            }
            if use_title_field:
                params["field"] = "ti"
            if year is not None:
                params["mindate"] = str(year)
                params["maxdate"] = str(year)
            return params

        queries = [
            _build_params(f'"{clean_title}"', True),
            _build_params(clean_title, True),
            _build_params(clean_title, False),
        ]

        for params in queries:
            data = self._request_json(self._SEARCH_URL, params)
            idlist = data.get("esearchresult", {}).get("idlist", [])
            if idlist:
                return idlist[0]

        if year is not None:
            # Final fallback: drop the year restriction entirely.
            params = _build_params(clean_title, True)
            params.pop("mindate", None)
            params.pop("maxdate", None)
            data = self._request_json(self._SEARCH_URL, params)
            idlist = data.get("esearchresult", {}).get("idlist", [])
            if idlist:
                return idlist[0]

        return None

    def _fetch_summary(self, pubmed_id: str) -> _PubMedDetails | None:
        params = {
            "db": "pubmed",
            "retmode": "json",
            "id": pubmed_id,
        }
        data = self._request_json(self._SUMMARY_URL, params)
        result = data.get("result", {})
        record = result.get(pubmed_id)
        if not isinstance(record, dict):
            return None
        doi: Optional[str] = None
        article_ids = record.get("articleids", [])
        if isinstance(article_ids, list):
            for item in article_ids:
                if not isinstance(item, dict):
                    continue
                if item.get("idtype") == "doi":
                    value = item.get("value")
                    if isinstance(value, str) and value.strip():
                        doi = value.strip()
                        break

        raw_authors = record.get("authors", [])
        authors: List[str] = []
        if isinstance(raw_authors, list):
            for author in raw_authors:
                if isinstance(author, dict):
                    name = author.get("name")
                    if isinstance(name, str) and name.strip():
                        authors.append(name.strip())

        journal = record.get("fulljournalname") or record.get("source")
        if isinstance(journal, str):
            journal = journal.strip() or None
        else:
            journal = None

        title = record.get("title")
        if isinstance(title, str):
            title = title.strip() or None
        else:
            title = None

        pubdate = record.get("pubdate") or record.get("sortpubdate")
        year = _extract_year(pubdate)

        return _PubMedDetails(
            pubmed_id=pubmed_id,
            doi=doi,
            authors=authors or None,
            journal=journal,
            title=title,
            year=year,
        )

    def summary_by_pmid(self, pubmed_id: Optional[str]) -> _PubMedDetails | None:
        if not pubmed_id:
            return None
        try:
            return self._fetch_summary(str(pubmed_id))
        except Exception:
            return None

    def _request_json(self, url: str, params: Dict[str, str]) -> dict:
        if self.api_key:
            params.setdefault("api_key", self.api_key)
        if self.email:
            params.setdefault("email", self.email)
        retries = 3
        for attempt in range(retries):
            self._respect_rate_limit()
            response = self._session.get(url, params=params, timeout=15)
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:  # pragma: no cover - network dependent
                status = exc.response.status_code if exc.response else None
                if status == 429 and attempt < retries - 1:
                    time.sleep(max(self.min_delay, 1.0))
                    continue
                raise
            data = response.json()
            if isinstance(data, dict) and "error" in data:
                message = str(data.get("error", ""))
                if "rate limit" in message.lower() and attempt < retries - 1:
                    # Back off a bit longer before retrying
                    time.sleep(max(self.min_delay, 0.5))
                    continue
            return data
        return data

    def _respect_rate_limit(self) -> None:
        if self.min_delay <= 0:
            self._last_request = time.monotonic()
            return
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self._last_request = time.monotonic()


def _ensure_env_loaded() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    if load_dotenv is not None:
        try:
            cwd_env = Path(".env")
            if cwd_env.exists():
                load_dotenv(dotenv_path=cwd_env)
            else:
                load_dotenv()
        except Exception:
            pass
    _DOTENV_LOADED = True
