from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import requests
from scholarly import scholarly
from tqdm import tqdm


@dataclass(frozen=True)
class Publications:
    """Container for author-level publication data."""

    by_author: Dict[str, List[dict]]
    people: List[dict]
    name_col: str = "name"
    group_col: Optional[str] = None

    @classmethod
    def from_data(
        cls,
        publications_by_author: Dict[str, List[dict]]
        | Iterable[tuple[str, List[dict]]],
        people: Iterable[dict],
        *,
        name_col: str = "name",
        group_col: Optional[str] = None,
    ) -> "Publications":
        return cls(
            by_author=dict(publications_by_author),
            people=list(people),
            name_col=name_col,
            group_col=group_col,
        )

    def mapping(self) -> Dict[str, List[dict]]:
        return self.by_author

    def people_rows(self) -> List[dict]:
        return self.people

    def resolve_name_column(self, fallback: str = "name") -> str:
        return self.name_col or fallback

    def resolve_group_column(self, fallback: Optional[str] = None) -> Optional[str]:
        return self.group_col if self.group_col is not None else fallback

    def exclude_authors(self, names: Iterable[str]) -> "Publications":
        """Return a new instance excluding the specified authors."""

        to_remove: Set[str] = {str(name).strip() for name in names if name is not None}
        to_remove.discard("")
        if not to_remove:
            return self

        filtered_by_author = {
            author: pubs
            for author, pubs in self.by_author.items()
            if author not in to_remove
        }
        resolved_name_col = self.resolve_name_column("name")
        filtered_people = [
            row
            for row in self.people
            if str(row.get(resolved_name_col, "")).strip() not in to_remove
        ]
        return Publications(
            by_author=filtered_by_author,
            people=filtered_people,
            name_col=self.name_col,
            group_col=self.group_col,
        )


_RE_SPACES = re.compile(r"\s+")
_RE_PUNCT = re.compile(r"[^\w\s]")


def normalize_title(title: str, drop_subtitle: bool = False) -> str:
    t = (title or "").strip().lower()
    if drop_subtitle and ":" in t:
        t = t.split(":", 1)[0]
    t = _RE_PUNCT.sub(" ", t)
    t = _RE_SPACES.sub(" ", t).strip()
    return t


def normalize_year(year) -> Optional[int]:
    try:
        y = int(str(year))
        return y if 1800 <= y <= 2100 else None
    except Exception:
        return None


def _is_blank(s: Optional[str]) -> bool:
    if s is None:
        return True
    s = str(s).strip().lower()
    return s in {"", "none", "nan", "na", "n/a", "null"}


def _fetch_google_scholar_pubs(
    scholar_id: str, *, sleep_range=(1.0, 2.5)
) -> List[dict]:
    for attempt in range(5):
        try:
            author = scholarly.search_author_id(scholar_id)
            author = scholarly.fill(author, sections=["publications"])
            pubs = []
            for p in author.get("publications", []):
                bib = p.get("bib", {})
                title = bib.get("title") or ""
                year = normalize_year(bib.get("pub_year"))
                if title.strip():
                    pubs.append({"title": title, "year": year})
            time.sleep(random.uniform(*sleep_range))
            return pubs
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return []


def _fetch_semantic_scholar_pubs(semantic_id: str, *, sleep_s=0.5) -> List[dict]:
    url = f"https://api.semanticscholar.org/graph/v1/author/{semantic_id}"
    params = {"fields": "papers.title,papers.year"}
    headers = {"User-Agent": "cooplot/0.2 (+https://example.com)"}
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            if r.status_code == 200:
                data = r.json()
                out = []
                for p in data.get("papers", []):
                    title = (p.get("title") or "").strip()
                    year = normalize_year(p.get("year"))
                    if title:
                        out.append({"title": title, "year": year})
                time.sleep(sleep_s)
                return out
        except Exception:
            pass
        time.sleep(sleep_s * (attempt + 1))
    return []


def scrape_all(
    people: List[dict],
    *,
    name_col: str = "name",
    scholar_col: str = "scholar_id",
    semantic_col: str = "semantic_id",
    cache_dir: str | Path = ".cache/cooplot",
    drop_subtitle: bool = False,
    fallback_semantic_if_empty: bool = False,
    group_col: str | None = None,
) -> Publications:
    """
    Returns a :class:`Publications` instance containing the normalized
    author-to-publication mapping.
    Strategy:
      - If Google Scholar ID present → use GS.
      - Else if Semantic ID present → use S2.
      - If `fallback_semantic_if_empty` and GS returned 0 pubs → try S2.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out: Dict[str, List[dict]] = {}
    people_rows = list(people)

    for p in tqdm(people_rows, desc="Scraping"):
        name = p[name_col]
        gs_id = p.get(scholar_col, "")
        s2_id = p.get(semantic_col, "")

        cache_file = cache / f"{name}.json"
        if cache_file.exists():
            pubs = json.loads(cache_file.read_text(encoding="utf-8"))
            out[name] = pubs
            continue

        pubs: List[dict] = []
        used_source = None

        if not _is_blank(gs_id):
            pubs = _fetch_google_scholar_pubs(gs_id)
            used_source = "gs"

        if (used_source is None and not _is_blank(s2_id)) or (
            fallback_semantic_if_empty
            and used_source == "gs"
            and len(pubs) == 0
            and not _is_blank(s2_id)
        ):
            pubs = _fetch_semantic_scholar_pubs(s2_id)
            used_source = "s2"

        # normalize + dedupe titles within person
        seen = set()
        normed: List[dict] = []
        for item in pubs:
            raw = item["title"]
            year = item.get("year")
            nt = normalize_title(raw, drop_subtitle=drop_subtitle)
            key = (nt, year)
            if nt and key not in seen:
                seen.add(key)
                normed.append({"title": raw, "norm_title": nt, "year": year})

        cache_file.write_text(
            json.dumps(normed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        out[name] = normed

    return Publications(
        by_author=out,
        people=people_rows,
        name_col=name_col,
        group_col=group_col,
    )
