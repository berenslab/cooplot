from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_window(window: str) -> Tuple[int, int]:
    w = window.strip()
    if "-" in w:
        a, b = w.split("-", 1)
        return int(a), int(b)
    # single year shorthand
    y = int(w)
    return y, y


def _lastname(name: str) -> str:
    return name.split()[-1].lower()


def _titles_in_window(pubs: List[dict], lo: int, hi: int) -> set[str]:
    titles = set()
    for p in pubs:
        y = p.get("year")
        if isinstance(y, int) and lo <= y <= hi:
            t = (p.get("norm_title") or "").strip()
            if t:
                titles.add(t)
    return titles


def _normalize_group(value: Optional[str]) -> str:
    if value is None:
        return "Unlabeled"
    g = str(value).strip()
    return g if g else "Unlabeled"


def _prepare_labels_and_groups(
    people: List[dict], name_col: str, group_col: Optional[str]
) -> Tuple[List[str], Dict[str, str]]:
    labels = sorted([p[name_col] for p in people], key=_lastname)
    group_map: Dict[str, str] = {}
    if group_col:
        for person in people:
            name = person.get(name_col)
            if not isinstance(name, str):
                continue
            group_map[name] = _normalize_group(person.get(group_col))
    return labels, group_map


def _titles_for_windows(
    publications_by_author: Dict[str, List[dict]],
    labels: List[str],
    windows: List[str],
) -> Dict[str, Dict[str, set[str]]]:
    title_sets_by_window: Dict[str, Dict[str, set[str]]] = {}
    for win in windows:
        lo, hi = parse_window(win)
        title_sets_by_window[win] = {
            name: _titles_in_window(publications_by_author.get(name, []), lo, hi)
            for name in labels
        }
    return title_sets_by_window


def build_matrices(
    publications_by_author: Dict[str, List[dict]],
    people: List[dict],
    windows: List[str],
    *,
    name_col: str = "name",
    group_col: Optional[str] = None,
) -> Dict[str, dict]:
    """Return co-authorship matrices for each window using author-level data.

    The result is ``{window: {"labels": [...], "matrix": [[...]]}}`` with
    labels ordered by alphabetical last name. When ``group_col`` is provided the
    output also includes a ``label_to_group`` mapping so callers can color the
    visualization by group.
    """

    labels, group_map = _prepare_labels_and_groups(people, name_col, group_col)
    mats: Dict[str, dict] = {}

    title_sets_by_window = _titles_for_windows(publications_by_author, labels, windows)

    for win in windows:
        title_sets = title_sets_by_window[win]
        n = len(labels)
        M = np.zeros((n, n), dtype=int)
        for i in range(n):
            ti = title_sets[labels[i]]
            for j in range(i, n):
                tj = title_sets[labels[j]]
                M[i, j] = M[j, i] = len(ti & tj)
        np.fill_diagonal(M, 0)

        entry = {"labels": labels, "matrix": M.tolist()}
        if group_col:
            entry["label_to_group"] = {
                label: group_map.get(label, "Unlabeled") for label in labels
            }
        mats[win] = entry

    return mats
