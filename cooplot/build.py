from __future__ import annotations

from typing import Dict, List, Tuple

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


def build_matrices(
    publications_by_author: Dict[str, List[dict]],
    people: List[dict],
    windows: List[str],
    *,
    name_col: str = "name",
) -> Dict[str, dict]:
    """
    Returns {window: {"labels": [...], "matrix": [[...],[...],...]}}
    Ordering is stable alphabetical by last name.
    """
    labels = sorted([p[name_col] for p in people], key=_lastname)
    mats: Dict[str, dict] = {}

    for win in windows:
        lo, hi = parse_window(win)
        title_sets = {
            name: _titles_in_window(publications_by_author.get(name, []), lo, hi)
            for name in labels
        }
        n = len(labels)
        M = np.zeros((n, n), dtype=int)
        for i in range(n):
            ti = title_sets[labels[i]]
            for j in range(i, n):
                tj = title_sets[labels[j]]
                M[i, j] = M[j, i] = len(ti & tj)
        np.fill_diagonal(M, 0)
        mats[win] = {"labels": labels, "matrix": M.tolist()}
    return mats
