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


def build_matrices(
    publications_by_author: Dict[str, List[dict]],
    people: List[dict],
    windows: List[str],
    *,
    name_col: str = "name",
    group_col: Optional[str] = None,
    aggregate_groups: bool = False,
) -> Dict[str, dict]:
    """
    Returns {window: {"labels": [...], "matrix": [[...],[...],...]}}
    Ordering is stable alphabetical by last name.
    """
    labels = sorted([p[name_col] for p in people], key=_lastname)
    group_map: Dict[str, str] = {}
    if group_col:
        for person in people:
            name = person.get(name_col)
            if not isinstance(name, str):
                continue
            group_map[name] = _normalize_group(person.get(group_col))
    if aggregate_groups and not group_map:
        raise ValueError("aggregate_groups=True requires a valid group_col with values.")
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
        if aggregate_groups:
            group_labels: List[str] = []
            group_title_sets: Dict[str, set[str]] = {}
            for label in labels:
                group_label = group_map.get(label, "Unlabeled")
                if group_label not in group_labels:
                    group_labels.append(group_label)
                titles = group_title_sets.setdefault(group_label, set())
                titles.update(title_sets[label])
            g_count = len(group_labels)
            G = np.zeros((g_count, g_count), dtype=int)
            for idx_i, group_i in enumerate(group_labels):
                titles_i = group_title_sets.get(group_i, set())
                for idx_j in range(idx_i + 1, g_count):
                    group_j = group_labels[idx_j]
                    titles_j = group_title_sets.get(group_j, set())
                    overlap = len(titles_i & titles_j)
                    if overlap <= 0:
                        continue
                    G[idx_i, idx_j] = G[idx_j, idx_i] = overlap
            mats[win] = {
                "labels": group_labels,
                "matrix": G.tolist(),
                "label_to_group": {g: g for g in group_labels},
            }
        else:
            entry = {"labels": labels, "matrix": M.tolist()}
            if group_col:
                entry["label_to_group"] = {
                    label: group_map.get(label, "Unlabeled") for label in labels
                }
            mats[win] = entry
    return mats
