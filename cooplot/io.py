from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple


def read_rows_with_header(
    path: str | Path, delimiter: str = ";"
) -> Tuple[List[str], List[Dict[str, str]]]:
    p = Path(path)
    with p.open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter=delimiter)
        header = next(r, None)
        if not header:
            raise ValueError("CSV has no header row.")
        header = [h.strip() for h in header]
        rows: List[Dict[str, str]] = []
        for row in r:
            if not row:
                continue
            # pad/truncate
            row = (row + [""] * len(header))[: len(header)]
            rows.append({h: v.strip() for h, v in zip(header, row)})
    if not rows:
        raise ValueError("CSV has no data rows.")
    return header, rows
