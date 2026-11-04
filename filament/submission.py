"""Write and verify the competition CSV (filament_id, segmentation_rle)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .rle import decode_counts, encode_counts

HEADER = "filament_id,segmentation_rle"


def rows_for_image(stem: str, instances) -> list[tuple[str, str]]:
    """`<stem>_<k>` ids starting at 1 (the format in the host's example), one row per instance."""
    return [(f"{stem}_{k}", encode_counts(m)) for k, m in enumerate(instances, 1)]


def write_submission(rows: list[tuple[str, str]], path: Path) -> None:
    """Write raw lines.

    COCO RLE counts use ASCII 48..111, which may include a backslash but never ',' or
    quotes, so no CSV quoting/escaping is needed (the host forbids quotes, and an
    escapechar would corrupt the backslashes).
    """
    for fid, counts in rows:
        if "," in counts or '"' in counts or "'" in counts or "_" not in fid:
            raise ValueError(f"unexpected characters in row {fid}")
    with open(path, "w", newline="") as fh:
        fh.write(HEADER + "\n")
        fh.writelines(f"{fid},{counts}\n" for fid, counts in rows)


def verify_submission(path: Path, rows: list[tuple[str, str]], expected_stems: set[str],
                      n_decode: int = 5) -> pd.DataFrame:
    """Read the file back the way the host does (pandas defaults) and check it matches."""
    sub = pd.read_csv(path)
    if list(sub.columns) != HEADER.split(","):
        raise RuntimeError(f"bad header {list(sub.columns)}")
    if len(sub) != len(rows) or sub["segmentation_rle"].tolist() != [r[1] for r in rows]:
        raise RuntimeError("CSV round trip changed the RLE strings")
    stems = sub["filament_id"].str.split("_", n=1).str[0]   # the host parses on the FIRST '_'
    if not stems.isin(expected_stems).all():
        raise RuntimeError("filament_id does not start with a known image id")
    if sub["filament_id"].duplicated().any():
        raise RuntimeError("duplicate filament_id")
    for counts, (_, orig) in zip(sub["segmentation_rle"].head(n_decode), rows[:n_decode]):
        if not np.array_equal(decode_counts(counts), decode_counts(orig)):
            raise RuntimeError("decoded mask differs after CSV round trip")
    return sub
