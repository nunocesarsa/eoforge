"""Match the bands a model needs to the bands of an image.

Band names in cards use Sentinel-2 style (B02, B03, B04, B08, B8A, B11, B12 ...). Image band
descriptions are matched after normalising common spellings: B2 = B02 = blue, NIR = B08,
nir_narrow = B8A, swir1 = B11 ... Prefixed names such as 'dry_B04' (seasonal stacks made by
fetch_landcovernet.py) are matched with a chosen prefix.
"""
from __future__ import annotations

import re

ALIASES = {
    "COASTAL": "B01", "AEROSOL": "B01",
    "BLUE": "B02", "GREEN": "B03", "RED": "B04",
    "REDEDGE1": "B05", "RE1": "B05", "REDEDGE2": "B06", "RE2": "B06", "REDEDGE3": "B07", "RE3": "B07",
    "NIR": "B08", "NIRBROAD": "B08", "NIRNARROW": "B8A", "NARROWNIR": "B8A", "NIR08": "B8A", "RE4": "B8A",
    "WATERVAPOUR": "B09", "WATERVAPOR": "B09", "CIRRUS": "B10",
    "SWIR1": "B11", "SWIR16": "B11", "SWIR2": "B12", "SWIR22": "B12",
}


def canonical(name: str) -> str:
    """'b2' -> 'B02', 'B8a' -> 'B8A', 'nir_narrow' -> 'B8A', 'Red' -> 'B04'."""
    n = re.sub(r"[\s_\-.]", "", str(name)).upper()
    if n in ALIASES:
        return ALIASES[n]
    m = re.fullmatch(r"B0?(\d{1,2})(A?)", n)
    if m:
        return f"B{int(m.group(1)):02d}{m.group(2)}" if not m.group(2) else f"B{int(m.group(1))}A"
    return n


def split_prefix(desc: str) -> tuple[str, str]:
    """'dry_B04' -> ('dry', 'B04'); 'B04' -> ('', 'B04')."""
    m = re.match(r"^(.*?)[_\-]?(B0?\d{1,2}A?|blue|green|red|nir\w*|swir\w*|re\d|rededge\d)$", desc or "", re.I)
    if m and m.group(1):
        return m.group(1).rstrip("_-"), canonical(m.group(2))
    return "", canonical(desc or "")


def prefixes(descriptions: list[str]) -> list[str]:
    seen = []
    for d in descriptions:
        p, _ = split_prefix(d)
        if p not in seen:
            seen.append(p)
    return seen


def match_bands(required: list[str], descriptions: list[str], prefix: str | None = None):
    """Return (indices, missing). indices are 1-based band numbers in the order of `required`.

    Bands without descriptions are not matched by name; the caller (app/CLI) can set them by hand.
    """
    table = {}
    for i, d in enumerate(descriptions, 1):
        if not d:
            continue
        p, c = split_prefix(d)
        if prefix is None or p == prefix:
            table.setdefault(c, i)
    idx, missing = [], []
    for r in required:
        c = canonical(r)
        if c in table:
            idx.append(table[c])
        else:
            idx.append(None)
            missing.append(r)
    return idx, missing


def parse_band_list(text: str) -> list[int]:
    """'2,3,4,8' -> [2, 3, 4, 8] (1-based)."""
    return [int(x) for x in re.split(r"[,\s]+", text.strip()) if x]


S2_L1C_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
S2_L2A_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def guess_descriptions(descriptions: list[str], required: list[str] | None = None):
    """Names for images whose bands have no descriptions. Returns (names, note or None).

    13 bands = all Sentinel-2 L1C bands in order (e.g. Sen1Floods11), 12 = L2A without B10;
    an image with exactly as many bands as the model needs is taken to be in the model's order
    (e.g. Fields of The World chips: B04, B03, B02, B08).
    """
    if any(descriptions):
        return list(descriptions), None
    n = len(descriptions)
    if n == 13:
        return list(S2_L1C_ORDER), "Bands have no names; assumed the 13 Sentinel-2 L1C bands in order (B01 ... B12)."
    if n == 12:
        return list(S2_L2A_ORDER), "Bands have no names; assumed the 12 Sentinel-2 L2A bands in order (B01 ... B12, no B10)."
    if required and n == len(required):
        return [canonical(r) for r in required], ("Bands have no names; assumed they are in the model's order ("
                                                  + ", ".join(required) + ").")
    return list(descriptions), None
